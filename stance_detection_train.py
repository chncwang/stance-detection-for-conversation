import datetime
import random
import check_point
import math
import sys
import classifier
import torch
import torchtext
import configs
import dataset
import itertools
import collections
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
import logging
import log_config
import os
import utils
import imp

logger = utils.getLogger(__file__)

hyper_params = imp.load_source("module.name", sys.argv[1])

def printHyperParams():
    logger.info("batch_size:%d", hyper_params.batch_size)
    logger.info("seed:%d", hyper_params.seed)
    logger.info("embedding_tuning:%r", hyper_params.embedding_tuning)
    logger.info("min_freq:%d", hyper_params.min_freq)
    logger.info("word_dim:%d", hyper_params.word_dim)
    logger.info("hidden_dim:%d", hyper_params.hidden_dim)
    logger.info("dropout:%f", hyper_params.dropout)
    logger.info("learning_rate:%f", hyper_params.learning_rate)
    logger.info("weight_decay:%f", hyper_params.weight_decay)
    logger.info("clip_grad:%f", hyper_params.clip_grad)
    logger.info("layer:%d", hyper_params.layer)

printHyperParams()

torch.manual_seed(hyper_params.seed)

posts = dataset.readConversationSentences("/var/wqs/weibo_dialogue/posts-bpe")
responses = dataset.readConversationSentences("/var/wqs/weibo_dialogue/responses-bpe")

def readSamples(path):
    return dataset.readSamples(path, posts, responses)

def maxLen(samples):
    return max([len(x.post) + len(x.response) for x in samples]) + 1

training_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/overall_filtered_train")
dev_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/overall_filtered_dev")
test_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/overall_filtered_test")
g_max_len = max([maxLen(training_samples), maxLen(dev_samples), maxLen(test_samples)])
logger.info("max len of the whole dataset:%d", g_max_len)

to_build_vocb_samples = None
if hyper_params.embedding_tuning:
    to_build_vocb_samples = training_samples
else:
    to_build_vocb_samples = training_samples + dev_samples + test_samples

def sentenceToCounter(sentence, counter):
    words = sentence.split(" ")
    return counter.update(words)

counter = collections.Counter()
for idx, sample in enumerate(to_build_vocb_samples):
    sentenceToCounter(sample.post + " <sep> " + sample.response, counter)

def oovCount(sentence, counter):
    words = sentence.split(" ")
    oov_count = 0
    for word in words:
        if counter[word] < hyper_params.min_freq:
            oov_count += 1
    return oov_count, len(words)

oov_count = 0
all_words_count = 0
for idx, sample in enumerate(to_build_vocb_samples):
    t = oovCount(sample.post + " <sep> " + sample.response, counter)
    oov_count += t[0]
    all_words_count += t[1]

logger.info("oov:%f", oov_count / float(all_words_count))
word_vectors = torchtext.vocab.Vectors("/var/wqs/cn_embeddings/sgns.weibo.bigram-char")

if not hyper_params.embedding_tuning:
    for k in counter.keys():
        if counter[k] < hyper_params.min_freq and k in word_vectors.stoi:
            counter[k] = 10000

vocab = torchtext.vocab.Vocab(counter, min_freq = hyper_params.min_freq)
logger.info("vocab len:%d", len(vocab))
vocab.load_vectors(word_vectors)
embedding_table = nn.Embedding.from_pretrained(vocab.vectors,
        freeze = hyper_params.embedding_tuning)

def word_indexes(words, stoi):
    return [stoi[word] for word in words]

def pad_batch(word_ids_arr, lens):
    tensor = torch.ones(len(word_ids_arr), max(lens), dtype = int)
    for idx, (ids, seq_len) in enumerate(zip(word_ids_arr, lens)):
        x = torch.LongTensor(ids)
        tensor[idx, :seq_len] = x
    return tensor

def buildDataset(samples, stoi, apply_mask = 0.0):
    sentences = [s.post + " <sep> " + s.response for s in samples]
    words_arr = [s.split(" ") for s in sentences]
    sentences_indexes_arr = [word_indexes(s, stoi) for s in words_arr]
    if apply_mask > 0:
        dataset.applyLangMask(sentences_indexes_arr, stoi["<mask>"], len(vocab),
                max([len(x) for x in words_arr]), rg = random, p = apply_mask)
    sentence_lens = [len(s) for s in words_arr]
    labels = [int(s.stance) for s in samples]
    sentence_tensor = pad_batch(sentences_indexes_arr, sentence_lens)
    src_key_padding_mask = utils.srcMask(sentences_indexes_arr, sentence_lens)
    label_tensor = torch.LongTensor(labels)

    return dataset.StanceDetectionDataset(sentence_tensor, sentence_lens, src_key_padding_mask,
            label_tensor)

if g_max_len >= configs.MAX_LEN_FOR_POSITIONAL_ENCODING:
    logger.error("g_max_len:%d MAX_LEN_FOR_POSITIONAL_ENCODING:%d", g_max_len,
            configs.MAX_LEN_FOR_POSITIONAL_ENCODING)
    sys.exit(1)

model = classifier.TransformerClassifier(embedding_table,
        configs.MAX_LEN_FOR_POSITIONAL_ENCODING).to(device = configs.device)

lm_model, _, vocab, _ = check_point.loadCheckPoint(configs.model_file)
model.embedding = lm_model.embedding
model.input_linear = lm_model.input_linear
model.transformer = lm_model.transformer
del lm_model

training_set = buildDataset(training_samples, vocab.stoi, apply_mask = 1)

data_loader_params = { "batch_size": hyper_params.batch_size, "shuffle": True }
training_generator = torch.utils.data.DataLoader(training_set, **data_loader_params)

def setGradRequired(weights, required):
    for w in weights:
        if isinstance(w, list):
            setGradRequired(w, required)
        else:
            w.requires_grad = required

if hyper_params.gradual_unfreeze:
    model.embedding.weight.requires_grad = False
    for l in model.transformer.layers:
        setGradRequired(l.parameters(), False)

PAD_ID = vocab.stoi["<pad>"]

CPU_DEVICE = torch.device("cpu")

def evaluate(model, samples):
    model.eval()
    with torch.no_grad():
        evaluation_set = buildDataset(samples, vocab.stoi)
        evaluation_loader_params = { "batch_size": configs.evaluation_batch_size,
                "shuffle": False }
        evaluation_generator = torch.utils.data.DataLoader(evaluation_set,
            **evaluation_loader_params)
        predicted_idxes = []
        ground_truths = []
        loss_sum = 0.0
        for sentence_tensor, sentence_lens, src_key_padding_mask, label_tensor\
                in evaluation_generator:
            sentence_tensor = sentence_tensor.to(device = configs.device)
            predicted = model(sentence_tensor, sentence_lens, src_key_padding_mask)
            loss = nn.CrossEntropyLoss()(predicted, label_tensor.to(device = configs.device))
            loss_sum += loss
            predicted_idx = torch.max(predicted, 1)[1]
            predicted_idxes += list(predicted_idx.to(device = CPU_DEVICE).data.int())
            ground_truths += list(label_tensor.to(device = CPU_DEVICE).int())
    model.train()
    return metrics.f1_score(ground_truths, predicted_idxes, average = None),\
            loss_sum / len(ground_truths) * configs.evaluation_batch_size

step = 0

def isMlpToLabelLayer(group):
    return math.fabs(group["lr"] - hyper_params.learning_rate - 1e-8) < 1e-10

param_lr_list = [{"params": model.mlp_to_label.parameters(),
        "lr": hyper_params.learning_rate + 1e-8}]
lr = hyper_params.learning_rate
logger.debug("layer count:%d", len(model.transformer.layers))
for i in range(hyper_params.layer - 1, -1, -1):
    logger.info("layer %d initial lr:%f", i, lr)
    param_lr_list.append({"params": model.transformer.layers[i].parameters(),
            "lr": lr})
    lr *= hyper_params.layer_lr_decay

optimizer = optim.Adam(param_lr_list, lr = hyper_params.learning_rate, betas = (0.9, 0.98),
        eps = 1e-9, weight_decay = hyper_params.weight_decay)

initial_lr_dict = {}
for g in optimizer.param_groups:
    initial_lr_dict[id(g)] = g["lr"]

stagnation_epochs = 0
best_epoch_i = 0
best_dev_macro, best_test_macro = 0.0, 0.0
mask_p = 1

for epoch_i in itertools.count(0):
    if stagnation_epochs >= 2000:
        break
    batch_i = -1
    predicted_idxes = []
    ground_truths = []
    loss_sum = 0.0
    logger.info("epoch:%d batch count:%d", epoch_i,
            len(training_samples) / hyper_params.batch_size)

    if 0 < epoch_i <= hyper_params.layer:
        setGradRequired(model.transformer.layers[hyper_params.layer - epoch_i].parameters(), True)
    if epoch_i == 1:
        step = 0

    for sentence_tensor, sentence_lens, src_key_padding_mask, label_tensor in training_generator:
        batch_i += 1
        step += 1
        cut = hyper_params.warm_up_steps * hyper_params.cut_frac
        p = (step / cut) if step < cut else 1 - (step - cut) / (cut * (1.0 / hyper_params.cut_frac\
                - 1))
        p = (1 + p * (hyper_params.ratio - 1)) / hyper_params.ratio
        if p < 0:
            break

        should_print = batch_i * hyper_params.batch_size % 100 == 0
        if should_print:
            words = [vocab.itos[x] for x in sentence_tensor[0] if x != PAD_ID]
            logger.info("sentence:%s", " ".join(words))
            logger.info("step:%d p:%f", step, p)

        model.zero_grad()
        sentence_tensor = sentence_tensor.to(device = configs.device)
        predicted = model(sentence_tensor, sentence_lens, src_key_padding_mask)
        label_tensor = label_tensor.to(device = configs.device)
        loss = nn.CrossEntropyLoss()(predicted, label_tensor)
        if math.isnan(loss):
            logger.error("predicted:%f", predicted)
            logger.error("label_tensor:%f", label_tensor)
            sys.exit(1)
        loss.backward()
        if hyper_params.clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), hyper_params.clip_grad)
        for g in optimizer.param_groups:
            logger.debug("g:%s", g)
            g["lr"] = p * initial_lr_dict[id(g)]
            if should_print:
                logger.info("per layer lr:%f", g["lr"])
        optimizer.step()
        predicted_idx = torch.max(predicted, 1)[1]
        predicted_idxes += list(predicted_idx.to(device = CPU_DEVICE).data.
                int())
        ground_truths += list(label_tensor.to(device = CPU_DEVICE).int())
        loss_sum += loss
        if should_print:
            acc = metrics.accuracy_score(ground_truths, predicted_idxes)
            logger.info("acc:%f correct:%d total:%d", acc, acc * len(ground_truths),
                    len(ground_truths))
            logger.info("loss:%f", loss_sum / len(ground_truths) * hyper_params.batch_size)

    acc = metrics.accuracy_score(ground_truths, predicted_idxes)
    logger.info("whole avg acc:%f correct:%d total:%d", acc, acc * len(ground_truths), len(ground_truths))
    logger.info("whole avg loss:%f", loss_sum / len(ground_truths) * hyper_params.batch_size)
    logger.info("evaluating dev set...")
    dev_score, dev_loss = evaluate(model, dev_samples)
    logger.info("dev:%s loss:%f", dev_score, dev_loss)
    dev_macro = 0.5 * (dev_score[0] + dev_score[1])
    logger.info("dev macro:%f", dev_macro)

    logger.info("evaluating test set...")
    test_score, test_loss = evaluate(model, test_samples)
    logger.info("test:%s loss:%f", test_score, test_loss)
    test_macro = 0.5 * (test_score[0] + test_score[1])
    logger.info("test macro:%f", test_macro)

    if dev_macro > best_dev_macro:
        best_epoch_i = epoch_i
        best_dev_macro = dev_macro
        best_test_macro = test_macro
        logger.info("new best results")
        logger.info("laozhongyi_%f", best_dev_macro)
        stagnation_epochs = 0
    else:
        stagnation_epochs += 1
        logger.info("stagnation_epochs:%d", stagnation_epochs)
    logger.info("best epoch:%d dev_macro:%f test_macro:%f", best_epoch_i, best_dev_macro,
            best_test_macro)

    mask_p *= 0.5
    training_set = buildDataset(training_samples, vocab.stoi, apply_mask = mask_p)

    data_loader_params = { "batch_size": hyper_params.batch_size, "shuffle": True }
    training_generator = torch.utils.data.DataLoader(training_set, **data_loader_params)
