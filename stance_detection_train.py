import datetime
import sys
import classifier_module
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
import logging
import log_config
import lm_utils

logger = utils.getLogger(__file__)

hyper_params = imp.load_source("module.name", sys.argv[1])

utils.printStanceDetectionHyperParams(hyper_params)

torch.manual_seed(hyper_params.seed)

posts = dataset.readConversationSentences("/var/wqs/weibo_dialogue/posts-bpe")
responses = dataset.readConversationSentences(
        "/var/wqs/weibo_dialogue/responses-bpe")

def readSamples(path):
    return dataset.readSamples(path, posts, responses)

training_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/"\
                "overall_filtered_train")
dev_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/"\
                "overall_filtered_dev")
test_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/"\
                "overall_filtered_test")

to_build_vocb_samples = None
if hyper_params.embedding_tuning:
    to_build_vocb_samples = training_samples
else:
    to_build_vocb_samples = training_samples + dev_samples + test_samples

def sentenceToCounter(sentence, counter):
    words = sentence.split(" ")
    return counter.update(words)

def concatSentence(post, response):
    return "<begin> " + post + " <sep> " + response + " <end>"

counter = collections.Counter()
for idx, sample in enumerate(to_build_vocb_samples):
    sentenceToCounter(concatSentence(sample.post, sample.response), counter)

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
    t = oovCount(concatSentence(sample.post, sample.response), counter)
    oov_count += t[0]
    all_words_count += t[1]

logger.info("oov:%f", oov_count / float(all_words_count))
word_vectors = torchtext.vocab.Vectors(
        "/var/wqs/cn_embeddings/sgns.weibo.bigram-char")

if not hyper_params.embedding_tuning:
    for k in counter.keys():
        if counter[k] < hyper_params.min_freq and k in word_vectors.stoi:
            counter[k] = 10000

logger.info("pretrained_model loading...")
pretrained_model, stored_opt, vocab, _ = lm_utils.loadLmCheckPoint(
        "/var/wqs/pretrained/lstm/model-5-2020-08-17-21-35", hyper_params)
del stored_opt
logger.info("pretrained_model loaded")
# vocab = torchtext.vocab.Vocab(counter, min_freq = hyper_params.min_freq)
logger.info("vocab len:%d", len(vocab))
# vocab.load_vectors(word_vectors)
embedding_table = nn.Embedding.from_pretrained(vocab.vectors,
        freeze = not hyper_params.embedding_tuning).to(device = configs.device)

def word_indexes(words, stoi):
    return [stoi[word] for word in words]

def pad_batch(word_ids_arr, lenghs):
    max_len = max(lenghs)
    tensor = torch.ones(len(word_ids_arr), max_len, dtype = int)
    for idx, (ids, seq_len) in enumerate(zip(word_ids_arr, lenghs)):
        x = torch.LongTensor(ids)
        tensor[idx, : seq_len] = x

    return tensor

def buildDataset(samples, stoi):
    sentences = [concatSentence(s.post, s.response)  for s in samples]
    words_arr = [s.split(" ") for s in sentences]
    sentence_lens = [len(s) for s in words_arr]

    l2r_sentences_indexes_arr = [word_indexes(s, stoi) for s in words_arr]
    l2r_sentence_tensor = pad_batch(l2r_sentences_indexes_arr, sentence_lens)
    labels = [int(s.stance) for s in samples]

    for words in words_arr:
        words.reverse()
    r2l_sentences_indexes_arr = [word_indexes(s, stoi) for s in words_arr]
    r2l_sentence_tensor = pad_batch(r2l_sentences_indexes_arr, sentence_lens)

    label_tensor = torch.LongTensor(labels)

    return dataset.StanceDetectionDataset(l2r_sentence_tensor,
            r2l_sentence_tensor, sentence_lens, label_tensor)

training_set = buildDataset(training_samples, vocab.stoi)

data_loader_params = {
        "batch_size": hyper_params.batch_size,
        "shuffle": True }
training_generator = torch.utils.data.DataLoader(training_set,
        **data_loader_params)

model = classifier_module.LSTMClassifier(embedding_table).to(
        device = configs.device)
model.l2r_lstm = pretrained_model.l2r_lstm
model.r2l_lstm = pretrained_model.r2l_lstm
model.embedding = pretrained_model.embedding
del pretrained_model
# logger.debug("model params:%s", list(model.parameters()))

optimizer = optim.Adam(model.parameters(), lr = hyper_params.learning_rate,
        weight_decay = hyper_params.weight_decay)
PAD_ID = vocab.stoi["<pad>"]
if PAD_ID != 1:
    logger.error("pad id should be 1, but is %d", PAD_ID)
    sys.exit(1)

CPU_DEVICE = torch.device("cpu")

def evaluate(model, samples):
    model.eval()
    with torch.no_grad():
        evaluation_set = buildDataset(samples, vocab.stoi)
        evaluation_loader_params = {
                "batch_size": configs.evaluation_batch_size,
                "shuffle": False }
        evaluation_generator = torch.utils.data.DataLoader(evaluation_set,
            **evaluation_loader_params)
        predicted_idxes = []
        ground_truths = []
        for l2r_sentence_tensor, r2l_sentence_tensor, sentence_lens,\
                label_tensor in evaluation_generator:
            l2r_sentence_tensor = l2r_sentence_tensor.to(
                    device = configs.device)
            r2l_sentence_tensor = r2l_sentence_tensor.to(
                    device = configs.device)
            predicted = model(l2r_sentence_tensor, r2l_sentence_tensor,
                    sentence_lens)
            predicted_idx = torch.max(predicted, 1)[1]
            predicted_idxes += list(predicted_idx.to(device = CPU_DEVICE).data.
                    int())
            ground_truths += list(label_tensor.to(device = CPU_DEVICE).int())
    model.train()
    return metrics.f1_score(ground_truths, predicted_idxes, average = None)

stagnation_epochs = 0
best_epoch_i = 0
step = 0
best_dev_macro, best_test_macro = 0.0, 0.0
for epoch_i in itertools.count(0):
    if stagnation_epochs >= 10:
        break
    batch_i = -1
    predicted_idxes = []
    ground_truths = []
    loss_sum = 0.0
    logger.info("epoch:%d batch count:%d", epoch_i,
            len(training_samples) / hyper_params.batch_size)
    for l2r_sentence_tensor, r2l_sentence_tensor, sentence_lens, label_tensor\
            in training_generator:
        batch_i += 1
        step += 1

        t = step + 1
        cut = hyper_params.T * hyper_params.cut_frac
        p = (t / cut) if step < cut else 1 - (t - cut) /\
                (cut * (1.0 / hyper_params.cut_frac - 1));
        learning_rate = hyper_params.learning_rate * (1 + p *
                (hyper_params.ratio - 1)) / hyper_params.ratio
        for g in optimizer.param_groups:
            g["lr"] = learning_rate
        logger.debug("learning_rate:%f", learning_rate)

        should_print = batch_i * hyper_params.batch_size % 1000 == 0
        if should_print:
            logger.info("learning_rate:%f", learning_rate)

        if should_print:
            words = [vocab.itos[x] for x in l2r_sentence_tensor[0]\
                    if x != PAD_ID]
            logger.info("sentence:%s", " ".join(words))

        model.zero_grad()
        l2r_sentence_tensor = l2r_sentence_tensor.to(device = configs.device)
        r2l_sentence_tensor = r2l_sentence_tensor.to(device = configs.device)
        predicted = model(l2r_sentence_tensor, r2l_sentence_tensor,
                sentence_lens)
        label_tensor = label_tensor.to(device = configs.device)
        loss = nn.CrossEntropyLoss()(predicted, label_tensor)
        loss.backward()
        if hyper_params.clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(),
                    hyper_params.clip_grad)
        optimizer.step()
        predicted_idx = torch.max(predicted, 1)[1]
        predicted_idxes += list(predicted_idx.to(device = CPU_DEVICE).data.
                int())
        ground_truths += list(label_tensor.to(device = CPU_DEVICE).int())
        loss_sum += loss
        if should_print:
            acc = metrics.accuracy_score(ground_truths, predicted_idxes)
            logger.info("acc:%f correct:%d total:%d", acc,
                    acc * len(ground_truths), len(ground_truths))

    acc = metrics.accuracy_score(ground_truths, predicted_idxes)
    logger.info("acc:%f correct:%d total:%d", acc, acc * len(ground_truths),
            len(ground_truths))
    logger.info("evaluating dev set...")
    dev_score = evaluate(model, dev_samples)
    logger.info("dev:%s", dev_score)
    dev_macro = sum(dev_score[:2]) / 2.0
    logger.info("dev macro:%f", dev_macro)

    logger.info("evaluating test set...")
    test_score = evaluate(model, test_samples)
    logger.info("test:%s", test_score)
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
    logger.info("best epoch:%d dev_macro:%f test_macro:%f", best_epoch_i,
            best_dev_macro, best_test_macro)
