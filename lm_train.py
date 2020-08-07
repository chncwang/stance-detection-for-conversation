import datetime
import sys
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
import lm_module

logger = utils.getLogger(__file__)

logger.info("torch version:%s", torch.__version__)

hyper_params = imp.load_source("module.name", sys.argv[1])

utils.printLmHyperParams(hyper_params)

torch.manual_seed(hyper_params.seed)

posts = dataset.readConversationSentences("/var/wqs/weibo_dialogue/posts-bpe")
responses = dataset.readConversationSentences(
        "/var/wqs/weibo_dialogue/responses-bpe")

def readSentences(path, rate = 1.0):
    logger.info("rate:%f", rate)
    return dataset.readLmSentences(path, posts, responses, rate = rate)

training_samples = readSentences("/var/wqs/stance-lm/train", rate = 0.01)
logger.info("traning samples count:%d", len(training_samples))
dev_samples = readSentences("/var/wqs/stance-lm/dev")
logger.info("dev samples count:%d", len(dev_samples))

to_build_vocb_samples = None
if hyper_params.embedding_tuning:
    to_build_vocb_samples = training_samples
else:
    to_build_vocb_samples = training_samples + dev_samples

def sentenceToCounter(sentence, counter):
    words = sentence.split(" ")
    return counter.update(words)

counter = collections.Counter()
logger.info("building vocabulary...")
for idx, sample in enumerate(to_build_vocb_samples):
    sentenceToCounter(sample, counter)

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
    t = oovCount(sample, counter)
    oov_count += t[0]
    all_words_count += t[1]

logger.info("oov:%f", oov_count / float(all_words_count))
word_vectors = torchtext.vocab.Vectors(
        "/var/wqs/cn_embeddings/sgns.weibo.bigram-char")

if not hyper_params.embedding_tuning:
    for k in counter.keys():
        if counter[k] < hyper_params.min_freq and k in word_vectors.stoi:
            counter[k] = 10000

vocab = torchtext.vocab.Vocab(counter, min_freq = hyper_params.min_freq)
logger.info("vocab len:%d", len(vocab))
vocab.load_vectors(word_vectors)
embedding_table = nn.Embedding.from_pretrained(vocab.vectors,
        freeze = hyper_params.embedding_tuning).to(device = configs.device)

def word_indexes(words, stoi):
    return [stoi[word] for word in words]

def pad_batch(word_ids_arr, lenghs):
    tensor = torch.ones(len(word_ids_arr), max(lenghs), dtype = int)
    for idx, (ids, seq_len) in enumerate(zip(word_ids_arr, lenghs)):
        x = torch.LongTensor(ids)
        tensor[idx, :seq_len] = x
    return tensor

def buildDataset(samples, stoi):
    words_arr = [s.split(" ") for s in samples]
    src_sentences_indexes_arr = [word_indexes(s[:-1], stoi) for s in words_arr]
    tgt_sentences_indexes_arr = [word_indexes(s[1:], stoi) for s in words_arr]
    sentence_lens = [len(s) - 1 for s in words_arr]

    src_sentence_tensor = pad_batch(src_sentences_indexes_arr, sentence_lens)
    tgt_sentence_tensor = pad_batch(tgt_sentences_indexes_arr, sentence_lens)

    return dataset.LmDataset(src_sentence_tensor, tgt_sentence_tensor,
            sentence_lens)

training_set = buildDataset(training_samples, vocab.stoi)

data_loader_params = {
        "batch_size": hyper_params.batch_size,
        "shuffle": True }
training_generator = torch.utils.data.DataLoader(training_set,
        **data_loader_params)

model = lm_module.LstmLm(embedding_table, len(vocab)).to(
        device = configs.device)
optimizer = optim.Adam(model.parameters(), lr = hyper_params.learning_rate,
        weight_decay = hyper_params.weight_decay)
PAD_ID = vocab.stoi["<pad>"]
if PAD_ID != 1:
    logger.error("PAD_ID is %d", PAD_ID)
    sys.exit(1)

CPU_DEVICE = torch.device("cpu")

def evaluate(model, samples):
    evaluation_set = buildDataset(samples, vocab.stoi)
    evaluation_loader_params = {
            "batch_size": configs.evaluation_batch_size,
            "shuffle": False }
    evaluation_generator = torch.utils.data.DataLoader(evaluation_set,
        **evaluation_loader_params)
    predicted_idxes = []
    ground_truths = []
    for sentence_tensor, sentence_lens, label_tensor in evaluation_generator:
        sentence_tensor = sentence_tensor.to(device = configs.device)
        predicted = model(sentence_tensor, sentence_lens)
        predicted_idx = torch.max(predicted, 1)[1]
        predicted_idxes += list(predicted_idx.to(device = CPU_DEVICE).data.
                int())
        ground_truths += list(label_tensor.to(device = CPU_DEVICE).int())
    return metrics.f1_score(ground_truths, predicted_idxes, average = None)

stagnation_epochs = 0
best_epoch_i = 0
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
    for src_tensor, tgt_tensor, lens in training_generator:
        batch_i += 1

        should_print = batch_i * hyper_params.batch_size % 1000 == 0
        if should_print:
            words = [vocab.itos[x] for x in src_tensor[0] if x != PAD_ID]
            logger.info("sentence:%s", " ".join(words))

        model.zero_grad()
        src_tensor = src_tensor.to(device = configs.device)
        predicted = model(src_tensor, lens)
        max_len_in_batch = predicted.size()[2]
        tgt_tensor = tgt_tensor.to(device = configs.device)
        logger.debug("predicted size:%s tgt_tensor:%s", predicted.size(),
                tgt_tensor.size())
        logger.debug("tgt_tensor size:%s", tgt_tensor.size())
        tgt_tensor = torch.split(tgt_tensor,
                [max_len_in_batch, tgt_tensor.size()[1] - max_len_in_batch],
                1)[0]
        logger.debug("tgt_tensor size:%s", tgt_tensor.size())
        loss = nn.NLLLoss()(predicted, tgt_tensor)
        loss.backward()
        if hyper_params.clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(),
                    hyper_params.clip_grad)
        optimizer.step()
        predicted_idx = torch.max(predicted, 1)[1]
        predicted_idx = predicted_idx.to(device = CPU_DEVICE).tolist()
        for x in predicted_idx:
            predicted_idxes += x
        logger.debug("predicted_idxes:%s", predicted_idxes)
        ground_truth = tgt_tensor.to(device = CPU_DEVICE).tolist()
        for x in ground_truth:
            ground_truths += x
        logger.debug("ground_truths:%s", ground_truths)
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
    dev_macro = 0.5 * (dev_score[0] + dev_score[1])
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
