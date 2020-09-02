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
        "/var/wqs/conversation-stance-corpus/overall_filtered/to_annotate")
test_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/overall_filtered_test")
g_max_len = max([maxLen(training_samples), maxLen(dev_samples), maxLen(test_samples)])
logger.info("max len of the whole dataset:%d", g_max_len)

def concatSentencePair(p, r):
    return "<cls> " + p + " <sep> " + r

def word_indexes(words, stoi):
    return [stoi[word] for word in words]

def pad_batch(word_ids_arr, lens):
    tensor = torch.ones(len(word_ids_arr), max(lens), dtype = int)
    for idx, (ids, seq_len) in enumerate(zip(word_ids_arr, lens)):
        x = torch.LongTensor(ids)
        tensor[idx, :seq_len] = x
    return tensor

def buildDataset(samples, stoi, apply_mask = 0.0):
    sentences = [concatSentencePair(s.post, s.response) for s in samples]
    words_arr = [s.split(" ") for s in sentences]
    sentences_indexes_arr = [word_indexes(s, stoi) for s in words_arr]
    sentence_lens = [len(s) for s in words_arr]
    labels = [int(s.stance) for s in samples]
    sentence_tensor = pad_batch(sentences_indexes_arr, sentence_lens)
    src_key_padding_mask = utils.srcMask(sentences_indexes_arr, sentence_lens)
    label_tensor = torch.LongTensor(labels)
    post_ids = [s.post_id for s in samples]
    response_ids = [s.response_id for s in samples]

    return dataset.StanceDetectionDataset(sentence_tensor, sentence_lens, src_key_padding_mask,
            label_tensor, post_ids, response_ids)

if g_max_len >= configs.MAX_LEN_FOR_POSITIONAL_ENCODING:
    logger.error("g_max_len:%d MAX_LEN_FOR_POSITIONAL_ENCODING:%d", g_max_len,
            configs.MAX_LEN_FOR_POSITIONAL_ENCODING)
    sys.exit(1)

model, _, vocab, _ = check_point.loadStanceDetectionCheckPoint(
        "/var/wqs/pretrained/transformer-classifier/model-0-2020-09-02-18-11")

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
        softmax = nn.Softmax(dim = 1)
        for sentence_tensor, sentence_lens, src_key_padding_mask, label_tensor, post_id,\
                response_id in evaluation_generator:
            sentence_tensor = sentence_tensor.to(device = configs.device)
            predicted = model(sentence_tensor, sentence_lens, src_key_padding_mask)
            predicted = softmax(predicted)
            logger.info("post_id:%d res_id%d predicted:%s", post_id, response_id, predicted)
            loss = nn.CrossEntropyLoss()(predicted, label_tensor.to(device = configs.device))
            loss_sum += loss
            predicted_idx = torch.max(predicted, 1)[1]
            predicted_idxes += list(predicted_idx.to(device = CPU_DEVICE).data.int())
            ground_truths += list(label_tensor.to(device = CPU_DEVICE).int())
    model.train()
    return metrics.f1_score(ground_truths, predicted_idxes, average = None),\
            loss_sum / len(ground_truths) * configs.evaluation_batch_size

dev_score, dev_loss = evaluate(model, dev_samples)
