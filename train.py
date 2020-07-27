import datetime
import sys
import model
import torch
import torchtext
import configs
import dataset
import hyper_params
import itertools
import collections
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
import logging
import log_config
import os
import utils

logger = utils.getLogger(__file__)

torch.manual_seed(hyper_params.seed)

posts = dataset.readConversationSentences("/var/wqs/weibo_dialogue/posts")
responses = dataset.readConversationSentences(
        "/var/wqs/weibo_dialogue/responses")

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

counter = collections.Counter()
for idx, sample in enumerate(to_build_vocb_samples):
    sentenceToCounter(sample.post, counter)
    sentenceToCounter(sample.response, counter)

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
    t = oovCount(sample.post, counter)
    oov_count += t[0]
    all_words_count += t[1]

logger.info("oov:%f", oov_count / float(all_words_count))

vocab = torchtext.vocab.Vocab(counter, min_freq = hyper_params.min_freq)
logger.info("vocab len:%d", len(vocab))
embedding_table = nn.Embedding(len(vocab), hyper_params.word_dim,
        padding_idx = 0)
embedding_table.weight.data.uniform_(-1, 1)

def word_indexes(words, stoi):
    return [stoi[word] for word in words]

def pad_batch(word_ids_arr, lenghs):
    tensor = torch.ones(len(word_ids_arr), max(lenghs), dtype = int)
    for idx, (ids, seq_len) in enumerate(zip(word_ids_arr, lenghs)):
        tensor[idx, :seq_len] = torch.LongTensor(ids)
    return tensor

def buildDataset(samples, stoi):
    post_lens = [0] * len(samples)
    post_indexes_arr = [None] * len(samples)
    response_lens = [0] * len(samples)
    response_indexes_arr = [None] * len(samples)
    labels = [None] * len(samples)

    for i, sample in enumerate(samples):
        post_words = sample.post.split(" ")
        post_len = len(post_words)
        post_lens[i] = post_len
        post_indexes = word_indexes(post_words, stoi)
        post_indexes_arr[i] = post_indexes
        response_words = sample.response.split(" ")
        response_len = len(response_words)
        response_lens[i] = response_len
        response_indexes = word_indexes(response_words, stoi)
        response_indexes_arr[i] = response_indexes
        labels[i] = int(sample.stance)

    post_tensor = pad_batch(post_indexes_arr, post_lens)
    response_tensor = pad_batch(response_indexes_arr, response_lens)
    label_tensor = torch.LongTensor(labels)

    return dataset.Dataset(post_tensor, post_lens, response_tensor,
            response_lens, label_tensor)

training_set = buildDataset(training_samples, vocab.stoi)

data_loader_params = {
        "batch_size": hyper_params.batch_size,
        "shuffle": True }
training_generator = torch.utils.data.DataLoader(training_set,
        **data_loader_params)

model = model.LSTMClassifier(len(vocab)).to(device = configs.device)
optimizer = optim.Adam(model.parameters(), lr = hyper_params.learning_rate,
        weight_decay = hyper_params.weight_decay)
PAD_ID = vocab.stoi["<pad>"]

CPU_DEVICE = torch.device("cpu")

def evaluate(model, samples):
    evaluation_set = buildDataset(samples)
    evaluation_loader_params = {
            "batch_size": configs.evaluation_batch_size,
            "shuffle": False }
    evaluation_generator = torch.utils.data.DataLoader(evaluation_set,
        **evaluation_loader_params)
    predicted_idxes = []
    ground_truths = []
    for post_tensor, post_lengths, response_tensor, response_lengths,\
            label_tensor in evaluation_generator:
        post_tensor = post_tensor.to(device = configs.device)
        response_tensor = response_tensor.to(device = configs.device)
        predicted = model(post_tensor, post_lengths, response_tensor,
                response_lengths)
        predicted_idx = torch.max(predicted, 1)[1]
        predicted_idxes += list(predicted_idx.to(device = CPU_DEVICE).data.
                int())
        ground_truths += list(label_tensor.to(device = CPU_DEVICE).int())


for epoch_i in itertools.count(0):
    if epoch_i > 10:
        break
    batch_i = -1
    predicted_idxes = []
    ground_truths = []
    loss_sum = 0.0
    for post_tensor, post_lengths, response_tensor, response_lengths,\
            label_tensor in training_generator:
        batch_i += 1

        should_print = batch_i * hyper_params.batch_size % 100 == 0
        if should_print:
            post_words = [vocab.itos[x] for x in post_tensor[0] if x != PAD_ID]
            logger.info("post:%s", " ".join(post_words))
            response_words = [vocab.itos[x] for x in response_tensor[0]
                    if x != PAD_ID]
            logger.info("response:%s", " ".join(response_words))

        model.zero_grad()
        post_tensor = post_tensor.to(device = configs.device)
        response_tensor = response_tensor.to(device = configs.device)
        predicted = model(post_tensor, post_lengths, response_tensor,
                response_lengths)
        label_tensor = label_tensor.to(device = configs.device)
        loss = nn.CrossEntropyLoss()(predicted, label_tensor)
        loss.backward()
        optimizer.step()
        predicted_idx = torch.max(predicted, 1)[1]
        predicted_idxes += list(predicted_idx.to(device = CPU_DEVICE).data.
                int())
        ground_truths += list(label_tensor.to(device = CPU_DEVICE).int())
        loss_sum += loss
        if should_print:
            acc = metrics.accuracy_score(ground_truths, predicted_idxes)
            logger.info("acc:%f", acc)