import datetime
import math
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
import pickle

logger = utils.getLogger(__file__)

logger.info("torch version:%s", torch.__version__)

hyper_params = imp.load_source("module.name", sys.argv[1])

utils.printLmHyperParams(hyper_params)
utils.printConfigs()

torch.manual_seed(hyper_params.seed)

posts = dataset.readConversationSentences("/var/wqs/weibo_dialogue/posts-bpe")
responses = dataset.readConversationSentences(
        "/var/wqs/weibo_dialogue/responses-bpe")

def readSentences(path, rate = 1.0):
    logger.info("rate:%f", rate)
    return dataset.readLmSentences(path, posts, responses, rate = rate)

training_samples = readSentences("/var/wqs/stance-lm/train",
        rate = configs.lm_training_set_rate)
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

vocab, embedding_table = None, None
if configs.model_file is None:
    vocab = torchtext.vocab.Vocab(counter, min_freq = hyper_params.min_freq)
    logger.info("vocab len:%d", len(vocab))
    vocab.load_vectors(word_vectors)
    embedding_table = nn.Embedding.from_pretrained(vocab.vectors,
            freeze = not hyper_params.embedding_tuning).to(
                    device = configs.device)

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

    l2r_src_sentences_indexes_arr =\
            [word_indexes(s[:-1], stoi) for s in words_arr]
    l2r_tgt_sentences_indexes_arr =\
            [word_indexes(s[1:], stoi) for s in words_arr]
    sentence_lens = [len(s) - 1 for s in words_arr]

    l2r_src_sentence_tensor = pad_batch(l2r_src_sentences_indexes_arr,
            sentence_lens)
    l2r_tgt_sentence_tensor = pad_batch(l2r_tgt_sentences_indexes_arr,
            sentence_lens)

    for words in words_arr:
        words.reverse()

    r2l_src_sentences_indexes_arr =\
            [word_indexes(s[:-1], stoi) for s in words_arr]
    r2l_tgt_sentences_indexes_arr =\
            [word_indexes(s[1:], stoi) for s in words_arr]

    r2l_src_sentence_tensor = pad_batch(r2l_src_sentences_indexes_arr,
            sentence_lens)
    r2l_tgt_sentence_tensor = pad_batch(r2l_tgt_sentences_indexes_arr,
            sentence_lens)

    return dataset.LmDataset(l2r_src_sentence_tensor, l2r_tgt_sentence_tensor,
            r2l_src_sentence_tensor, r2l_tgt_sentence_tensor, sentence_lens)

model, optimizer, learning_rate = None, None, None
if configs.model_file is None:
    model = lm_module.LstmLm(embedding_table, len(vocab)).to(
            device = configs.device)
    learning_rate = hyper_params.learning_rate
    optimizer = optim.Adam(model.parameters(), lr = learning_rate,
            weight_decay = hyper_params.weight_decay)
else:
    logger.info("loading %s...", configs.model_file)
    model, optimizer, vocab, learning_rate = utils.loadLmCheckPoint(
            configs.model_file)

training_set = buildDataset(training_samples, vocab.stoi)

data_loader_params = {
        "batch_size": hyper_params.batch_size,
        "shuffle": True }
training_generator = torch.utils.data.DataLoader(training_set,
        **data_loader_params)

PAD_ID = vocab.stoi["<pad>"]
if PAD_ID != 1:
    logger.error("PAD_ID is %d", PAD_ID)
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
        loss_sums = [0.0, 0.0]
        dataset_len_sum = 0
        for l2r_src_tensor, l2r_tgt_tensor, r2l_src_tensor, r2l_tgt_tensor,\
                lens in evaluation_generator:
            l2r_src_tensor = l2r_src_tensor.to(device = configs.device)
            r2l_src_tensor = r2l_src_tensor.to(device = configs.device)
            logger.debug("src_tensor size:%s tgt_tensor size:%s",
                    l2r_src_tensor.size(), l2r_tgt_tensor.size())
            logger.debug("lens:%s", lens)
            predicted = model(l2r_src_tensor, r2l_src_tensor, lens)
            l2r_tgt_tensor = l2r_tgt_tensor.to(device = configs.device)
            r2l_tgt_tensor = r2l_tgt_tensor.to(device = configs.device)
            for direction_i, (tgt_tensor, logsoftmax) in enumerate(zip(
                    [l2r_tgt_tensor, r2l_tgt_tensor], predicted)):
                ids_list = []
                len_sum = 0
                for i, sentence_ids in enumerate(torch.split(tgt_tensor, 1)):
                    sentence_ids = sentence_ids.reshape(sentence_ids.size()[1])
                    length = lens[i]
                    len_sum += length
                    non_padded = torch.split(sentence_ids,
                            [length, tgt_tensor.size()[1] - length], 0)[0]
                    ids_list.append(non_padded)
                ids_tuple = tuple(ids_list)
                concated = torch.cat(ids_tuple)
                loss = nn.NLLLoss()(logsoftmax, concated)
                loss_sums[direction_i] += loss * len_sum
                if direction_i == 0:
                    dataset_len_sum += len_sum
    model.train()
    return [math.exp(s / dataset_len_sum) for s in loss_sums]

stagnation_epochs = 0
best_epoch_i = 0
best_dev_ppl = 1e100
for epoch_i in itertools.count(0):
    if stagnation_epochs >= 2:
        break
    batch_i = -1
    loss_sums = [0.0, 0.0]
    logger.info("epoch:%d batch count:%d", epoch_i,
            len(training_samples) / hyper_params.batch_size)
    logger.info("learning_rate:%f", learning_rate)
    total_token_count = 0
    total_hit_counts = [0, 0]
    for l2r_src_tensor, l2r_tgt_tensor, r2l_src_tensor, r2l_tgt_tensor,\
            lens in training_generator:
        batch_i += 1

        should_print = batch_i * hyper_params.batch_size % 1000 == 0
        if should_print:
            logger.info("batch_i:%d", batch_i)
            words = [vocab.itos[x] for x in l2r_src_tensor[0] if x != PAD_ID]
            logger.info("sentence:%s", " ".join(words))

        model.zero_grad()
        l2r_src_tensor = l2r_src_tensor.to(device = configs.device)
        r2l_src_tensor = r2l_src_tensor.to(device = configs.device)
        predicted = model(l2r_src_tensor, r2l_src_tensor, lens)
        logger.debug("predicted size:%s", predicted[0].size())
        l2r_tgt_tensor = l2r_tgt_tensor.to(device = configs.device)
        r2l_tgt_tensor = r2l_tgt_tensor.to(device = configs.device)
        logger.debug("tgt_tensor size:%s", l2r_tgt_tensor.size())

        losses = [0.0] * 2
        concated_arr = [None] * 2
        for direction_i, (tgt_tensor, logsoftmax) in\
                enumerate(zip([l2r_tgt_tensor, r2l_tgt_tensor], predicted)):
            ids_list = []
            first_len = 0
            for i, sentence_ids in enumerate(torch.split(tgt_tensor, 1)):
                sentence_ids = sentence_ids.reshape(sentence_ids.size()[1])
                logger.debug("sentence_ids size:%s", sentence_ids.size())
                length = lens[i]
                logger.debug("length:%d", length)
                if i == 0:
                    first_len = length
                non_padded = torch.split(sentence_ids,
                        [length, tgt_tensor.size()[1] - length], 0)[0]
                logger.debug("non_padded size:%s", non_padded.size())
                ids_list.append(non_padded)
            ids_tuple = tuple(ids_list)
            concated = torch.cat(ids_tuple)
            concated_arr[direction_i] = concated
            logger.debug("concated size:%s predicted size:%s", concated.size(),
                    logsoftmax[0].size())
            loss = nn.NLLLoss()(logsoftmax, concated)
            losses[direction_i] = float(loss)
            loss.backward()

        if hyper_params.clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(),
                    hyper_params.clip_grad)
        optimizer.step()

        for direction_i in range(0, 2):
            predicted_idx = torch.max(predicted[direction_i], 1)[1]
            predicted_idx = predicted_idx.to(device = CPU_DEVICE).tolist()
            if should_print:
                words = [vocab.itos[x] for x in predicted_idx[: first_len]]
                logger.info("predicted:%s", " ".join(words))
            logger.debug("predicted_idx len:%d", len(predicted_idx))
            ground_truth = concated_arr[direction_i].to(
                    device = CPU_DEVICE).tolist()
            logger.debug("ground_truth len:%d", len(ground_truth))
            token_count_in_batch = len(ground_truth)
            if direction_i == 0:
                total_token_count += token_count_in_batch
            loss_sums[direction_i] += losses[direction_i] *\
                    token_count_in_batch
            acc = metrics.accuracy_score(ground_truth, predicted_idx)
            total_hit_counts[direction_i] += acc * token_count_in_batch
            if should_print:
                ppl = math.exp(loss_sums[direction_i] / total_token_count)
                logger.info("%s ppl:%f acc:%f correct:%d total:%d",
                        "l2r" if direction_i == 0 else "r2l", ppl, acc,
                        total_hit_counts[direction_i], total_token_count)

    logger.info("evaluating dev set...")
    dev_ppls = evaluate(model, dev_samples)
    logger.info("dev ppls:%s", dev_ppls)
    dev_ppl = 0.5 * sum(dev_ppls)

    if dev_ppl < best_dev_ppl:
        best_epoch_i = epoch_i
        best_dev_ppl = dev_ppl
        logger.info("new best results")
        logger.info("laozhongyi_%f", best_dev_ppl)
        stagnation_epochs = 0
        utils.saveCheckPoint(model, optimizer, vocab, learning_rate, epoch_i)
    else:
        stagnation_epochs += 1
        logger.info("stagnation_epochs:%d", stagnation_epochs)
    logger.info("best epoch:%d dev_ppl:%f", best_epoch_i, best_dev_ppl)

    learning_rate = (learning_rate - hyper_params.min_learning_rate) *\
            hyper_params.lr_decay + hyper_params.min_learning_rate
    logger.info("new learning rate:%f", learning_rate)
    for g in optimizer.param_groups:
        g["lr"] = learning_rate
