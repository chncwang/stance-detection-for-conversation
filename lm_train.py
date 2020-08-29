import datetime
import common
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
import random
import check_point

logger = utils.getLogger(__file__)

logger.info("torch version:%s", torch.__version__)

hyper_params = imp.load_source("module.name", sys.argv[1])

utils.printLmHyperParams(hyper_params)

posts = dataset.readConversationSentences("/var/wqs/weibo_dialogue/posts-bpe")
responses = dataset.readConversationSentences("/var/wqs/weibo_dialogue/responses-bpe")

def readSentences(path, rate = 1.0):
    logger.info("rate:%f", rate)
    return dataset.readLmSentences(path, posts, responses, rate = rate)

training_samples = readSentences("/var/wqs/stance-lm/train", rate = configs.lm_training_set_rate)
logger.info("traning samples count:%d", len(training_samples))
dev_samples = readSentences("/var/wqs/stance-lm/dev")
logger.info("dev samples count:%d", len(dev_samples))

corpus_max_len = max([len(s) for s in (training_samples + dev_samples)])
logger.info("corpus_max_len:%d", corpus_max_len)

to_build_vocb_samples = None
if hyper_params.embedding_tuning:
    to_build_vocb_samples = training_samples
else:
    to_build_vocb_samples = training_samples + dev_samples

def sentenceToCounter(sentence, counter):
    words = sentence.split(" ")
    return counter.update(words)

counter = collections.Counter()
counter["<mask>"] = 10000
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
word_vectors = torchtext.vocab.Vectors("/var/wqs/cn_embeddings/sgns.weibo.bigram-char")

if not hyper_params.embedding_tuning:
    for k in counter.keys():
        if counter[k] < hyper_params.min_freq and k in word_vectors.stoi:
            counter[k] = 10000

def isZero(tensor):
    for i in range(tensor.size()[0]):
        if abs(tensor[i]) > 1e-10:
            return False
    return True

vocab, embedding_table = None, None
if configs.model_file is None:
    vocab = torchtext.vocab.Vocab(counter, min_freq = hyper_params.min_freq)
    logger.info("vocab len:%d", len(vocab))
    logger.debug("stoi len:%d", len(vocab.stoi))
    logger.debug("unk id:%d sep id:%d", vocab.stoi["<unk>"], vocab.stoi["<sep>"])
    vocab.load_vectors(word_vectors)
    logger.debug("unk id:%d sep id:%d", vocab.stoi["<unk>"], vocab.stoi["<sep>"])
    embedding_table = nn.Embedding.from_pretrained(vocab.vectors,
            freeze = not hyper_params.embedding_tuning).to(device = configs.device)
    logger.debug("unk:%s sep:%s pad:%s", embedding_table(torch.LongTensor([0]).to(
            device = configs.device)),
            embedding_table(torch.LongTensor([4]).to(device = configs.device)),
            embedding_table(torch.LongTensor([1]).to(device = configs.device)))

def targetIdsList(src_sentence_ids_arr, prediction_positions_arr):
    target_ids_arr = [None] * len(src_sentence_ids_arr)
    for i, (ids, positions) in enumerate(zip(src_sentence_ids_arr, prediction_positions_arr)):
        logger.debug("positions:%s", positions)
        target_ids = torch.LongTensor([ids[p] for p in positions])
        target_ids_arr[i] = target_ids
    return target_ids_arr

model, optimizer, step = None, None, None
if configs.model_file is None:
    model = lm_module.TransformerLm(embedding_table, len(vocab),
            configs.MAX_LEN_FOR_POSITIONAL_ENCODING).to( device = configs.device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = hyper_params.weight_decay)
    step = 0
else:
    logger.info("loading %s...", configs.model_file)
    model, optimizer, vocab, step = check_point.loadCheckPoint(configs.model_file)
    step += 1

def buildDatasetAndGenerator(samples, stoi, vocab_len):
    training_set = dataset.buildLmDataset(samples, stoi, vocab_len)

    data_loader_params = { "batch_size": hyper_params.batch_size, "shuffle": True }
    training_generator = torch.utils.data.DataLoader(training_set, **data_loader_params)
    return training_set, training_generator

training_set, training_generator = buildDatasetAndGenerator(training_samples, vocab.stoi,
        len(vocab))

PAD_ID = vocab.stoi["<pad>"]
if PAD_ID != 1:
    logger.error("PAD_ID is %d", PAD_ID)
    sys.exit(1)

CPU_DEVICE = torch.device("cpu")

stagnation_epochs = 0
best_epoch_i = 0
best_dev_ppl = 1e100
ppl = 0
for epoch_i in itertools.count(0):
    if stagnation_epochs >= 100:
        break
    batch_i = -1
    loss_sum = 0.0
    logger.info("epoch:%d batch count:%d", epoch_i,
            len(training_samples) / hyper_params.batch_size)
    total_token_count = 0
    total_hit_count = 0

    if step > hyper_params.warm_up_steps:
        lr = hyper_params.min_learning_rate + (lr - hyper_params.min_learning_rate) *\
                hyper_params.lr_decay

    for src_tensor, tgt_tensor, src_key_padding_mask, prediction_positions_arr, lens\
            in training_generator:
        logger.debug("prediction_positions_arr size:%s", prediction_positions_arr.size())
        step += 1
        batch_i += 1
        should_print = batch_i * hyper_params.batch_size % 1000 == 0
        if step < hyper_params.warm_up_steps:
            lr = hyper_params.learning_rate * min(1, step / hyper_params.warm_up_steps)

        if should_print:
            logger.info("lr:%f", lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        logger.debug("src_tensor size:%s", src_tensor.size())
        logger.debug("src_key_padding_mask size:%s", src_key_padding_mask.size())

        if should_print:
            logger.info("batch_i:%d", batch_i)
            t = src_tensor[0]
            words = [vocab.itos[x] for x in t if x != PAD_ID]
            logger.info("sentence:%s", " ".join(words))
            words = [vocab.itos[x] for x in tgt_tensor[0] if x != PAD_ID]
            logger.info("sentence:%s", " ".join(words))

        model.zero_grad()
        src_tensor = src_tensor.to(device = configs.device)
        predicted = model(src_tensor, lens, src_key_padding_mask, prediction_positions_arr)
        logger.debug("predicted size:%s", predicted.size())
        logger.debug("original predicted:%s", predicted)
        tgt_tensor = tgt_tensor.to(device = configs.device)
        logger.debug("tgt_tensor size:%s", tgt_tensor.size())
        tgt_tensor = tgt_tensor[prediction_positions_arr]
        logger.debug("prediction_positions_arr size:%s", prediction_positions_arr.size())
        logger.debug("prediction_positions_arr:%s", prediction_positions_arr)
        logger.debug("tgt_tensor size:%s", tgt_tensor.size())
        loss = nn.NLLLoss()(predicted, tgt_tensor)
        loss.backward()

        if hyper_params.clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), hyper_params.clip_grad)
        optimizer.step()

        predicted_idx = torch.max(predicted, 1)[1]
        predicted_idx = predicted_idx.to(device = CPU_DEVICE).tolist()
        if should_print:
            prediction_positions = prediction_positions_arr[0]
            first_len = 0
            for b in prediction_positions.tolist():
                if b:
                    first_len += 1
            words = [vocab.itos[x] for x in predicted_idx[: first_len]]
            first_tensor = src_tensor[0]
            src_words = [vocab.itos[x] for x in src_tensor[0].tolist()[: lens[0]]]
            words_i = 0
            for i, w in enumerate(src_words):
                if prediction_positions[i]:
                    src_words[i] = words[words_i]
                    words_i += 1
            logger.info("predicted:%s", " ".join(src_words))
        logger.debug("predicted_idx len:%d", len(predicted_idx))
        logger.debug("tgt_tensor size:%s", tgt_tensor.size())
        logger.debug("prediction_positions_arr size:%s", prediction_positions_arr.size())
        ground_truth = tgt_tensor.tolist()
        logger.debug("ground_truth len:%d", len(ground_truth))
        token_count_in_batch = len(ground_truth)
        total_token_count += token_count_in_batch
        loss_sum += loss * token_count_in_batch
        acc = metrics.accuracy_score(ground_truth, predicted_idx)
        total_hit_count += acc * token_count_in_batch
        ppl = math.exp(loss_sum / total_token_count)
        if should_print:
            logger.info("ppl:%f acc:%f correct:%d total:%d", ppl,
                    float(total_hit_count) / total_token_count,
                    total_hit_count, total_token_count)
    logger.info("ppl:%f acc:%f correct:%d total:%d", ppl,
            float(total_hit_count) / total_token_count, total_hit_count, total_token_count)
    logger.debug("stoi len:%d", len(vocab.stoi))

    logger.debug("vocab len:%d", len(vocab))
    logger.debug("stoi len:%d", len(vocab.stoi))
    logger.info("evaluating dev set...")
    dev_ppl = common.evaluate(model, dev_samples, vocab)
    logger.info("dev ppl:%s", dev_ppl)

    if dev_ppl < best_dev_ppl:
        best_epoch_i = epoch_i
        best_dev_ppl = dev_ppl
        logger.info("new best results")
        logger.info("laozhongyi_%f", best_dev_ppl)
        stagnation_epochs = 0
        check_point.saveCheckPoint(model, optimizer, vocab, step, epoch_i)
    else:
        stagnation_epochs += 1
        logger.info("stagnation_epochs:%d", stagnation_epochs)
    logger.info("best epoch:%d dev_ppl:%f", best_epoch_i, best_dev_ppl)

    logger.debug("vocab len:%d", len(vocab))
    logger.debug("stoi len:%d", len(vocab.stoi))
    training_set, training_generator = buildDatasetAndGenerator(training_samples, vocab.stoi,
            len(vocab))
