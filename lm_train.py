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
import random

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

training_samples = readSentences("/var/wqs/stance-lm/train",
        rate = configs.lm_training_set_rate)
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
word_vectors = torchtext.vocab.Vectors(
        "/var/wqs/cn_embeddings/sgns.weibo.bigram-char")

if not hyper_params.embedding_tuning:
    for k in counter.keys():
        if counter[k] < hyper_params.min_freq and k in word_vectors.stoi:
            counter[k] = 10000

def isZero(tensor):
    for i in range(tensor.size()[0]):
        if abs(tensor[i]) > 1e-10:
            return False
    return True

# def randomInitUnk(vocab_):
#     vocab = torchtext.vocab.Vocab(counter, min_freq = hyper_params.min_freq)
#     logger.info("vocab len:%d", len(vocab))
#     logger.debug("stoi len:%d", len(vocab.stoi))
#     logger.debug("unk id:%d sep id:%d", vocab.stoi["<unk>"],
#             vocab.stoi["<sep>"])
#     vocab.load_vectors(word_vectors)
#     logger.debug("unk id:%d sep id:%d", vocab.stoi["<unk>"],
#             vocab.stoi["<sep>"])
#     embedding_table = nn.Embedding.from_pretrained(vocab.vectors,
#             freeze = True).to(device = configs.device)
#     l = embedding_table.weight.tolist()
#     logger.debug("weight size: %s", embedding_table.weight.size())
#     dim = embedding_table.weight.size()[1]
#     sums = [0.0] * dim
#     for v in l:
#         for i in range(dim):
#             sums[i] += v[i]
#     avgs = [x / len(l) for x in sums]
#     logger.debug("avgs:%s", avgs)
#     del vocab
#     del embedding_table
#     avgs = torch.FloatTensor(avgs)
#     emb_vectors = vocab_.vectors
#     logger.debug("emb_vectors:%s", emb_vectors)
#     pad_id = vocab_.stoi["<pad>"]
#     for i in range(emb_vectors.size()[0]):
#         if i != pad_id and isZero(emb_vectors[i]):
#             emb_vectors[i] = avgs

vocab, embedding_table = None, None
if configs.model_file is None:
    vocab = torchtext.vocab.Vocab(counter, min_freq = hyper_params.min_freq)
    logger.info("vocab len:%d", len(vocab))
    logger.debug("stoi len:%d", len(vocab.stoi))
    logger.debug("unk id:%d sep id:%d", vocab.stoi["<unk>"],
            vocab.stoi["<sep>"])
    vocab.load_vectors(word_vectors)
#     randomInitUnk(vocab)
    logger.debug("unk id:%d sep id:%d", vocab.stoi["<unk>"],
            vocab.stoi["<sep>"])
    embedding_table = nn.Embedding.from_pretrained(vocab.vectors,
            freeze = not hyper_params.embedding_tuning).to(
                    device = configs.device)
    logger.debug("unk:%s sep:%s pad:%s",
            embedding_table(torch.LongTensor([0]).to(device = configs.device)),
            embedding_table(torch.LongTensor([4]).to(device = configs.device)),
            embedding_table(torch.LongTensor([1]).to(device = configs.device)))

def wordIndexes(words, stoi, vocab_len):
    return [stoi[word] for word in words]

def pad_batch(word_ids_arr, lenghs):
    tensor = torch.ones(len(word_ids_arr), max(lenghs), dtype = int)
    for idx, (ids, seq_len) in enumerate(zip(word_ids_arr, lenghs)):
        x = torch.LongTensor(ids)
        tensor[idx, :seq_len] = x
    return tensor

def applyLangMask(ids_arr, mask_id, vocab_size, max_len):
    prediction_positions_arr = [None] * len(ids_arr)

    logger.debug("ids_arr len:%d", len(ids_arr))
    for ids_arr_i, ids in enumerate(ids_arr):
        if ids_arr_i % 10000 == 0:
            logger.info("applying mask... %f", float(ids_arr_i) / len(ids_arr))
        l = []
        for i in range(max_len):
            r = random.random()
            if i >= len(ids):
                l.append(False)
                continue
            if r < 0.15:
                l.append(True)
                if r < 0.12:
                    ids[i] = mask_id
                elif r < 0.135:
                    ids[i] = random.randint(0, vocab_size - 1)
            else:
                l.append(False)
        prediction_positions_arr[ids_arr_i] = l

    result = torch.BoolTensor(prediction_positions_arr)
    logger.debug("result size:%s", result.size())
    return result

def targetIdsList(src_sentence_ids_arr, prediction_positions_arr):
    target_ids_arr = [None] * len(src_sentence_ids_arr)
    for i, (ids, positions) in enumerate(zip(src_sentence_ids_arr,
            prediction_positions_arr)):
        logger.debug("positions:%s", positions)
        target_ids = torch.LongTensor([ids[p] for p in positions])
        target_ids_arr[i] = target_ids
    return target_ids_arr

def buildDataset(samples, stoi, vocab_len):
    words_arr = [s.split(" ") for s in samples]

    logger.info("transfering words to ids...")
    src_sentences_indexes_arr = [wordIndexes(s, stoi, vocab_len) for s in\
            words_arr]
    tgt_sentences_indexes_arr = [wordIndexes(s, stoi, vocab_len) for s in\
            words_arr]
    mask_id = stoi["<mask>"]
    sentence_lens = [len(s) for s in words_arr]
    prediction_positions_arr = applyLangMask(src_sentences_indexes_arr,
            mask_id, vocab_len, max(sentence_lens))
    logger.debug("prediction_positions_arr len:%d",
            len(prediction_positions_arr))
    logger.debug("prediction_positions_arr size:%s",
                prediction_positions_arr.size())
    src_key_padding_mask = utils.srcMask(src_sentences_indexes_arr,
            sentence_lens)
    src_sentence_tensor = pad_batch(src_sentences_indexes_arr, sentence_lens)
    tgt_sentence_tensor = pad_batch(tgt_sentences_indexes_arr, sentence_lens)

    return dataset.LmDataset(src_sentence_tensor, tgt_sentence_tensor,
            src_key_padding_mask, prediction_positions_arr, sentence_lens)

def saveCheckPoint(model, optimizer, vocab, step, epoch):
    state = {"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "vocab": vocab}
    path = "model-" + str(epoch) + "-" + datetime.datetime.now().strftime(
            "%Y-%m-%d-%H-%M")
    logger.info("path:%s", path)
    logger.info("saving model...")
    torch.save(state, path)

def loadCheckPoint(path):

    state = torch.load(path)
    vocab = state["vocab"]
    embedding_table = nn.Embedding(len(vocab), hyper_params.word_dim)
    model = lm_module.TransformerLm(embedding_table, len(vocab),
            configs.MAX_LEN_FOR_POSITIONAL_ENCODING).to(
                    device = configs.device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3,
            weight_decay = hyper_params.weight_decay)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    step = state["step"]

    return model, optimizer, vocab, step

model, optimizer, step = None, None, None
if configs.model_file is None:
    model = lm_module.TransformerLm(embedding_table, len(vocab),
            configs.MAX_LEN_FOR_POSITIONAL_ENCODING).\
                    to( device = configs.device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3,
            weight_decay = hyper_params.weight_decay)
    step = 0
else:
    logger.info("loading %s...", configs.model_file)
    model, optimizer, vocab, step = loadCheckPoint(configs.model_file)

def buildDatasetAndGenerator(samples, stoi, vocab_len):
    training_set = buildDataset(samples, stoi, vocab_len)

    data_loader_params = {
            "batch_size": hyper_params.batch_size,
            "shuffle": True }
    training_generator = torch.utils.data.DataLoader(training_set,
            **data_loader_params)
    return training_set, training_generator

training_set, training_generator = buildDatasetAndGenerator(training_samples,
        vocab.stoi, len(vocab))

PAD_ID = vocab.stoi["<pad>"]
if PAD_ID != 1:
    logger.error("PAD_ID is %d", PAD_ID)
    sys.exit(1)

CPU_DEVICE = torch.device("cpu")

def evaluate(model, samples):
    model.eval()
    with torch.no_grad():
        for i in range(10):
            logger.debug("vocab len:%d", len(vocab))
            logger.debug("stoi len:%d", len(vocab.stoi))
            evaluation_set = buildDataset(samples, vocab.stoi, len(vocab))
            logger.debug("vocab len:%d", len(vocab))
            logger.debug("stoi len:%d", len(vocab.stoi))
            evaluation_loader_params = {
                    "batch_size": int(max(1, hyper_params.batch_size / 4)),
                    "shuffle": False }
            logger.debug("evaluation_generator...")
            evaluation_generator = torch.utils.data.DataLoader(evaluation_set,
                **evaluation_loader_params)
            loss_sum = 0.0
            dataset_len_sum = 0
            for src_tensor, tgt_tensor, src_key_padding_mask,\
                    prediction_positions_arr, lens in evaluation_generator:
                logger.debug("src_tensor.to...")
                src_tensor = src_tensor.to(device = configs.device)
                logger.debug("src_tensor size:%s tgt_tensor size:%s",
                        src_tensor.size(), tgt_tensor.size())
                logger.debug("predicted...")
                predicted = model(src_tensor, lens, src_key_padding_mask,
                        prediction_positions_arr)
                logger.debug("tgt_tensor...")
                tgt_tensor = tgt_tensor[prediction_positions_arr]
                tgt_tensor = tgt_tensor.to(device = configs.device)
                loss = nn.NLLLoss()(predicted, tgt_tensor)
                logger.debug("tgt_tensor size:%s", tgt_tensor.size())
                len_sum = len(tgt_tensor)
                loss_sum += loss * len_sum
                dataset_len_sum += len_sum
    model.train()
    return math.exp(loss_sum / dataset_len_sum)

stagnation_epochs = 0
best_epoch_i = 0
best_dev_ppl = 1e100
ppl = 0
for epoch_i in itertools.count(0):
#     if stagnation_epochs >= 2:
#         break
    batch_i = -1
    loss_sum = 0.0
    logger.info("epoch:%d batch count:%d", epoch_i,
            len(training_samples) / hyper_params.batch_size)
    total_token_count = 0
    total_hit_count = 0
    for src_tensor, tgt_tensor, src_key_padding_mask,\
            prediction_positions_arr, lens in training_generator:
        logger.debug("prediction_positions_arr size:%s",
                prediction_positions_arr.size())
        step += 1
        batch_i += 1
        should_print = batch_i * hyper_params.batch_size % 1000 == 0
        lr = math.pow(hyper_params.hidden_dim, -0.5) * min(pow(step, -0.5),
                step * pow(hyper_params.warm_up_steps, -1.5))
        lr *= hyper_params.learning_rate
        if should_print:
            logger.info("lr:%f", lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        logger.debug("src_tensor size:%s", src_tensor.size())
        logger.debug("src_key_padding_mask size:%s",
                src_key_padding_mask.size())

        if should_print:
            logger.info("batch_i:%d", batch_i)
            t = src_tensor[0]
            words = [vocab.itos[x] for x in t if x != PAD_ID]
            logger.info("sentence:%s", " ".join(words))
            words = [vocab.itos[x] for x in tgt_tensor[0] if x != PAD_ID]
            logger.info("sentence:%s", " ".join(words))

        model.zero_grad()
        src_tensor = src_tensor.to(device = configs.device)
        predicted = model(src_tensor, lens, src_key_padding_mask,
                prediction_positions_arr)
        logger.debug("predicted size:%s", predicted.size())
        logger.debug("original predicted:%s", predicted)
        tgt_tensor = tgt_tensor.to(device = configs.device)
        logger.debug("tgt_tensor size:%s", tgt_tensor.size())
        tgt_tensor = tgt_tensor[prediction_positions_arr]
        logger.debug("prediction_positions_arr size:%s",
                prediction_positions_arr.size())
        logger.debug("prediction_positions_arr:%s", prediction_positions_arr)
        logger.debug("tgt_tensor size:%s", tgt_tensor.size())
        loss = nn.NLLLoss()(predicted, tgt_tensor)
        loss.backward()

        if hyper_params.clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(),
                    hyper_params.clip_grad)
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
            src_words = [vocab.itos[x] for x in src_tensor[0].tolist()\
                    [: lens[0]]]
            words_i = 0
            for i, w in enumerate(src_words):
                if prediction_positions[i]:
                    src_words[i] = words[words_i]
                    words_i += 1
            logger.info("predicted:%s", " ".join(src_words))
        logger.debug("predicted_idx len:%d", len(predicted_idx))
        logger.debug("tgt_tensor size:%s", tgt_tensor.size())
        logger.debug("prediction_positions_arr size:%s",
                prediction_positions_arr.size())
        ground_truth = tgt_tensor.tolist()
        logger.debug("ground_truth len:%d", len(ground_truth))
        token_count_in_batch = len(ground_truth)
        total_token_count += token_count_in_batch
        loss_sum += loss * token_count_in_batch
        acc = metrics.accuracy_score(ground_truth, predicted_idx)
        total_hit_count += acc * token_count_in_batch
        ppl = math.exp(loss_sum / total_token_count)
    logger.info("ppl:%f acc:%f correct:%d total:%d", ppl,
            float(total_hit_count) / total_token_count, total_hit_count,
            total_token_count)
    logger.debug("stoi len:%d", len(vocab.stoi))

    logger.debug("vocab len:%d", len(vocab))
    logger.debug("stoi len:%d", len(vocab.stoi))
    logger.info("evaluating dev set...")
#     dev_ppl = evaluate(model, dev_samples)
    dev_ppl = 1
    logger.info("dev ppl:%s", dev_ppl)

    if dev_ppl < best_dev_ppl:
        best_epoch_i = epoch_i
        best_dev_ppl = dev_ppl
        logger.info("new best results")
        logger.info("laozhongyi_%f", best_dev_ppl)
        stagnation_epochs = 0
#         saveCheckPoint(model, optimizer, vocab, lr, epoch_i)
    else:
        stagnation_epochs += 1
        logger.info("stagnation_epochs:%d", stagnation_epochs)
    logger.info("best epoch:%d dev_ppl:%f", best_epoch_i, best_dev_ppl)

    logger.debug("vocab len:%d", len(vocab))
    logger.debug("stoi len:%d", len(vocab.stoi))
#     training_set, training_generator = buildDatasetAndGenerator(
#             training_samples, vocab.stoi, len(vocab))
