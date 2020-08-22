import logging
import torch
import os

def getLogger(filename):
    self_module = os.path.basename(filename[:-3])
    return logging.getLogger(self_module)

logger = getLogger(__file__)

def printLmHyperParams(hyper_params):
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

def srcMask(word_ids_arr, lens):
    max_len = max(lens)
    tensor = torch.zeros(len(word_ids_arr), max_len, dtype = torch.bool)
    for idx, (ids, seq_len) in enumerate(zip(word_ids_arr, lens)):
        tensor[idx, seq_len:] = torch.FloatTensor([True] * (max_len - seq_len))
    return tensor
