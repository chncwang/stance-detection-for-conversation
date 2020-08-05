import logging
import utils
import os

def getLogger(filename):
    self_module = os.path.basename(filename[:-3])
    return logging.getLogger(self_module)

logger = getLogger(__file__)

def printStanceDetectionHyperParams(hyper_params):
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
