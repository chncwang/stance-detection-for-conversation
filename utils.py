import logging
import utils
import os
import configs

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
    logger.info("min_learning_rate:%f", hyper_params.min_learning_rate)
    logger.info("lr_decay:%f", hyper_params.lr_decay)
    logger.info("weight_decay:%f", hyper_params.weight_decay)
    logger.info("clip_grad:%f", hyper_params.clip_grad)
    logger.info("layer:%d", hyper_params.layer)

def printConfigs():
    logger.info("device:%s", configs.device)
    logger.info("evaluation_batch_size:%d", configs.evaluation_batch_size)
    logger.info("lm_left_to_right:%r", configs.lm_left_to_right)
    logger.info("lm_training_set_rate:%f", configs.lm_training_set_rate)

def loadLmCheckPoint(path):
    state = torch.load(path)
    vocab = state["vocab"]
    embedding_table = nn.Embedding(len(vocab), hyper_params.word_dim)
    model = lm_module.LstmLm(embedding_table, len(vocab)).to(
            device = configs.device)
    optimizer = optim.Adam(model.parameters(), lr = hyper_params.learning_rate,
            weight_decay = hyper_params.weight_decay)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    learning_rate = state["learning_rate"]

    return model, optimizer, vocab, learning_rate
