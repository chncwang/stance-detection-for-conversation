import datetime
import classifier
import configs
import imp
import sys
import torch
import utils
import torch.nn as nn
import lm_module
import torch.optim as optim

logger = utils.getLogger(__file__)

hyper_params = imp.load_source("module.name", sys.argv[1])

def saveCheckPoint(model, optimizer, vocab, step, epoch):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
            "vocab": vocab}
    path = "model-" + str(epoch) + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger.info("path:%s", path)
    logger.info("saving model...")
    torch.save(state, path)

def loadCheckPoint(path):
    state = torch.load(path)
    vocab = state["vocab"]
    embedding_table = nn.Embedding(len(vocab), hyper_params.word_dim)
    model = lm_module.TransformerLm(embedding_table, len(vocab),
            configs.MAX_LEN_FOR_POSITIONAL_ENCODING).to(device = configs.device)
    optimizer = optim.Adam(model.parameters(), lr = hyper_params.learning_rate,
            weight_decay = hyper_params.weight_decay)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    step = state["step"]

    return model, optimizer, vocab, step

def loadStanceDetectionCheckPoint(path):
    state = torch.load(path)
    vocab = state["vocab"]
    embedding_table = nn.Embedding(len(vocab), hyper_params.word_dim)
    model = classifier.TransformerClassifier(embedding_table,
            configs.MAX_LEN_FOR_POSITIONAL_ENCODING).to(device = configs.device)
    model.load_state_dict(state["model"])
    step = state["step"]

    return model, None, vocab, step
