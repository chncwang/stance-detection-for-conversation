import logging
import utils
import os
import configs
import torch
import torch.nn as nn
import torch.optim as optim
import lm_module

def getLogger(filename):
    self_module = os.path.basename(filename[:-3])
    return logging.getLogger(self_module)

logger = getLogger(__file__)

def loadLmCheckPoint(path, hyper_params):
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
