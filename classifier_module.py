import torch
import torch.nn as nn
import torch.autograd as autograd
import configs
import functools
import torch.nn.utils.rnn as rnn
import imp
import sys
import logging
import log_config
import utils

logger = utils.getLogger(__file__)
hyper_params = imp.load_source("module.name", sys.argv[1])

class LSTMClassifier(nn.Module):
    def __init__(self, embedding):
        super(LSTMClassifier, self).__init__()
        randn = lambda : torch.randn(hyper_params.layer,
                hyper_params.batch_size, hyper_params.hidden_dim,
                requires_grad = True).to(device = configs.device)
        self.initial_hiddens = (randn(), randn())
        self.embedding = embedding
        self.left_to_right_lstm = nn.LSTM(hyper_params.word_dim,
                hyper_params.hidden_dim, bidirectional = False,
                num_layers = hyper_params.layer)
        logger.debug("left_to_right_lstm all_weights:%s",
                self.left_to_right_lstm.all_weights)
        self.dropout = nn.Dropout(p = hyper_params.dropout, inplace = True)
        self.mlp_to_label = nn.Linear(hyper_params.hidden_dim, 3)

    def forward(self, sentence_tensor, lens):
        batch_size = sentence_tensor.size()[0]
        word_vectors = self.embedding(sentence_tensor)
        self.dropout(word_vectors)
        packed_tensor = rnn.pack_padded_sequence(word_vectors, lens,
                batch_first = True, enforce_sorted = False)
        hiddens = self.left_to_right_lstm(packed_tensor,
                self.initial_hiddens)[0]
        hiddens = rnn.pad_packed_sequence(hiddens, batch_first = True)[0]
        self.dropout(hiddens)
        hiddens = hiddens.permute(0, 2, 1)
        seq_len = hiddens.size()[2]
        sentence_pooled = nn.MaxPool1d(seq_len)(hiddens)
        sentence_pooled = sentence_pooled.permute(0, 2, 1)
        output = self.mlp_to_label(sentence_pooled)
        return output.view(batch_size, 3)
