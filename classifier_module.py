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
        self.l2r_initial_hiddens = (randn(), randn())
        self.r2l_initial_hiddens = (randn(), randn())
        self.embedding = embedding
        self.l2r_lstm = nn.LSTM(hyper_params.word_dim,
                hyper_params.hidden_dim, bidirectional = False,
                num_layers = hyper_params.layer)
        self.r2l_lstm = nn.LSTM(hyper_params.word_dim,
                hyper_params.hidden_dim, bidirectional = False,
                num_layers = hyper_params.layer)
        self.dropout = nn.Dropout(p = hyper_params.dropout, inplace = True)
        self.mlp_to_label = nn.Linear(2 * hyper_params.hidden_dim, 3)

    def forward(self, l2r_sentence_tensor, r2l_sentence_tensor, lens):
        batch_size = l2r_sentence_tensor.size()[0]
        hiddens_arr = [None] * 2
        for i, (sentence_tensor, lstm, initial_hiddens) in\
                enumerate(zip([l2r_sentence_tensor, r2l_sentence_tensor],
                        [self.l2r_lstm, self.r2l_lstm],
                        [self.l2r_initial_hiddens, self.r2l_initial_hiddens])):
            word_vectors = self.embedding(sentence_tensor)
            self.dropout(word_vectors)
            packed_tensor = rnn.pack_padded_sequence(word_vectors, lens,
                    batch_first = True, enforce_sorted = False)
            hiddens = lstm(packed_tensor, initial_hiddens)[0]
            hiddens = rnn.pad_packed_sequence(hiddens, batch_first = True)[0]
            hiddens_arr[i] = hiddens

        reversal_hiddens = hiddens_arr[1]
        logger.debug("reversal_hiddens:%s", reversal_hiddens)
        logger.debug("reversal_hiddens size:%s", reversal_hiddens.size())
        hiddens_list = [None] * batch_size
        for i in range(batch_size):
            hiddens = reversal_hiddens[i]
            logger.debug("hiddens:%s", hiddens)
            logger.debug("hiddens size:%s", hiddens.size())
            words, paddings = torch.split(hiddens, [lens[i],
                    hiddens.size()[0] - lens[i]])
            logger.debug("words:%s", words)
            logger.debug("words size:%s", words.size())
            logger.debug("paddings:%s", paddings)
            logger.debug("paddings size:%s", paddings.size())
            words = torch.flip(words, [0])
            logger.debug("flipped words:%s", words)
            hiddens = torch.cat((words, paddings), 0)
            logger.debug("hiddens size:%s", hiddens.size())
            hiddens_list[i] = hiddens
        restored_hiddens = torch.cat(hiddens_list, 0)
        hiddens_arr[1] = reversal_hiddens

        hiddens = torch.cat(hiddens_arr, 2)
        logger.debug("hiddens size:%s", hiddens.size())
        hiddens = hiddens.permute(0, 2, 1)
        seq_len = hiddens.size()[2]
        sentence_pooled = nn.MaxPool1d(seq_len)(hiddens)
        sentence_pooled = sentence_pooled.permute(0, 2, 1)
        logger.debug("sentence pooled size:%s", sentence_pooled.size())
        output = self.mlp_to_label(sentence_pooled)
        return output.view(batch_size, 3)
