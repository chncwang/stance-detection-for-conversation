import torch
import torch.nn as nn
import torch.autograd as autograd
import configs
import functools
import torch.nn.utils.rnn as rnn
import imp
import sys
import utils

logger = utils.getLogger(__file__)
hyper_params = imp.load_source("module.name", sys.argv[1])

class LstmLm(nn.Module):
    def __init__(self, embedding, vocab_len):
        super(LstmLm, self).__init__()
        randn = lambda : torch.randn(hyper_params.layer,
                hyper_params.batch_size, hyper_params.hidden_dim,
                requires_grad = True).to(device = configs.device)
        self.initial_hiddens = (randn(), randn())
        self.embedding = embedding
        self.lstm = nn.LSTM(hyper_params.word_dim,
                hyper_params.hidden_dim, num_layers = hyper_params.layer)
        logger.debug("vocab_len:%d", vocab_len)
        self.hidden_to_embedding = nn.Linear(hyper_params.hidden_dim,
                hyper_params.word_dim)
        embedding_to_word_id = nn.Linear(hyper_params.word_dim, vocab_len,
                bias = False)
        embedding_to_word_id.weight = embedding.weight
        self.embedding_to_word_id = embedding_to_word_id

    def forward(self, src_sentence_tensor, sentence_lens):
        word_vectors = self.embedding(src_sentence_tensor)
        nn.Dropout(p = hyper_params.dropout, inplace = True)(word_vectors)
        packed_tensor = rnn.pack_padded_sequence(word_vectors, sentence_lens,
                batch_first = True, enforce_sorted = False)
        hiddens = self.lstm(packed_tensor, self.initial_hiddens)[0]
        hiddens = rnn.pad_packed_sequence(hiddens, batch_first = True)[0]
        nn.Dropout(p = hyper_params.dropout, inplace = True)(hiddens)
        logger.debug("hiddens size:%s", hiddens.size())
        logger.debug("hiddens%s", hiddens)
        embeddings = self.hidden_to_embedding(hiddens)
        logger.debug("embeddings:%s", embeddings)
        word_ids = self.embedding_to_word_id(embeddings)
        logger.debug("word_ids size:%s", word_ids.size())
        logger.debug("word_ids:%s", word_ids)
        word_ids = word_ids.permute(0, 2, 1)
        logger.debug("word_ids size:%s", word_ids.size())
        return nn.LogSoftmax(2)(word_ids)
