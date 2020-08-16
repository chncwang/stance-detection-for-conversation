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
        self.l2r_lstm = nn.LSTM(hyper_params.word_dim,
                hyper_params.hidden_dim, num_layers = hyper_params.layer)
        self.r2l_lstm = nn.LSTM(hyper_params.word_dim,
                hyper_params.hidden_dim, num_layers = hyper_params.layer)
        logger.debug("vocab_len:%d", vocab_len)
        self.hidden_to_embedding = nn.Linear(hyper_params.hidden_dim,
                hyper_params.word_dim)
        embedding_to_word_id = nn.Linear(hyper_params.word_dim, vocab_len,
                bias = False)
        embedding_to_word_id.load_state_dict(embedding.state_dict())
        self.embedding_to_word_id = embedding_to_word_id
        self.dropout = nn.Dropout(p = hyper_params.dropout, inplace = True)
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, l2r_src_sentence_tensor, r2l_src_sentence_tensor,
            sentence_lens):
        results = [None] * 2
        for results_i, (src_sentence_tensor, lstm) in\
                enumerate(zip(
                        [l2r_src_sentence_tensor, r2l_src_sentence_tensor],
                        [self.l2r_lstm, self.r2l_lstm])):
            word_vectors = self.embedding(src_sentence_tensor)
            self.dropout(word_vectors)
            packed_tensor = rnn.pack_padded_sequence(word_vectors,
                    sentence_lens, batch_first = True, enforce_sorted = False)
            hiddens = lstm(packed_tensor, self.initial_hiddens)[0]
            hiddens = rnn.pad_packed_sequence(hiddens, batch_first = True)[0]
            self.dropout(hiddens)
            logger.debug("hiddens size:%s", hiddens.size())
            embeddings = self.hidden_to_embedding(hiddens)
            word_ids = self.embedding_to_word_id(embeddings)
            logger.debug("word_ids size:%s", word_ids.size())
            len_list = sentence_lens.tolist()
            logger.debug("len_list:%s", len_list)
            len_sum = sum(len_list)
            len_max = max(len_list)
            to_cat_list = []
            for idx, (vector, length) in enumerate(zip(torch.split(word_ids, 1),
                    sentence_lens.tolist())):
                logger.debug("idx:%d", idx)
                vector = vector.reshape(vector.size()[1], vector.size()[2])
                logger.debug("vector size:%s", vector.size())
                t = torch.split(vector, [length, len_max - length])
                logger.debug("t[0] size:%s", t[0].size())
                to_cat_list.append(t[0])
            to_cat_tuple = tuple(to_cat_list)
            concated = torch.cat(to_cat_tuple, 0)
            logger.debug("concated size:%s", concated.size())
            results[results_i] = self.log_softmax(concated)
        return results
