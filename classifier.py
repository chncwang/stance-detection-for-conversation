import torch
import math
import torch.nn as nn
import torch.autograd as autograd
import configs
import functools
import torch.nn.utils.rnn as rnn
import utils
import imp
import sys
import positional

logger = utils.getLogger(__file__)

hyper_params = imp.load_source("module.name", sys.argv[1])

class TransformerClassifier(nn.Module):
    def __init__(self, embedding, max_len_for_positional_encoding):
        super(TransformerClassifier, self).__init__()
        self.embedding = embedding
        self.input_linear = nn.Linear(hyper_params.word_dim, hyper_params.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hyper_params.hidden_dim, 8)
        self.transformer = nn.TransformerEncoder(encoder_layer, hyper_params.layer)
        self.mlp_to_label = nn.Linear(hyper_params.hidden_dim * 2, 3)
        self.dropout = nn.Dropout(p = hyper_params.dropout, inplace = True)
        self.positional_encoding = positional.PositionalEncoding(hyper_params.hidden_dim,
                dropout = hyper_params.dropout, max_len = max_len_for_positional_encoding)

    def forward(self, sentence_tensor, sentence_lens, src_key_padding_mask):
        logger.debug("src_key_padding_mask size:%s", src_key_padding_mask.size())
        src_key_padding_mask = src_key_padding_mask.to(device = configs.device)
        logger.debug("sentence_tensor size:%s", sentence_tensor.size())
        batch_size = sentence_tensor.size()[0]
        word_vectors = self.embedding(sentence_tensor).to(device = configs.device)
        word_vectors = self.input_linear(word_vectors)
        word_vectors = word_vectors * math.sqrt(hyper_params.hidden_dim)
        word_vectors = word_vectors.permute(1, 0, 2)
        word_vectors = self.positional_encoding(word_vectors)
        self.dropout(word_vectors)
        hiddens = self.transformer(word_vectors, src_key_padding_mask = src_key_padding_mask)
        self.dropout(hiddens)
        logger.debug("hiddens size:%s", hiddens.size())
        hiddens = hiddens.permute(1, 0, 2)
        logger.debug("hiddens size:%s", hiddens.size())
        max_len = hiddens.size()[1]

        for idx, sent_hiddens in enumerate(hiddens):
            x = torch.FloatTensor([-math.inf] * hyper_params.hidden_dim *
                    (max_len - sentence_lens[idx])).view(max_len - sentence_lens[idx],
                                    hyper_params.hidden_dim)
            logger.debug("x size:%s hiddens[idx] size:%s len:%d", x.size(), hiddens[idx].size(),
                    sentence_lens[idx])
            sent_hiddens[sentence_lens[idx]:] = x

        hiddens = hiddens.permute(0, 2, 1)
        logger.debug("hiddens:%s", hiddens)
        max_pooled = nn.MaxPool1d(max_len)(hiddens)
        max_pooled = max_pooled.permute(0, 2, 1)
        avg_pooled = nn.AvgPool1d(max_len)(hiddens)
        avg_pooled = avg_pooled.permute(0, 2, 1)
        pooled = torch.cat((max_pooled, avg_pooled), 2)
        output = self.mlp_to_label(pooled)
        return output.view(batch_size, 3)
