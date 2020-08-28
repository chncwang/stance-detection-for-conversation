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

class TransformerLm(nn.Module):
    def __init__(self, embedding, vocab_len, max_len_for_positional_encoding):
        super(TransformerLm, self).__init__()
        self.embedding = embedding
        self.input_linear = nn.Linear(hyper_params.word_dim, hyper_params.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hyper_params.hidden_dim, 8, activation = "gelu")
        self.dropout = nn.Dropout(p = hyper_params.dropout, inplace = True)
        self.positional_encoding = positional.PositionalEncoding(hyper_params.hidden_dim,
                dropout = hyper_params.dropout, max_len = max_len_for_positional_encoding)
        self.transformer = nn.TransformerEncoder(encoder_layer, hyper_params.layer)
        self.hidden_to_embedding = nn.Linear(hyper_params.hidden_dim, hyper_params.word_dim)
        embedding_to_word_id = nn.Linear(hyper_params.word_dim, vocab_len, bias = False)
        embedding_to_word_id.weight = embedding.weight
        self.embedding_to_word_id = embedding_to_word_id
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, sentence_tensor, sentence_lens, src_key_padding_mask,
            prediction_position_tensors):
        logger.debug("src_key_padding_mask size:%s", src_key_padding_mask.size())
        src_key_padding_mask = src_key_padding_mask.to(device = configs.device)
        logger.debug("sentence_tensor:%s", sentence_tensor)
        logger.debug("sentence_tensor size:%s", sentence_tensor.size())
        batch_size = sentence_tensor.size()[0]
        word_vectors = self.embedding(sentence_tensor).to(device = configs.device)
        word_vectors = self.input_linear(word_vectors)
        word_vectors = word_vectors * math.sqrt(hyper_params.hidden_dim)
        word_vectors = word_vectors.permute(1, 0, 2)
        logger.debug("word_vectors size:%s", word_vectors.size())
        word_vectors = self.positional_encoding(word_vectors)
        hiddens = self.transformer(word_vectors, src_key_padding_mask = src_key_padding_mask)
        hiddens = hiddens.permute(1, 0, 2)
        logger.debug("hiddens size:%s", hiddens.size())
        hiddens = hiddens[prediction_position_tensors]
        self.dropout(hiddens)
        decoder_embeddings = self.hidden_to_embedding(hiddens)
        word_ids = self.embedding_to_word_id(decoder_embeddings)

        return self.log_softmax(word_ids)
