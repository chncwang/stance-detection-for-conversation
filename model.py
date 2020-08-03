import torch
import math
import torch.nn as nn
import torch.autograd as autograd
import hyper_params
import configs
import functools
import torch.nn.utils.rnn as rnn
import utils

logger = utils.getLogger(__file__)

class TransformerClassifier(nn.Module):
    def __init__(self, embedding):
        super(TransformerClassifier, self).__init__()
        self.embedding = embedding
        self.input_linear = nn.Linear(hyper_params.word_dim,
                hyper_params.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hyper_params.hidden_dim, 8)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                hyper_params.layer)
        self.mlp_to_label = nn.Linear(hyper_params.hidden_dim, 3)

    def forwardSentenceToTransformer(self, input_tensor, lens,
            src_key_padding_mask,
            transformer):
        nn.Dropout(p = hyper_params.dropout, inplace = True)(input_tensor)
        input_tensor = self.input_linear(input_tensor).permute(1, 0, 2)
        logger.debug("input_tensor size:%s", input_tensor.size())
        return transformer(input_tensor,
                src_key_padding_mask = src_key_padding_mask)

    def passSentenceToPooled(self, sentence, lens, src_key_padding_mask,
            transformer):
        word_vectors = self.embedding(sentence).to(device = configs.device)
        hiddens = self.forwardSentenceToTransformer(word_vectors, lens,
                src_key_padding_mask, transformer)
        nn.Dropout(p = hyper_params.dropout, inplace = True)(hiddens)
        logger.debug("hiddens size:%s", hiddens.size())
        hiddens = hiddens.permute(1, 0, 2)
        logger.debug("hiddens size:%s", hiddens.size())
        max_len = hiddens.size()[1]

        for idx, sent_hiddens in enumerate(hiddens):
            x = torch.FloatTensor([-math.inf] * hyper_params.hidden_dim *
                    (max_len - lens[idx])).view(max_len - lens[idx],
                            hyper_params.hidden_dim)
            logger.debug("x size:%s hiddens[idx] size:%s len:%d", x.size(),
                    hiddens[idx].size(), lens[idx])
            sent_hiddens[lens[idx]:] = x

        hiddens = hiddens.permute(0, 2, 1)
        logger.debug("hiddens:%s", hiddens)
        return nn.MaxPool1d(max_len)(hiddens)

    def forward(self, sentence_tensor, sentence_lens, src_key_padding_mask):
        logger.debug("src_key_padding_mask size:%s",
                src_key_padding_mask.size())
        src_key_padding_mask = src_key_padding_mask.to(device = configs.device)
        logger.debug("sentence_tensor size:%s", sentence_tensor.size())
        batch_size = sentence_tensor.size()[0]
        sentence_pooled = self.passSentenceToPooled(sentence_tensor,
                sentence_lens, src_key_padding_mask, self.transformer)
        logger.debug("sentence_pooled size:%s", sentence_pooled.size())
        sentence_pooled = sentence_pooled.permute(0, 2, 1)
        output = self.mlp_to_label(sentence_pooled)
        return output.view(batch_size, 3)
