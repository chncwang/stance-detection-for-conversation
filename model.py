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

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, embedding, max_len_for_positional_encoding):
        super(TransformerClassifier, self).__init__()
        self.embedding = embedding
        self.input_linear = nn.Linear(hyper_params.word_dim,
                hyper_params.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hyper_params.hidden_dim, 8)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                hyper_params.layer)
        self.mlp_to_label = nn.Linear(hyper_params.hidden_dim, 3)
        self.dropout = nn.Dropout(p = hyper_params.dropout, inplace = True)
        self.postional_encoding = PositionalEncoding(hyper_params.hidden_dim,
                dropout = hyper_params.dropout,
                max_len = max_len_for_positional_encoding)

    def forwardSentenceToTransformer(self, input_tensor, lens,
            src_key_padding_mask,
            transformer):
        input_tensor = input_tensor.permute(1, 0, 2)
        logger.debug("input_tensor size:%s", input_tensor.size())
        return transformer(input_tensor,
                src_key_padding_mask = src_key_padding_mask)

    def passSentenceToPooled(self, sentence, lens, src_key_padding_mask,
            transformer):
        word_vectors = self.embedding(sentence).to(device = configs.device)
        word_vectors = self.input_linear(word_vectors)
        word_vectors = self.postional_encoding(word_vectors)
        hiddens = self.forwardSentenceToTransformer(word_vectors, lens,
                src_key_padding_mask, transformer)
        self.dropout(hiddens)
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
