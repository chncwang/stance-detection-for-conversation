import torch
import torch.nn as nn
import torch.autograd as autograd
import hyper_params
import configs
import functools
import torch.nn.utils.rnn as rnn

class LSTMClassifier(nn.Module):
    def __init__(self, embedding):
        super(LSTMClassifier, self).__init__()
        randn = lambda : torch.randn(2 * hyper_params.layer,
                hyper_params.batch_size, hyper_params.hidden_dim,
                requires_grad = True).to(device = configs.device)
        self.init_hiddens = (randn(), randn())
        self.embedding = embedding
        self.lstm = nn.LSTM(hyper_params.word_dim,
                hyper_params.hidden_dim, bidirectional = True,
                num_layers = hyper_params.layer)
        self.mlp_to_label = nn.Linear(hyper_params.hidden_dim * 2, 3)

    def forwardSentenceToLstm(self, input_tensor, lengths, lstm, init_hiddens):
        nn.Dropout(p = hyper_params.dropout, inplace = True)(input_tensor)
        packed_tensor = rnn.pack_padded_sequence(input_tensor, lengths,
                batch_first = True, enforce_sorted = False)
        return lstm(packed_tensor, init_hiddens)[0]

    def passSentenceToPooled(self, sentence, lengths, lstm, init_hiddens):
        word_vectors = self.embedding(sentence).to(device = configs.device)
        hiddens = self.forwardSentenceToLstm(word_vectors, lengths, lstm,
                init_hiddens)
        hiddens = rnn.pad_packed_sequence(hiddens, batch_first = True)[0]
        nn.Dropout(p = hyper_params.dropout, inplace = True)(hiddens)
        hiddens = hiddens.permute(0, 2, 1)
        seq_len = hiddens.size()[2]
        return nn.MaxPool1d(seq_len)(hiddens)

    def forward(self, sentence_tensor, sentence_lens):
        batch_size = sentence_tensor.size()[0]
        sentence_pooled = self.passSentenceToPooled(sentence_tensor,
                sentence_lens, self.lstm, self.init_hiddens)
        sentence_pooled = sentence_pooled.permute(0, 2, 1)
        output = self.mlp_to_label(sentence_pooled)
        return output.view(batch_size, 3)
