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
        randn = lambda : torch.randn(2, hyper_params.batch_size,
                hyper_params.hidden_dim, requires_grad = True).to(
                        device = configs.device)
        self.post_init_hiddens = (randn(), randn())
        self.response_init_hiddens = (randn(), randn())
        self.embedding = embedding
        self.post_lstm = nn.LSTM(hyper_params.word_dim,
                hyper_params.hidden_dim, bidirectional = True)
        self.response_lstm = nn.LSTM(hyper_params.word_dim,
                hyper_params.hidden_dim, bidirectional = True)
        self.post_hidden_mlp = nn.Linear(hyper_params.hidden_dim * 4,
                hyper_params.hidden_dim)
        self.mlp_to_label = nn.Linear(hyper_params.hidden_dim, 3)

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

    def forward(self, post_tensor, post_lengths, response_tensor,
            response_lengths):
        batch_size = post_tensor.size()[0]
        post_pooled = self.passSentenceToPooled(post_tensor, post_lengths,
                self.post_lstm, self.post_init_hiddens)
        response_pooled = self.passSentenceToPooled(response_tensor,
                response_lengths, self.response_lstm,
                self.response_init_hiddens)
        concat = torch.cat((post_pooled, response_pooled), 1)
        concat = concat.permute(0, 2, 1)
        output = self.post_hidden_mlp(concat)
        output = nn.ReLU()(output)
        output = self.mlp_to_label(output)
        return output.view(batch_size, 3)
