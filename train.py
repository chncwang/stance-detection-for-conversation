import sys
import model
import torch
import torchtext
import configs
import dataset
import hyper_params
import itertools
import collections
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(hyper_params.seed)

posts = dataset.readConversationSentences("/var/wqs/weibo_dialogue/posts")
responses = dataset.readConversationSentences(
        "/var/wqs/weibo_dialogue/responses")

def readSamples(path):
    return dataset.readSamples(path, posts, responses)

training_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/"\
                "overall_filtered_train")
dev_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/"\
                "overall_filtered_dev")
test_samples = readSamples(
        "/var/wqs/conversation-stance-corpus/overall_filtered/"\
                "overall_filtered_test")

to_build_vocb_samples = None
if hyper_params.embedding_tuning:
    to_build_vocb_samples = training_samples
else:
    to_build_vocb_samples = training_samples + dev_samples + test_samples

def sentenceToCounter(sentence, counter):
    words = sentence.split(" ")
    return counter.update(words)

counter = collections.Counter()
for idx, sample in enumerate(to_build_vocb_samples):
    sentenceToCounter(sample.post, counter)
    sentenceToCounter(sample.response, counter)

def oovCount(sentence, counter):
    words = sentence.split(" ")
    oov_count = 0
    for word in words:
        if counter[word] < hyper_params.min_freq:
            oov_count += 1
    return oov_count, len(words)

oov_count = 0
all_words_count = 0
for idx, sample in enumerate(to_build_vocb_samples):
    t = oovCount(sample.post, counter)
    oov_count += t[0]
    all_words_count += t[1]

print("oov:", oov_count / float(all_words_count))

vocab = torchtext.vocab.Vocab(counter, min_freq = hyper_params.min_freq)
print("vocab len:", len(vocab))
embedding_table = nn.Embedding(len(vocab), hyper_params.word_dim)
embedding_table.weight.data.uniform_(-1, 1)

def word_indexes(words, stoi):
    return [stoi[word] for word in words]

def pad_batch(word_ids_arr, lenghs):
    tensor = torch.ones(len(word_ids_arr), max(lenghs), dtype = int)
    for idx, (ids, seq_len) in enumerate(zip(word_ids_arr, lenghs)):
        tensor[idx, :seq_len] = torch.LongTensor(ids)
    return tensor

def buildDataset(samples, stoi):
    post_lens = [0] * len(samples)
    post_indexes_arr = [None] * len(samples)
    response_lens = [0] * len(samples)
    response_indexes_arr = [None] * len(samples)
    labels = [0] * len(samples)

    for i, sample in enumerate(samples):
        post_words = sample.post.split(" ")
        post_len = len(post_words)
        post_lens[i] = post_len
        post_indexes = word_indexes(post_words, stoi)
        post_indexes_arr[i] = post_indexes
        response_words = sample.response.split(" ")
        response_len = len(response_words)
        response_lens[i] = response_len
        response_indexes = word_indexes(response_words, stoi)
        response_indexes_arr[i] = response_indexes
        labels[i] = int(sample.stance)

    post_tensor = pad_batch(post_indexes_arr, post_lens)
    response_tensor = pad_batch(response_indexes_arr, response_lens)
    label_tensor = torch.LongTensor(labels)

    return dataset.Dataset(post_tensor, post_lens, response_tensor,
            response_lens, label_tensor)

training_set = buildDataset(training_samples, vocab.stoi)

data_loader_params = {
        "batch_size": hyper_params.batch_size,
        "shuffle": True }
training_generator = torch.utils.data.DataLoader(training_set,
        **data_loader_params)

model = model.LSTMClassifier(len(vocab))
optimizer = optim.Adam(model.parameters(), lr = hyper_params.learning_rate,
        weight_decay = hyper_params.weight_decay)
PAD_ID = vocab.stoi["<pad>"]

for epoch_i in itertools.count(0):
    if epoch_i > 10:
        break
    batch_i = -1
    for post_tensor, post_lengths, response_tensor, response_lengths,\
            label_tensor in training_generator:
        batch_i += 1

        should_print = batch_i * hyper_params.batch_size % 10000 == 0
        if should_print:
            post_words = [vocab.itos[x] for x in post_tensor[0] if x != PAD_ID]
            print("post:", " ".join(post_words))
            response_words = [vocab.itos[x] for x in response_tensor[0]
                    if x != PAD_ID]
            print("response:", " ".join(response_words))

        predecited = model(post_tensor, post_lengths, response_tensor,
                response_lengths)

        sys.exit(0)
