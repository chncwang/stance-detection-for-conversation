import sample
import random
import torch
import utils

logger = utils.getLogger(__file__)

def readLines(path):
    return open(path).read().splitlines()

def readConversationSentences(path):
    return [line.split("##", 1)[1] for line in readLines(path)]

def readSamples(path, posts, responses):
    lines = readLines(path)
    fragments = [line.split(" ") for line in lines]
    return [sample.Sample(posts[int(x[0])], responses[int(x[1])],
            sample.toStance(x[2])) for x in fragments]

class StanceDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, sentence_tensor, sentence_lengths, src_mask, label_tensor):
        self.sentence_tensor = sentence_tensor
        self.sentence_lengths = sentence_lengths
        self.src_mask = src_mask
        self.label_tensor = label_tensor

    def __len__(self):
        return len(self.label_tensor)

    def __getitem__(self, idx):
        return self.sentence_tensor[idx], self.sentence_lengths[idx], self.src_mask[idx],\
                self.label_tensor[idx]

def readLmSentences(path, posts, responses, rate = 1.0):
    lines = readLines(path)
    sentences = []
    for i, line in enumerate(lines):
        if i > rate * len(lines):
            break
        strs = line.split(" ")
        pi, ri = int(strs[0]), int(strs[1])
        p, r = posts[pi], responses[ri]
        sentences.append("<cls> " + p + " <sep> " + r)
    logger.debug("len:%d sentences:%s", len(sentences), sentences[:5])
    return sentences

class LmDataset(torch.utils.data.Dataset):
    def __init__(self, src_sentence_tensor, tgt_ids_arr, src_key_padding_mask,
            prediction_positions_arr, sentence_lens):
        self.src_sentence_tensor = src_sentence_tensor
        self.tgt_ids_arr = tgt_ids_arr
        self.src_key_padding_mask = src_key_padding_mask
        self.prediction_positions_arr = prediction_positions_arr
        self.sentence_lens = sentence_lens

    def __len__(self):
        return len(self.sentence_lens)

    def __getitem__(self, idx):
        return self.src_sentence_tensor[idx], self.tgt_ids_arr[idx],\
                self.src_key_padding_mask[idx], self.prediction_positions_arr[idx],\
                self.sentence_lens[idx]

def wordIndexes(words, stoi, vocab_len):
    return [stoi[word] for word in words]

def pad_batch(word_ids_arr, lenghs):
    tensor = torch.ones(len(word_ids_arr), max(lenghs), dtype = int)
    for idx, (ids, seq_len) in enumerate(zip(word_ids_arr, lenghs)):
        x = torch.LongTensor(ids)
        tensor[idx, :seq_len] = x
    return tensor

def applyLangMask(ids_arr, mask_id, vocab_size, max_len, rg, p = 1.0):
    prediction_positions_arr = [None] * len(ids_arr)

    logger.debug("ids_arr len:%d", len(ids_arr))
    for ids_arr_i, ids in enumerate(ids_arr):
        if ids_arr_i % 10000 == 0:
            logger.info("applying mask... %f", float(ids_arr_i) / len(ids_arr))
        l = []
        masked = False
        while not masked:
            l = []
            for i in range(max_len):
                r = rg.random()
                if i >= len(ids):
                    l.append(False)
                    continue
                if r < 0.15 * p:
                    masked = True
                    l.append(True)
                    if r < 0.12 * p:
                        ids[i] = mask_id
                    elif r < 0.135 * p:
                        ids[i] = rg.randint(0, vocab_size - 1)
                else:
                    l.append(False)
            if p < 1:
                break
        prediction_positions_arr[ids_arr_i] = l

    result = torch.BoolTensor(prediction_positions_arr)
    logger.debug("result size:%s", result.size())
    return result

def buildLmDataset(samples, stoi, vocab_len, rg = random):
    words_arr = [s.split(" ") for s in samples]

    logger.info("transfering words to ids...")
    src_sentences_indexes_arr = [wordIndexes(s, stoi, vocab_len) for s in words_arr]
    tgt_sentences_indexes_arr = [wordIndexes(s, stoi, vocab_len) for s in words_arr]
    mask_id = stoi["<mask>"]
    sentence_lens = [len(s) for s in words_arr]
    prediction_positions_arr = applyLangMask(src_sentences_indexes_arr, mask_id, vocab_len,
            max(sentence_lens), rg)
    logger.debug("prediction_positions_arr len:%d", len(prediction_positions_arr))
    logger.debug("prediction_positions_arr size:%s", prediction_positions_arr.size())
    src_key_padding_mask = utils.srcMask(src_sentences_indexes_arr, sentence_lens)
    src_sentence_tensor = pad_batch(src_sentences_indexes_arr, sentence_lens)
    tgt_sentence_tensor = pad_batch(tgt_sentences_indexes_arr, sentence_lens)

    return LmDataset(src_sentence_tensor, tgt_sentence_tensor, src_key_padding_mask,
            prediction_positions_arr, sentence_lens)
