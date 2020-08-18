import sample
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
    def __init__(self, l2r_sentence_tensor, r2l_sentence_tensor, sentence_lens,
            label_tensor):
        self.l2r_sentence_tensor = l2r_sentence_tensor
        self.r2l_sentence_tensor = r2l_sentence_tensor
        self.sentence_lens = sentence_lens
        self.label_tensor = label_tensor

    def __len__(self):
        return len(self.label_tensor)

    def __getitem__(self, idx):
        return self.l2r_sentence_tensor[idx], self.r2l_sentence_tensor[idx],\
                self.sentence_lens[idx], self.label_tensor[idx]

def readLmSentences(path, posts, responses, rate = 1.0):
    lines = readLines(path)
    sentences = []
    for i, line in enumerate(lines):
        if i > rate * len(lines):
            break
        strs = line.split(" ")
        pi, ri = int(strs[0]), int(strs[1])
        p, r = posts[pi], responses[ri]
        sentences.append("<begin> " + p + " <sep> " + r + " <end>")
    logger.debug("len:%d sentences:%s", len(sentences), sentences[:5])
    return sentences

class LmDataset(torch.utils.data.Dataset):
    def __init__(self, src_sentence_tensor, tgt_sentence_tensor,
            sentence_lens):
        self.src_sentence_tensor = src_sentence_tensor
        self.tgt_sentence_tensor = tgt_sentence_tensor
        self.sentence_lens = sentence_lens

    def __len__(self):
        return len(self.sentence_lens)

    def __getitem__(self, idx):
        return self.src_sentence_tensor[idx], self.tgt_sentence_tensor[idx],\
                self.sentence_lens[idx]
