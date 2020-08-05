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
    def __init__(self, sentence_tensor, sentence_lengths, label_tensor):
        self.sentence_tensor = sentence_tensor
        self.sentence_lengths = sentence_lengths
        self.label_tensor = label_tensor

    def __len__(self):
        return len(self.label_tensor)

    def __getitem__(self, idx):
        return self.sentence_tensor[idx], self.sentence_lengths[idx],\
                self.label_tensor[idx]

def readLmSentences(path, posts, responses):
    lines = readLines(path)
    sentences = [None] * len(lines)
    for i, line in enumerate(lines):
        strs = line.split(" ")
        pi, ri = int(strs[0]), int(strs[1])
        p, r = posts[pi], responses[ri]
        sentences[i] = "<begin> " + p + " <sep> " + r + " <end>"
    logger.debug("len:%d sentences:%s", len(sentences), sentences[:5])
    return sentences

class LmDataset(torch.utils.data.Dataset):
    def __init__(self, sentence_tensor, sentence_lengths):
        self.sentence_tensor = sentence_tensor
        self.sentence_lengths = sentence_lengths

    def __len__(self):
        return len(self.sentence_lengths)

    def __getitem__(self, idx):
        return self.sentence_tensor[idx], self.sentence_lengths[idx]
