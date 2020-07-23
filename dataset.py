import sample
import torch

def readLines(path):
    return open(path).read().splitlines()

def readConversationSentences(path):
    return [line.split("##", 1)[1] for line in readLines(path)]

def readSamples(path, posts, responses):
    lines = readLines(path)
    fragments = [line.split(" ") for line in lines]
    return [sample.Sample(posts[int(x[0])], responses[int(x[1])],
            sample.toStance(x[2])) for x in fragments]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, post_tensor, post_lengths, response_tensor,
            response_lengths, label_tensor):
        self.post_tensor = post_tensor
        self.post_lengths = post_lengths
        self.response_tensor = response_tensor
        self.response_lengths = response_lengths
        self.label_tensor = label_tensor

    def __len__(self):
        return len(self.label_tensor)

    def __getitem__(self, idx):
        return self.post_tensor[idx], self.post_lengths[idx],\
                self.response_tensor[idx], self.response_lengths[idx],\
                self.label_tensor[idx]
