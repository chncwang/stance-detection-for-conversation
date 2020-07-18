import torch
import dataset

posts = dataset.readConversationSentences("/var/wqs/weibo_dialogue/posts")
responses = dataset.readConversationSentences("/var/wqs/weibo_dialogue/responses")
training_samples = dataset.readTrainingSamples(\
        "/var/wqs/conversation-stance-corpus/overall_filtered/overall_filtered_train",
        posts, responses)
training_set = dataset.Dataset(training_samples)
