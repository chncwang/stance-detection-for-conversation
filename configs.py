import torch

device = torch.device("cuda:0")
# device = torch.device("cpu")
evaluation_batch_size = 64
MAX_LEN_FOR_POSITIONAL_ENCODING = 600
lm_training_set_rate = 0.001
model_file = None
model_file = "/home/chncwang/Downloads/model-1-2020-08-31-15-38"
# model_dir = "/home/chncwang/Downloads/stance-detection-for-conversation-2/"
