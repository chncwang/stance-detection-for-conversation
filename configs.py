import torch

device = torch.device("cuda:0")
# device = torch.device("cpu")
evaluation_batch_size = 128
MAX_LEN_FOR_POSITIONAL_ENCODING = 600
lm_training_set_rate = 1
model_file = None
# model_file = "./model-0-2020-08-24-18-25"
