import torch

device = torch.device("cuda:0")
# device = torch.device("cpu")
evaluation_batch_size = 128
MAX_LEN_FOR_POSITIONAL_ENCODING = 200
lm_training_set_rate = 0.01
model_file = None
