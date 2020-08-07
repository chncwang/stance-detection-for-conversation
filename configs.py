import torch

device = torch.device("cuda:0")
evaluation_batch_size = 128
lm_left_to_right = True
lm_training_set_rate = 0.01
