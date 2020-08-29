import torch
import sys
import random
import configs
import utils
import dataset
import torch.nn as nn
import math

logger = utils.getLogger(__file__)

def evaluate(model, samples, vocab):
    model.eval()
    rg = random.Random(0)
    loss_sum = 0.0
    dataset_len_sum = 0
    with torch.no_grad():
        for i in range(10):
            logger.debug("vocab len:%d", len(vocab))
            logger.debug("stoi len:%d", len(vocab.stoi))
            evaluation_set = dataset.buildLmDataset(samples, vocab.stoi, len(vocab))
            logger.debug("vocab len:%d", len(vocab))
            logger.debug("stoi len:%d", len(vocab.stoi))
            evaluation_loader_params = { "batch_size": int(max(1, 1)),
                    "shuffle": False }
            logger.debug("evaluation_generator...")
            evaluation_generator = torch.utils.data.DataLoader(evaluation_set,
                **evaluation_loader_params)
            for src_tensor, tgt_tensor, src_key_padding_mask, prediction_positions_arr, lens\
                    in evaluation_generator:
                logger.debug("src_tensor.to...")
                src_tensor = src_tensor.to(device = configs.device)
                logger.debug("src_tensor size:%s tgt_tensor size:%s", src_tensor.size(),
                        tgt_tensor.size())
                logger.debug("predicted...")
                predicted = model(src_tensor, lens, src_key_padding_mask, prediction_positions_arr)
                logger.debug("tgt_tensor...")
                tgt_tensor = tgt_tensor[prediction_positions_arr]
                tgt_tensor = tgt_tensor.to(device = configs.device)
                loss = nn.NLLLoss()(predicted, tgt_tensor)
                logger.debug("loss:%f", loss)
                if math.isnan(loss):
                    logger.error("predicted: %s", [math.exp(x) for x in predicted.tolist()])
                    sys.exit(1)
                logger.debug("tgt_tensor size:%s", tgt_tensor.size())
                len_sum = len(tgt_tensor)
                logger.debug("len_sum:%d", len_sum)
                loss_sum += loss * len_sum
                dataset_len_sum += len_sum
    model.train()
    logger.debug("loss_sum:%f dataset_len_sum:%d", loss_sum, dataset_len_sum)
    return math.exp(loss_sum / dataset_len_sum)
