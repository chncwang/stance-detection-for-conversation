import configs
import dataset
import os
import utils
import check_point
import common

model_dir_path = configs.model_dir
logger = utils.getLogger(__file__)

posts = dataset.readConversationSentences("/var/wqs/weibo_dialogue/posts-bpe")
responses = dataset.readConversationSentences("/var/wqs/weibo_dialogue/responses-bpe")
dev_samples = dataset.readLmSentences("/var/wqs/stance-lm/dev", posts, responses, 1)
logger.info("dev samples count:%d", len(dev_samples))

for filename in os.listdir(model_dir_path):
    if filename.startswith("model-"):
        logger.info(filename)
        model, optimizer, vocab, step = check_point.loadCheckPoint(model_dir_path + filename)
        ppl = common.evaluate(model, dev_samples, vocab)
        logger.info("%s ppl:%f", model_dir_path + filename, ppl)
