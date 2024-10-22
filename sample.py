import enum
import traceback
import sys
import log_config
import utils

logger = utils.getLogger(__file__)

class Stance(enum.IntEnum):
    FAVOR = 0
    AGAINST = 1
    NEUTRAL = 2

def toStance(str):
    if str == "f":
        return Stance.FAVOR
    elif str == "a":
        return Stance.AGAINST
    elif str == "n" or str == "u":
        return Stance.NEUTRAL
    else:
        logger.error(str)
        traceback.print_stack()
        sys.exit(1)

class Sample:
    def __init__(self, post, response, stance):
        self.post = post
        self.response = response
        self.stance = stance

    def __str__(self):
        return "post:{} response:{} stance:{}".format(self.post, self.response,
                self.stance)
