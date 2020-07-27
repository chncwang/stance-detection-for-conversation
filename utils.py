import logging
import os

def getLogger(filename):
    self_module = os.path.basename(filename[:-3])
    return logging.getLogger(self_module)
