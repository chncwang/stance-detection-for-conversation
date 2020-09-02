from logging import getLogger, DEBUG, INFO, CRITICAL, ERROR, WARNING
import logging
import datetime

FORMAT = "%(levelname)s - %(asctime)s - %(filename)s - %(funcName)s -\
%(lineno)d - %(message)s"
LOG_FILENAME = "log-{}".format(datetime.datetime.now()).replace(" ", "-")
logging.basicConfig(format = FORMAT,
#         filename = LOG_FILENAME,
#         filename = "log",
        level = logging.INFO)

# getLogger("stance_detection_train").setLevel(DEBUG)
# getLogger("dataset").setLevel(DEBUG)
# getLogger("lm_train").setLevel(DEBUG)
# getLogger("lm_module").setLevel(DEBUG)
# getLogger("positional").setLevel(DEBUG)
# getLogger("evaluate").setLevel(DEBUG)
# getLogger("common").setLevel(DEBUG)
# getLogger("classifier").setLevel(DEBUG)
