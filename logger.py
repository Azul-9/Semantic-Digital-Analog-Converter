"""
used to get logger
"""

import logging
import logging.handlers
import colorlog
from pathlib import Path

def get_logger(name:str, path:Path):
    assert name is not None, "need a name for logger"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:  # it's important to avoid outputing same log
        log_handler = logging.FileHandler(filename=path, mode='w')  # w means overwriting exists log file
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

        log_handler_std = colorlog.StreamHandler()
        log_handler_std.setLevel(logging.INFO)
        log_handler_std.setFormatter(colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(message)s"))

        logger.addHandler(log_handler)
        # logger.addHandler(log_handler_std)

    return logger