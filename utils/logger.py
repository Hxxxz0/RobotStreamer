import logging
import os
import sys


def get_logger(out_dir):
    logger = logging.getLogger("MotionDiffusionCore")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_hdlr)
        logger.addHandler(strm_hdlr)
    return logger
