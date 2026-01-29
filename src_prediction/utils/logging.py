import os
import warnings
import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("src_prediction")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def suppress_warnings() -> None:
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
