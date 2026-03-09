import os
import logging

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

formatter = logging.Formatter("[%(asctime)s] %(name)s | %(message)s", datefmt="%H:%M:%S")

# Console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# File
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "rag.log"), encoding="utf-8")
file_handler.setFormatter(formatter)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger
