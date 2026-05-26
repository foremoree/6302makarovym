import logging
import logging.config
import json
import os

def setup_logging():
    logger = logging.getLogger("metetl")

    config_path = "logging_config.json"

    os.makedirs("logs", exist_ok=True)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logging.config.dictConfig(config)
    return logger