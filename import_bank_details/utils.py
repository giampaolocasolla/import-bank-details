import logging
from typing import Dict

import yaml

# Get the logger instance
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load the configuration from the YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as stream:
        try:
            config = yaml.safe_load(stream)
            logger.info("Configuration file loaded successfully.")
            return config
        except yaml.YAMLError as exc:
            logger.error("Error loading configuration file:", exc)
            raise
