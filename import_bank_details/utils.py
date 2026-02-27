import logging
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the configuration from the YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML content is invalid.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
            logger.info("Configuration file loaded successfully.")
            return config  # type: ignore[no-any-return]
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as exc:
        logger.error(f"Error loading configuration file: {exc}")
        raise
