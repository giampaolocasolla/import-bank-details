import logging
import os
from datetime import datetime
from logging import FileHandler


def setup_logging(log_dir: str = ".log") -> None:
    """
    Set up logging configuration to include timestamps and display logs in the terminal.
    Each run creates a new log file with a timestamped filename.

    Args:
        log_dir: Directory to store log files. Defaults to ".log".
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"app_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Suppress verbose logging from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
