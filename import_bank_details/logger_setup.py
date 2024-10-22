# logger_setup.py
import logging
from logging.handlers import RotatingFileHandler


def setup_logging() -> None:
    """
    Set up logging configuration to include timestamps and display logs in the terminal.
    A rotating file handler is also used to limit the log file size and keep backups.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set root logger to DEBUG to capture all levels

    # Create handlers - one for writing to log files and another for streaming to the console
    file_handler = RotatingFileHandler("app.log", maxBytes=1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)  # Set file handler to DEBUG level

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Set stream handler to INFO level

    # Define the logging format to include timestamps
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the root logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
