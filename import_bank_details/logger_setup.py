import logging
import os
from datetime import datetime
from logging import FileHandler


def setup_logging() -> None:
    """
    Set up logging configuration to include timestamps and display logs in the terminal.
    Each run creates a new log file with a timestamped filename.
    The log file has a maximum size of 20 MB without splitting into backup files.
    Logs are stored in the `.log` directory.
    """
    # Define the log directory
    log_directory = ".log"

    # Create the log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Generate a timestamp for the log file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # e.g., 2023-09-24_15-30-45
    log_file_name = f"app_{timestamp}.log"
    log_file_path = os.path.join(log_directory, log_file_name)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set root logger to DEBUG to capture all levels

    # Create a rotating file handler with a maximum size of 20 MB and no backups
    file_handler = FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Set file handler to DEBUG level

    # Create a stream handler for console output
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
