"""Tests for the logger_setup module."""

import logging
import os

from import_bank_details.logger_setup import setup_logging


def test_setup_logging_basic(tmp_path):
    """Test the basic functionality of setup_logging."""
    log_dir = str(tmp_path / "test_logs")

    # Save original handlers to restore later
    logger = logging.getLogger()
    original_handlers = logger.handlers.copy()
    original_level = logger.level

    try:
        setup_logging(log_dir=log_dir)

        # Check that the log directory was created
        assert os.path.exists(log_dir)

        # Check that handlers were added
        new_handlers = [h for h in logger.handlers if h not in original_handlers]
        assert len(new_handlers) == 2

        # Check handler types
        handler_types = {type(h) for h in new_handlers}
        assert logging.FileHandler in handler_types
        assert logging.StreamHandler in handler_types

        # Check that root logger level is DEBUG
        assert logger.level == logging.DEBUG
    finally:
        # Restore original handlers
        for handler in logger.handlers[:]:
            if handler not in original_handlers:
                handler.close()
                logger.removeHandler(handler)
        logger.setLevel(original_level)


def test_setup_logging_creates_log_file(tmp_path):
    """Test that setup_logging creates a log file and writes to it."""
    log_dir = str(tmp_path / "logs")
    logger = logging.getLogger()
    original_handlers = logger.handlers.copy()
    original_level = logger.level

    try:
        setup_logging(log_dir=log_dir)

        # Write a test message
        test_message = "Test message for logging"
        logger.info(test_message)

        # Check the log file was created
        log_files = os.listdir(log_dir)
        assert len(log_files) == 1
        assert log_files[0].startswith("app_")
        assert log_files[0].endswith(".log")

        # Check the message is in the log file
        with open(os.path.join(log_dir, log_files[0]), "r") as f:
            log_content = f.read()
            assert test_message in log_content
    finally:
        for handler in logger.handlers[:]:
            if handler not in original_handlers:
                handler.close()
                logger.removeHandler(handler)
        logger.setLevel(original_level)
