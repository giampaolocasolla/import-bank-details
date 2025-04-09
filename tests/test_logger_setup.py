"""Tests for the logger_setup module."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from import_bank_details.logger_setup import setup_logging


def test_setup_logging_basic():
    """Test the basic functionality of setup_logging without file operations."""
    # Save original handlers to restore later
    original_handlers = logging.getLogger().handlers.copy()

    # Test with mocked file handler
    with patch("logging.FileHandler", MagicMock()):
        with patch("os.makedirs") as mock_makedirs:
            # Call the setup_logging function
            setup_logging()

            # Check if the log directory creation was attempted
            mock_makedirs.assert_called_once_with(".log", exist_ok=True)

            # Get the root logger
            logger = logging.getLogger()

            # Check if the logger level is set to DEBUG
            assert logger.level == logging.DEBUG

            # Check if handlers were added (should be 2)
            assert len(logger.handlers) > len(original_handlers)

    # Clean up - restore original handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        if handler not in original_handlers:
            logger.removeHandler(handler)


@pytest.mark.skip(reason="Log file path is inconsistent in CI environment")
def test_setup_logging(tmpdir):
    """Test the setup_logging function."""
    # Change current working directory to tmpdir to create log files there
    original_cwd = os.getcwd()
    os.chdir(tmpdir)

    try:
        # Call the setup_logging function
        setup_logging()

        # Check if the log directory was created
        assert os.path.exists(".log")

        # Get the root logger
        logger = logging.getLogger()

        # Check if the logger level is set to DEBUG
        assert logger.level == logging.DEBUG

        # Check if handlers are set up correctly
        has_file_handler = False
        has_stream_handler = False

        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                has_file_handler = True
                # The file handler should have DEBUG level (10)
                assert handler.level <= logging.DEBUG
                # Check if the log file was created
                log_file_path = handler.baseFilename
                assert os.path.exists(log_file_path)

            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                has_stream_handler = True
                # The stream handler should have INFO level (20)
                assert handler.level <= logging.INFO

        # Assert that both handlers are present
        assert has_file_handler, "FileHandler was not created"
        assert has_stream_handler, "StreamHandler was not created"

        # Test logging functionality
        test_message = "Test message for logging"
        logger.info(test_message)

        # Check if the message is in the log file
        log_files = os.listdir(os.path.join(tmpdir, ".log"))
        assert len(log_files) > 0

        with open(os.path.join(tmpdir, ".log", log_files[0]), "r") as f:
            log_content = f.read()
            assert test_message in log_content

    finally:
        # Reset handlers to avoid affecting other tests
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Change back to the original working directory
        os.chdir(original_cwd)
