import os
import sys
import logging
from typing import Optional


class UTF8StreamHandler(logging.StreamHandler):
    """Custom stream handler that forces UTF-8 encoding on Windows"""

    def __init__(self, stream=None):
        super().__init__(stream)
        if hasattr(self.stream, 'reconfigure'):
            try:
                self.stream.reconfigure(encoding='utf-8', errors='replace')
            except:
                pass


def setup_logger(name: str,
                 log_file: Optional[str] = None,
                 level: int = logging.INFO,
                 log_dir: str = 'logs') -> logging.Logger:
    """
    Set up a logger with UTF-8 support and consistent formatting.

    Args:
        name: Logger name (usually __name__)
        log_file: Optional log filename (if None, uses name + '.log')
        level: Logging level
        log_dir: Directory for log files

    Returns:
        Configured logger
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename if not provided
    if log_file is None:
        log_file = f"{name.replace('.', '_')}.log"

    log_path = os.path.join(log_dir, log_file)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler with UTF-8
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler with UTF-8
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one with default settings."""
    return logging.getLogger(name) if logging.getLogger(name).handlers else setup_logger(name)
