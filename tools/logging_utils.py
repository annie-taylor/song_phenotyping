import os
import sys
import logging
from typing import Optional
import platform


class UTF8StreamHandler(logging.StreamHandler):
    """Custom stream handler that handles UTF-8 encoding robustly on Windows"""

    def __init__(self, stream=None):
        super().__init__(stream)
        self.encoding_fixed = False

        # Try to fix encoding
        if hasattr(self.stream, 'reconfigure'):
            try:
                self.stream.reconfigure(encoding='utf-8', errors='replace')
                self.encoding_fixed = True
            except Exception:
                pass

        # Alternative approach for Windows/PyCharm
        if not self.encoding_fixed and platform.system() == "Windows":
            try:
                # Try to wrap the stream with UTF-8 encoding
                import codecs
                if hasattr(self.stream, 'buffer'):
                    self.stream = codecs.getwriter('utf-8')(self.stream.buffer, errors='replace')
                    self.encoding_fixed = True
            except Exception:
                pass

    def emit(self, record):
        """Override emit to handle encoding errors gracefully"""
        try:
            super().emit(record)
        except UnicodeEncodeError:
            # Fallback: replace emojis with ASCII alternatives
            if hasattr(record, 'msg'):
                record.msg = self._replace_emojis(str(record.msg))
            try:
                super().emit(record)
            except Exception as e:
                # Last resort: write to stderr without emojis
                try:
                    safe_msg = self._replace_emojis(str(record.getMessage()))
                    print(f"LOGGING ERROR - {safe_msg}", file=sys.stderr)
                except:
                    print(f"LOGGING ERROR - Unable to display message", file=sys.stderr)

    def _replace_emojis(self, text: str) -> str:
        """Replace common emojis with ASCII alternatives"""
        emoji_map = {
            '🚀': '[START]',
            '🎵': '[MUSIC]',
            '🔍': '[SEARCH]',
            '📊': '[STATS]',
            '📁': '[FOLDER]',
            '🐦': '[BIRD]',
            '🔄': '[PROCESS]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARN]',
            '⏭️': '[SKIP]',
            '💥': '[CRASH]',
            '💾': '[SAVE]',
            '📄': '[FILE]',
            '🎯': '[TARGET]',
            '📋': '[COPY]',
            '📈': '[GRAPH]',
            '🔧': '[TOOL]',
            '⏰': '[TIME]',
        }

        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)

        return text


def setup_logger(name: str,
                 log_file: Optional[str] = None,
                 level: int = logging.INFO,
                 log_dir: str = 'logs',
                 use_emojis: Optional[bool] = None) -> logging.Logger:
    """
    Set up a logger with UTF-8 support and consistent formatting.

    Args:
        name: Logger name (usually __name__)
        log_file: Optional log filename (if None, uses name + '.log')
        level: Logging level
        log_dir: Directory for log files
        use_emojis: Force emoji usage on/off (None = auto-detect)

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

    # File handler with UTF-8 (files usually handle UTF-8 better)
    try:
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create UTF-8 file handler: {e}", file=sys.stderr)
        # Fallback to default encoding
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler with UTF-8
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one with default settings."""
    return logging.getLogger(name) if logging.getLogger(name).handlers else setup_logger(name)


# Convenience function for testing emoji support
def test_emoji_support():
    """Test if current environment supports emoji logging"""
    test_logger = setup_logger('emoji_test', 'emoji_test.log')
    try:
        test_logger.info("🚀 Testing emoji support")
        return True
    except Exception:
        return False