import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import colorlog

def setup_logger(name: str = "kairos_pipeline", log_file: str = "logs/system.log", level: str = "INFO") -> logging.Logger:
    """
    Sets up a thread-safe, colorized console logger alongside a rotating file logger.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Convert string log level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicating handlers if logger is re-initialized
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Console Handler with Colorized Output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # 2. Rotating File Handler (10MB max size, keeping up to 5 backup logs)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(numeric_level)

    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(process)d:%(threadName)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

# Module-level default logger instance
logger = setup_logger()