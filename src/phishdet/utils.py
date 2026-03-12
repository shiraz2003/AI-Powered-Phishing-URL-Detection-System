"""Utility helpers: logging configuration and miscellaneous functions."""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout with a standard format.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__`` of the calling module).
    level:
        Logging level (default: ``logging.INFO``).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def validate_url(url: str) -> bool:
    """Return *True* when *url* starts with a recognised scheme.

    Parameters
    ----------
    url:
        The URL string to validate.
    """
    return isinstance(url, str) and url.strip().startswith(("http://", "https://"))
