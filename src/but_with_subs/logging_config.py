"""Central logging configuration for the but_with_subs package.

This module provides a single entry point for configuring logging
across the entire application. Call configure_logging() early in
your application startup to ensure all loggers share the same
handlers and formatting.
"""

import logging
import sys
from functools import cache

logger = logging.getLogger(__package__)


def configure_logging() -> None:
    """Configure logging for the but_with_subs package."""
    # Set up the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s • %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(fmt=formatter)
    root_logger.addHandler(handler)

    # Ignore other loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)


@cache
def log_once(message: str, level: int) -> None:
    """Log a message once, regardless of the number of times it is called.

    Args:
        message:
            The message to log.
        level:
            The logging level to use.
    """
    logger.log(msg=message, level=level)
