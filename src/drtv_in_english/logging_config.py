"""Central logging configuration for the drtv_in_english package.

This module provides a single entry point for configuring logging
across the entire application. Call configure_logging() early in
your application startup to ensure all loggers share the same
handlers and formatting.
"""

import logging
import sys

logger = logging.getLogger(__package__)


def configure_logging() -> None:
    """Configure logging for the drtv_in_english package."""
    root_logger = logging.getLogger()

    # Remove any existing custom handlers to ensure idempotency
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)
            handler.close()

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
