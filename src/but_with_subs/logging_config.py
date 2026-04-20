"""Central logging configuration for the but_with_subs package.

This module provides a single entry point for configuring logging
across the entire application. Call configure_logging() early in
your application startup to ensure all loggers share the same
handlers and formatting.
"""

import logging
import sys


def configure_logging() -> None:
    """Configure logging for the but_with_subs package."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s • %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(fmt=formatter)
    root_logger.addHandler(hdlr=handler)
