"""Central logging configuration for the but_with_subs package.

This module provides a single entry point for configuring logging
across the entire application. Call configure_logging() early in
your application startup to ensure all loggers share the same
handlers and formatting.
"""

import logging
import sys

logger = logging.getLogger(__package__)


def configure_logging() -> None:
    """Configure logging for the but_with_subs package.

    Sets up a StreamHandler with a formatter that includes the
    logger name, level, and message. The default level is DEBUG.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt="%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
