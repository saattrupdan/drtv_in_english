"""Hardware-specific device information."""

from functools import cache

import torch

from .logging_config import logger


@cache
def get_device() -> torch.device:
    """Get the device to use for inference.

    Returns:
        The device to use for inference.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device
