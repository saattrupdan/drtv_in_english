"""Test script to demonstrate the download function in action.

Usage:
    uv run src/scripts/download_video.py [url]
"""

import logging

import click

from but_with_subs import download

logger = logging.getLogger(__package__)


@click.command()
@click.argument("url", required=True)
def main(url: str) -> None:
    """Run a test download from a URL."""
    logger.info(f"Downloading from {url}...")
    gen = download(url=url)
    while True:
        progress = next(gen)
        logger.info(progress)


if __name__ == "__main__":
    main()
