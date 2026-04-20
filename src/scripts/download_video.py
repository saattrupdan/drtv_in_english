"""Test script to demonstrate the download function in action.

Usage:
    uv run src/scripts/download_video.py [url]
"""

import logging

import click
from tqdm.auto import tqdm

from but_with_subs import download

logger = logging.getLogger(__package__)


@click.command()
@click.argument("url", required=True)
def main(url: str) -> None:
    """Run a test download from a URL."""
    logger.info(f"Downloading from {url}...")
    with tqdm(total=100, unit="%", desc="Download progress") as pbar:
        download(
            url=url,
            progress_hook=lambda p: pbar.update(int(100 * p.percentage - pbar.n)),
        )


if __name__ == "__main__":
    main()
