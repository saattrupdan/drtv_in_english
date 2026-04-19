"""Test script to demonstrate the download function in action.

Usage:
    uv run src/scripts/download_video.py [url]
"""

import logging
import sys

from but_with_subs.downloading import download

logger = logging.getLogger(__package__)


def main() -> None:
    """Run a test download from a URL.

    Sets up logging, gets the URL from command-line arguments or uses a
    default, iterates over the download generator to display progress,
    and logs the final file result.
    """
    import but_with_subs  # noqa: F401

    default_url: str = "https://www.dr.dk/drtv/serie/kommissionen_589959"
    url: str = sys.argv[1] if len(sys.argv) > 1 else default_url

    logger.info("Starting test download from URL: %s", url)

    gen = download(url=url)
    try:
        while True:
            progress = next(gen)
            logger.info("Progress: %s", progress)
    except StopIteration as exc:
        final_file = exc.value

    logger.info("Final result: %s", final_file)


if __name__ == "__main__":
    main()
