"""End-to-end pipeline: DR URL → translated subtitles.

Downloads a DR TV video + its Danish subtitles, translates the subtitles
via an LLM, and writes a ``.<lang>.vtt`` file next to the video.

Usage:
    uv run src/scripts/run_pipeline.py <DR URL> --language en
"""

import logging
import os
import sys

import click
from dotenv import load_dotenv
from tqdm.auto import tqdm

from danglish import (
    build_client,
    configure_logging,
    correct_and_translate,
    download,
    parse_external_vtt,
    write_vtt_file,
)

load_dotenv()

logger = logging.getLogger("danglish")

configure_logging()


@click.command()
@click.argument("url", required=True)
@click.option(
    "--language",
    type=str,
    default="en",
    show_default=True,
    help="Target language for translation (e.g. 'en' for English).",
)
@click.option(
    "--max-parallel",
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of LLM requests in flight at once.",
)
def main(url: str, language: str, max_parallel: int) -> None:
    """Download a DR video and produce translated subtitles end-to-end."""
    logger.info(f"Downloading from {url}...")
    with tqdm(total=100, unit="%", desc="Download progress") as pbar:
        file = download(
            url=url,
            progress_hook=lambda p: pbar.update(int(100 * p.percentage - pbar.n)),
        )

    if file.video_path is None:
        logger.error("Download did not produce a video file")
        sys.exit(1)

    if file.subtitles_path is None:
        logger.error("No Danish subtitles available for this video")
        sys.exit(1)

    logger.info(f"Using source subtitles: {file.subtitles_path}")
    chunks = parse_external_vtt(path=file.subtitles_path)
    logger.info(f"Parsed {len(chunks)} cues from source subtitles")

    llm_client = build_client()
    llm_model = os.environ["LLM_MODEL"]

    with tqdm(total=len(chunks), desc="Translating", unit="chunk") as pbar:

        def _on_progress(ratio: float) -> None:
            pbar.n = int(ratio * len(chunks))
            pbar.refresh()

        chunks = correct_and_translate(
            chunks,
            target_language=language,
            client=llm_client,
            model=llm_model,
            max_parallel=max_parallel,
            on_progress=_on_progress,
        )

    output_path = file.video_path.with_suffix(f".{language}.vtt")
    write_vtt_file(chunks=chunks, path=output_path)
    logger.info(f"Wrote translated subtitles to {output_path}")


if __name__ == "__main__":
    main()
