"""Translate subtitle files using transformer models.

Usage:
    uv run src/scripts/translate_subtitles.py [subtitle_path] [--source-lang LANG]
    [--target-lang LANG] [--model MODEL]
"""

import logging
import sys
from pathlib import Path

import click

from but_with_subs.translation import translate_subtitles

logger = logging.getLogger(__package__)


@click.command()
@click.argument("subtitle_path", type=str)
@click.option(
    "--source-lang",
    type=str,
    default="dan",
    show_default=True,
    help="Source language code (e.g., 'dan' for Danish).",
)
@click.option(
    "--target-lang",
    type=str,
    default="eng",
    show_default=True,
    help="Target language code (e.g., 'eng' for English).",
)
@click.option(
    "--model",
    type=str,
    default="alirezamsh/small100",
    show_default=True,
    help="HuggingFace model ID for translation.",
)
@click.option(
    "--output-path",
    type=str,
    default=None,
    show_default=True,
    help="Output path for translated subtitles. Default: input_path_translated.vtt",
)
def main(
    subtitle_path: str,
    source_lang: str,
    target_lang: str,
    model: str,
    output_path: str | None,
) -> None:
    """Translate subtitle files between languages."""
    path = Path(subtitle_path)
    if not path.is_file():
        logger.error(f"File not found: {subtitle_path}")
        sys.exit(1)

    result = translate_subtitles(
        input_path=path,
        output_path=output_path,
        source_lang=source_lang,
        target_lang=target_lang,
        model_id=model,
    )

    logger.info(f"Translation complete: {result}")


if __name__ == "__main__":
    main()
