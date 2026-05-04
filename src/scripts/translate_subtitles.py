"""Translate text files using transformer models.

Usage:
    uv run src/scripts/translate_subtitles.py [file_path] [--target-lang LANG]
    [--model MODEL] [--batch-size N]
"""

import logging
import sys
from pathlib import Path

import click

from but_with_subs import configure_logging
from but_with_subs.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_TRANSLATION_MODEL,
)
from but_with_subs.translation import translate_chunks
from but_with_subs.vtt import parse_vtt_file, write_vtt_file

logger = logging.getLogger(__package__)

configure_logging()


@click.command()
@click.argument("file_path", type=str)
@click.option(
    "--target-lang",
    type=str,
    default=DEFAULT_TARGET_LANGUAGE,
    show_default=True,
    help="Target language code (e.g., 'en' for English).",
)
@click.option(
    "--model",
    type=str,
    default=DEFAULT_TRANSLATION_MODEL,
    show_default=True,
    help="HuggingFace model ID for translation.",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    help="Number of texts to translate in each batch.",
)
@click.option(
    "--output-path",
    type=str,
    default=None,
    show_default=True,
    help="Output path for translated file. Default: input_path_translated.vtt",
)
def main(
    file_path: str,
    target_lang: str,
    model: str,
    batch_size: int,
    output_path: str | None,
) -> None:
    """Translate text between languages."""
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    if output_path is None:
        output_path = str(path.with_stem(path.stem + "_translated"))

    logger.info(f"Parsing file: {path}")
    chunks = parse_vtt_file(path)

    text_count = len([c for c in chunks if c.text])
    logger.info(f"Translating {text_count} text segments to {target_lang}")
    translated_chunks = translate_chunks(
        chunks, target_lang, batch_size, model_id=model
    )

    write_vtt_file(translated_chunks, Path(output_path))

    logger.info(f"Translation complete: {output_path}")


if __name__ == "__main__":
    main()
