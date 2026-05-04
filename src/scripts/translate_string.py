"""Translate a single string into a target language.

Usage:
    uv run src/scripts/translate_string.py [TEXT] [TARGET_LANG]
"""

import click
from transformers import M2M100ForConditionalGeneration

from but_with_subs import configure_logging
from but_with_subs.constants import TRANSLATION_MODEL
from but_with_subs.device import get_device
from but_with_subs.tokenization_small100 import SMALL100Tokenizer
from but_with_subs.translation import translate_single

import bits_and_bobs as bnb

configure_logging()


@click.command()
@click.argument("text", required=True)
@click.argument("target_lang", required=True)
def translate_string(text: str, target_lang: str) -> None:
    """Translate a single string into a target language.

    Loads the translation model and tokenizer, then translates the given
    text into the specified target language.

    Args:
        text:
            The text string to translate.
        target_lang:
            Target language code (e.g., 'en', 'da', 'fr').
    """
    device = get_device()

    logger = bnb.get_logger("but_with_subs")
    logger.info(f"Loading translation model: {TRANSLATION_MODEL}")
    with bnb.no_terminal_output():
        model = M2M100ForConditionalGeneration.from_pretrained(TRANSLATION_MODEL)
        tokenizer = SMALL100Tokenizer.from_pretrained(TRANSLATION_MODEL)
        model = model.to(device)

    logger.info(f"Translating to {target_lang}")
    translated = translate_single(model, tokenizer, text, target_lang)
    print(translated)


if __name__ == "__main__":
    translate_string()
