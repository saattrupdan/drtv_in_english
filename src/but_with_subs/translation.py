"""Translation utilities powered by LLMs."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from .llm import LLMConfig, query_llm


class TranslatedText(BaseModel):
    """A model representing translated text."""

    text: str = ""


async def translate(
    text: str,
    target_language: str,
    llm_config: LLMConfig,
    *,
    llm_model: Optional[str] = None,
    api_base: Optional[str] = None,
) -> str:
    """Translate text to the target language using an LLM.

    Args:
        text: The source text to translate.
        target_language: The target language for translation (e.g. 'French').
        llm_config: Configuration for the LLM.
        llm_model: Optional explicit model name override.
        api_base: Optional explicit API base URL override.

    Returns:
        The translated text with casing, punctuation, and transcription
        artifacts corrected.
    """
    prompt = (
        f"Translate the following text to {target_language}.\n"
        "Also fix casing, punctuation, and any transcription artifacts.\n"
        "Only output the translated text.\n"
        f"\nInput: {text}"
    )

    config = llm_config.model_copy(update={"response_model": TranslatedText})

    response = await query_llm(prompt, config)

    if isinstance(response, TranslatedText):
        return response.text

    return str(response)
