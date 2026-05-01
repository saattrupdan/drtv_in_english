"""Translation utilities powered by LLMs."""

from __future__ import annotations

from textwrap import dedent

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
    llm_model: str | None = None,
    api_base: str | None = None,
) -> str:
    """Translate text to the target language using an LLM.

    Args:
        text:
            The source text to translate.
        target_language:
            The target language for translation (e.g. 'French').
        llm_config:
            Configuration for the LLM.
        llm_model:
            Optional explicit model name override.
        api_base:
            Optional explicit API base URL override.

    Returns:
        The translated text with casing, punctuation, and transcription
        artifacts corrected.
    """
    prompt = dedent(f"""
        Translate the following text to {target_language}.
        Also fix casing, punctuation, and any transcription artifacts.
        Only output the translated text.

        Input: {text}
    """).strip()

    update: dict[str, object] = {"response_model": TranslatedText}
    if llm_model is not None:
        update["model"] = llm_model
    if api_base is not None:
        update["api_base"] = api_base
    config = llm_config.model_copy(update=update)

    response = await query_llm(prompt, config)

    if isinstance(response, TranslatedText):
        return response.text

    return str(response)
