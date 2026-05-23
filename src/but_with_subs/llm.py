"""LLM-based ASR corrector and translator.

Replaces the small100 model with an OpenAI-compatible API call that
corrects ASR errors and translates into the target language in a single
pass, using a sliding window of surrounding chunks for context.

Environment variables (read by build_client):
    LLM_BASE_URL  – OpenAI-compatible endpoint (e.g. https://api.openai.com/v1)
    LLM_API_KEY   – API key
    LLM_MODEL     – Model name (e.g. gpt-4o-mini, qwen2.5-7b-instruct)
"""

from __future__ import annotations

import json
import logging
import os
import typing as t

import openai
from pydantic import BaseModel, ValidationError

from .data_models import Chunk
from .logging_config import logger

_OPENAI_CLIENT: openai.OpenAI | None = None


class CorrectedChunk(BaseModel):
    """Structured output from the LLM: corrected + translated text for one chunk."""

    text: str


def build_client() -> openai.OpenAI:
    """Build an OpenAI-compatible client from environment variables.

    Reads LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL from the environment.
    Raises ValueError if any required variable is missing.

    Returns:
        An OpenAI client configured with the correct base URL and API key.
    """
    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        raise ValueError("LLM_BASE_URL is required (e.g. https://api.openai.com/v1)")

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY is required")

    model = os.environ.get("LLM_MODEL")
    if not model:
        raise ValueError("LLM_MODEL is required (e.g. gpt-4o-mini)")

    return openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=3)


def correct_and_translate(
    chunks: list[Chunk],
    target_language: str,
    *,
    client: openai.OpenAI,
    model: str = "gpt-4o-mini",
    context_window: int = 6,
    on_progress: t.Callable[[float], None] | None = None,
) -> list[Chunk]:
    """Rewrite each chunk's text so it is both ASR-corrected and translated.

    Operates over sliding windows of ``context_window`` chunks before and after
    the target chunk so the LLM can fix ASR errors using surrounding context
    (proper nouns, agreement, missing function words). Returns a new list of
    Chunk objects with the same timing/audio/speaker but updated ``text``.

    Args:
        chunks:
            List of Chunk objects with transcribed (Danish) text.
        target_language:
            ISO-639-1 target language code (e.g. ``"en"``).
        client:
            Pre-built OpenAI client.
        model:
            Model name for the API call.
        context_window:
            Number of adjacent chunks to include before and after the target
            chunk (default 6). The full request window is
            ``2 * context_window + 1`` chunks.
        on_progress:
            Optional callback invoked with a float 0..1 after each chunk.

    Returns:
        A new list of Chunk objects with corrected + translated text.
    """
    if not chunks:
        return []

    n = len(chunks)
    result: list[Chunk] = []

    # System prompt shared by every request.
    system_prompt = (
        "You are an expert translator and editor of Danish TV subtitles. "
        "The source text was transcribed by an ASR model and may contain errors "
        "(missing words, mangled proper nouns, literal translations of idioms). "
        "Your task is to: "
        "1) Correct any ASR errors using the surrounding context. "
        "2) Translate the corrected text into the requested target language. "
        "3) Preserve the original meaning, speaker intent, and subtitle length. "
        "4) Do not translate proper nouns (keep them as-is). "
        "5) Return ONLY the JSON object for the single requested chunk."
    )

    for i, chunk in enumerate(chunks):
        # Build the window: context_window before, current, context_window after.
        window_start = max(0, i - context_window)
        window_end = min(n, i + context_window + 1)  # inclusive end index
        window = chunks[window_start:window_end]

        # Build the numbered list for the LLM.
        numbered = ""
        for j, wc in enumerate(window):
            start = f"{wc.start_time:.3f}" if wc.start_time else "0.0"
            end = f"{wc.end_time:.3f}" if wc.end_time else "0.0"
            text = wc.text or "[no text]"
            numbered += (
                f"{window_start + j}: [{start} -> {end}] {text}\n"
            )

        # Identify the centre chunk index within the window.
        centre_offset = i - window_start  # index of the target chunk in the window

        user_prompt = (
            f"Target language: {target_language}\n\n"
            f"Chunks (return ONLY the JSON for chunk {centre_offset}):\n\n"
            f"{numbered}"
            f"---\n"
            f"Return a JSON object with a single key 'text' containing the "
            f"corrected and translated text for chunk {centre_offset}."
        )

        # API call with JSON mode.
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            raw = response.choices[0].message.content
            if raw is None:
                raise ValueError("Empty response")

            parsed: dict[str, str] = json.loads(raw)
            corrected_text = parsed.get("text", "")
            if not corrected_text:
                raise ValueError("Empty text field")

            # Validate with Pydantic.
            CorrectedChunk.model_validate({"text": corrected_text})

        except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as exc:
            logger.warning(
                f"LLM JSON parse failed for chunk {i} (text={chunk.text!r}): {exc}"
            )
            corrected_text = chunk.text or ""

        # Build the new chunk (copy, don't mutate).
        new_chunk = chunk.model_copy()
        new_chunk.text = corrected_text
        result.append(new_chunk)

        # Progress callback.
        if on_progress is not None:
            on_progress((i + 1) / n)

    return result
