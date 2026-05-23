"""LLM-based ASR corrector and translator.

Replaces the small100 model with an OpenAI-compatible API call that
corrects ASR errors and translates into the target language in a single
pass, using a sliding window of surrounding chunks for context.

Environment variables (read by build_client):
    LLM_BASE_URL  -- OpenAI-compatible endpoint (e.g. https://api.openai.com/v1)
    LLM_API_KEY   -- API key
    LLM_MODEL     -- Model name (e.g. gpt-4o-mini, qwen2.5-7b-instruct)
"""

from __future__ import annotations

import collections.abc as c
import json
import os

import openai
from pydantic import BaseModel, ValidationError

from .data_models import Chunk
from .logging_config import logger


class CorrectedChunk(BaseModel):
    """Structured output from the LLM: corrected and translated text for one chunk."""

    text: str


def build_client() -> openai.OpenAI:
    """Build an OpenAI-compatible client from environment variables.

    Reads LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL from the environment.

    Returns:
        An OpenAI client configured with the correct base URL and API key.

    Raises:
        ValueError:
            If any of LLM_BASE_URL, LLM_API_KEY, or LLM_MODEL is missing.
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
    on_progress: c.Callable[[float], None] | None = None,
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
        model (optional):
            Model name for the API call. Defaults to ``"gpt-4o-mini"``.
        context_window (optional):
            Number of adjacent chunks to include before and after the target
            chunk. The full request window is ``2 * context_window + 1``
            chunks. Defaults to 6.
        on_progress (optional):
            Callback invoked with a float in ``[0, 1]`` after each chunk.
            Defaults to None.

    Returns:
        A new list of Chunk objects with corrected and translated text.
    """
    if not chunks:
        return []

    n = len(chunks)
    result: list[Chunk] = []

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
        window_start = max(0, i - context_window)
        window_end = min(n, i + context_window + 1)
        window = chunks[window_start:window_end]

        numbered = ""
        for j, wc in enumerate(window):
            start = f"{wc.start_time:.3f}" if wc.start_time else "0.0"
            end = f"{wc.end_time:.3f}" if wc.end_time else "0.0"
            text = wc.text or "[no text]"
            numbered += f"{window_start + j}: [{start} -> {end}] {text}\n"

        centre_offset = i - window_start

        user_prompt = (
            f"Target language: {target_language}\n\n"
            f"Chunks (return ONLY the JSON for chunk {centre_offset}):\n\n"
            f"{numbered}"
            f"---\n"
            f"Return a JSON object with a single key 'text' containing the "
            f"corrected and translated text for chunk {centre_offset}."
        )

        corrected_text = _request_correction(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=chunk.text or "",
            chunk_index=i,
        )

        new_chunk = chunk.model_copy()
        new_chunk.text = corrected_text
        result.append(new_chunk)

        if on_progress is not None:
            on_progress((i + 1) / n)

    return result


def _request_correction(
    *,
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    fallback: str,
    chunk_index: int,
) -> str:
    """Call the LLM for one chunk and return its corrected text or a fallback.

    Args:
        client:
            Pre-built OpenAI client.
        model:
            Model name for the API call.
        system_prompt:
            System message describing the correction and translation task.
        user_prompt:
            User message containing the windowed chunks.
        fallback:
            Text to return if the LLM response is malformed or empty.
        chunk_index:
            Zero-based index of the chunk being processed, used in log messages.

    Returns:
        The corrected and translated text, or ``fallback`` on parse failure.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            timeout=30.0,
        )
        raw = response.choices[0].message.content
        if raw is None:
            logger.warning(f"LLM returned no content for chunk {chunk_index}")
            return fallback

        parsed: dict[str, str] = json.loads(raw)
        corrected_text = parsed.get("text", "")
        if not corrected_text:
            logger.warning(f"LLM returned empty text for chunk {chunk_index}")
            return fallback

        CorrectedChunk.model_validate({"text": corrected_text})
        return corrected_text

    except (json.JSONDecodeError, ValidationError, KeyError) as exc:
        logger.warning(f"LLM JSON parse failed for chunk {chunk_index}: {exc}")
        return fallback
    except openai.OpenAIError as exc:
        logger.warning(f"LLM API call failed for chunk {chunk_index}: {exc}")
        return fallback
