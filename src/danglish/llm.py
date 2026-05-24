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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    return openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0)


SYSTEM_PROMPT = (
    "You are an expert translator and editor of Danish TV subtitles. "
    "The source text was transcribed by an ASR model and may contain errors "
    "(missing words, mangled proper nouns, literal translations of idioms). "
    "Your task is to: "
    "1) Correct any ASR errors using the surrounding context. "
    "2) Translate the corrected text into the requested target language. "
    "3) Preserve the original meaning, speaker intent, and subtitle length. "
    "4) Do not translate proper nouns (keep them as-is). "
    "5) Return ONLY the requested JSON object, with no extra prose."
)


def correct_and_translate(
    chunks: list[Chunk],
    target_language: str,
    *,
    client: openai.OpenAI,
    model: str = "gpt-4o-mini",
    context_window: int = 6,
    batch_size: int = 5,
    max_parallel: int = 20,
    on_progress: c.Callable[[float], None] | None = None,
) -> list[Chunk]:
    """Rewrite each chunk's text so it is both ASR-corrected and translated.

    Chunks are grouped into batches of ``batch_size``. For each batch the
    LLM receives ``context_window`` chunks of surrounding context before
    and after, so it can fix ASR errors (proper nouns, agreement, missing
    function words) using the broader transcript. A single LLM call
    returns translations for the whole batch.

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
            Number of adjacent chunks to include before and after each
            batch. Defaults to 6.
        batch_size (optional):
            Number of target chunks per LLM request. Higher values mean
            fewer requests but larger prompts and outputs. Defaults to 5.
        max_parallel (optional):
            Maximum number of LLM requests in flight at once. Defaults to 20.
        on_progress (optional):
            Callback invoked with a float in ``[0, 1]`` after each batch.
            Defaults to None.

    Returns:
        A new list of Chunk objects with corrected and translated text.

    Raises:
        ValueError:
            If ``batch_size`` is less than 1.
    """
    if not chunks:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_parallel < 1:
        raise ValueError("max_parallel must be at least 1")

    n = len(chunks)
    result: list[Chunk] = list(chunks)

    batches: list[tuple[list[int], str]] = []
    for batch_start in range(0, n, batch_size):
        batch_end = min(n, batch_start + batch_size)
        window_start = max(0, batch_start - context_window)
        window_end = min(n, batch_end + context_window)

        numbered = ""
        for j in range(window_start, window_end):
            wc = chunks[j]
            start = f"{wc.start_time:.3f}" if wc.start_time else "0.0"
            end = f"{wc.end_time:.3f}" if wc.end_time else "0.0"
            text = wc.text or "[no text]"
            numbered += f"{j}: [{start} -> {end}] {text}\n"

        target_indices = list(range(batch_start, batch_end))
        target_ids = ", ".join(str(i) for i in target_indices)

        user_prompt = (
            f"Target language: {target_language}\n\n"
            f"Chunks (translate ONLY the chunks with ids {target_ids}):\n\n"
            f"{numbered}"
            f"---\n"
            f"Return a JSON object of the form "
            f'{{"translations": {{"<id>": "<corrected and translated text>", ...}}}} '
            f"covering exactly the requested ids ({target_ids})."
        )
        batches.append((target_indices, user_prompt))

    completed = 0
    progress_lock = threading.Lock()

    def _run_batch(item: tuple[list[int], str]) -> tuple[list[int], dict[int, str]]:
        target_indices, user_prompt = item
        translations = _request_batch(
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            target_indices=target_indices,
        )
        return target_indices, translations

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(_run_batch, item) for item in batches]
        for future in as_completed(futures):
            target_indices, translations = future.result()
            for idx in target_indices:
                corrected = translations.get(idx)
                if not corrected:
                    corrected = chunks[idx].text or ""
                new_chunk = chunks[idx].model_copy()
                new_chunk.text = corrected
                result[idx] = new_chunk

            if on_progress is not None:
                with progress_lock:
                    completed += 1
                    on_progress(min(1.0, completed / len(batches)))

    return result


def _request_batch(
    *,
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    target_indices: list[int],
) -> dict[int, str]:
    """Call the LLM for one batch and return ``{chunk_index: corrected_text}``.

    Any kind of failure (network error, malformed JSON, missing keys)
    results in an empty dict; the caller falls back to the original text
    for the missing indices. This keeps one bad batch from killing the
    whole run.

    Args:
        client:
            Pre-built OpenAI client.
        model:
            Model name for the API call.
        system_prompt:
            System message describing the correction and translation task.
        user_prompt:
            User message containing the windowed chunks.
        target_indices:
            Chunk ids the LLM is expected to translate in this batch.

    Returns:
        Mapping from chunk index to corrected text. May be empty or
        partial on failure.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=1.0,
            timeout=600.0,
        )
        raw = response.choices[0].message.content
        if raw is None:
            logger.warning(f"LLM returned no content for batch {target_indices}")
            return {}

        parsed = json.loads(_extract_json(raw))
        translations_raw = parsed.get("translations") or parsed
        if not isinstance(translations_raw, dict):
            logger.warning(
                f"LLM response for batch {target_indices} is not a JSON object"
            )
            return {}

        translations: dict[int, str] = {}
        for key, value in translations_raw.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if idx not in target_indices:
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            CorrectedChunk.model_validate({"text": value})
            translations[idx] = value
        return translations

    except (json.JSONDecodeError, ValidationError, KeyError) as exc:
        logger.warning(f"LLM JSON parse failed for batch {target_indices}: {exc}")
        return {}
    except openai.OpenAIError as exc:
        logger.warning(f"LLM API call failed for batch {target_indices}: {exc}")
        return {}


def _extract_json(raw: str) -> str:
    """Extract the outermost JSON object from a model response.

    Local models occasionally wrap JSON in code fences or add a preamble
    even when asked for a strict JSON object. This trims to the first
    ``{`` and the matching final ``}`` so ``json.loads`` succeeds.

    Args:
        raw:
            Raw text returned by the LLM.

    Returns:
        A substring that should parse as JSON. May still be invalid if
        the model returned no braces at all -- the caller catches that.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return raw
    return raw[start : end + 1]
