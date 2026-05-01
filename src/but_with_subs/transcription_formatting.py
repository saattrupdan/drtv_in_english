"""Format raw transcriptions into subtitle-ready segments.

This module uses an LLM to format raw word-level transcriptions into
properly punctuated, properly casemapped subtitle segments.
"""

import logging
from textwrap import dedent

from pydantic import BaseModel
from tqdm.auto import tqdm

from .llm import LLMConfig, QueryLLMBatchItem, query_llm_batch
from .transcribing import Transcription

logger = logging.getLogger(__package__)


class InvalidInputError(ValueError):
    """Raised when the transcription input is invalid.

    This is used to signal that one or more transcriptions passed to the
    formatting pipeline contain empty or otherwise unacceptable content.

    Attributes:
        message:
            A human-readable explanation of the validation failure.
    """

    pass


class TranscribedSegmentsResponse(BaseModel):
    """Response containing a list of formatted segments.

    Attributes:
        segments:
            List of formatted subtitle segments.
    """

    segments: list[Transcription]


def _build_prompt(chunk_transcriptions: list[list[Transcription]]) -> str:
    """Build an LLM prompt for formatting raw transcriptions.

    Args:
        chunk_transcriptions:
            A list of chunk transcription lists. Each inner list contains
            Transcription objects from a single audio chunk.

    Returns:
        A prompt string suitable for sending to an LLM.
    """
    for chunk in chunk_transcriptions:
        for transcription in chunk:
            if not transcription.text:
                logger.warning(
                    "Skipping transcription with empty text at "
                    f"start_time={transcription.start_time}"
                )
                break

    lines: list[str] = [
        "You are a subtitle formatting assistant. Format the following "
        "raw word-level transcription into clean subtitle-ready segments."
    ]

    lines.append(
        dedent("""
        Rules:
        - Fix all casing and punctuation.
        - Split into coherent short segments suitable for subtitle display.
        - Remove filler words (um, uh, you know, like) and stutters.
        - You need to translate the
        - Keep the meaning of the original words intact.
        - Each segment should be have at most 12 words
        - Prioritise that sentences are shown together, rather than broken up across
          several segments.
        - Return valid JSON with an array of segments matching the schema:
          {"segments": [{"text": str, "start_time": float, "end_time": float}]}
        - start_time and end_time are in seconds.
        """).strip()
    )

    lines.append("\nRaw transcription:\n")

    chunk_start = 1
    for chunk_index, chunk in enumerate(chunk_transcriptions, start=1):
        if len(chunk_transcriptions) > 1:
            lines.append(f"Chunk {chunk_index}:\n")
        for transcription in chunk:
            lines.append(
                f"{chunk_start}. {transcription.text!r} "
                f"[{transcription.start_time:.2f}-{transcription.end_time:.2f}]"
            )
            chunk_start += 1

    return "\n".join(lines)


async def format_transcriptions(
    chunk_transcriptions: list[list[Transcription]], llm_config: LLMConfig
) -> list[Transcription]:
    """Format raw transcriptions into subtitle-ready segments using an LLM.

    Flattens the chunked transcriptions, sends them to an LLM for formatting,
    then maps LLM position indices back to real timestamps.

    Args:
        chunk_transcriptions:
            A list of chunk transcription lists. Each inner list contains
            Transcription objects from a single audio chunk.
        llm_config:
            Configuration for the LLM API call.

    Returns:
        A list of ``Transcription`` objects with formatted text and correct
        timestamps.
    """
    # Flatten all words and build a position-to-transcription lookup.
    position_to_transcription: dict[int, Transcription] = {}
    position_index = 1

    for chunk in chunk_transcriptions:
        for transcription in chunk:
            position_to_transcription[position_index] = transcription
            position_index += 1

    # Split into batches and build prompts
    batches: list[list[list[Transcription]]] = [
        chunk_transcriptions[i : i + 4] for i in range(0, len(chunk_transcriptions), 4)
    ]

    config = llm_config.model_copy(update={"response_model": TranscribedSegmentsResponse})
    items = [
        QueryLLMBatchItem(prompt=_build_prompt(batch), config=config)
        for batch in tqdm(batches, desc="Processing transcription batches")
    ]

    results = await query_llm_batch(items)

    all_segments: list[Transcription] = []
    for batch_idx, response in enumerate(results):
        if isinstance(response, str):
            logger.warning("LLM returned raw string instead of structured data")
            continue

        if response is None:
            logger.warning(
                "LLM returned None for batch %d, skipping. "
                "response_model=%s, prompt_len=%d. "
                "Check LLM provider and prompt context.",
                batch_idx,
                TranscribedSegmentsResponse.__name__,
                len(items[batch_idx].prompt),
            )
            continue

        all_segments.extend(response.segments)

    return all_segments
