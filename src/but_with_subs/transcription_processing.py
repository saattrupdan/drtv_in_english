"""Process raw transcriptions into subtitle-ready segments."""

import logging
from textwrap import dedent

from .data_models import (
    LLMConfig,
    QueryLLMBatchItem,
    TranscribedSegmentsResponse,
    Transcription,
)
from .llm import query_llm_batch

logger = logging.getLogger(__package__)


async def process_transcriptions(
    chunk_transcriptions: list[list[Transcription]],
    target_language: str,
    llm_config: LLMConfig,
) -> list[Transcription]:
    """Process raw transcriptions into subtitle-ready segments using an LLM.

    Args:
        chunk_transcriptions:
            A list of chunk transcription lists. Each inner list contains
            Transcription objects from a single audio chunk.
        target_language:
            The target language for the subtitles.
        llm_config:
            Configuration for the LLM API call.

    Returns:
        A list of ``Transcription`` objects with processed text and correct timestamps.
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

    config = llm_config.model_copy(
        update={"response_model": TranscribedSegmentsResponse}
    )
    items = [
        QueryLLMBatchItem(
            prompt=_build_prompt(
                chunk_transcriptions=batch, target_language=target_language
            ),
            config=config,
        )
        for batch in batches
    ]

    results = await query_llm_batch(
        items=items, desc="Processing transcription batches"
    )

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


def _build_prompt(
    chunk_transcriptions: list[list[Transcription]], target_language: str
) -> str:
    """Build an LLM prompt for processing raw transcriptions.

    Args:
        chunk_transcriptions:
            A list of chunk transcription lists. Each inner list contains
            Transcription objects from a single audio chunk.
        target_language:
            The target language for the subtitles.

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
        dedent(f"""
        You are a subtitle processing assistant. Process the following raw word-level
        transcription into clean subtitle-ready segments.

        Rules:
        - Fix all casing and punctuation.
        - Split into coherent short segments suitable for subtitle display.
        - Remove filler words (um, uh, you know, like) and stutters.
        - You need to translate the segments into {target_language}
        - Keep the meaning of the original words intact.
        - Each segment should be have at most 12 words
        - Prioritise that sentences are shown together, rather than broken up across
          several segments.
        - Return valid JSON with an array of segments matching the schema:
          {{"segments": [{{"text": str, "start_time": float, "end_time": float}}]}}
        - start_time and end_time are in seconds.

        Raw transcription:
        """)
    ]

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
