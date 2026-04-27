"""Format raw transcriptions into subtitle-ready segments.

This module uses an LLM to format raw word-level transcriptions into
properly punctuated, properly casemapped subtitle segments.
"""

import logging

from pydantic import BaseModel
from tqdm.auto import tqdm

from .llm import LLMConfig, query_llm
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


class FormattedSegment(BaseModel):
    """A single formatted segment with text and timing.

    Attributes:
        text:
            The formatted text for this segment.
        start_position:
            1-based inclusive position of the first word in the flattened
            word list.
        end_position:
            1-based inclusive position of the last word in the flattened
            word list.
        start_time:
            Approximate start time of the segment in seconds.
        end_time:
            Approximate end time of the segment in seconds.
    """

    text: str
    start_position: int
    end_position: int
    start_time: float
    end_time: float


class TranscribedSegmentsResponse(BaseModel):
    """Response containing a list of formatted segments.

    Attributes:
        segments:
            List of formatted subtitle segments.
    """

    segments: list[FormattedSegment]


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
        "Rules:"
        "\n- Fix all casing and punctuation."
        "\n- Split into coherent short segments suitable for subtitle display."
        "\n- Remove filler words (um, uh, you know, like) and stutters."
        "\n- Keep the meaning of the original words intact."
        "\n- Each segment should be 6-12 words long at most."
        "\n- Return valid JSON with an array of segments matching this "
        'schema: {"segments": [{"text": str, '
        '"start_position": int, "end_position": int, '
        '"start_time": float, "end_time": float}]}'
        "\n- start_position and end_position are 1-based inclusive indices "
        "into the flattened word list below."
        "\n- start_time and end_time are in seconds."
    )

    lines.append("\nRaw transcription:\n")

    chunk_start = 1

    for chunk_index, chunk in enumerate(chunk_transcriptions, start=1):
        if len(chunk_transcriptions) > 1:
            lines.append(f"Chunk {chunk_index}:\n")

        for transcription in chunk:
            lines.append(
                f'{chunk_start}. "{transcription.text}" '
                f"[{transcription.start_time:.2f}-"
                f"{transcription.end_time:.2f}]"
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

    # Split into batches of 4 chunks.
    batches: list[list[list[Transcription]]] = [
        chunk_transcriptions[i : i + 4] for i in range(0, len(chunk_transcriptions), 4)
    ]

    all_segments: list[FormattedSegment] = []

    for batch_idx, batch in tqdm(
        enumerate(batches), desc="Processing transcription batches"
    ):
        segments = await _process_batch(batch, batch_idx, llm_config)
        all_segments.extend(segments)

    # Map position intervals back to timestamps.
    formatted_transcriptions: list[Transcription] = []
    for segment in all_segments:
        start_word = position_to_transcription.get(segment.start_position)
        end_word = position_to_transcription.get(segment.end_position)

        if start_word is None or end_word is None:
            logger.warning(
                "Position %d-%d not found in transcription, skipping segment",
                segment.start_position,
                segment.end_position,
            )
            continue

        formatted_transcriptions.append(
            Transcription(
                start_time=start_word.start_time,
                end_time=end_word.end_time,
                text=segment.text,
            )
        )

    return formatted_transcriptions


async def _process_batch(
    batch: list[list[Transcription]], batch_idx: int, llm_config: LLMConfig
) -> list[FormattedSegment]:
    """Process a single batch of chunks through the LLM.

    Args:
        batch:
            A list of 4 or fewer chunk transcription lists.
        batch_idx:
            Zero-based index of the batch within the full set of batches.
        llm_config:
            Configuration for the LLM API call.

    Returns:
        A list of formatted segments from the LLM response.
    """
    prompt = _build_prompt(batch)
    config = llm_config.model_copy(
        update={"response_model": TranscribedSegmentsResponse}
    )
    response = await query_llm(prompt=prompt, config=config)

    if isinstance(response, str):
        logger.warning("LLM returned raw string instead of structured data")
        return []

    if response is None:
        logger.warning(
            "LLM returned None for batch %d, skipping. "
            "response_model=%s, prompt_len=%d. "
            "Check LLM provider and prompt context.",
            batch_idx,
            TranscribedSegmentsResponse.__name__,
            len(prompt),
        )
        return []

    return response.segments
