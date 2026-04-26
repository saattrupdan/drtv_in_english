"""Tests for the transcription formatting module.

This module contains tests for the ``_build_prompt`` and ``_process_batch``
functions, covering prompt generation, empty text handling, various LLM
response types, and progress callback support.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from but_with_subs.llm import LLMConfig
from but_with_subs.transcribing import Transcription
from but_with_subs.transcription_formatting import (
    FormattedSegment,
    TranscribedSegmentsResponse,
    _build_prompt,
    _process_batch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_config(
    model: str = "gpt-4",
    temperature: float = 0.0,
    max_tokens: int = 64,
    api_base: str = "http://localhost:8000",
) -> LLMConfig:
    """Create a minimal LLMConfig for testing.

    Args:
        model: Model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens.
        api_base: API base URL.

    Returns:
        An LLMConfig instance.
    """
    return LLMConfig(
        model=model, temperature=temperature, max_tokens=max_tokens, api_base=api_base
    )


def _make_transcription(
    text: str = "hello", start_time: float = 1.0, end_time: float = 2.0
) -> Transcription:
    """Create a minimal Transcription for testing.

    Args:
        text: The transcribed text.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        A Transcription instance.
    """
    return Transcription(start_time=start_time, end_time=end_time, text=text)


# ---------------------------------------------------------------------------
# _build_prompt() tests
# ---------------------------------------------------------------------------


def test_build_prompt_includes_chunk_labels() -> None:
    """Test that _build_prompt includes Chunk {N}: labels.

    Verifies that when multiple chunk transcription lists are provided,
    the prompt includes chunk header labels.
    """
    chunk_transcriptions: list[list[Transcription]] = [
        [_make_transcription(text="hello")],
        [_make_transcription(text="world")],
    ]

    prompt = _build_prompt(chunk_transcriptions=chunk_transcriptions)

    assert "Chunk 1:" in prompt
    assert "Chunk 2:" in prompt


def test_build_prompt_handles_empty_text() -> None:
    """Test that _build_prompt handles empty transcription text gracefully.

    Verifies that passing a transcription with ``text=""`` does not raise.
    """
    empty_transcription = Transcription(start_time=0.0, end_time=1.0, text="")
    chunk_transcriptions: list[list[Transcription]] = [[empty_transcription]]

    prompt = _build_prompt(chunk_transcriptions=chunk_transcriptions)

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_prompt_includes_transcription_text() -> None:
    """Test that _build_prompt includes the transcription text.

    Verifies that the raw transcription text appears in the generated prompt.
    """
    chunk_transcriptions: list[list[Transcription]] = [
        [_make_transcription(text="hello world")]
    ]

    prompt = _build_prompt(chunk_transcriptions=chunk_transcriptions)

    assert "hello world" in prompt
    assert "Raw transcription:" in prompt


def test_build_prompt_includes_schema() -> None:
    """Test that _build_prompt includes the JSON schema description.

    Verifies that the prompt includes the expected schema fields for segments.
    """
    chunk_transcriptions: list[list[Transcription]] = [
        [_make_transcription(text="hello")]
    ]

    prompt = _build_prompt(chunk_transcriptions=chunk_transcriptions)

    assert '"segments"' in prompt
    assert '"text": str' in prompt
    assert '"start_time": float' in prompt
    assert '"end_time": float' in prompt


# ---------------------------------------------------------------------------
# _process_batch() tests
# ---------------------------------------------------------------------------


@patch("but_with_subs.transcription_formatting.query_llm", new_callable=AsyncMock)
def test_process_batch_with_mocked_llm(mock_query_llm: AsyncMock) -> None:
    """Test that _process_batch returns FormattedSegment instances.

    Verifies that when ``query_llm`` returns a structured
    ``TranscribedSegmentsResponse``, ``_process_batch`` returns a list
    of ``FormattedSegment`` objects.
    """
    llm_config = _make_llm_config()
    mock_query_llm.return_value = TranscribedSegmentsResponse(segments=[])

    batch: list[list[Transcription]] = [[_make_transcription(text="hello")]]
    result = asyncio.run(_process_batch(batch=batch, llm_config=llm_config))

    assert isinstance(result, list)


@patch("but_with_subs.transcription_formatting.query_llm", new_callable=AsyncMock)
def test_process_batch_with_raw_string(mock_query_llm: AsyncMock) -> None:
    """Test that _process_batch returns [] when query_llm returns a raw string.

    If ``query_llm`` returns a plain string instead of structured data,
    ``_process_batch`` should log a warning and return an empty list.
    """
    llm_config = _make_llm_config()
    mock_query_llm.return_value = "raw unrecognized response"

    batch: list[list[Transcription]] = [[_make_transcription(text="hello")]]
    result = asyncio.run(_process_batch(batch=batch, llm_config=llm_config))

    assert result == []


@patch("but_with_subs.transcription_formatting.query_llm", new_callable=AsyncMock)
def test_process_batch_with_progress_callback(mock_query_llm: AsyncMock) -> None:
    """Test that _process_batch passes progress_callback to query_llm.

    Verifies that a provided ``progress_callback`` is forwarded to
    ``query_llm`` so progress events are received.
    """
    llm_config = _make_llm_config()
    mock_query_llm.return_value = TranscribedSegmentsResponse(
        segments=[
            FormattedSegment(
                text="Hello world",
                start_position=1,
                end_position=2,
                start_time=1.0,
                end_time=2.0,
            )
        ]
    )

    batch: list[list[Transcription]] = [[_make_transcription(text="hello")]]
    events: list = []

    def callback(progress: object) -> None:
        events.append(progress)

    asyncio.run(
        _process_batch(batch=batch, llm_config=llm_config, progress_callback=callback)
    )

    assert mock_query_llm.call_args.kwargs["progress_callback"] == callback
