"""Tests for the transcription formatting module.

This module contains tests for the ``_build_prompt`` and ``format_transcriptions``
functions, covering prompt generation, empty text handling, and various LLM
response types.
"""

from unittest.mock import AsyncMock, patch

from but_with_subs.data_models import (
    LLMConfig,
    TranscribedSegmentsResponse,
    Transcription,
)
from but_with_subs.transcription_formatting import _build_prompt, format_transcriptions

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
# format_transcriptions() tests
# ---------------------------------------------------------------------------


@patch("but_with_subs.transcription_formatting.query_llm_batch", new_callable=AsyncMock)
async def test_format_transcriptions_with_mocked_llm(mock_query_llm: AsyncMock) -> None:
    """Test that format_transcriptions returns formatted transcription segments.

    Verifies that when ``query_llm_batch`` returns a structured
    ``TranscribedSegmentsResponse``, ``format_transcriptions`` returns a list
    of ``Transcription`` objects.
    """
    llm_config = _make_llm_config()
    mock_query_llm.return_value = [TranscribedSegmentsResponse(segments=[])]

    chunk_transcriptions: list[list[Transcription]] = [
        [_make_transcription(text="hello")]
    ]
    result = await format_transcriptions(
        chunk_transcriptions=chunk_transcriptions, llm_config=llm_config
    )

    assert isinstance(result, list)


@patch("but_with_subs.transcription_formatting.query_llm_batch", new_callable=AsyncMock)
async def test_format_transcriptions_with_raw_string(mock_query_llm: AsyncMock) -> None:
    """Test that format_transcriptions skips raw string responses.

    If ``query_llm_batch`` returns a plain string instead of structured data,
    ``format_transcriptions`` should log a warning and skip the batch.
    """
    llm_config = _make_llm_config()
    mock_query_llm.return_value = ["raw unrecognized response"]

    chunk_transcriptions: list[list[Transcription]] = [
        [_make_transcription(text="hello")]
    ]
    result = await format_transcriptions(
        chunk_transcriptions=chunk_transcriptions, llm_config=llm_config
    )

    assert result == []


@patch("but_with_subs.transcription_formatting.query_llm_batch", new_callable=AsyncMock)
async def test_format_transcriptions_with_none_response(
    mock_query_llm: AsyncMock,
) -> None:
    """Test that format_transcriptions skips None responses.

    If ``query_llm_batch`` returns None, ``format_transcriptions`` should log a
    warning and skip the batch.
    """
    llm_config = _make_llm_config()
    mock_query_llm.return_value = [None]

    chunk_transcriptions: list[list[Transcription]] = [
        [_make_transcription(text="hello")]
    ]
    result = await format_transcriptions(
        chunk_transcriptions=chunk_transcriptions, llm_config=llm_config
    )

    assert result == []


@patch("but_with_subs.transcription_formatting.query_llm_batch", new_callable=AsyncMock)
async def test_format_transcriptions_returns_segments(
    mock_query_llm: AsyncMock,
) -> None:
    """Test that format_transcriptions returns Transcription segments from the LLM.

    Verifies that formatted segments from the LLM response are returned
    correctly.
    """
    llm_config = _make_llm_config()
    expected_segment = _make_transcription(text="Hello, world!")
    mock_query_llm.return_value = [
        TranscribedSegmentsResponse(segments=[expected_segment])
    ]

    chunk_transcriptions: list[list[Transcription]] = [
        [_make_transcription(text="hello")]
    ]
    result = await format_transcriptions(
        chunk_transcriptions=chunk_transcriptions, llm_config=llm_config
    )

    assert len(result) == 1
    assert result[0].text == "Hello, world!"
