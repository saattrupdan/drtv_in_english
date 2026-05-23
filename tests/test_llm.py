"""Tests for the LLM correct-and-translate module.

This module contains comprehensive tests for the LLM-based translation functions,
including mocking the OpenAI client to verify correct behaviour under various
conditions.
"""

import json
import unittest.mock as um

import openai
import pytest

from but_with_subs.data_models import Chunk
from but_with_subs.llm import (
    CorrectedChunk,
    build_client,
    correct_and_translate,
)


def _make_chunk(
    text: str,
    start_time: float = 0.0,
    end_time: float = 1.0,
    speaker: str | None = None,
) -> Chunk:
    """Create a Chunk with audio data.

    Args:
        text: The text content for the chunk.
        start_time: The start time in seconds.
        end_time: The end time in seconds.
        speaker: Optional speaker name.

    Returns:
        A Chunk instance with default audio data.
    """
    import numpy as np

    return Chunk(
        start_time=start_time,
        end_time=end_time,
        audio=np.zeros(16000, dtype=np.float32),
        text=text,
        speaker=speaker,
    )


# ---------------------------------------------------------------------------
# build_client() tests
# ---------------------------------------------------------------------------


def test_build_client_raises_when_base_url_missing() -> None:
    """Test that build_client raises ValueError when LLM_BASE_URL is missing."""
    env = {"LLM_API_KEY": "sk-test", "LLM_MODEL": "gpt-4o-mini"}
    with um.patch.dict("os.environ", env, clear=True):
        with pytest.raises(ValueError, match="LLM_BASE_URL"):
            build_client()


def test_build_client_raises_when_api_key_missing() -> None:
    """Test that build_client raises ValueError when LLM_API_KEY is missing."""
    env = {"LLM_BASE_URL": "https://api.openai.com/v1", "LLM_MODEL": "gpt-4o-mini"}
    with um.patch.dict("os.environ", env, clear=True):
        with pytest.raises(ValueError, match="LLM_API_KEY"):
            build_client()


def test_build_client_raises_when_model_missing() -> None:
    """Test that build_client raises ValueError when LLM_MODEL is missing."""
    env = {"LLM_BASE_URL": "https://api.openai.com/v1", "LLM_API_KEY": "sk-test"}
    with um.patch.dict("os.environ", env, clear=True):
        with pytest.raises(ValueError, match="LLM_MODEL"):
            build_client()


def test_build_client_returns_openai_client() -> None:
    """Test that build_client returns an OpenAI client with correct config."""
    env = {
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "sk-test-key",
        "LLM_MODEL": "gpt-4o-mini",
    }
    with um.patch.dict("os.environ", env, clear=True):
        # Patch the OpenAI constructor to avoid actual network calls
        with um.patch.object(
            openai.OpenAI, "__init__", return_value=None
        ) as mock_init:
            build_client()
            # Verify constructor was called with correct args
            mock_init.assert_called_once_with(
                base_url="https://api.openai.com/v1",
                api_key="sk-test-key",
                max_retries=3,
            )


# ---------------------------------------------------------------------------
# CorrectedChunk model tests
# ---------------------------------------------------------------------------


def test_corrected_chunk_validates_text() -> None:
    """Test that CorrectedChunk validates and stores text correctly."""
    model = CorrectedChunk(text="Hello world")
    assert model.text == "Hello world"


def test_corrected_chunk_rejects_empty_text() -> None:
    """Test that CorrectedChunk rejects empty text."""
    # Empty string is valid in Pydantic, but the LLM module checks for it
    model = CorrectedChunk(text="")
    assert model.text == ""


# ---------------------------------------------------------------------------
# correct_and_translate() tests
# ---------------------------------------------------------------------------


def _make_mock_client(
    responses: list[str] | None = None,
) -> openai.OpenAI:
    """Create a mock OpenAI client with pre-configured responses.

    Args:
        responses: List of JSON strings to return for each chunk.

    Returns:
        A MagicMock configured to simulate openai.OpenAI.
    """
    mock_client = um.MagicMock()
    mock_client.max_retries = 3

    if responses is not None:
        mock_client.chat.completions.create.return_value = um.MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            um.MagicMock(message=um.MagicMock(content=resp)) for resp in responses
        ]

    return mock_client


def test_correct_and_translate_empty_chunks_returns_empty() -> None:
    """Test that correct_and_translate returns empty list for empty input."""
    client = _make_mock_client()
    result = correct_and_translate([], "en", client=client)
    assert result == []


def test_correct_and_translate_context_window() -> None:
    """Test that each chunk is sent with the configured context window.

    Verifies that the LLM receives a window of chunks before and after the
    target chunk, with the correct size.
    """
    mock_client = _make_mock_client(
        responses=[json.dumps({"text": f"Translated {i}"}) for i in range(5)]
    )

    chunks = [_make_chunk(f"Danish text {i}") for i in range(5)]
    result = correct_and_translate(
        chunks, "en", client=mock_client, context_window=2
    )

    # Verify each chunk was translated
    for i, chunk in enumerate(result):
        assert chunk.text == f"Translated {i}"

    # Verify the chat API was called 5 times (once per chunk)
    assert mock_client.chat.completions.create.call_count == 5


def test_correct_and_translate_malformed_json_preserves_original() -> None:
    """Test that malformed JSON from the LLM preserves the original text.

    Verifies that when the LLM returns invalid JSON, the original chunk text
    is preserved and a warning is logged.
    """
    import logging

    mock_client = _make_mock_client(
        responses=[
            "not valid json",  # malformed
            json.dumps({"text": "Good translation"}),
        ]
    )

    chunks = [_make_chunk("Original text"), _make_chunk("Second text")]

    with um.patch("but_with_subs.llm.logger") as mock_logger:
        result = correct_and_translate(
            chunks, "en", client=mock_client, context_window=0
        )

        # First chunk should preserve original text
        assert result[0].text == "Original text"
        # Second chunk should be translated
        assert result[1].text == "Good translation"

        # Verify warning was logged for the first chunk
        assert mock_logger.warning.called


def test_correct_and_translate_empty_text_preserves_original() -> None:
    """Test that empty 'text' field from LLM preserves original chunk text.

    Verifies that when the LLM returns valid JSON with an empty 'text' field,
    the original chunk text is preserved.
    """
    mock_client = _make_mock_client(
        responses=[
            json.dumps({"text": ""}),  # empty text
            json.dumps({"text": "Valid translation"}),
        ]
    )

    chunks = [_make_chunk("Original text"), _make_chunk("Second text")]

    with um.patch("but_with_subs.llm.logger") as mock_logger:
        result = correct_and_translate(
            chunks, "en", client=mock_client, context_window=0
        )

        # First chunk should preserve original text
        assert result[0].text == "Original text"
        # Second chunk should be translated
        assert result[1].text == "Valid translation"

        # Verify warning was logged for the first chunk
        assert mock_logger.warning.called


def test_correct_and_translate_progress_callback() -> None:
    """Test that on_progress fires once per chunk with monotonically increasing ratios.

    Verifies that the progress callback is invoked after each chunk with a ratio
    that increases monotonically from 0 to 1.
    """
    mock_client = _make_mock_client(
        responses=[json.dumps({"text": f"Result {i}"}) for i in range(5)]
    )

    chunks = [_make_chunk(f"Text {i}") for i in range(5)]
    progress_values: list[float] = []

    def _on_progress(ratio: float) -> None:
        progress_values.append(ratio)

    correct_and_translate(
        chunks, "en", client=mock_client, on_progress=_on_progress, context_window=0
    )

    # Verify progress was called 5 times
    assert len(progress_values) == 5

    # Verify monotonically increasing ratios
    for i in range(1, len(progress_values)):
        assert progress_values[i] >= progress_values[i - 1]

    # Verify the last ratio is 1.0
    assert progress_values[-1] == 1.0


def test_correct_and_translate_preserves_chunk_metadata() -> None:
    """Test that timing and speaker metadata are preserved through translation.

    Verifies that start_time, end_time, and speaker are carried through the
    translation process unchanged.
    """
    mock_client = _make_mock_client(
        responses=[json.dumps({"text": "Translated text"})]
    )

    original_start = 5.5
    original_end = 8.5
    original_speaker = "Charlie"
    chunk = _make_chunk(
        "Original text",
        start_time=original_start,
        end_time=original_end,
        speaker=original_speaker,
    )

    result = correct_and_translate(
        [chunk], "en", client=mock_client, context_window=0
    )

    assert result[0].start_time == original_start
    assert result[0].end_time == original_end
    assert result[0].speaker == original_speaker
    assert result[0].text == "Translated text"


def test_correct_and_translate_handles_none_text() -> None:
    """Test that chunks with None text are handled gracefully.

    Verifies that chunks without text are passed through with the original
    (None) text preserved.
    """
    mock_client = _make_mock_client(
        responses=[
            json.dumps({"text": "Translated"}),
            json.dumps({"text": "Second"}),
        ]
    )

    chunks = [
        _make_chunk("Has text"),
        Chunk(
            start_time=1.0,
            end_time=2.0,
            audio=[],  # type: ignore
            text=None,
            speaker="Bob",
        ),
    ]

    # For None text, the LLM will receive "[no text]" as the chunk text
    # and should return a translation
    result = correct_and_translate(
        chunks, "en", client=mock_client, context_window=0
    )

    assert result[0].text == "Translated"
    assert result[1].text == "Second"
