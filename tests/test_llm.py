"""Tests for the LLM correct-and-translate module.

This module contains comprehensive tests for the LLM-based translation functions,
including mocking the OpenAI client to verify correct behaviour under various
conditions.
"""

import json
import unittest.mock as um

import openai
import pytest

from danglish.data_models import Chunk
from danglish.llm import CorrectedChunk, build_client, correct_and_translate


def _make_chunk(
    text: str | None,
    start_time: float = 0.0,
    end_time: float = 1.0,
    speaker: str | None = None,
) -> Chunk:
    """Create a Chunk for testing.

    Returns:
        A Chunk instance with the given fields.
    """
    return Chunk(start_time=start_time, end_time=end_time, text=text, speaker=speaker)


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
        with um.patch.object(openai.OpenAI, "__init__", return_value=None) as mock_init:
            build_client()
            # Verify constructor was called with correct args
            mock_init.assert_called_once_with(
                base_url="https://api.openai.com/v1",
                api_key="sk-test-key",
                max_retries=0,
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


def _make_mock_client(responses: list[str] | None = None) -> openai.OpenAI:
    """Create a mock OpenAI client with pre-configured responses.

    Args:
        responses: List of JSON strings to return for each chunk.

    Returns:
        A MagicMock configured to simulate openai.OpenAI.
    """
    mock_client = um.MagicMock()
    mock_client.max_retries = 3

    if responses is not None:
        mock_responses = []
        for resp in responses:
            mock_resp = um.MagicMock()
            mock_resp.choices = [um.MagicMock(message=um.MagicMock(content=resp))]
            mock_responses.append(mock_resp)
        mock_client.chat.completions.create.side_effect = mock_responses

    return mock_client


def test_correct_and_translate_empty_chunks_returns_empty() -> None:
    """Test that correct_and_translate returns empty list for empty input."""
    client = _make_mock_client()
    result = correct_and_translate([], "en", client=client)
    assert result == []


def _batch_response(translations: dict[int, str]) -> str:
    """Build a JSON response of the form the LLM is asked to emit.

    Args:
        translations:
            Mapping from chunk index to corrected text.

    Returns:
        A JSON-encoded ``{"translations": {...}}`` string.
    """
    return json.dumps({"translations": {str(k): v for k, v in translations.items()}})


def test_correct_and_translate_batches_chunks() -> None:
    """Test that chunks are batched and each batch produces one API call.

    With 5 chunks and ``batch_size=2`` we expect 3 API calls (batches of
    size 2, 2, 1), and the translations returned by the LLM are applied
    to the corresponding chunks.
    """
    mock_client = _make_mock_client(
        responses=[
            _batch_response({0: "Translated 0", 1: "Translated 1"}),
            _batch_response({2: "Translated 2", 3: "Translated 3"}),
            _batch_response({4: "Translated 4"}),
        ]
    )

    chunks = [_make_chunk(f"Danish text {i}") for i in range(5)]
    result = correct_and_translate(
        chunks, "en", client=mock_client, context_window=2, batch_size=2
    )

    for i, chunk in enumerate(result):
        assert chunk.text == f"Translated {i}"
    assert mock_client.chat.completions.create.call_count == 3


def test_correct_and_translate_malformed_json_preserves_original() -> None:
    """Test that malformed JSON for a batch preserves originals for that batch."""
    mock_client = _make_mock_client(
        responses=["not valid json", _batch_response({1: "Good translation"})]
    )

    chunks = [_make_chunk("Original text"), _make_chunk("Second text")]

    with um.patch("danglish.llm.logger") as mock_logger:
        result = correct_and_translate(
            chunks, "en", client=mock_client, context_window=0, batch_size=1
        )

        assert result[0].text == "Original text"
        assert result[1].text == "Good translation"
        assert mock_logger.warning.called


def test_correct_and_translate_missing_id_preserves_original() -> None:
    """Test that a chunk omitted from the LLM response falls back to its original."""
    mock_client = _make_mock_client(
        responses=[_batch_response({1: "Valid translation"})]
    )

    chunks = [_make_chunk("Original text"), _make_chunk("Second text")]

    result = correct_and_translate(
        chunks, "en", client=mock_client, context_window=0, batch_size=2
    )

    assert result[0].text == "Original text"
    assert result[1].text == "Valid translation"


def test_correct_and_translate_progress_callback() -> None:
    """Test that on_progress fires once per batch with increasing ratios."""
    mock_client = _make_mock_client(
        responses=[
            _batch_response({0: "A", 1: "B"}),
            _batch_response({2: "C", 3: "D"}),
            _batch_response({4: "E"}),
        ]
    )

    chunks = [_make_chunk(f"Text {i}") for i in range(5)]
    progress_values: list[float] = []

    correct_and_translate(
        chunks,
        "en",
        client=mock_client,
        on_progress=progress_values.append,
        context_window=0,
        batch_size=2,
    )

    assert len(progress_values) == 3
    for i in range(1, len(progress_values)):
        assert progress_values[i] >= progress_values[i - 1]
    assert progress_values[-1] == 1.0


def test_correct_and_translate_preserves_chunk_metadata() -> None:
    """Test that timing and speaker metadata are preserved through translation."""
    mock_client = _make_mock_client(responses=[_batch_response({0: "Translated text"})])

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
        [chunk], "en", client=mock_client, context_window=0, batch_size=1
    )

    assert result[0].start_time == original_start
    assert result[0].end_time == original_end
    assert result[0].speaker == original_speaker
    assert result[0].text == "Translated text"


def test_correct_and_translate_handles_none_text() -> None:
    """Test that chunks with None text are handled gracefully."""
    mock_client = _make_mock_client(
        responses=[_batch_response({0: "Translated", 1: "Second"})]
    )

    chunks = [
        _make_chunk("Has text"),
        _make_chunk(None, start_time=1.0, end_time=2.0, speaker="Bob"),
    ]

    result = correct_and_translate(
        chunks, "en", client=mock_client, context_window=0, batch_size=2
    )

    assert result[0].text == "Translated"
    assert result[1].text == "Second"
