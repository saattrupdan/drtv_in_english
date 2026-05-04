"""Tests for the translation module.

This module contains comprehensive tests for the translation functions,
including mocking the transformer model and tokenizer to verify correct
behaviour under various conditions.
"""

import unittest.mock as um

import numpy as np
import torch

from but_with_subs.data_models import Chunk
from but_with_subs.translation import (
    _translate_batch,
    translate_chunks,
    translate_single,
)


def _make_mock_model() -> um.MagicMock:
    """Create a mock model with sensible generate behaviour.

    Returns:
        A MagicMock configured to simulate M2M100ForConditionalGeneration.
    """
    mock = um.MagicMock()
    mock.device = torch.device("cpu")
    mock_sequences = um.MagicMock()
    mock_sequences.sequences = torch.tensor([[1, 2, 3, 4]])
    mock.generate.return_value = mock_sequences
    return mock


def _make_mock_tokenizer() -> um.MagicMock:
    """Create a mock tokenizer with sensible encode/decode behaviour.

    Returns:
        A MagicMock configured to simulate SMALL100Tokenizer.
    """
    mock = um.MagicMock()
    # The tokenizer is called as tokenizer(text, return_tensors="pt")
    # and the result has .to(device) called on it. The result of .to()
    # must be dict-like so that **encoded works in model.generate(**encoded).
    mock_return = um.MagicMock()
    mock_return.to.return_value = um.MagicMock()
    mock.return_value = mock_return
    mock.batch_decode.return_value = ["Translated text"]
    return mock


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
    return Chunk(
        start_time=start_time,
        end_time=end_time,
        audio=np.zeros(16000, dtype=np.float32),
        text=text,
        speaker=speaker,
    )


def _run_translate_chunks(
    chunks: list[Chunk],
    target_lang: str = "fr",
    batch_size: int = 2,
    mock_model: um.MagicMock | None = None,
    mock_tokenizer: um.MagicMock | None = None,
) -> list[Chunk]:
    """Helper to run translate_chunks with standard mocks.

    Args:
        chunks: List of chunks to translate.
        target_lang: Target language code.
        batch_size: Batch size for translation.
        mock_model: Optional pre-configured mock model.
        mock_tokenizer: Optional pre-configured mock tokenizer.

    Returns:
        List of translated chunks.
    """
    if mock_model is None:
        mock_model = _make_mock_model()
    if mock_tokenizer is None:
        mock_tokenizer = _make_mock_tokenizer()

    # Create fresh mocks for the class-level patches that have from_pretrained
    mock_model_cls = um.MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_model

    mock_tokenizer_cls = um.MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

    with (
        um.patch(
            "but_with_subs.translation.M2M100ForConditionalGeneration", mock_model_cls
        ),
        um.patch("but_with_subs.translation.SMALL100Tokenizer", mock_tokenizer_cls),
        um.patch("but_with_subs.translation.get_device") as mock_device,
        um.patch("but_with_subs.translation.bnb.no_terminal_output") as mock_ctx,
        um.patch("but_with_subs.translation.logger"),
    ):
        mock_device.return_value = torch.device("cpu")
        mock_ctx.return_value.__enter__ = um.MagicMock(return_value=None)
        mock_ctx.return_value.__exit__ = um.MagicMock(return_value=None)

        return list(
            translate_chunks(
                chunks=chunks, target_lang=target_lang, batch_size=batch_size
            )
        )


# ---------------------------------------------------------------------------
# translate_chunks() tests
# ---------------------------------------------------------------------------


def test_translate_chunks_does_not_mutate_originals() -> None:
    """Regression test: translate_chunks must not mutate original Chunk objects."""
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Translated 1", "Translated 2"]

    chunks = [_make_chunk("Original text"), _make_chunk("Another text")]
    original_texts = [c.text for c in chunks]

    results = _run_translate_chunks(
        chunks=chunks, batch_size=2, mock_tokenizer=mock_tokenizer
    )

    for orig, chunk in zip(original_texts, chunks):
        assert chunk.text == orig, (
            f"Original chunk was mutated: {orig!r} -> {chunk.text!r}"
        )

    for chunk in results:
        assert chunk.text not in original_texts


def test_translate_chunks_empty_chunks_yields_nothing() -> None:
    """Test that translate_chunks yields nothing when given empty chunks.

    Verifies that calling translate_chunks with an empty list of chunks
    produces no output.
    """
    results = _run_translate_chunks(chunks=[])

    assert results == []


def test_translate_chunks_no_text_chunks_yields_nothing() -> None:
    """Test that chunks without text produce no output.

    Verifies that when no chunks have text, translate_chunks returns
    without yielding anything.
    """
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=1.0,
            audio=np.zeros(16000, dtype=np.float32),
            text=None,
            speaker=None,
        ),
        Chunk(
            start_time=1.0,
            end_time=2.0,
            audio=np.zeros(16000, dtype=np.float32),
            text=None,
            speaker="Alice",
        ),
    ]

    results = _run_translate_chunks(chunks=chunks)

    assert results == []


def test_translate_chunks_yields_translated_chunks() -> None:
    """Test that translate_chunks yields translated chunks.

    Verifies that the generator yields translated chunks in order.
    """
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Bonjour"]

    chunk = _make_chunk("Hello")
    results = _run_translate_chunks(
        chunks=[chunk], target_lang="fr", mock_tokenizer=mock_tokenizer
    )

    assert len(results) == 1
    assert results[0].text == "Bonjour"


def test_translate_chunks_batch_size_handling() -> None:
    """Test that batch_size=2 with 5 chunks processes in 3 batches.

    Verifies that 5 chunks with batch_size=2 are split into batches
    of sizes 2, 2, and 1.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    decode_calls: list[list[str]] = [["T1", "T2"], ["T3", "T4"], ["T5"]]
    mock_tokenizer.batch_decode.side_effect = decode_calls

    chunks = [_make_chunk(f"Text {i}") for i in range(5)]

    results = _run_translate_chunks(
        chunks=chunks,
        batch_size=2,
        mock_model=mock_model,
        mock_tokenizer=mock_tokenizer,
    )

    assert len(results) == 5
    assert results[0].text == "T1"
    assert results[1].text == "T2"
    assert results[2].text == "T3"
    assert results[3].text == "T4"
    assert results[4].text == "T5"
    # batch_decode should be called 3 times (2+2+1)
    assert mock_tokenizer.batch_decode.call_count == 3


def test_translate_chunks_preserves_metadata() -> None:
    """Test that start_time, end_time, and speaker are preserved.

    Verifies that chunk metadata is carried through the translation
    process unchanged.
    """
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Traduit"]

    original_start = 5.5
    original_end = 8.5
    original_speaker = "Charlie"
    chunk = Chunk(
        start_time=original_start,
        end_time=original_end,
        audio=np.zeros(16000, dtype=np.float32),
        text="Original text",
        speaker=original_speaker,
    )

    results = _run_translate_chunks(
        chunks=[chunk], target_lang="fr", mock_tokenizer=mock_tokenizer
    )

    assert results[0].start_time == original_start
    assert results[0].end_time == original_end
    assert results[0].speaker == original_speaker
    assert results[0].text == "Traduit"


def test_translate_chunks_mixed_text_and_none() -> None:
    """Test translate_chunks with mixed text and None chunks.

    Verifies that only chunks with text are translated, and that
    chunks without text are skipped entirely.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    # Two text chunks are processed together in one batch of size 2
    mock_tokenizer.batch_decode.return_value = ["Translated 1", "Translated 2"]

    chunks = [
        _make_chunk("Has text"),
        Chunk(
            start_time=1.0,
            end_time=2.0,
            audio=np.zeros(16000, dtype=np.float32),
            text=None,
            speaker="Bob",
        ),
        _make_chunk("Also has text"),
    ]

    results = _run_translate_chunks(
        chunks=chunks,
        batch_size=2,
        mock_model=mock_model,
        mock_tokenizer=mock_tokenizer,
    )

    # Only 2 chunks with text should be yielded
    assert len(results) == 2
    assert results[0].text == "Translated 1"
    assert results[1].text == "Translated 2"


def test_translate_chunks_single_chunk() -> None:
    """Test translate_chunks with a single chunk.

    Verifies that a single chunk is translated correctly without
    triggering multi-batch logic.
    """
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Single translation"]

    chunk = _make_chunk("Single text")

    results = _run_translate_chunks(
        chunks=[chunk], target_lang="de", batch_size=3, mock_tokenizer=mock_tokenizer
    )

    assert len(results) == 1
    assert results[0].text == "Single translation"


def test_translate_chunks_multiple_batches_yield_order() -> None:
    """Test that chunks are yielded in order across multiple batches.

    Verifies that when processing multiple batches, the output chunks
    maintain the original ordering.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    decode_calls: list[list[str]] = [["French A", "French B"], ["French C", "French D"]]
    mock_tokenizer.batch_decode.side_effect = decode_calls

    chunks = [_make_chunk(f"Original {i}") for i in range(4)]

    results = _run_translate_chunks(
        chunks=chunks,
        batch_size=2,
        mock_model=mock_model,
        mock_tokenizer=mock_tokenizer,
    )

    assert len(results) == 4
    assert results[0].text == "French A"
    assert results[1].text == "French B"
    assert results[2].text == "French C"
    assert results[3].text == "French D"


# ---------------------------------------------------------------------------
# translate_single() tests
# ---------------------------------------------------------------------------


def test_translate_single_returns_translated_text() -> None:
    """Test that translate_single returns the translated text.

    Verifies that calling translate_single with a mocked model and
    tokenizer returns the expected translated string.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()

    result = translate_single(
        model=mock_model, tokenizer=mock_tokenizer, text="Hello world"
    )

    assert result == "Translated text"


def test_translate_single_calls_model_generate() -> None:
    """Test that translate_single calls model.generate with correct inputs.

    Verifies that the model.generate method is called with the encoded
    input from the tokenizer.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()

    translate_single(model=mock_model, tokenizer=mock_tokenizer, text="Test text")

    assert mock_model.generate.called


def test_translate_single_passes_text_to_tokenizer() -> None:
    """Test that translate_single passes the correct text to the tokenizer.

    Verifies that the input text is passed as a keyword argument to
    the tokenizer.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()

    translate_single(model=mock_model, tokenizer=mock_tokenizer, text="Specific text")

    mock_tokenizer.assert_called_once_with("Specific text", return_tensors="pt")


# ---------------------------------------------------------------------------
# _translate_batch() tests
# ---------------------------------------------------------------------------


def test_translate_batch_single_text_uses_single_path() -> None:
    """Test that _translate_batch uses the single-text code path.

    Verifies that when given a single text, the tokenizer is called
    without padding/truncation arguments.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Single result"]

    result = _translate_batch(
        model=mock_model, tokenizer=mock_tokenizer, texts=["Hello"]
    )

    assert result == ["Single result"]
    # Single text path: no padding/truncation
    mock_tokenizer.assert_called_once_with("Hello", return_tensors="pt")


def test_translate_batch_multiple_texts_uses_batch_path() -> None:
    """Test that _translate_batch uses the batch code path.

    Verifies that when given multiple texts, the tokenizer is called
    with padding and truncation enabled.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Result A", "Result B"]

    result = _translate_batch(
        model=mock_model, tokenizer=mock_tokenizer, texts=["Hello", "World"]
    )

    assert result == ["Result A", "Result B"]
    # Batch path: with padding and truncation
    mock_tokenizer.assert_called_once_with(
        ["Hello", "World"], return_tensors="pt", padding=True, truncation=True
    )


def test_translate_batch_empty_list_uses_batch_path() -> None:
    """Test that _translate_batch with an empty list uses the batch path.

    Verifies that an empty list bypasses the single-text branch and
    enters the batch code path.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = []

    result = _translate_batch(model=mock_model, tokenizer=mock_tokenizer, texts=[])

    assert result == []
    # Empty list: len != 1, so batch path is taken
    # The tokenizer is called with padding and truncation
    mock_tokenizer.assert_called_once()
    call_kwargs = mock_tokenizer.call_args
    assert call_kwargs.kwargs.get("padding") is True


def test_translate_batch_multiple_texts_returns_correct_count() -> None:
    """Test that _translate_batch returns one result per input text.

    Verifies that the number of output translations matches the
    number of input texts.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["A", "B", "C", "D", "E"]

    texts = [f"Text {i}" for i in range(5)]
    result = _translate_batch(model=mock_model, tokenizer=mock_tokenizer, texts=texts)

    assert len(result) == 5
    assert result == ["A", "B", "C", "D", "E"]


def test_translate_batch_sends_encoded_tensor_to_device() -> None:
    """Test that encoded tensors are sent to the correct device.

    Verifies that the .to(device) method is called on the encoded
    tensor before passing it to model.generate.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Result"]
    mock_tokenizer.return_value = um.MagicMock()
    mock_tokenizer.return_value.to.return_value = um.MagicMock()

    _translate_batch(model=mock_model, tokenizer=mock_tokenizer, texts=["Hello"])

    assert mock_tokenizer.return_value.to.called


def test_translate_batch_single_text_sends_tensor_to_device() -> None:
    """Test that single text encoded tensor is sent to the correct device.

    Verifies that for single-text batches, the encoded tensor is still
    transferred to the model's device.
    """
    mock_model = _make_mock_model()
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Result"]
    mock_tokenizer.return_value = um.MagicMock()
    mock_tokenizer.return_value.to.return_value = um.MagicMock()

    _translate_batch(model=mock_model, tokenizer=mock_tokenizer, texts=["Single"])

    assert mock_tokenizer.return_value.to.called


def test_translate_batch_handles_tensor_return_from_generate() -> None:
    """Test that _translate_batch handles a raw Tensor from model.generate().

    Regression test for the bug where model.generate() returns a Tensor
    directly instead of an object with a .sequences attribute, causing
    an AttributeError.
    """
    mock_model = _make_mock_model()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Single result"]

    result = _translate_batch(
        model=mock_model, tokenizer=mock_tokenizer, texts=["Hello"]
    )

    assert result == ["Single result"]


def test_translate_batch_handles_tensor_return_in_batch_mode() -> None:
    """Test _translate_batch handles raw Tensor from model.generate() in batch mode.

    Regression test for the bug where model.generate() returns a Tensor
    directly instead of an object with a .sequences attribute, causing
    an AttributeError in the batch code path.
    """
    mock_model = _make_mock_model()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mock_tokenizer = _make_mock_tokenizer()
    mock_tokenizer.batch_decode.return_value = ["Result A", "Result B"]

    result = _translate_batch(
        model=mock_model, tokenizer=mock_tokenizer, texts=["Hello", "World"]
    )

    assert result == ["Result A", "Result B"]
