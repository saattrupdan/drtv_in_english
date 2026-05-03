"""Tests for the text_chunking module.

This module contains tests for the _split_text and chunk_transcriptions functions.
"""

import unittest.mock as um

from but_with_subs.data_models import Transcription
from but_with_subs.text_chunking import _split_text, chunk_transcriptions

# ---------------------------------------------------------------------------
# _split_text tests
# ---------------------------------------------------------------------------


def test_split_text_empty_returns_empty_list() -> None:
    """Test _split_text returns an empty list for empty text."""
    result = _split_text(text="", max_words=5)

    assert result == []


def test_split_text_whitespace_only_returns_single() -> None:
    """Test _split_text returns a single segment for whitespace-only text."""
    result = _split_text(text="   ", max_words=5)

    assert result == ["   "]


def test_split_text_short_text_returns_single_segment() -> None:
    """Test _split_text returns a single segment when text is short enough."""
    text = "hello world"
    result = _split_text(text=text, max_words=10)

    assert result == ["hello world"]


def test_split_text_exact_max_words_returns_single_segment() -> None:
    """Test _split_text returns a single segment when word count equals max_words."""
    text = "one two three four five"
    result = _split_text(text=text, max_words=5)

    assert result == ["one two three four five"]


def test_split_text_long_text_chunks_by_words() -> None:
    """Test _split_text splits long text into word-sized chunks."""
    text = "one two three four five six seven eight"
    result = _split_text(text=text, max_words=4)

    assert len(result) == 2
    assert result[0] == "one two three four"
    assert result[1] == "five six seven eight"


def test_split_text_long_text_with_sentences() -> None:
    """Test _split_text respects sentence boundaries when sentences are short enough.

    Mocks nltk.sent_tokenize to return expected sentence segments and verifies
    that short sentences are returned as-is without further splitting.
    """
    text = "First sentence. Second sentence."

    with um.patch(
        "but_with_subs.text_chunking.nltk.sent_tokenize",
        return_value=["First sentence.", "Second sentence."],
    ):
        result = _split_text(text=text, max_words=5)

    assert len(result) == 2
    assert result[0] == "First sentence."
    assert result[1] == "Second sentence."


def test_split_text_long_sentence_splits_by_punctuation() -> None:
    """Test _split_text splits long sentences on punctuation pauses."""
    text = "Hello world, how are you? I am fine, thanks."
    result = _split_text(text=text, max_words=3)

    assert len(result) >= 2


def test_split_text_sentence_too_long_chunks_by_word_count() -> None:
    """Test _split_text falls back to word count when sentences are too long."""
    text = "A very long sentence with many words that exceeds the limit"
    result = _split_text(text=text, max_words=5)

    assert len(result) >= 1
    for segment in result:
        assert len(segment.split()) <= 5


# ---------------------------------------------------------------------------
# chunk_transcriptions tests
# ---------------------------------------------------------------------------


def test_chunk_transcriptions_single_short_returns_single() -> None:
    """Test chunk_transcriptions returns a single segment for short text.

    Verifies that when the combined transcription text is short enough,
    a single Transcription is returned without any modification.
    """
    transcriptions = [
        Transcription(start_time=0.0, end_time=0.5, text="hello"),
        Transcription(start_time=0.5, end_time=1.0, text="world"),
    ]
    mock_punctuator = um.Mock()
    mock_punctuator.punctuate = um.Mock(side_effect=lambda text: text)
    with um.patch(
        "but_with_subs.text_chunking.PunctFixer", return_value=mock_punctuator
    ):
        result = chunk_transcriptions(transcriptions=transcriptions, max_words=10)

    assert len(result) == 1
    assert result[0].text == "hello world"


def test_chunk_transcriptions_long_text_returns_multiple_segments() -> None:
    """Test chunk_transcriptions splits long text into multiple segments.

    Creates a list of transcriptions with combined text exceeding max_words,
    and verifies that the result contains multiple segments.
    """
    transcriptions = [
        Transcription(start_time=0.0, end_time=0.5, text="one"),
        Transcription(start_time=0.5, end_time=1.0, text="two"),
        Transcription(start_time=1.0, end_time=1.5, text="three"),
        Transcription(start_time=1.5, end_time=2.0, text="four"),
        Transcription(start_time=2.0, end_time=2.5, text="five"),
        Transcription(start_time=2.5, end_time=3.0, text="six"),
    ]
    mock_punctuator = um.Mock()
    mock_punctuator.punctuate = um.Mock(side_effect=lambda text: text)
    with um.patch(
        "but_with_subs.text_chunking.PunctFixer", return_value=mock_punctuator
    ):
        result = chunk_transcriptions(transcriptions=transcriptions, max_words=3)

    assert len(result) >= 2


def test_chunk_transcriptions_preserves_time_ranges() -> None:
    """Test chunk_transcriptions preserves start and end times from matching words.

    Verifies that the returned segments have start_time from the first word's
    start_time and end_time from the last word's end_time.
    """
    transcriptions = [
        Transcription(start_time=0.0, end_time=0.5, text="hello"),
        Transcription(start_time=0.5, end_time=1.0, text="beautiful"),
        Transcription(start_time=1.0, end_time=1.5, text="world"),
    ]
    mock_punctuator = um.Mock()
    mock_punctuator.punctuate = um.Mock(side_effect=lambda text: text)
    with um.patch(
        "but_with_subs.text_chunking.PunctFixer", return_value=mock_punctuator
    ):
        result = chunk_transcriptions(transcriptions=transcriptions, max_words=2)

    assert len(result) == 2
    assert result[0].start_time == 0.0
    assert result[0].end_time == 1.0
    assert result[1].start_time == 1.0
    assert result[1].end_time == 1.5


def test_chunk_transcriptions_handles_punctuation_in_segments() -> None:
    """Test chunk_transcriptions handles punctuation in generated segments.

    Verifies that segments with punctuation are created correctly and
    time ranges are preserved based on word matching.
    """
    transcriptions = [
        Transcription(start_time=0.0, end_time=0.5, text="hello"),
        Transcription(start_time=0.5, end_time=1.0, text="world"),
    ]
    mock_punctuator = um.Mock()
    mock_punctuator.punctuate = um.Mock(side_effect=lambda text: text)
    with um.patch(
        "but_with_subs.text_chunking.PunctFixer", return_value=mock_punctuator
    ):
        result = chunk_transcriptions(transcriptions=transcriptions, max_words=3)

    assert len(result) == 1
    assert "hello" in result[0].text
    assert "world" in result[0].text


def test_chunk_transcriptions_with_multiple_segments() -> None:
    """Test chunk_transcriptions creates multiple segments from long input.

    Creates transcriptions with enough words to exceed max_words,
    and verifies that multiple segments are created with correct
    time ranges and text content.
    """
    transcriptions = [
        Transcription(start_time=0.0, end_time=0.5, text="first"),
        Transcription(start_time=0.5, end_time=1.0, text="second"),
        Transcription(start_time=1.0, end_time=1.5, text="third"),
        Transcription(start_time=1.5, end_time=2.0, text="fourth"),
        Transcription(start_time=2.0, end_time=2.5, text="fifth"),
        Transcription(start_time=2.5, end_time=3.0, text="sixth"),
    ]
    mock_punctuator = um.Mock()
    mock_punctuator.punctuate = um.Mock(side_effect=lambda text: text)
    with um.patch(
        "but_with_subs.text_chunking.PunctFixer", return_value=mock_punctuator
    ):
        result = chunk_transcriptions(transcriptions=transcriptions, max_words=2)

    assert len(result) >= 2
    for segment in result:
        assert segment.start_time < segment.end_time
        assert len(segment.text.split()) <= 2
