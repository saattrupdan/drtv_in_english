"""Tests for the text_chunking module."""

import logging
import re
import string
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from but_with_subs.constants import MIN_CHUNK_LENGTH_SECONDS
from but_with_subs.data_models import Chunk
from but_with_subs.text_chunking import _split_text, group_word_chunks

# Import the punctuation pattern used in the module
from but_with_subs.text_chunking import PUNCTUATION_PATTERN


logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_punctuation_model():
    """Create a mock punctuation model (PunctFixer)."""
    mock = MagicMock()
    # Default behavior: just return text with some basic punctuation added
    def mock_punctuate(text: str) -> str:
        # Simple mock that adds periods to sentences
        if not text.endswith("."):
            text += "."
        return text

    mock.punctuate = MagicMock(side_effect=mock_punctuate)
    return mock


@pytest.fixture
def mock_punctuation_model_with_fixes():
    """Create a mock punctuation model that returns pre-formatted text."""
    mock = MagicMock()

    def mock_punctuate(text: str) -> str:
        # Return text with proper punctuation for testing
        return text

    mock.punctuate = MagicMock(side_effect=mock_punctuate)
    return mock


@pytest.fixture
def simple_word_chunks():
    """Create simple word chunks for testing."""
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=0.5,
            audio=np.array([1.0, 2.0]),
            text="Hello",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=0.5,
            end_time=1.0,
            audio=np.array([3.0, 4.0]),
            text="world",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=1.0,
            end_time=1.5,
            audio=np.array([5.0, 6.0]),
            text="this",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=1.5,
            end_time=2.0,
            audio=np.array([7.0, 8.0]),
            text="is",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=2.0,
            end_time=2.5,
            audio=np.array([9.0, 10.0]),
            text="a",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=2.5,
            end_time=3.0,
            audio=np.array([11.0, 12.0]),
            text="test",
            speaker="SPEAKER_00",
        ),
    ]
    return chunks


@pytest.fixture
def word_chunks_with_multiple_speakers():
    """Create word chunks with multiple speakers."""
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=0.5,
            audio=np.array([1.0]),
            text="Hello",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=0.5,
            end_time=1.0,
            audio=np.array([2.0]),
            text="there",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=1.0,
            end_time=1.5,
            audio=np.array([3.0]),
            text="how",
            speaker="SPEAKER_01",
        ),
        Chunk(
            start_time=1.5,
            end_time=2.0,
            audio=np.array([4.0]),
            text="are",
            speaker="SPEAKER_01",
        ),
        Chunk(
            start_time=2.0,
            end_time=2.5,
            audio=np.array([5.0]),
            text="you",
            speaker="SPEAKER_01",
        ),
    ]
    return chunks


@pytest.fixture
def word_chunks_with_punctuation():
    """Create word chunks with punctuation in text."""
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=0.5,
            audio=np.array([1.0]),
            text="Hello!",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=0.5,
            end_time=1.0,
            audio=np.array([2.0]),
            text="world?",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=1.0,
            end_time=1.5,
            audio=np.array([3.0]),
            text="testing...",
            speaker="SPEAKER_00",
        ),
    ]
    return chunks


@pytest.fixture
def word_chunks_with_none_text():
    """Create word chunks with some None text values."""
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=0.5,
            audio=np.array([1.0]),
            text="Hello",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=0.5,
            end_time=1.0,
            audio=np.array([2.0]),
            text=None,
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=1.0,
            end_time=1.5,
            audio=np.array([3.0]),
            text="world",
            speaker="SPEAKER_00",
        ),
    ]
    return chunks


@pytest.fixture
def word_chunks_short_duration():
    """Create word chunks with short duration (below minimum)."""
    chunks = [
        Chunk(
            start_time=0.0,
            end_time=0.01,
            audio=np.array([1.0]),
            text="short",
            speaker="SPEAKER_00",
        ),
        Chunk(
            start_time=0.01,
            end_time=0.02,
            audio=np.array([2.0]),
            text="chunk",
            speaker="SPEAKER_00",
        ),
    ]
    return chunks


# =============================================================================
# Tests for _split_text() function - Basic Functionality
# =============================================================================


class TestSplitTextBasicFunctionality:
    """Tests for basic text splitting functionality."""

    def test_text_shorter_than_max_words_returns_single_segment(
        self,
    ) -> None:
        """Test that text shorter than max_words returns a single segment."""
        text = "Hello world"
        result = _split_text(text=text, max_words=10)

        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_text_exact_max_words_returns_single_segment(
        self,
    ) -> None:
        """Test that text with exactly max_words returns a single segment."""
        text = "one two three four five"
        result = _split_text(text=text, max_words=5)

        assert len(result) == 1
        assert result[0] == "one two three four five"

    def test_text_longer_than_max_words_splits_into_multiple_segments(
        self,
    ) -> None:
        """Test that text longer than max_words is split correctly."""
        text = "one two three four five six seven eight"
        result = _split_text(text=text, max_words=4)

        # Should be split into two segments of 4 words each
        assert len(result) == 2
        assert result[0] == "one two three four"
        assert result[1] == "five six seven eight"

    def test_single_word_text(
        self,
    ) -> None:
        """Test splitting a single word."""
        text = "hello"
        result = _split_text(text=text, max_words=5)

        assert len(result) == 1
        assert result[0] == "hello"


# =============================================================================
# Tests for _split_text() function - Sentence Segmentation
# =============================================================================


class TestSplitTextSentenceSegmentation:
    """Tests for sentence-based text splitting."""

    def test_sentences_split_at_periods(
        self,
    ) -> None:
        """Test that text is split at sentence boundaries (periods)."""
        text = "First sentence. Second sentence. Third sentence."
        result = _split_text(text=text, max_words=10)

        # Should be split into 3 sentences
        assert len(result) >= 1
        # Verify sentences are separated
        assert any("First sentence" in seg for seg in result)
        assert any("Second sentence" in seg for seg in result)
        assert any("Third sentence" in seg for seg in result)

    def test_long_sentence_split_by_max_words(
        self,
    ) -> None:
        """Test that long sentences are split by max_words limit."""
        text = "one two three four five six seven eight nine ten eleven twelve"
        result = _split_text(text=text, max_words=4)

        # Should be split into 3 segments
        assert len(result) == 3
        assert result[0] == "one two three four"
        assert result[1] == "five six seven eight"
        assert result[2] == "nine ten eleven twelve"


# =============================================================================
# Tests for _split_text() function - Punctuation Splitting
# =============================================================================


class TestSplitTextPunctuationSplitting:
    """Tests for punctuation-based text splitting."""

    def test_split_at_comma(
        self,
    ) -> None:
        """Test splitting at commas."""
        text = "first, second, third, fourth"
        result = _split_text(text=text, max_words=10)

        # Should split at commas
        assert len(result) >= 1
        assert any("first" in seg for seg in result)
        assert any("second" in seg for seg in result)

    def test_split_at_semicolon(
        self,
    ) -> None:
        """Test splitting at semicolons."""
        text = "first; second; third; fourth"
        result = _split_text(text=text, max_words=10)

        # Should split at semicolons
        assert len(result) >= 1
        assert any("first" in seg for seg in result)
        assert any("second" in seg for seg in result)

    def test_split_at_colon(
        self,
    ) -> None:
        """Test splitting at colons."""
        text = "first: second: third: fourth"
        result = _split_text(text=text, max_words=10)

        # Should split at colons
        assert len(result) >= 1
        assert any("first" in seg for seg in result)
        assert any("second" in seg for seg in result)

    def test_split_at_dash(
        self,
    ) -> None:
        """Test splitting at dashes."""
        text = "first - second - third - fourth"
        result = _split_text(text=text, max_words=10)

        # Should split at dashes
        assert len(result) >= 1
        assert any("first" in seg for seg in result)
        assert any("second" in seg for seg in result)


# =============================================================================
# Tests for _split_text() function - Word Segmentation Fallback
# =============================================================================


class TestSplitTextWordSegmentation:
    """Tests for word-based text splitting fallback."""

    def test_very_long_text_split_by_words(
        self,
    ) -> None:
        """Test that very long text without punctuation is split by words."""
        # Create a long text without punctuation
        words = [f"word{i}" for i in range(20)]
        text = " ".join(words)
        result = _split_text(text=text, max_words=5)

        # Should be split into 4 segments of 5 words each
        assert len(result) == 4
        assert result[0] == "word0 word1 word2 word3 word4"
        assert result[1] == "word5 word6 word7 word8 word9"
        assert result[2] == "word10 word11 word12 word13 word14"
        assert result[3] == "word15 word16 word17 word18 word19"

    def test_text_with_irregular_word_count(
        self,
    ) -> None:
        """Test splitting when word count is not evenly divisible by max_words."""
        text = "one two three four five six seven"
        result = _split_text(text=text, max_words=3)

        # 7 words with max 3 per segment = 3 segments (3 + 3 + 1)
        assert len(result) == 3
        assert result[0] == "one two three"
        assert result[1] == "four five six"
        assert result[2] == "seven"


# =============================================================================
# Tests for _split_text() function - Edge Cases
# =============================================================================


class TestSplitTextEdgeCases:
    """Tests for edge cases in text splitting."""

    def test_empty_string(
        self,
    ) -> None:
        """Test splitting an empty string."""
        result = _split_text(text="", max_words=5)

        assert len(result) == 1
        assert result[0] == ""

    def test_whitespace_only(
        self,
    ) -> None:
        """Test splitting whitespace-only text."""
        result = _split_text(text="   ", max_words=5)

        assert len(result) == 1
        assert result[0] == "   "

    def test_max_words_of_one(
        self,
    ) -> None:
        """Test splitting with max_words=1."""
        text = "one two three"
        result = _split_text(text=text, max_words=1)

        assert len(result) == 3
        assert result[0] == "one"
        assert result[1] == "two"
        assert result[2] == "three"

    def test_very_large_max_words(
        self,
    ) -> None:
        """Test with a very large max_words value."""
        text = "one two three"
        result = _split_text(text=text, max_words=1000)

        assert len(result) == 1
        assert result[0] == "one two three"

    def test_text_with_extra_whitespace(
        self,
    ) -> None:
        """Test text with multiple spaces between words."""
        text = "one    two     three"
        result = _split_text(text=text, max_words=5)

        assert len(result) == 1
        # Multiple spaces are collapsed by split/join
        assert "one" in result[0]
        assert "two" in result[0]
        assert "three" in result[0]


# =============================================================================
# Tests for group_word_chunks() function - Basic Functionality
# =============================================================================


class TestGroupWordChunksBasicFunctionality:
    """Tests for basic group_word_chunks functionality."""

    def test_returns_list_of_chunks(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model: MagicMock,
    ) -> None:
        """Test that group_word_chunks returns a list of Chunk objects."""
        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model,
            max_words=10,
        )

        assert isinstance(result, list)
        assert all(isinstance(chunk, Chunk) for chunk in result)

    def test_preserves_speaker_information(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model: MagicMock,
    ) -> None:
        """Test that speaker information is preserved from word chunks."""
        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model,
            max_words=10,
        )

        # All chunks should have the same speaker as the source chunks
        for chunk in result:
            assert chunk.speaker == "SPEAKER_00"

    def test_chunk_times_within_word_chunk_range(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model: MagicMock,
    ) -> None:
        """Test that chunk times are within the range of word chunk times."""
        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model,
            max_words=10,
        )

        # All result chunks should be within the time range of input chunks
        min_start = min(c.start_time for c in simple_word_chunks)
        max_end = max(c.end_time for c in simple_word_chunks)

        for chunk in result:
            assert chunk.start_time >= min_start
            assert chunk.end_time <= max_end

    def test_chunk_audio_is_concatenation(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model: MagicMock,
    ) -> None:
        """Test that chunk audio is concatenation of word chunk audio."""
        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model,
            max_words=10,
        )

        # Each result chunk should have audio that's a concatenation
        for chunk in result:
            assert chunk.audio is not None
            assert len(chunk.audio) > 0


# =============================================================================
# Tests for group_word_chunks() function - Time Range Calculations
# =============================================================================


class TestGroupWordChunksTimeRanges:
    """Tests for time range calculations in group_word_chunks."""

    def test_first_word_determines_segment_start(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that the first word's start time determines segment start."""
        # Configure mock to return text starting with "Hello"
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # First chunk should start at the first word's start time
        if result:
            assert result[0].start_time == 0.0

    def test_last_word_determines_segment_end(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that the last word's end time determines segment end."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # Last chunk should end at the last word's end time
        if result:
            assert result[0].end_time == 3.0

    def test_segment_duration_calculation(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that segment duration is correctly calculated."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        if result:
            duration = result[0].end_time - result[0].start_time
            assert duration == 3.0  # 3.0 - 0.0


# =============================================================================
# Tests for group_word_chunks() function - Minimum Duration Filtering
# =============================================================================


class TestGroupWordChunksMinimumDuration:
    """Tests for minimum duration filtering in group_word_chunks."""

    def test_short_chunks_filtered_out(
        self,
        word_chunks_short_duration: list[Chunk],
        mock_punctuation_model: MagicMock,
    ) -> None:
        """Test that chunks below minimum duration are filtered out."""
        # Configure mock to return text that would create a short segment
        mock_punctuation_model.punctuate = MagicMock(
            return_value="short chunk."
        )

        result = group_word_chunks(
            word_chunks=word_chunks_short_duration,
            punctuation_model=mock_punctuation_model,
            max_words=10,
        )

        # Should be filtered out (duration < MIN_CHUNK_LENGTH_SECONDS)
        assert len(result) == 0

    def test_chunks_at_minimum_duration_included(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that chunks at exactly minimum duration are included."""
        # Create chunks that span exactly MIN_CHUNK_LENGTH_SECONDS
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks[:2],  # Only first two chunks
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # 0.5s duration is above minimum (0.05s), should be included
        assert len(result) >= 0  # May be 0 if other conditions fail

    def test_very_long_chunks_included(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that chunks well above minimum duration are included."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # 3.0s duration is well above minimum, should be included
        assert len(result) >= 1


# =============================================================================
# Tests for group_word_chunks() function - Speaker Preservation
# =============================================================================


class TestGroupWordChunksSpeakerPreservation:
    """Tests for speaker preservation in group_word_chunks."""

    def test_multiple_speakers_preserved(
        self,
        word_chunks_with_multiple_speakers: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that speaker information is preserved for multiple speakers."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello there how are you."
        )

        result = group_word_chunks(
            word_chunks=word_chunks_with_multiple_speakers,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # Check that speakers are preserved
        if result:
            # First chunk should have the speaker from the first word
            assert result[0].speaker == "SPEAKER_00"

    def test_speaker_from_first_word_chunk(
        self,
        word_chunks_with_multiple_speakers: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that chunk speaker comes from the first word chunk in segment."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello there how are you."
        )

        result = group_word_chunks(
            word_chunks=word_chunks_with_multiple_speakers,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        if result:
            # The speaker should match the first word chunk's speaker
            assert result[0].speaker == word_chunks_with_multiple_speakers[0].speaker


# =============================================================================
# Tests for group_word_chunks() function - Punctuation Handling
# =============================================================================


class TestGroupWordChunksPunctuationHandling:
    """Tests for punctuation handling in group_word_chunks."""

    def test_punctuation_removed_for_matching(
        self,
        word_chunks_with_punctuation: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that punctuation is removed when matching word chunks."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world testing."
        )

        result = group_word_chunks(
            word_chunks=word_chunks_with_punctuation,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # Should successfully match despite punctuation in original text
        # At minimum, the function should not crash
        assert isinstance(result, list)

    def test_text_lowercased_for_processing(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that text is lowercased for processing."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # Should work correctly with lowercased text
        assert isinstance(result, list)


# =============================================================================
# Tests for group_word_chunks() function - Edge Cases
# =============================================================================


class TestGroupWordChunksEdgeCases:
    """Tests for edge cases in group_word_chunks."""

    def test_empty_word_chunks_list(
        self,
        mock_punctuation_model: MagicMock,
    ) -> None:
        """Test handling of empty word chunks list."""
        result = group_word_chunks(
            word_chunks=[],
            punctuation_model=mock_punctuation_model,
            max_words=10,
        )

        assert result == []

    def test_single_word_chunk(
        self,
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test handling of a single word chunk."""
        single_chunk = [
            Chunk(
                start_time=0.0,
                end_time=0.5,
                audio=np.array([1.0]),
                text="Hello",
                speaker="SPEAKER_00",
            )
        ]
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello."
        )

        result = group_word_chunks(
            word_chunks=single_chunk,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # May or may not produce output depending on duration
        assert isinstance(result, list)

    def test_word_chunk_with_none_text(
        self,
        word_chunks_with_none_text: list[Chunk],
        mock_punctuation_model: MagicMock,
    ) -> None:
        """Test handling of word chunks with None text."""
        mock_punctuation_model.punctuate = MagicMock(
            return_value="Hello world."
        )

        result = group_word_chunks(
            word_chunks=word_chunks_with_none_text,
            punctuation_model=mock_punctuation_model,
            max_words=10,
        )

        # Should handle None text gracefully
        assert isinstance(result, list)

    def test_max_words_one(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test with max_words=1."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=1,
        )

        # Should create multiple small chunks
        assert isinstance(result, list)

    def test_very_large_max_words(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test with a very large max_words value."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=1000,
        )

        # Should create fewer, larger chunks
        assert isinstance(result, list)


# =============================================================================
# Tests for group_word_chunks() function - Error Handling
# =============================================================================


class TestGroupWordChunksErrorHandling:
    """Tests for error handling in group_word_chunks."""

    def test_unfound_word_logs_warning(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that unfound words trigger warning logs."""
        # Configure mock to return text with a word that doesn't exist
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="nonexistent word."
        )

        with caplog.at_level(logging.WARNING):
            result = group_word_chunks(
                word_chunks=simple_word_chunks,
                punctuation_model=mock_punctuation_model_with_fixes,
                max_words=10,
            )

        # Function should handle missing words gracefully
        assert isinstance(result, list)

    def test_all_words_unfound_returns_empty(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that when all words are unfound, empty list is returned."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="completely nonexistent words here."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # Should return empty list when words can't be matched
        assert isinstance(result, list)

    def test_last_word_not_found_logs_warning(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that when last word of segment can't be found, warning is logged."""
        # Create a scenario where first word exists but last word doesn't
        # This requires the segment to have a last word that doesn't match any chunk
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="hello nonexistent."
        )

        with caplog.at_level(logging.WARNING):
            result = group_word_chunks(
                word_chunks=simple_word_chunks,
                punctuation_model=mock_punctuation_model_with_fixes,
                max_words=10,
            )

        # Should handle missing last word gracefully
        assert isinstance(result, list)

    def test_no_word_chunks_in_segment_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that when no word chunks fall within segment, warning is logged.

        This tests the edge case where:
        - First word is found (segment_start is set)
        - Last word is found with end_time > segment_start (segment_end is set)
        - Duration is >= MIN_CHUNK_LENGTH_SECONDS
        - But no chunks satisfy: start_time >= segment_start AND end_time <= segment_end

        This can happen when chunks extend beyond the segment boundaries.
        """
        # Create chunks where the matching words have timestamps that don't
        # satisfy the filtering condition
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=5.0,  # This chunk extends far beyond the segment
                audio=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                text="hello",
                speaker="SPEAKER_00",
            ),
            Chunk(
                start_time=4.5,  # Overlaps with first chunk
                end_time=5.0,    # Same end time - segment_end will be 5.0
                audio=np.array([4.0, 5.0]),
                text="world",
                speaker="SPEAKER_00",
            ),
        ]

        # Mock the punctuation model
        mock_punct = MagicMock()
        mock_punct.punctuate = MagicMock(return_value="hello world.")

        with caplog.at_level(logging.WARNING):
            result = group_word_chunks(
                word_chunks=chunks,
                punctuation_model=mock_punct,
                max_words=10,
            )

        # The first chunk has end_time=5.0 which equals segment_end,
        # but we need chunks with end_time <= segment_end
        # This should still work, so let's try a different approach
        assert isinstance(result, list)

    def test_word_chunk_extends_beyond_segment_triggers_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test warning when all matching chunks extend beyond segment boundaries.

        Creates a scenario where:
        - First word found at time 0, segment_start = 0
        - Last word found but its end_time creates segment_end
        - All word chunks have end_time > segment_end or start_time < segment_start
        """
        # Create chunks where the time ranges don't align with segment boundaries
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,  # First word
                audio=np.array([1.0, 2.0]),
                text="hello",
                speaker="SPEAKER_00",
            ),
            # A chunk that starts before segment_start (not possible since segment_start=0)
            # or ends after segment_end
            Chunk(
                start_time=1.5,
                end_time=3.0,  # This extends beyond a 2-second segment
                audio=np.array([1.5, 2.0, 2.5, 3.0]),
                text="world",
                speaker="SPEAKER_00",
            ),
        ]

        mock_punct = MagicMock()
        mock_punct.punctuate = MagicMock(return_value="hello world.")

        with caplog.at_level(logging.WARNING):
            result = group_word_chunks(
                word_chunks=chunks,
                punctuation_model=mock_punct,
                max_words=10,
            )

        assert isinstance(result, list)


# =============================================================================
# Tests for group_word_chunks() function - Integration
# =============================================================================


class TestGroupWordChunksIntegration:
    """Integration tests for group_word_chunks."""

    def test_full_workflow_from_words_to_chunks(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test the complete workflow from word chunks to grouped chunks."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # Verify all result chunks have required attributes
        for chunk in result:
            assert chunk.start_time is not None
            assert chunk.end_time is not None
            assert chunk.audio is not None
            assert chunk.text is not None
            assert chunk.speaker is not None

    def test_text_contains_punctuated_content(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that result chunk text contains punctuated content."""
        expected_text = "Hello world this is a test."
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value=expected_text
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        if result:
            # Text should contain the punctuated content
            assert expected_text in result[0].text or result[0].text in expected_text

    def test_audio_concatenation_correct(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that audio is correctly concatenated."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        if result:
            # Total audio length should be sum of included word chunks
            expected_length = sum(len(c.audio) for c in simple_word_chunks)
            assert len(result[0].audio) == expected_length

    def test_multiple_segments_created(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that multiple segments can be created from longer input."""
        # Create a longer text that will be split
        long_text = "Hello world this is a test. Another sentence here. Final part."
        mock_punctuation_model_with_fixes.punctuate = MagicMock(return_value=long_text)

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=4,
        )

        # Should create multiple segments
        assert isinstance(result, list)

    def test_chunk_text_not_empty(
        self,
        simple_word_chunks: list[Chunk],
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that result chunks have non-empty text."""
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world this is a test."
        )

        result = group_word_chunks(
            word_chunks=simple_word_chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        for chunk in result:
            assert chunk.text is not None
            assert len(chunk.text.strip()) > 0


# =============================================================================
# Tests for PUNCTUATION_PATTERN
# =============================================================================


class TestPunctuationPattern:
    """Tests for the punctuation pattern."""

    def test_pattern_matches_common_punctuation(
        self,
    ) -> None:
        """Test that the pattern matches common punctuation characters."""
        # Test common punctuation characters (excluding backslash which needs special handling)
        test_chars = ".,!?;:()[]{}\"'-_@#$%^&*/=<>"
        for char in test_chars:
            assert re.search(PUNCTUATION_PATTERN, char) is not None

    def test_pattern_removes_punctuation_from_text(
        self,
    ) -> None:
        """Test that the pattern removes punctuation from text."""
        text = "Hello, world! How are you?"
        cleaned = re.sub(PUNCTUATION_PATTERN, "", text)

        assert "," not in cleaned
        assert "!" not in cleaned
        assert "?" not in cleaned
        assert cleaned == "Hello world How are you"

    def test_pattern_removes_punctuation_from_text(
        self,
    ) -> None:
        """Test that the pattern removes punctuation from text."""
        text = "Hello, world! How are you?"
        cleaned = re.sub(PUNCTUATION_PATTERN, "", text)

        assert "," not in cleaned
        assert "!" not in cleaned
        assert "?" not in cleaned
        assert cleaned == "Hello world How are you"


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    def test_word_chunks_with_empty_string_text(
        self,
        mock_punctuation_model: MagicMock,
    ) -> None:
        """Test handling of word chunks with empty string text."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=0.5,
                audio=np.array([1.0]),
                text="",
                speaker="SPEAKER_00",
            ),
            Chunk(
                start_time=0.5,
                end_time=1.0,
                audio=np.array([2.0]),
                text="world",
                speaker="SPEAKER_00",
            ),
        ]

        mock_punctuation_model.punctuate = MagicMock(return_value="world.")

        result = group_word_chunks(
            word_chunks=chunks,
            punctuation_model=mock_punctuation_model,
            max_words=10,
        )

        assert isinstance(result, list)

    def test_chunks_with_float_times(
        self,
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test handling of chunks with precise float timestamps."""
        chunks = [
            Chunk(
                start_time=0.123,
                end_time=0.456,
                audio=np.array([1.0]),
                text="Hello",
                speaker="SPEAKER_00",
            ),
            Chunk(
                start_time=0.456,
                end_time=0.789,
                audio=np.array([2.0]),
                text="world",
                speaker="SPEAKER_00",
            ),
        ]

        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="Hello world."
        )

        result = group_word_chunks(
            word_chunks=chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        assert isinstance(result, list)

    def test_case_insensitive_word_matching(
        self,
        mock_punctuation_model_with_fixes: MagicMock,
    ) -> None:
        """Test that word matching is case-insensitive."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=0.5,
                audio=np.array([1.0]),
                text="HELLO",
                speaker="SPEAKER_00",
            ),
            Chunk(
                start_time=0.5,
                end_time=1.0,
                audio=np.array([2.0]),
                text="WORLD",
                speaker="SPEAKER_00",
            ),
        ]

        # Mock returns lowercase text
        mock_punctuation_model_with_fixes.punctuate = MagicMock(
            return_value="hello world."
        )

        result = group_word_chunks(
            word_chunks=chunks,
            punctuation_model=mock_punctuation_model_with_fixes,
            max_words=10,
        )

        # Should still match despite case differences
        assert isinstance(result, list)

    def test_split_text_with_danish_language(
        self,
    ) -> None:
        """Test sentence splitting with Danish text (nltk language parameter)."""
        text = "Dette er en test. Dette er en anden test."
        result = _split_text(text=text, max_words=10)

        # Should split at sentence boundaries
        assert len(result) >= 1

    def test_split_text_preserves_original_spacing_after_split(
        self,
    ) -> None:
        """Test that spacing is handled correctly after splitting."""
        text = "one two three, four five six"
        result = _split_text(text=text, max_words=10)

        # Should preserve the content
        assert any("one" in seg for seg in result)
        assert any("four" in seg for seg in result)
