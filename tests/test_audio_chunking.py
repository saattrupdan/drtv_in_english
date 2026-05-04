"""Tests for the audio_chunking module."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from but_with_subs.audio_chunking import chunk_by_audio
from but_with_subs.constants import MIN_CHUNK_LENGTH_SECONDS
from but_with_subs.data_models import Chunk

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_speech_timestamps_single_speaker():
    """Create mock speech timestamps for a single speaker."""
    # Format: (Turn, speaker_id) where Turn has start and end attributes
    mock_turn1 = MagicMock()
    mock_turn1.start = 0.5
    mock_turn1.end = 2.5

    mock_turn2 = MagicMock()
    mock_turn2.start = 3.0
    mock_turn2.end = 5.0

    return [(mock_turn1, "SPEAKER_00"), (mock_turn2, "SPEAKER_00")]


@pytest.fixture
def mock_speech_timestamps_multiple_speakers():
    """Create mock speech timestamps for multiple speakers."""
    mock_turn1 = MagicMock()
    mock_turn1.start = 0.5
    mock_turn1.end = 1.5

    mock_turn2 = MagicMock()
    mock_turn2.start = 1.5
    mock_turn2.end = 2.5

    mock_turn3 = MagicMock()
    mock_turn3.start = 2.5
    mock_turn3.end = 4.0

    mock_turn4 = MagicMock()
    mock_turn4.start = 4.5
    mock_turn4.end = 6.0

    return [
        (mock_turn1, "SPEAKER_00"),
        (mock_turn2, "SPEAKER_01"),
        (mock_turn3, "SPEAKER_00"),
        (mock_turn4, "SPEAKER_01"),
    ]


@pytest.fixture
def mock_speech_timestamps_short_segments():
    """Create mock speech timestamps with segments below minimum duration."""
    mock_turn1 = MagicMock()
    mock_turn1.start = 0.0
    mock_turn1.end = 0.03  # Below MIN_CHUNK_LENGTH_SECONDS (0.05)

    mock_turn2 = MagicMock()
    mock_turn2.start = 0.1
    mock_turn2.end = 0.12  # Below minimum

    mock_turn3 = MagicMock()
    mock_turn3.start = 0.5
    mock_turn3.end = 1.0  # Above minimum

    return [
        (mock_turn1, "SPEAKER_00"),
        (mock_turn2, "SPEAKER_00"),
        (mock_turn3, "SPEAKER_00"),
    ]


@pytest.fixture
def mock_audio_10_seconds():
    """Create a 10-second audio array at 16kHz."""
    duration = 10.0
    sample_rate = 16_000
    n_samples = int(duration * sample_rate)
    # Create a simple sine wave
    t = np.linspace(0, duration, n_samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


@pytest.fixture
def mock_audio_5_seconds():
    """Create a 5-second audio array at 16kHz."""
    duration = 5.0
    sample_rate = 16_000
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


@pytest.fixture
def mock_audio_silent():
    """Create a silent audio array."""
    sample_rate = 16_000
    n_samples = int(5.0 * sample_rate)
    return np.zeros(n_samples, dtype=np.float32)


@pytest.fixture
def mock_audio_very_short():
    """Create a very short audio array (below minimum chunk length)."""
    sample_rate = 16_000
    n_samples = int(0.02 * sample_rate)  # 20ms
    return np.sin(2 * np.pi * 440 * np.linspace(0, 0.02, n_samples)).astype(np.float32)


# =============================================================================
# Tests for chunk_by_audio() function - Basic Functionality
# =============================================================================


class TestChunkByAudioBasicFunctionality:
    """Tests for basic chunking functionality."""

    def test_chunks_returned_as_list_of_chunk_objects(
        self,
        mock_audio_10_seconds: np.ndarray,
        mock_speech_timestamps_single_speaker: list,
    ) -> None:
        """Test that chunk_by_audio returns a list of Chunk objects."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_speech_timestamps_single_speaker
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert isinstance(chunks, list)
            assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_chunk_count_matches_valid_speech_segments(
        self,
        mock_audio_10_seconds: np.ndarray,
        mock_speech_timestamps_single_speaker: list,
    ) -> None:
        """Test that chunk count matches the number of valid speech segments."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_speech_timestamps_single_speaker
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # Both segments are above minimum duration
            assert len(chunks) == 2

    def test_chunk_start_and_end_times_match_speech_timestamps(
        self,
        mock_audio_10_seconds: np.ndarray,
        mock_speech_timestamps_single_speaker: list,
    ) -> None:
        """Test that chunk times match the speech timestamps from pipeline."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_speech_timestamps_single_speaker
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert chunks[0].start_time == 0.5
            assert chunks[0].end_time == 2.5
            assert chunks[1].start_time == 3.0
            assert chunks[1].end_time == 5.0

    def test_pipeline_called_with_correct_waveform_and_sample_rate(
        self,
        mock_audio_10_seconds: np.ndarray,
        mock_speech_timestamps_single_speaker: list,
    ) -> None:
        """Test that pipeline is called with correct waveform and sample rate."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_speech_timestamps_single_speaker
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunk_by_audio(audio=mock_audio_10_seconds)

            # Verify the pipeline was called
            assert mock_pipeline_instance.call_count > 0
            call_args = mock_pipeline_instance.call_args

            # Check that waveform was passed correctly
            waveform = call_args[0][0]["waveform"]
            assert isinstance(waveform, torch.Tensor)
            assert waveform.shape[0] == 1  # Batch dimension
            assert waveform.shape[1] == mock_audio_10_seconds.shape[0]

            # Check sample rate
            assert call_args[0][0]["sample_rate"] == 16_000


# =============================================================================
# Tests for chunk_by_audio() function - Multiple Speakers
# =============================================================================


class TestChunkByAudioMultipleSpeakers:
    """Tests for handling multiple speakers."""

    def test_speaker_assignment_from_pipeline(
        self,
        mock_audio_10_seconds: np.ndarray,
        mock_speech_timestamps_multiple_speakers: list,
    ) -> None:
        """Test that speakers are correctly assigned from pipeline output."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_speech_timestamps_multiple_speakers
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert chunks[0].speaker == "SPEAKER_00"
            assert chunks[1].speaker == "SPEAKER_01"
            assert chunks[2].speaker == "SPEAKER_00"
            assert chunks[3].speaker == "SPEAKER_01"

    def test_multiple_speakers_create_separate_chunks(
        self,
        mock_audio_10_seconds: np.ndarray,
        mock_speech_timestamps_multiple_speakers: list,
    ) -> None:
        """Test that different speakers create separate chunks even if adjacent."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_speech_timestamps_multiple_speakers
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # All 4 turns should be separate chunks (all above minimum duration)
            assert len(chunks) == 4

    def test_overlapping_speakers_handled_sequentially(
        self,
        mock_audio_10_seconds: np.ndarray,
        mock_speech_timestamps_multiple_speakers: list,
    ) -> None:
        """Test handling of speakers with overlapping/adjacent speech."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_speech_timestamps_multiple_speakers
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # Check that adjacent segments are handled correctly
            # Turn 1 ends at 1.5, Turn 2 starts at 1.5 (adjacent)
            assert chunks[0].end_time == 1.5
            assert chunks[1].start_time == 1.5

            # Turn 2 ends at 2.5, Turn 3 starts at 2.5 (adjacent)
            assert chunks[1].end_time == 2.5
            assert chunks[2].start_time == 2.5


# =============================================================================
# Tests for chunk_by_audio() function - Minimum Duration Filtering
# =============================================================================


class TestChunkByAudioMinimumDuration:
    """Tests for minimum duration filtering."""

    def test_segments_below_minimum_duration_filtered_out(
        self,
        mock_audio_10_seconds: np.ndarray,
        mock_speech_timestamps_short_segments: list,
    ) -> None:
        """Test that segments below minimum duration are filtered out."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_speech_timestamps_short_segments
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # Only the third segment (0.5-1.0) should be included
            assert len(chunks) == 1
            assert chunks[0].start_time == 0.5
            assert chunks[0].end_time == 1.0

    def test_segment_at_minimum_duration_included(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test that segments at exactly minimum duration are included."""
        mock_turn = MagicMock()
        mock_turn.start = 0.0
        mock_turn.end = MIN_CHUNK_LENGTH_SECONDS  # Exactly at minimum

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # Should be included (at minimum, not below)
            assert len(chunks) == 1

    def test_segment_just_above_minimum_duration_included(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test that segments just above minimum duration are included."""
        mock_turn = MagicMock()
        mock_turn.start = 0.0
        mock_turn.end = MIN_CHUNK_LENGTH_SECONDS + 0.01  # Just above minimum

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert len(chunks) == 1

    def test_segment_just_below_minimum_duration_excluded(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test that segments just below minimum duration are excluded."""
        mock_turn = MagicMock()
        mock_turn.start = 0.0
        mock_turn.end = MIN_CHUNK_LENGTH_SECONDS - 0.01  # Just below minimum

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert len(chunks) == 0


# =============================================================================
# Tests for chunk_by_audio() function - Audio Extraction Accuracy
# =============================================================================


class TestChunkByAudioExtractionAccuracy:
    """Tests for audio extraction accuracy."""

    def test_chunk_audio_correctly_extracted(
        self, mock_audio_5_seconds: np.ndarray
    ) -> None:
        """Test that audio is correctly extracted for each chunk."""
        mock_turn = MagicMock()
        mock_turn.start = 1.0
        mock_turn.end = 3.0

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_5_seconds)

            # Extracted audio should be from 1.0 to 3.0 seconds
            expected_start = int(1.0 * 16_000)
            expected_end = int(3.0 * 16_000)
            expected_audio = mock_audio_5_seconds[expected_start:expected_end]

            assert len(chunks) == 1
            np.testing.assert_array_equal(chunks[0].audio, expected_audio)

    def test_chunk_audio_length_matches_duration(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test that extracted audio length matches the chunk duration."""
        mock_turn = MagicMock()
        mock_turn.start = 2.0
        mock_turn.end = 5.5

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            duration = 5.5 - 2.0  # 3.5 seconds
            expected_length = int(duration * 16_000)

            assert len(chunks[0].audio) == expected_length

    def test_chunk_audio_dtype_preserved(
        self, mock_audio_5_seconds: np.ndarray
    ) -> None:
        """Test that audio dtype is preserved in chunks."""
        mock_turn = MagicMock()
        mock_turn.start = 0.0
        mock_turn.end = 2.0

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_5_seconds)

            assert chunks[0].audio.dtype == mock_audio_5_seconds.dtype


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestChunkByAudioEdgeCases:
    """Tests for edge cases."""

    def test_silent_audio_returns_empty_chunks(
        self, mock_audio_silent: np.ndarray
    ) -> None:
        """Test that silent audio with no speech returns empty list."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            # No speech detected
            mock_result.speaker_diarization = []
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_silent)

            assert chunks == []

    def test_very_short_audio_returns_empty_chunks(
        self, mock_audio_very_short: np.ndarray
    ) -> None:
        """Test that very short audio returns empty list."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            # Speech detected but below minimum duration
            mock_turn = MagicMock()
            mock_turn.start = 0.0
            mock_turn.end = 0.02
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_very_short)

            assert chunks == []

    def test_continuous_speech_creates_single_chunk(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test that continuous speech without pauses creates a single chunk."""
        mock_turn = MagicMock()
        mock_turn.start = 0.0
        mock_turn.end = 10.0

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert len(chunks) == 1
            assert chunks[0].start_time == 0.0
            assert chunks[0].end_time == 10.0

    def test_empty_speech_timestamps_returns_empty_list(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test that empty speech timestamps returns empty chunk list."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = []
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert chunks == []

    def test_all_segments_filtered_returns_empty_list(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test that when all segments are filtered, empty list is returned."""
        # Create multiple short segments all below minimum
        mock_turns = []
        for i in range(5):
            mock_turn = MagicMock()
            mock_turn.start = i * 0.02
            mock_turn.end = i * 0.02 + 0.01  # All 10ms segments
            mock_turns.append((mock_turn, "SPEAKER_00"))

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_turns
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert chunks == []


# =============================================================================
# Tests for Time Range Calculations
# =============================================================================


class TestChunkByAudioTimeRanges:
    """Tests for time range calculations."""

    def test_chunk_duration_calculation(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test that chunk duration is correctly calculated."""
        mock_turn = MagicMock()
        mock_turn.start = 1.5
        mock_turn.end = 4.75

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            expected_duration = 4.75 - 1.5
            actual_duration = chunks[0].end_time - chunks[0].start_time
            assert actual_duration == expected_duration

    def test_non_overlapping_chunks(self, mock_audio_10_seconds: np.ndarray) -> None:
        """Test that chunks do not overlap in time."""
        mock_turns = [
            (MagicMock(start=0.0, end=2.0), "SPEAKER_00"),
            (MagicMock(start=2.5, end=4.5), "SPEAKER_00"),
            (MagicMock(start=5.0, end=7.0), "SPEAKER_00"),
        ]

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_turns
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # Verify no overlap
            for i in range(len(chunks) - 1):
                assert chunks[i].end_time <= chunks[i + 1].start_time

    def test_adjacent_chunks_handled_correctly(
        self,
        mock_audio_10_seconds: np.ndarray,
        mock_speech_timestamps_multiple_speakers: list,
    ) -> None:
        """Test that adjacent chunks (end of one = start of next) are handled."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_speech_timestamps_multiple_speakers
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # Turn 1 ends at 1.5, Turn 2 starts at 1.5
            assert chunks[0].end_time == chunks[1].start_time
            # Turn 2 ends at 2.5, Turn 3 starts at 2.5
            assert chunks[1].end_time == chunks[2].start_time


# =============================================================================
# Integration Tests
# =============================================================================


class TestChunkByAudioIntegration:
    """Integration tests for the audio chunking module."""

    def test_full_workflow_from_audio_to_chunks(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test the complete workflow from audio input to chunk output."""
        # Create realistic speech timestamps
        mock_turns = [
            (MagicMock(start=0.5, end=2.0), "SPEAKER_00"),
            (MagicMock(start=2.5, end=4.0), "SPEAKER_01"),
            (MagicMock(start=4.5, end=6.5), "SPEAKER_00"),
            (MagicMock(start=7.0, end=9.0), "SPEAKER_01"),
        ]

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_turns
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            # Run the chunking
            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # Verify results
            assert len(chunks) == 4
            assert all(isinstance(chunk, Chunk) for chunk in chunks)

            # Verify each chunk has required attributes
            for chunk in chunks:
                assert chunk.start_time is not None
                assert chunk.end_time is not None
                assert chunk.audio is not None
                assert chunk.speaker is not None
                assert chunk.text is None  # Not set in chunk_by_audio

    def test_multiple_speakers_scenario(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test processing with multiple speakers throughout the audio."""
        mock_turns = [
            (MagicMock(start=0.0, end=1.0), "SPEAKER_00"),
            (MagicMock(start=1.0, end=2.0), "SPEAKER_01"),
            (MagicMock(start=2.0, end=3.0), "SPEAKER_02"),
            (MagicMock(start=3.0, end=4.0), "SPEAKER_00"),
            (MagicMock(start=4.0, end=5.0), "SPEAKER_01"),
            (MagicMock(start=5.0, end=6.0), "SPEAKER_02"),
        ]

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_turns
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert len(chunks) == 6
            speakers = [chunk.speaker for chunk in chunks]
            assert speakers == [
                "SPEAKER_00",
                "SPEAKER_01",
                "SPEAKER_02",
                "SPEAKER_00",
                "SPEAKER_01",
                "SPEAKER_02",
            ]

    def test_long_audio_processing(self, mock_audio_10_seconds: np.ndarray) -> None:
        """Test processing of longer audio with many segments."""
        # Simulate a long audio with many speech segments
        mock_turns = []
        for i in range(20):
            start = i * 0.5
            end = start + 0.4  # 400ms segments with 100ms gaps
            mock_turn = MagicMock()
            mock_turn.start = start
            mock_turn.end = end
            mock_turns.append((mock_turn, f"SPEAKER_{i % 3:02d}"))

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_turns
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # All segments are above minimum duration (0.4s > 0.05s)
            assert len(chunks) == 20

    def test_mixed_duration_segments(self, mock_audio_10_seconds: np.ndarray) -> None:
        """Test processing with mixed duration segments (some above, some below threshold)."""
        mock_turns = [
            (MagicMock(start=0.0, end=0.03), "SPEAKER_00"),  # Below minimum
            (MagicMock(start=0.5, end=1.5), "SPEAKER_00"),  # Above minimum
            (MagicMock(start=2.0, end=2.04), "SPEAKER_01"),  # Below minimum
            (MagicMock(start=3.0, end=5.0), "SPEAKER_00"),  # Above minimum
            (MagicMock(start=6.0, end=6.02), "SPEAKER_01"),  # Below minimum
            (MagicMock(start=7.0, end=9.0), "SPEAKER_00"),  # Above minimum
        ]

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_turns
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            # Only 3 segments should be included
            assert len(chunks) == 3
            assert chunks[0].start_time == 0.5
            assert chunks[1].start_time == 3.0
            assert chunks[2].start_time == 7.0


# =============================================================================
# Mocking Tests - Pipeline Integration
# =============================================================================


class TestPipelineMocking:
    """Tests verifying proper mocking of pyannote pipeline."""

    def test_pipeline_initialized_with_correct_model(
        self, mock_audio_10_seconds: np.ndarray
    ) -> None:
        """Test that pipeline is initialized with correct model."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = []
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunk_by_audio(audio=mock_audio_10_seconds)

            # Verify from_pretrained was called
            mock_pipeline_class.from_pretrained.assert_called_once()
            call_args = mock_pipeline_class.from_pretrained.call_args
            # Check model name contains expected identifier
            assert "pyannote" in call_args[0][0]
            assert "speaker-diarization" in call_args[0][0]

    def test_pipeline_to_device_called(self, mock_audio_10_seconds: np.ndarray) -> None:
        """Test that pipeline.to() is called for device placement."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = []
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            chunk_by_audio(audio=mock_audio_10_seconds)

            # Verify to() was called on the pipeline
            mock_pipeline_instance.to.assert_called_once()

    def test_progress_hook_used(self, mock_audio_10_seconds: np.ndarray) -> None:
        """Test that ProgressHook is used during pipeline execution."""
        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            with patch(
                "but_with_subs.audio_chunking.ProgressHook"
            ) as mock_progress_hook:
                mock_pipeline_instance = MagicMock()
                mock_result = MagicMock()
                mock_result.speaker_diarization = []
                mock_pipeline_instance.return_value = mock_result

                mock_pipeline_class.from_pretrained.return_value = (
                    mock_pipeline_instance
                )

                chunk_by_audio(audio=mock_audio_10_seconds)

                # Verify ProgressHook was used
                mock_progress_hook.assert_called()


# =============================================================================
# Logging Tests
# =============================================================================


class TestChunkByAudioLogging:
    """Tests for logging in chunk_by_audio."""

    def test_logging_on_successful_chunking(
        self, mock_audio_10_seconds: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that successful chunking is logged."""
        mock_turn = MagicMock()
        mock_turn.start = 0.0
        mock_turn.end = 5.0

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = [(mock_turn, "SPEAKER_00")]
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            with caplog.at_level(logging.INFO):
                chunks = chunk_by_audio(audio=mock_audio_10_seconds)

            assert len(chunks) == 1
            assert any(
                "Split audio into" in record.message for record in caplog.records
            )

    def test_logging_shows_chunk_count(
        self, mock_audio_10_seconds: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that log message includes the number of chunks."""
        mock_turns = [
            (MagicMock(start=0.0, end=2.0), "SPEAKER_00"),
            (MagicMock(start=2.5, end=4.5), "SPEAKER_01"),
        ]

        with patch("but_with_subs.audio_chunking.Pipeline") as mock_pipeline_class:
            mock_pipeline_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.speaker_diarization = mock_turns
            mock_pipeline_instance.return_value = mock_result

            mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance

            with caplog.at_level(logging.INFO):
                chunk_by_audio(audio=mock_audio_10_seconds)

            assert any(
                "Split audio into 2 chunks" in record.message
                for record in caplog.records
            )
