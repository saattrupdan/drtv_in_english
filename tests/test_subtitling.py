"""Comprehensive tests for the subtitling module.

This module contains unit tests, integration tests, and edge case tests
for the subtitling functionality including:
- Timestamp formatting
- Text escaping for WebVTT
- Overlapping speaker detection
- Speaker color assignment
- Full subtitle generation workflow
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from but_with_subs.constants import OVERLAPPING_SPEAKER_COLORS
from but_with_subs.data_models import Chunk
from but_with_subs.subtitling import (
    _apply_speaker_color,
    _assign_speaker_colors,
    _detect_overlapping_speakers,
    _escape_vtt_text,
    generate_subtitles,
)
from but_with_subs.vtt import format_vtt_timestamp

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing.

    Returns:
        List of sample Chunk objects for testing.
    """
    audio_data = np.array([0.1, 0.2, 0.3])
    return [
        Chunk(
            start_time=0.0,
            end_time=2.0,
            audio=audio_data,
            text="Hello world",
            speaker="Alice",
        ),
        Chunk(
            start_time=2.5,
            end_time=4.5,
            audio=audio_data,
            text="How are you?",
            speaker="Bob",
        ),
    ]


@pytest.fixture
def overlapping_chunks() -> list[Chunk]:
    """Create chunks with overlapping speakers.

    Returns:
        List of Chunk objects with overlapping time ranges.
    """
    audio_data = np.array([0.1, 0.2, 0.3])
    return [
        Chunk(
            start_time=0.0,
            end_time=3.0,
            audio=audio_data,
            text="First speaker talking",
            speaker="Alice",
        ),
        Chunk(
            start_time=2.0,
            end_time=5.0,
            audio=audio_data,
            text="Second speaker interrupts",
            speaker="Bob",
        ),
        Chunk(
            start_time=4.5,
            end_time=7.0,
            audio=audio_data,
            text="Third speaker joins",
            speaker="Charlie",
        ),
    ]


@pytest.fixture
def temp_audio_file() -> Path:
    """Create a temporary audio file for testing.

    Yields:
        Path to the temporary audio file.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio data")
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


# =============================================================================
# Unit Tests: format_vtt_timestamp
# =============================================================================


class TestFormatVttTimestamp:
    """Unit tests for format_vtt_timestamp function."""

    def test_zero_seconds(self) -> None:
        """Test formatting zero seconds."""
        result = format_vtt_timestamp(0.0)
        assert result == "00:00:00.000"

    def test_fractional_seconds(self) -> None:
        """Test formatting fractional seconds with milliseconds."""
        result = format_vtt_timestamp(0.123)
        assert result == "00:00:00.123"

    def test_one_second(self) -> None:
        """Test formatting exactly one second."""
        result = format_vtt_timestamp(1.0)
        assert result == "00:00:01.000"

    def test_minutes_and_seconds(self) -> None:
        """Test formatting minutes and seconds."""
        result = format_vtt_timestamp(125.0)
        assert result == "00:02:05.000"

    def test_hours_minutes_seconds(self) -> None:
        """Test formatting hours, minutes, and seconds."""
        result = format_vtt_timestamp(3661.0)
        assert result == "01:01:01.000"

    def test_large_seconds(self) -> None:
        """Test formatting large number of seconds."""
        result = format_vtt_timestamp(7265.5)
        assert result == "02:01:05.500"

    def test_rounding_milliseconds(self) -> None:
        """Test that milliseconds are properly rounded."""
        result = format_vtt_timestamp(1.9999)
        assert result == "00:00:02.000"

    def test_999_milliseconds(self) -> None:
        """Test 999 milliseconds formatting."""
        result = format_vtt_timestamp(0.999)
        assert result == "00:00:00.999"


# =============================================================================
# Unit Tests: _escape_vtt_text
# =============================================================================


class TestEscapeVttText:
    """Unit tests for _escape_vtt_text function."""

    def test_plain_text(self) -> None:
        """Test that plain text passes through unchanged."""
        result = _escape_vtt_text("Hello world")
        assert result == "Hello world"

    def test_ampersand(self) -> None:
        """Test escaping ampersand character."""
        result = _escape_vtt_text("A & B")
        assert result == "A &amp; B"

    def test_less_than(self) -> None:
        """Test escaping less-than character."""
        result = _escape_vtt_text("5 < 10")
        assert result == "5 &lt; 10"

    def test_greater_than(self) -> None:
        """Test escaping greater-than character."""
        result = _escape_vtt_text("10 > 5")
        assert result == "10 &gt; 5"

    def test_multiple_special_chars(self) -> None:
        """Test escaping multiple special characters."""
        result = _escape_vtt_text("A < B & C > D")
        assert result == "A &lt; B &amp; C &gt; D"

    def test_escaped_ampersand_first(self) -> None:
        """Test that ampersand is escaped first to prevent double-escaping."""
        result = _escape_vtt_text("&amp;")
        assert result == "&amp;amp;"

    def test_empty_string(self) -> None:
        """Test escaping empty string."""
        result = _escape_vtt_text("")
        assert result == ""

    def test_only_special_chars(self) -> None:
        """Test string with only special characters."""
        result = _escape_vtt_text("<>&")
        assert result == "&lt;&gt;&amp;"


# =============================================================================
# Unit Tests: _detect_overlapping_speakers
# =============================================================================


class TestDetectOverlappingSpeakers:
    """Unit tests for _detect_overlapping_speakers function."""

    def test_no_chunks(self) -> None:
        """Test detection with no chunks."""
        result = _detect_overlapping_speakers([])
        assert result == {}

    def test_single_chunk(self) -> None:
        """Test detection with single chunk."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=np.array([1]),
                text="test",
                speaker="Alice",
            )
        ]
        result = _detect_overlapping_speakers(chunks)
        assert result == {}

    def test_no_overlap(self) -> None:
        """Test detection when speakers do not overlap."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=np.array([1]),
                text="test",
                speaker="Alice",
            ),
            Chunk(
                start_time=3.0,
                end_time=5.0,
                audio=np.array([1]),
                text="test",
                speaker="Bob",
            ),
        ]
        result = _detect_overlapping_speakers(chunks)
        assert result == {}

    def test_simple_overlap(self) -> None:
        """Test detection of simple overlap between two speakers."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=3.0,
                audio=np.array([1]),
                text="test",
                speaker="Alice",
            ),
            Chunk(
                start_time=2.0,
                end_time=5.0,
                audio=np.array([1]),
                text="test",
                speaker="Bob",
            ),
        ]
        result = _detect_overlapping_speakers(chunks)
        assert "Alice" in result
        assert "Bob" in result

    def test_three_way_overlap(self) -> None:
        """Test detection of three-way speaker overlap."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=4.0,
                audio=np.array([1]),
                text="test",
                speaker="Alice",
            ),
            Chunk(
                start_time=2.0,
                end_time=5.0,
                audio=np.array([1]),
                text="test",
                speaker="Bob",
            ),
            Chunk(
                start_time=3.0,
                end_time=6.0,
                audio=np.array([1]),
                text="test",
                speaker="Charlie",
            ),
        ]
        result = _detect_overlapping_speakers(chunks)
        assert "Alice" in result
        assert "Bob" in result
        assert "Charlie" in result

    def test_chunks_without_speaker(self) -> None:
        """Test that chunks without speaker are ignored."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=np.array([1]),
                text="test",
                speaker=None,
            ),
            Chunk(
                start_time=1.5,
                end_time=3.0,
                audio=np.array([1]),
                text="test",
                speaker="Alice",
            ),
        ]
        result = _detect_overlapping_speakers(chunks)
        assert result == {}

    def test_touching_not_overlapping(self) -> None:
        """Test that touching segments (end == start) are not considered overlapping."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=np.array([1]),
                text="test",
                speaker="Alice",
            ),
            Chunk(
                start_time=2.0,
                end_time=4.0,
                audio=np.array([1]),
                text="test",
                speaker="Bob",
            ),
        ]
        result = _detect_overlapping_speakers(chunks)
        assert result == {}


# =============================================================================
# Unit Tests: _assign_speaker_colors
# =============================================================================


class TestAssignSpeakerColors:
    """Unit tests for _assign_speaker_colors function."""

    def test_empty_input(self) -> None:
        """Test with empty overlapping speakers dict."""
        result = _assign_speaker_colors({})
        assert result == {}

    def test_single_speaker(self) -> None:
        """Test that single speaker gets no color."""
        overlapping = {"Alice": [(0.0, 2.0)]}
        result = _assign_speaker_colors(overlapping)
        assert result == {}

    def test_two_speakers(self) -> None:
        """Test that second speaker gets first color."""
        overlapping = {"Alice": [(0.0, 2.0)], "Bob": [(1.0, 3.0)]}
        result = _assign_speaker_colors(overlapping)
        assert "Alice" not in result
        assert "Bob" in result
        assert result["Bob"] == OVERLAPPING_SPEAKER_COLORS[0]

    def test_multiple_speakers(self) -> None:
        """Test color cycling for multiple speakers."""
        overlapping = {
            "Alice": [(0.0, 2.0)],
            "Bob": [(1.0, 3.0)],
            "Charlie": [(2.0, 4.0)],
            "Diana": [(3.0, 5.0)],
        }
        result = _assign_speaker_colors(overlapping)
        assert "Alice" not in result
        assert result["Bob"] == OVERLAPPING_SPEAKER_COLORS[0]
        assert result["Charlie"] == OVERLAPPING_SPEAKER_COLORS[1]
        assert result["Diana"] == OVERLAPPING_SPEAKER_COLORS[2]

    def test_color_cycling(self) -> None:
        """Test that colors cycle when exceeding palette size."""
        overlapping = {
            "Alice": [(0.0, 1.0)],
            "Bob": [(0.1, 1.1)],
            "Charlie": [(0.2, 1.2)],
            "Diana": [(0.3, 1.3)],
            "Eve": [(0.4, 1.4)],
        }
        result = _assign_speaker_colors(overlapping)
        assert result["Eve"] == OVERLAPPING_SPEAKER_COLORS[3]

    def test_sorted_by_first_overlap(self) -> None:
        """Test that speakers are sorted by first overlap time."""
        overlapping = {
            "Charlie": [(3.0, 4.0)],
            "Alice": [(0.0, 1.0)],
            "Bob": [(1.5, 2.5)],
        }
        result = _assign_speaker_colors(overlapping)
        assert "Alice" not in result
        assert result["Bob"] == OVERLAPPING_SPEAKER_COLORS[0]
        assert result["Charlie"] == OVERLAPPING_SPEAKER_COLORS[1]


# =============================================================================
# Unit Tests: _apply_speaker_color
# =============================================================================


class TestApplySpeakerColor:
    """Unit tests for _apply_speaker_color function."""

    def test_basic_color_application(self) -> None:
        """Test basic color wrapping."""
        result = _apply_speaker_color("Hello", "#FF0000")
        assert result == "<c.ff0000>Hello</c>"

    def test_preserves_html_entities(self) -> None:
        """Test that already escaped text is preserved."""
        result = _apply_speaker_color("&lt;tag&gt;", "#00FF00")
        assert result == "<c.00ff00>&lt;tag&gt;</c>"

    def test_multiline_text(self) -> None:
        """Test color application to multiline text."""
        result = _apply_speaker_color("Line1\nLine2", "#0000FF")
        assert result == "<c.0000ff>Line1\nLine2</c>"


# =============================================================================
# Integration Tests: generate_subtitles
# =============================================================================


class TestGenerateSubtitles:
    """Integration tests for generate_subtitles function."""

    def test_generate_subtitles_creates_file(
        self, sample_chunks: Chunk, temp_audio_file: Path
    ) -> None:
        """Test that subtitle file is created."""
        result_path = generate_subtitles(sample_chunks, temp_audio_file)
        assert result_path.exists()
        assert result_path.suffix == ".vtt"

    def test_generate_subtitles_output_location(
        self, sample_chunks: Chunk, temp_audio_file: Path
    ) -> None:
        """Test that output file is in the same directory as audio."""
        result_path = generate_subtitles(sample_chunks, temp_audio_file)
        expected_path = temp_audio_file.with_suffix(".vtt")
        assert result_path == expected_path

    def test_generate_subtitles_content(
        self, sample_chunks: Chunk, temp_audio_file: Path
    ) -> None:
        """Test that generated file contains correct WebVTT content."""
        result_path = generate_subtitles(sample_chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")

        assert "1" in content
        assert "00:00:00.000 --> 00:00:02.000" in content
        assert "Hello world" in content
        assert "(Alice)" in content

    def test_generate_subtitles_uses_min_duration(self, temp_audio_file: Path) -> None:
        """Chunks shorter than MIN_CHUNK_DISPLAY_LENGTH_SECONDS are extended."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=0.1,
                audio=np.array([1]),
                text="Short",
                speaker="Alice",
            )
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")
        assert (
            "00:00:00.000 --> 00:00:00.500" in content
            or "00:00:00.000 --> 00:00:00.600" in content
        )

    def test_generate_subtitles_sorts_by_time(self, temp_audio_file: Path) -> None:
        """Test that chunks are sorted by start time in output."""
        chunks = [
            Chunk(
                start_time=5.0,
                end_time=7.0,
                audio=np.array([1]),
                text="Second",
                speaker="Bob",
            ),
            Chunk(
                start_time=1.0,
                end_time=3.0,
                audio=np.array([1]),
                text="First",
                speaker="Alice",
            ),
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")

        first_pos = content.find("First")
        second_pos = content.find("Second")
        assert first_pos < second_pos

    def test_generate_subtitles_escapes_special_chars(
        self, temp_audio_file: Path
    ) -> None:
        """Test that special characters are properly escaped."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=np.array([1]),
                text="A < B & C > D",
                speaker="Alice",
            )
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")

        assert "&lt; B &amp; C &gt; D" in content

    def test_generate_subtitles_handles_none_speaker(
        self, temp_audio_file: Path
    ) -> None:
        """Test handling of chunks without speaker information."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=np.array([1]),
                text="No speaker",
                speaker=None,
            )
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")

        assert "(N/A)" in content

    def test_generate_subtitles_applies_colors_for_overlapping(
        self, overlapping_chunks: Chunk, temp_audio_file: Path
    ) -> None:
        """Test that overlapping speakers get color styling."""
        result_path = generate_subtitles(overlapping_chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")

        assert "<c." in content

    def test_generate_subtitles_uses_correct_indexing(
        self, sample_chunks: Chunk, temp_audio_file: Path
    ) -> None:
        """Test that cues are numbered sequentially starting from 1."""
        result_path = generate_subtitles(sample_chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")

        assert "1 (" in content
        assert "2 (" in content

    def test_generate_subtitles_rejects_non_list_chunks(
        self, temp_audio_file: Path
    ) -> None:
        """Regression test: ensure generate_subtitles() gets a list of Chunks, not one.

        Previously, translated_chunks was reassigned to a single Chunk instead of
        being appended to a list, causing an AttributeError when sorting.
        """
        single_chunk = Chunk(
            start_time=0.0,
            end_time=2.0,
            audio=np.array([0.1, 0.2, 0.3]),
            text="Hello world",
            speaker="Alice",
        )
        with pytest.raises(AttributeError):
            generate_subtitles(chunks=single_chunk, audio_path=temp_audio_file)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for the subtitling module."""

    def test_empty_text_chunk(self, temp_audio_file: Path) -> None:
        """Test handling of chunks with empty text."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=np.array([1]),
                text="",
                speaker="Alice",
            )
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        assert result_path.exists()

    def test_special_unicode_characters(self, temp_audio_file: Path) -> None:
        """Test handling of Unicode characters."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=np.array([1]),
                text="Hello 😀 World 中文",
                speaker="Alice",
            )
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")
        assert "中文" in content

    def test_very_long_text(self, temp_audio_file: Path) -> None:
        """Test handling of very long text."""
        long_text = " ".join(["word"] * 100)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=10.0,
                audio=np.array([1]),
                text=long_text,
                speaker="Alice",
            )
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        assert result_path.exists()

    def test_very_long_duration(self, temp_audio_file: Path) -> None:
        """Test handling of very long duration timestamps."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=7200.0,
                audio=np.array([1]),
                text="Long",
                speaker="Alice",
            )
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")
        assert "02:" in content

    def test_negative_timestamp_handling(self, temp_audio_file: Path) -> None:
        """Test handling of negative start times."""
        chunks = [
            Chunk(
                start_time=-1.0,
                end_time=1.0,
                audio=np.array([1]),
                text="Starts before",
                speaker="Alice",
            )
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        assert result_path.exists()

    def test_identical_start_end_times(self, temp_audio_file: Path) -> None:
        """Test handling of chunks where start equals end."""
        chunks = [
            Chunk(
                start_time=5.0,
                end_time=5.0,
                audio=np.array([1]),
                text="Instant",
                speaker="Alice",
            )
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")
        assert "00:00:05.000 --> 00:00:05.500" in content

    def test_overlapping_same_speaker(self, temp_audio_file: Path) -> None:
        """Test handling when same speaker overlaps with themselves."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=3.0,
                audio=np.array([1]),
                text="Part 1",
                speaker="Alice",
            ),
            Chunk(
                start_time=2.0,
                end_time=5.0,
                audio=np.array([1]),
                text="Part 2",
                speaker="Alice",
            ),
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        assert result_path.exists()

    def test_many_chunks(self, temp_audio_file: Path) -> None:
        """Test handling of many chunks."""
        chunks = [
            Chunk(
                start_time=i * 0.5,
                end_time=(i + 1) * 0.5,
                audio=np.array([1]),
                text=f"Chunk {i}",
                speaker=f"Speaker {i % 3}",
            )
            for i in range(100)
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")
        assert "100" in content

    def test_file_overwritten_if_exists(
        self, sample_chunks: Chunk, temp_audio_file: Path
    ) -> None:
        """Test that existing subtitle file is overwritten."""
        generate_subtitles(sample_chunks, temp_audio_file)

        modified_chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=np.array([1]),
                text="Modified",
                speaker="Charlie",
            )
        ]
        result_path = generate_subtitles(modified_chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")

        assert "Modified" in content
        assert "Hello world" not in content


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the subtitling module."""

    def test_empty_chunks_raises_error(self, temp_audio_file: Path) -> None:
        """Test that empty chunks list raises ValueError."""
        with pytest.raises(ValueError, match="Transcriptions list must not be empty"):
            generate_subtitles([], temp_audio_file)

    def test_nonexistent_audio_file_raises_error(self, sample_chunks: Chunk) -> None:
        """Test that non-existent audio file raises FileNotFoundError."""
        nonexistent_path = Path("/nonexistent/path/audio.mp3")
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            generate_subtitles(sample_chunks, nonexistent_path)

    def test_directory_as_audio_path_creates_in_dir(
        self, sample_chunks: Chunk, tmp_path: Path
    ) -> None:
        """Test that passing a directory creates vtt file in that directory."""
        result_path = generate_subtitles(sample_chunks, tmp_path)
        # When given a directory, .with_suffix() creates the vtt file in that dir
        assert result_path.suffix == ".vtt"
        assert result_path.is_file()


# =============================================================================
# Stress Tests
# =============================================================================


class TestStressScenarios:
    """Stress tests for the subtitling module."""

    def test_rapid_fire_chunks(self, temp_audio_file: Path) -> None:
        """Test with many very short chunks in rapid succession."""
        chunks = [
            Chunk(
                start_time=i * 0.1,
                end_time=(i + 1) * 0.1,
                audio=np.array([1]),
                text=f"W{i}",
                speaker=f"S{i % 5}",
            )
            for i in range(50)
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        assert result_path.exists()

    def test_wide_time_range(self, temp_audio_file: Path) -> None:
        """Test with chunks spanning a wide time range."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=np.array([1]),
                text="Start",
                speaker="Alice",
            ),
            Chunk(
                start_time=1000.0,
                end_time=1001.0,
                audio=np.array([1]),
                text="Middle",
                speaker="Bob",
            ),
            Chunk(
                start_time=10000.0,
                end_time=10001.0,
                audio=np.array([1]),
                text="End",
                speaker="Charlie",
            ),
        ]
        result_path = generate_subtitles(chunks, temp_audio_file)
        content = result_path.read_text(encoding="utf-8")
        assert "00:00:00.000" in content
        assert "00:16:40.000" in content
        assert "02:46:40.000" in content
