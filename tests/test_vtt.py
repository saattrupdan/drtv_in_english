"""Comprehensive tests for the vtt module.

This module contains unit tests, integration tests, and edge case tests
for the WebVTT parsing and writing functionality including:
- Timestamp parsing and formatting
- Writing chunks to VTT files
- Parsing VTT files with various speaker formats
- Round-trip write and parse
- Edge cases such as empty files, unicode, and whitespace
"""

from pathlib import Path

import numpy as np
import pytest

from but_with_subs.data_models import Chunk
from but_with_subs.vtt import (
    format_vtt_timestamp,
    parse_vtt_file,
    parse_vtt_timestamp,
    write_vtt_file,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create sample chunks for testing.

    Returns:
        List of sample Chunk objects for testing.
    """
    audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
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


# =============================================================================
# Unit Tests: parse_vtt_timestamp
# =============================================================================


class TestParseVttTimestamp:
    """Unit tests for parse_vtt_timestamp function."""

    def test_zero_timestamp(self) -> None:
        """Test parsing zero timestamp."""
        result = parse_vtt_timestamp("00:00:00.000")
        assert result == 0.0

    def test_single_second(self) -> None:
        """Test parsing single second."""
        result = parse_vtt_timestamp("00:00:01.000")
        assert result == 1.0

    def test_one_minute(self) -> None:
        """Test parsing one minute."""
        result = parse_vtt_timestamp("00:01:00.000")
        assert result == 60.0

    def test_minutes_and_seconds(self) -> None:
        """Test parsing minutes and seconds."""
        result = parse_vtt_timestamp("00:01:30.500")
        assert result == 90.5

    def test_one_hour(self) -> None:
        """Test parsing one hour."""
        result = parse_vtt_timestamp("01:00:00.000")
        assert result == 3600.0

    def test_hours_minutes_seconds(self) -> None:
        """Test parsing hours, minutes, and seconds."""
        result = parse_vtt_timestamp("01:01:01.111")
        assert result == 3661.111

    def test_large_hours(self) -> None:
        """Test parsing large hour values."""
        result = parse_vtt_timestamp("02:30:45.999")
        assert result == 9045.999

    def test_milliseconds_precision(self) -> None:
        """Test millisecond precision."""
        result = parse_vtt_timestamp("00:00:00.001")
        assert result == 0.001

    def test_max_milliseconds(self) -> None:
        """Test 999 milliseconds."""
        result = parse_vtt_timestamp("00:00:00.999")
        assert result == 0.999

    def test_full_precision(self) -> None:
        """Test full precision with all fields."""
        result = parse_vtt_timestamp("01:23:45.678")
        assert result == 5025.678


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

    def test_nine_hundred_ninety_nine_milliseconds(self) -> None:
        """Test 999 milliseconds formatting."""
        result = format_vtt_timestamp(0.999)
        assert result == "00:00:00.999"

    def test_one_minute_thirty_seconds(self) -> None:
        """Test formatting one minute thirty seconds."""
        result = format_vtt_timestamp(90.25)
        assert result == "00:01:30.250"

    def test_one_hour_one_minute_one_second(self) -> None:
        """Test formatting hours, minutes, and seconds."""
        result = format_vtt_timestamp(3661.0)
        assert result == "01:01:01.000"

    def test_two_hours_one_minute_five_seconds_half(self) -> None:
        """Test formatting large seconds value."""
        result = format_vtt_timestamp(7265.5)
        assert result == "02:01:05.500"

    def test_rounding_milliseconds(self) -> None:
        """Test that milliseconds are properly rounded."""
        result = format_vtt_timestamp(1.9999)
        assert result == "00:00:02.000"

    def test_forty_five_seconds(self) -> None:
        """Test forty five seconds."""
        result = format_vtt_timestamp(45.0)
        assert result == "00:00:45.000"

    def test_half_hour(self) -> None:
        """Test half an hour."""
        result = format_vtt_timestamp(1800.0)
        assert result == "00:30:00.000"


# =============================================================================
# Unit Tests: write_vtt_file
# =============================================================================


class TestWriteVttFile:
    """Unit tests for write_vtt_file function."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """Test writing a single basic chunk."""
        audio_data = np.array([0.1], dtype=np.float32)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=audio_data,
                text="Hello world",
                speaker=None,
            )
        ]
        output_path = tmp_path / "basic.vtt"
        write_vtt_file(chunks=chunks, path=output_path)

        content = output_path.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT\n\n")
        assert "1\n" in content
        assert "00:00:00.000 --> 00:00:02.000" in content
        assert "Hello world" in content

    def test_multiple_chunks(self, tmp_path: Path) -> None:
        """Test writing multiple chunks."""
        audio_data = np.array([0.1], dtype=np.float32)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=audio_data,
                text="First chunk",
                speaker=None,
            ),
            Chunk(
                start_time=2.5,
                end_time=4.5,
                audio=audio_data,
                text="Second chunk",
                speaker=None,
            ),
            Chunk(
                start_time=5.0,
                end_time=7.0,
                audio=audio_data,
                text="Third chunk",
                speaker=None,
            ),
        ]
        output_path = tmp_path / "multi.vtt"
        write_vtt_file(chunks=chunks, path=output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "1\n" in content
        assert "2\n" in content
        assert "3\n" in content
        assert "First chunk" in content
        assert "Second chunk" in content
        assert "Third chunk" in content

    def test_chunks_with_speaker(self, tmp_path: Path) -> None:
        """Test writing chunks with speakers."""
        audio_data = np.array([0.1], dtype=np.float32)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=audio_data,
                text="Alice speaking",
                speaker="Alice",
            ),
        ]
        output_path = tmp_path / "speaker.vtt"
        write_vtt_file(chunks=chunks, path=output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "<v Alice>" in content

    def test_chunks_with_none_speaker(self, tmp_path: Path) -> None:
        """Test writing chunks with None speaker produces no speaker line."""
        audio_data = np.array([0.1], dtype=np.float32)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.0,
                audio=audio_data,
                text="No speaker",
                speaker=None,
            ),
        ]
        output_path = tmp_path / "nospeaker.vtt"
        write_vtt_file(chunks=chunks, path=output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "<v " not in content
        assert "1\n" in content
        assert "00:00:00.000 --> 00:00:02.000" in content
        assert "No speaker" in content

    def test_file_created_at_specified_path(self, tmp_path: Path) -> None:
        """Test that file is created at the exact specified path."""
        audio_data = np.array([0.1], dtype=np.float32)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=audio_data,
                text="test",
                speaker=None,
            )
        ]
        subdir = tmp_path / "sub"
        subdir.mkdir()
        output_path = subdir / "output.vtt"
        write_vtt_file(chunks=chunks, path=output_path)

        assert output_path.exists()

    def test_webvtt_header_present(self, tmp_path: Path) -> None:
        """Test that WEBVTT header is present at the start of the file."""
        audio_data = np.array([0.1], dtype=np.float32)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=audio_data,
                text="test",
                speaker=None,
            )
        ]
        output_path = tmp_path / "header.vtt"
        write_vtt_file(chunks=chunks, path=output_path)

        content = output_path.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT\n\n")


# =============================================================================
# Unit Tests: parse_vtt_file
# =============================================================================


class TestParseVttFile:
    """Unit tests for parse_vtt_file function."""

    def test_parse_basic_vtt(self, tmp_path: Path) -> None:
        """Test parsing a basic VTT file without speakers."""
        vtt_content = (
            "WEBVTT\n\n"
            "1\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Hello world\n"
        )
        vtt_path = tmp_path / "basic.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 2.0
        assert chunks[0].speaker is None

    def test_parse_vtt_with_parenthesised_speaker(
        self, tmp_path: Path
    ) -> None:
        """Test parsing VTT with (Speaker) format on cue line."""
        vtt_content = (
            "WEBVTT\n\n"
            "1 (Alice)\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Hello world\n"
        )
        vtt_path = tmp_path / "paren_speaker.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert len(chunks) == 1
        assert chunks[0].speaker == "Alice"
        assert chunks[0].text == "Hello world"

    def test_parse_vtt_with_v_tag_speaker(self, tmp_path: Path) -> None:
        """Test parsing VTT with <v Speaker> format."""
        vtt_content = (
            "WEBVTT\n\n"
            "1\n"
            "<v Bob>\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Hello Bob\n"
        )
        vtt_path = tmp_path / "v_tag_speaker.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert len(chunks) == 1
        assert chunks[0].speaker == "Bob"
        assert chunks[0].text == "Hello Bob"

    def test_parse_vtt_with_multiple_chunks(self, tmp_path: Path) -> None:
        """Test parsing VTT with multiple chunks."""
        vtt_content = (
            "WEBVTT\n\n"
            "1\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "First chunk\n"
            "\n"
            "2\n"
            "00:00:02.500 --> 00:00:04.500\n"
            "Second chunk\n"
        )
        vtt_path = tmp_path / "multi.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert len(chunks) == 2
        assert chunks[0].text == "First chunk"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 2.0
        assert chunks[1].text == "Second chunk"
        assert chunks[1].start_time == 2.5
        assert chunks[1].end_time == 4.5

    def test_parse_vtt_with_mixed_speaker_formats(
        self, tmp_path: Path
    ) -> None:
        """Test parsing VTT with mixed speaker formats."""
        vtt_content = (
            "WEBVTT\n\n"
            "1 (Alice)\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Alice speaks\n"
            "\n"
            "2\n"
            "<v Bob>\n"
            "00:00:02.500 --> 00:00:04.500\n"
            "Bob speaks\n"
            "\n"
            "3\n"
            "00:00:05.000 --> 00:00:07.000\n"
            "No speaker here\n"
        )
        vtt_path = tmp_path / "mixed.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert len(chunks) == 3
        assert chunks[0].speaker == "Alice"
        assert chunks[1].speaker == "Bob"
        assert chunks[2].speaker is None

    def test_parse_vtt_with_inline_speaker_in_text(
        self, tmp_path: Path
    ) -> None:
        """Test parsing VTT when speaker is embedded in text as (Speaker)."""
        vtt_content = (
            "WEBVTT\n\n"
            "1\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "(Alice) Hello world\n"
        )
        vtt_path = tmp_path / "inline_speaker.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert len(chunks) == 1
        assert chunks[0].speaker == "Alice"
        assert chunks[0].text == "Hello world"

    def test_parse_vtt_tracks_stripped_from_text(
        self, tmp_path: Path
    ) -> None:
        """Test that HTML-like tags are stripped from text during parsing."""
        vtt_content = (
            "WEBVTT\n\n"
            "1\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "<c.red>Hello</c> world\n"
        )
        vtt_path = tmp_path / "tags.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert chunks[0].text == "Hello world"

    def test_parse_vtt_with_multiple_chunks_different_speakers(
        self, tmp_path: Path
    ) -> None:
        """Test parsing VTT with multiple chunks having different speakers."""
        vtt_content = (
            "WEBVTT\n\n"
            "1 (Alice)\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Alice part\n"
            "\n"
            "2 (Bob)\n"
            "00:00:02.500 --> 00:00:04.500\n"
            "Bob part\n"
        )
        vtt_path = tmp_path / "multi_speaker.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert len(chunks) == 2
        assert chunks[0].speaker == "Alice"
        assert chunks[1].speaker == "Bob"

    def test_parse_vtt_with_multiline_text(
        self, tmp_path: Path
    ) -> None:
        """Test parsing VTT with multiline cue text."""
        vtt_content = (
            "WEBVTT\n\n"
            "1\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Line one\n"
            "Line two\n"
        )
        vtt_path = tmp_path / "multiline.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert len(chunks) == 1
        assert chunks[0].text == "Line one\nLine two"


# =============================================================================
# Round-Trip Tests
# =============================================================================


class TestRoundTrip:
    """Tests for write + parse round-trip integrity."""

    def test_single_chunk_round_trip(self, tmp_path: Path) -> None:
        """Test round-trip with a single chunk."""
        audio_data = np.array([0.1, 0.2], dtype=np.float32)
        original_chunks = [
            Chunk(
                start_time=0.0,
                end_time=2.5,
                audio=audio_data,
                text="Round trip test",
                speaker="Alice",
            )
        ]
        vtt_path = tmp_path / "roundtrip.vtt"
        write_vtt_file(chunks=original_chunks, path=vtt_path)
        parsed_chunks = parse_vtt_file(path=vtt_path)

        assert len(parsed_chunks) == 1
        assert parsed_chunks[0].text == "Round trip test"
        assert parsed_chunks[0].speaker == "Alice"
        assert parsed_chunks[0].start_time == 0.0
        assert parsed_chunks[0].end_time == 2.5

    def test_multiple_chunks_round_trip(self, tmp_path: Path) -> None:
        """Test round-trip with multiple chunks."""
        audio_data = np.array([0.1], dtype=np.float32)
        original_chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.5,
                audio=audio_data,
                text="First",
                speaker="Alice",
            ),
            Chunk(
                start_time=2.0,
                end_time=3.5,
                audio=audio_data,
                text="Second",
                speaker="Bob",
            ),
            Chunk(
                start_time=4.0,
                end_time=5.0,
                audio=audio_data,
                text="Third",
                speaker=None,
            ),
        ]
        vtt_path = tmp_path / "roundtrip_multi.vtt"
        write_vtt_file(chunks=original_chunks, path=vtt_path)
        parsed_chunks = parse_vtt_file(path=vtt_path)

        assert len(parsed_chunks) == 3
        assert parsed_chunks[0].text == "First"
        assert parsed_chunks[0].speaker == "Alice"
        assert parsed_chunks[1].text == "Second"
        assert parsed_chunks[1].speaker == "Bob"
        assert parsed_chunks[2].text == "Third"
        assert parsed_chunks[2].speaker is None

    def test_timestamp_precision_round_trip(self, tmp_path: Path) -> None:
        """Test that timestamp precision is preserved through round-trip."""
        audio_data = np.array([0.1], dtype=np.float32)
        original_chunks = [
            Chunk(
                start_time=1.234,
                end_time=5.678,
                audio=audio_data,
                text="Precision test",
                speaker=None,
            )
        ]
        vtt_path = tmp_path / "precision.vtt"
        write_vtt_file(chunks=original_chunks, path=vtt_path)
        parsed_chunks = parse_vtt_file(path=vtt_path)

        assert parsed_chunks[0].start_time == 1.234
        assert parsed_chunks[0].end_time == 5.678

    def test_speaker_preserved_round_trip(
        self, tmp_path: Path
    ) -> None:
        """Test that speaker info survives round-trip."""
        audio_data = np.array([0.1], dtype=np.float32)
        original_chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=audio_data,
                text="With speaker",
                speaker="Charlie",
            ),
            Chunk(
                start_time=1.5,
                end_time=2.5,
                audio=audio_data,
                text="Without speaker",
                speaker=None,
            ),
        ]
        vtt_path = tmp_path / "speaker_rt.vtt"
        write_vtt_file(chunks=original_chunks, path=vtt_path)
        parsed_chunks = parse_vtt_file(path=vtt_path)

        assert parsed_chunks[0].speaker == "Charlie"
        assert parsed_chunks[1].speaker is None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for vtt module."""

    def test_empty_vtt_file(self, tmp_path: Path) -> None:
        """Test parsing an empty VTT file."""
        vtt_path = tmp_path / "empty.vtt"
        vtt_path.write_text("", encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)
        assert len(chunks) == 0

    def test_vtt_with_header_only(self, tmp_path: Path) -> None:
        """Test parsing a VTT file with only the WEBVTT header."""
        vtt_content = "WEBVTT\n"
        vtt_path = tmp_path / "header_only.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)
        assert len(chunks) == 0

    def test_vtt_with_extra_whitespace(self, tmp_path: Path) -> None:
        """Test parsing VTT with extra blank lines and spaces."""
        vtt_content = (
            "WEBVTT\n\n\n\n"
            "1\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "  Hello world  \n"
            "\n\n"
        )
        vtt_path = tmp_path / "whitespace.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_unicode_text(self, tmp_path: Path) -> None:
        """Test parsing VTT with unicode text."""
        vtt_content = (
            "WEBVTT\n\n"
            "1\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Hello 世界 مرحبا שלום\n"
        )
        vtt_path = tmp_path / "unicode.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello 世界 مرحبا שלום"

    def test_unicode_speaker_name(self, tmp_path: Path) -> None:
        """Test parsing VTT with unicode speaker name."""
        vtt_content = (
            "WEBVTT\n\n"
            "1 (日本語)\n"
            "00:00:00.000 --> 00:00:02.000\n"
            "Unicode speaker test\n"
        )
        vtt_path = tmp_path / "unicode_speaker.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)
        assert len(chunks) == 1
        assert chunks[0].speaker == "日本語"

    def test_write_empty_chunks_list(self, tmp_path: Path) -> None:
        """Test writing an empty list of chunks."""
        output_path = tmp_path / "empty_write.vtt"
        write_vtt_file(chunks=[], path=output_path)

        content = output_path.read_text(encoding="utf-8")
        assert content == "WEBVTT\n\n"

    def test_chunk_with_none_text(self, tmp_path: Path) -> None:
        """Test writing a chunk with None text."""
        audio_data = np.array([0.1], dtype=np.float32)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=audio_data,
                text=None,
                speaker=None,
            )
        ]
        output_path = tmp_path / "none_text.vtt"
        write_vtt_file(chunks=chunks, path=output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "00:00:00.000 --> 00:00:01.000" in content

    def test_write_and_parse_none_text(self, tmp_path: Path) -> None:
        """Test round-trip with None text."""
        audio_data = np.array([0.1], dtype=np.float32)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=audio_data,
                text=None,
                speaker=None,
            )
        ]
        vtt_path = tmp_path / "none_text_rt.vtt"
        write_vtt_file(chunks=chunks, path=vtt_path)
        parsed = parse_vtt_file(path=vtt_path)

        assert len(parsed) == 1
        assert parsed[0].text == "None"

    def test_vtt_with_speaker_style_region(self, tmp_path: Path) -> None:
        """Test VTT with speaker style region after timestamp."""
        vtt_content = (
            "WEBVTT\n\n"
            "1\n"
            "00:00:00.000 --> 00:00:02.000 position:10%\n"
            "Styled subtitle\n"
        )
        vtt_path = tmp_path / "style_region.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)
        assert len(chunks) == 1
        assert chunks[0].text == "Styled subtitle"

    def test_vtt_with_all_speaker_formats_same_file(
        self, tmp_path: Path
    ) -> None:
        """Test VTT file mixing all speaker formats."""
        vtt_content = (
            "WEBVTT\n\n"
            "1 (Paren Speaker)\n"
            "00:00:00.000 --> 00:00:01.000\n"
            "Paren format\n"
            "\n"
            "2\n"
            "<v VTag Speaker>\n"
            "00:00:01.500 --> 00:00:02.500\n"
            "V tag format\n"
            "\n"
            "3\n"
            "00:00:03.000 --> 00:00:04.000\n"
            "(Inline Speaker) Inline format\n"
            "\n"
            "4\n"
            "00:00:04.500 --> 00:00:05.500\n"
            "No speaker\n"
        )
        vtt_path = tmp_path / "all_formats.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        chunks = parse_vtt_file(path=vtt_path)

        assert len(chunks) == 4
        assert chunks[0].speaker == "Paren Speaker"
        assert chunks[1].speaker == "VTag Speaker"
        assert chunks[2].speaker == "Inline Speaker"
        assert chunks[2].text == "Inline format"
        assert chunks[3].speaker is None

    def test_timestamp_edge_values(self, tmp_path: Path) -> None:
        """Test parse and format with extreme timestamp values."""
        audio_data = np.array([0.1], dtype=np.float32)
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=9999.0,
                audio=audio_data,
                text="Extreme duration",
                speaker=None,
            )
        ]
        vtt_path = tmp_path / "extreme.vtt"
        write_vtt_file(chunks=chunks, path=vtt_path)
        parsed = parse_vtt_file(path=vtt_path)

        assert parsed[0].start_time == 0.0
        assert parsed[0].end_time == 9999.0
