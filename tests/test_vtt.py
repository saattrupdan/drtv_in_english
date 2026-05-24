"""Tests for the vtt module."""

from pathlib import Path

from danglish.data_models import Chunk
from danglish.vtt import (
    format_vtt_timestamp,
    parse_external_vtt,
    parse_vtt_timestamp,
    write_vtt_file,
)

# =============================================================================
# parse_vtt_timestamp / format_vtt_timestamp
# =============================================================================


class TestParseVttTimestamp:
    """Unit tests for parse_vtt_timestamp function."""

    def test_zero_timestamp(self) -> None:
        """Zero is parsed as 0.0."""
        assert parse_vtt_timestamp("00:00:00.000") == 0.0

    def test_full_precision(self) -> None:
        """Hours, minutes, seconds and milliseconds combine correctly."""
        assert parse_vtt_timestamp("01:23:45.678") == 5025.678

    def test_milliseconds_precision(self) -> None:
        """1 ms is parsed as 0.001 s."""
        assert parse_vtt_timestamp("00:00:00.001") == 0.001


class TestFormatVttTimestamp:
    """Unit tests for format_vtt_timestamp function."""

    def test_zero_seconds(self) -> None:
        """Zero seconds renders as 00:00:00.000."""
        assert format_vtt_timestamp(0.0) == "00:00:00.000"

    def test_fractional(self) -> None:
        """Fractional seconds become milliseconds."""
        assert format_vtt_timestamp(0.123) == "00:00:00.123"

    def test_hours_minutes_seconds(self) -> None:
        """Hours, minutes, seconds and ms render correctly."""
        assert format_vtt_timestamp(3661.0) == "01:01:01.000"

    def test_rounding_milliseconds(self) -> None:
        """Sub-millisecond input is rounded."""
        assert format_vtt_timestamp(1.9999) == "00:00:02.000"


# =============================================================================
# write_vtt_file
# =============================================================================


class TestWriteVttFile:
    """Unit tests for write_vtt_file function."""

    def test_basic_write(self, tmp_path: Path) -> None:
        """A single chunk produces a valid WEBVTT cue."""
        chunks = [Chunk(start_time=0.0, end_time=2.0, text="Hello world", speaker=None)]
        output_path = tmp_path / "basic.vtt"
        write_vtt_file(chunks=chunks, path=output_path)

        content = output_path.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT\n\n")
        assert "1\n" in content
        assert "00:00:00.000 --> 00:00:02.000" in content
        assert "Hello world" in content

    def test_chunks_with_speaker(self, tmp_path: Path) -> None:
        """Speaker is rendered as a <v> tag."""
        chunks = [Chunk(start_time=0.0, end_time=2.0, text="hi", speaker="Alice")]
        output_path = tmp_path / "speaker.vtt"
        write_vtt_file(chunks=chunks, path=output_path)

        assert "<v Alice>" in output_path.read_text(encoding="utf-8")

    def test_empty_chunks_list(self, tmp_path: Path) -> None:
        """An empty input produces only the header."""
        output_path = tmp_path / "empty.vtt"
        write_vtt_file(chunks=[], path=output_path)
        assert output_path.read_text(encoding="utf-8") == "WEBVTT\n\n"


# =============================================================================
# parse_external_vtt
# =============================================================================


class TestParseExternalVtt:
    """Tests for parse_external_vtt covering DR-style cues."""

    def test_parse_basic(self, tmp_path: Path) -> None:
        """A single cue is parsed."""
        vtt_path = tmp_path / "basic.vtt"
        vtt_path.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello world\n", encoding="utf-8"
        )
        chunks = parse_external_vtt(path=vtt_path)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].start_time == 0.0
        assert chunks[0].end_time == 2.0

    def test_parse_multiple(self, tmp_path: Path) -> None:
        """Multiple cues separated by blank lines are parsed."""
        vtt_path = tmp_path / "multi.vtt"
        vtt_path.write_text(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.000\nFirst\n\n"
            "00:00:02.500 --> 00:00:04.500\nSecond\n",
            encoding="utf-8",
        )
        chunks = parse_external_vtt(path=vtt_path)
        assert [c.text for c in chunks] == ["First", "Second"]

    def test_multiline_text_joined(self, tmp_path: Path) -> None:
        """Multi-line cue text is joined with a single space."""
        vtt_path = tmp_path / "ml.vtt"
        vtt_path.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nLine one\nLine two\n",
            encoding="utf-8",
        )
        chunks = parse_external_vtt(path=vtt_path)
        assert chunks[0].text == "Line one Line two"

    def test_tags_stripped(self, tmp_path: Path) -> None:
        """HTML-like tags are removed."""
        vtt_path = tmp_path / "tags.vtt"
        vtt_path.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n<i>Hello</i> world\n",
            encoding="utf-8",
        )
        chunks = parse_external_vtt(path=vtt_path)
        assert chunks[0].text == "Hello world"

    def test_cue_settings_ignored(self, tmp_path: Path) -> None:
        """Cue settings after the timestamp are ignored."""
        vtt_path = tmp_path / "settings.vtt"
        vtt_path.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000 position:10%\nStyled\n",
            encoding="utf-8",
        )
        chunks = parse_external_vtt(path=vtt_path)
        assert chunks[0].text == "Styled"


# =============================================================================
# Round-trip
# =============================================================================


def test_round_trip(tmp_path: Path) -> None:
    """A chunk written to disk and parsed back preserves its fields."""
    original = [Chunk(start_time=1.234, end_time=5.678, text="Hi there", speaker=None)]
    vtt_path = tmp_path / "rt.vtt"
    write_vtt_file(chunks=original, path=vtt_path)
    parsed = parse_external_vtt(path=vtt_path)
    assert len(parsed) == 1
    assert parsed[0].text == "Hi there"
    assert parsed[0].start_time == 1.234
    assert parsed[0].end_time == 5.678
