"""Comprehensive tests for the translation module.

This module contains unit tests for helper functions, mocked tests for the
translation workflow, and error handling tests for the translation module.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from but_with_subs.data_models import Chunk
from but_with_subs.translation import (
    _format_vtt_timestamp,
    _parse_vtt_file,
    _parse_vtt_timestamp,
    _write_vtt_file,
    translate_subtitles,
    Translator,
)


# =============================================================================
# Unit Tests for Helper Functions
# =============================================================================

class TestParseVttTimestamp:
    """Tests for _parse_vtt_timestamp function."""

    def test_parse_simple_timestamp(self):
        """Test parsing a simple timestamp."""
        result = _parse_vtt_timestamp("00:00:01.500")
        assert result == 1.5

    def test_parse_timestamp_with_hours(self):
        """Test parsing timestamp with hours."""
        result = _parse_vtt_timestamp("01:30:45.123")
        assert result == 5445.123

    def test_parse_timestamp_zero(self):
        """Test parsing zero timestamp."""
        result = _parse_vtt_timestamp("00:00:00.000")
        assert result == 0.0

    def test_parse_timestamp_large(self):
        """Test parsing large timestamp."""
        result = _parse_vtt_timestamp("02:15:30.500")
        assert result == 8130.5

    def test_parse_timestamp_milliseconds(self):
        """Test parsing timestamp with various milliseconds."""
        result = _parse_vtt_timestamp("00:00:00.001")
        assert result == 0.001

    def test_parse_timestamp_edge_case(self):
        """Test parsing timestamp edge case with round numbers."""
        result = _parse_vtt_timestamp("00:01:00.000")
        assert result == 60.0


class TestFormatVttTimestamp:
    """Tests for _format_vtt_timestamp function."""

    def test_format_simple_seconds(self):
        """Test formatting simple seconds."""
        result = _format_vtt_timestamp(1.5)
        assert result == "00:00:01.500"

    def test_format_with_hours(self):
        """Test formatting timestamp with hours."""
        result = _format_vtt_timestamp(5445.123)
        assert result == "01:30:45.123"

    def test_format_zero(self):
        """Test formatting zero seconds."""
        result = _format_vtt_timestamp(0.0)
        assert result == "00:00:00.000"

    def test_format_large_timestamp(self):
        """Test formatting large timestamp."""
        result = _format_vtt_timestamp(8130.5)
        assert result == "02:15:30.500"

    def test_format_rounding(self):
        """Test rounding of milliseconds."""
        result = _format_vtt_timestamp(1.2567)
        assert result == "00:00:01.257"

    def test_format_exact_milliseconds(self):
        """Test exact millisecond formatting."""
        result = _format_vtt_timestamp(0.001)
        assert result == "00:00:00.001"

    def test_format_round_trip(self):
        """Test that format -> parse -> format is consistent."""
        original_seconds = 1234.567
        formatted = _format_vtt_timestamp(original_seconds)
        parsed = _parse_vtt_timestamp(formatted)
        assert abs(original_seconds - parsed) < 0.001


class TestParseVttFile:
    """Tests for _parse_vtt_file function."""

    def test_parse_simple_vtt(self, tmp_path):
        """Test parsing a simple VTT file."""
        vtt_content = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
Hello world

2
00:00:05.000 --> 00:00:08.000
This is a test
"""
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(vtt_content)

        chunks = _parse_vtt_file(vtt_file)

        assert len(chunks) == 2
        assert chunks[0].start_time == 1.0
        assert chunks[0].end_time == 4.0
        assert chunks[0].text == "Hello world"
        assert chunks[0].speaker is None
        assert chunks[1].text == "This is a test"

    def test_parse_vtt_with_speaker(self, tmp_path):
        """Test parsing VTT file with speaker information."""
        vtt_content = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
(John) Hello there

2
00:00:05.000 --> 00:00:08.000
(Mary) How are you?
"""
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(vtt_content)

        chunks = _parse_vtt_file(vtt_file)

        assert len(chunks) == 2
        assert chunks[0].speaker == "John"
        assert chunks[0].text == "Hello there"
        assert chunks[1].speaker == "Mary"
        assert chunks[1].text == "How are you?"

    def test_parse_empty_vtt(self, tmp_path):
        """Test parsing an empty VTT file."""
        vtt_content = "WEBVTT\n\n"
        vtt_file = tmp_path / "empty.vtt"
        vtt_file.write_text(vtt_content)

        chunks = _parse_vtt_file(vtt_file)

        assert len(chunks) == 0

    def test_parse_vtt_with_html_tags(self, tmp_path):
        """Test parsing VTT file with HTML tags that should be stripped."""
        vtt_content = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
This has <b>bold</b> and <i>italic</i> text
"""
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(vtt_content)

        chunks = _parse_vtt_file(vtt_file)

        assert len(chunks) == 1
        assert chunks[0].text == "This has bold and italic text"

    def test_parse_vtt_with_multiline_text(self, tmp_path):
        """Test parsing VTT file with multiline text."""
        vtt_content = """WEBVTT

1
00:00:01.000 --> 00:00:05.000
First line
Second line
Third line
"""
        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text(vtt_content)

        chunks = _parse_vtt_file(vtt_file)

        assert len(chunks) == 1
        assert "First line" in chunks[0].text
        assert "Second line" in chunks[0].text


class TestWriteVttFile:
    """Tests for _write_vtt_file function."""

    def test_write_simple_chunks(self, tmp_path):
        """Test writing simple chunks to VTT file."""
        chunks = [
            Chunk(
                start_time=1.0,
                end_time=4.0,
                audio=np.zeros(48000, dtype=np.float32),
                text="Hello world",
                speaker=None,
            ),
            Chunk(
                start_time=5.0,
                end_time=8.0,
                audio=np.zeros(48000, dtype=np.float32),
                text="This is a test",
                speaker=None,
            ),
        ]
        vtt_file = tmp_path / "output.vtt"

        _write_vtt_file(chunks, vtt_file)

        content = vtt_file.read_text()
        assert "WEBVTT" in content
        assert "00:00:01.000 --> 00:00:04.000" in content
        assert "00:00:05.000 --> 00:00:08.000" in content
        assert "Hello world" in content
        assert "This is a test" in content

    def test_write_with_speaker(self, tmp_path):
        """Test writing chunks with speaker information."""
        chunks = [
            Chunk(
                start_time=1.0,
                end_time=4.0,
                audio=np.zeros(48000, dtype=np.float32),
                text="Hello there",
                speaker="John",
            ),
        ]
        vtt_file = tmp_path / "output.vtt"

        _write_vtt_file(chunks, vtt_file)

        content = vtt_file.read_text()
        assert "<v John>" in content
        assert "Hello there" in content

    def test_write_empty_chunks(self, tmp_path):
        """Test writing empty chunk list."""
        chunks: list[Chunk] = []
        vtt_file = tmp_path / "empty.vtt"

        _write_vtt_file(chunks, vtt_file)

        content = vtt_file.read_text()
        assert content == "WEBVTT\n\n"


# =============================================================================
# Mocked Tests for Translation Workflow
# =============================================================================

class TestTranslator:
    """Tests for the Translator class with mocked pipeline."""

    def test_init_translator(self):
        """Test translator initialization."""
        with patch("but_with_subs.translation.pipeline") as mock_pipeline:
            mock_pipeline.return_value = MagicMock()
            translator = Translator(model_id="test/model")
            assert translator._pipeline is not None
            mock_pipeline.assert_called_once()

    def test_translate_text(self):
        """Test translating a single text."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"translation_text": "translated text"}]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translator = Translator()
            result = translator.translate_text("original text", "dan", "eng")

        assert result == "translated text"

    def test_translate_chunks(self):
        """Test translating multiple chunks."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"translation_text": "translated 1"},
            {"translation_text": "translated 2"},
        ]

        chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=np.zeros(16000, dtype=np.float32),
                text="original 1",
                speaker=None,
            ),
            Chunk(
                start_time=1.0,
                end_time=2.0,
                audio=np.zeros(16000, dtype=np.float32),
                text="original 2",
                speaker=None,
            ),
        ]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translator = Translator()
            result = translator.translate_chunks(chunks, "dan", "eng")

        assert len(result) == 2
        assert result[0].text == "translated 1"
        assert result[1].text == "translated 2"
        assert result[0].start_time == 0.0
        assert result[1].end_time == 2.0

    def test_translate_chunks_empty(self):
        """Test translating empty chunk list."""
        with patch("but_with_subs.translation.pipeline"):
            translator = Translator()
            result = translator.translate_chunks([], "dan", "eng")

        assert result == []

    def test_translate_chunks_with_none_text(self):
        """Test translating chunks where some have no text."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"translation_text": "translated"}]

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
                text="original",
                speaker=None,
            ),
        ]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translator = Translator()
            result = translator.translate_chunks(chunks, "dan", "eng")

        assert len(result) == 2
        assert result[0].text is None
        assert result[1].text == "translated"

    def test_translate_chunks_all_none_text_logs_warning_and_returns_original(self, caplog):
        """Test that translate_chunks logs warning and returns original when all chunks have None text.
        
        This tests lines 95-96 in translation.py where a warning is logged and
        the original chunks are returned when no chunks have text.
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
                speaker=None,
            ),
        ]

        with patch("but_with_subs.translation.pipeline"):
            translator = Translator()
            
            with caplog.at_level("WARNING"):
                result = translator.translate_chunks(chunks, "dan", "eng")

        # Verify warning was logged
        assert any("No chunks with text to translate" in record.message for record in caplog.records)
        
        # Verify original chunks are returned unchanged
        assert result is chunks
        assert len(result) == 2
        assert all(c.text is None for c in result)

    def test_translate_batch_processing(self):
        """Test that batch processing works correctly."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"translation_text": f"translated {i}"} for i in range(20)
        ]

        chunks = [
            Chunk(
                start_time=float(i),
                end_time=float(i + 1),
                audio=np.zeros(16000, dtype=np.float32),
                text=f"original {i}",
                speaker=None,
            )
            for i in range(20)
        ]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translator = Translator()
            result = translator.translate_chunks(chunks, "dan", "eng", batch_size=5)

        assert len(result) == 20
        assert mock_pipeline.call_count == 4  # 20 chunks / 5 batch_size


class TestTranslateSubtitles:
    """Tests for the translate_subtitles function with mocked dependencies."""

    def test_translate_subtitles_basic(self, tmp_path):
        """Test basic subtitle translation workflow."""
        input_vtt = tmp_path / "input.vtt"
        input_vtt.write_text("""WEBVTT

1
00:00:01.000 --> 00:00:04.000
Hej verden
""")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"translation_text": "Hello world"}]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            output_path = translate_subtitles(
                input_vtt,
                source_lang="dan",
                target_lang="eng",
                model_id="test/model",
            )

        assert output_path.exists()
        content = output_path.read_text()
        assert "WEBVTT" in content
        assert "Hello world" in content

    def test_translate_subtitles_custom_output_path(self, tmp_path):
        """Test translation with custom output path."""
        input_vtt = tmp_path / "input.vtt"
        input_vtt.write_text("WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nTest\n")

        output_vtt = tmp_path / "custom_output.vtt"

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"translation_text": "Test"}]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            result = translate_subtitles(input_vtt, output_vtt)

        assert result == output_vtt
        assert output_vtt.exists()

    def test_translate_subtitles_auto_output_path(self, tmp_path):
        """Test translation with auto-generated output path."""
        input_vtt = tmp_path / "subtitles.vtt"
        input_vtt.write_text("WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nTest\n")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"translation_text": "Test"}]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            result = translate_subtitles(input_vtt)

        expected = tmp_path / "subtitles_translated.vtt"
        assert result == expected
        assert result.exists()

    def test_translate_subtitles_with_speaker(self, tmp_path):
        """Test translation preserving speaker information."""
        input_vtt = tmp_path / "input.vtt"
        input_vtt.write_text("""WEBVTT

1
00:00:01.000 --> 00:00:04.000
(Johan) Hej hvordan går det?
""")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"translation_text": "Hi how are you?"}]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            output_path = translate_subtitles(input_vtt)

        content = output_path.read_text()
        assert "<v Johan>" in content
        assert "Hi how are you?" in content


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in the translation module."""

    def test_translate_subtitles_file_not_found(self):
        """Test error when input file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            translate_subtitles("/nonexistent/path/file.vtt")

    def test_translate_text_exception(self):
        """Test handling of translation exceptions."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Translation failed")

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translator = Translator()
            with pytest.raises(Exception, match="Translation failed"):
                translator.translate_text("test", "dan", "eng")

    def test_translate_chunks_batch_fallback(self, caplog):
        """Test fallback to individual translation when batch fails."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = [
            Exception("Batch failed"),
            [{"translation_text": "fallback 1"}],
            [{"translation_text": "fallback 2"}],
        ]

        chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=np.zeros(16000, dtype=np.float32),
                text="original 1",
                speaker=None,
            ),
            Chunk(
                start_time=1.0,
                end_time=2.0,
                audio=np.zeros(16000, dtype=np.float32),
                text="original 2",
                speaker=None,
            ),
        ]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translator = Translator()
            result = translator.translate_chunks(chunks, "dan", "eng")

        assert len(result) == 2
        assert result[0].text == "fallback 1"
        assert result[1].text == "fallback 2"

    def test_translate_chunks_all_failures(self, caplog):
        """Test handling when all translations fail."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = Exception("Always fails")

        chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=np.zeros(16000, dtype=np.float32),
                text="original 1",
                speaker=None,
            ),
        ]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translator = Translator()
            result = translator.translate_chunks(chunks, "dan", "eng")

        # Should return original text when translation fails
        assert len(result) == 1
        assert result[0].text == "original 1"

    def test_parse_vtt_invalid_timestamp(self, tmp_path):
        """Test error handling for invalid timestamp format."""
        vtt_content = """WEBVTT

1
invalid-timestamp --> 00:00:04.000
Test
"""
        vtt_file = tmp_path / "invalid.vtt"
        vtt_file.write_text(vtt_content)

        # Invalid timestamps are simply not matched by the regex, so no chunks parsed
        chunks = _parse_vtt_file(vtt_file)
        assert len(chunks) == 0

    def test_parse_vtt_malformed_content(self, tmp_path):
        """Test parsing malformed VTT content."""
        vtt_content = "This is not valid VTT content at all"
        vtt_file = tmp_path / "malformed.vtt"
        vtt_file.write_text(vtt_content)

        chunks = _parse_vtt_file(vtt_file)
        assert len(chunks) == 0

    def test_write_vtt_invalid_path(self, tmp_path):
        """Test error when writing to invalid path."""
        chunks = [
            Chunk(
                start_time=0.0,
                end_time=1.0,
                audio=np.zeros(16000, dtype=np.float32),
                text="test",
                speaker=None,
            ),
        ]

        # Try to write to a path that's a file, not a directory
        invalid_path = tmp_path / "file.txt" / "subdir" / "output.vtt"

        with pytest.raises(FileNotFoundError):
            _write_vtt_file(chunks, invalid_path)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the translation module."""

    def test_full_round_trip(self, tmp_path):
        """Test complete round trip: parse -> translate -> write -> parse."""
        original_vtt = tmp_path / "original.vtt"
        original_vtt.write_text("""WEBVTT

1
00:00:01.000 --> 00:00:04.000
Hej verden
2
00:00:05.000 --> 00:00:08.000
(Hans) Hvordan har du det?
""")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"translation_text": "Hello world"},
            {"translation_text": "How are you?"},
        ]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translated_vtt = translate_subtitles(original_vtt)
            chunks = _parse_vtt_file(translated_vtt)

        assert len(chunks) == 2
        assert chunks[0].text == "Hello world"
        assert chunks[1].speaker == "Hans"
        assert chunks[1].text == "How are you?"
        assert chunks[0].start_time == 1.0
        assert chunks[0].end_time == 4.0

    def test_preserves_timing_structure(self, tmp_path):
        """Test that translation preserves timing structure."""
        original_vtt = tmp_path / "original.vtt"
        original_vtt.write_text("""WEBVTT

1
00:00:00.500 --> 00:00:02.750
Segment one
2
00:00:02.750 --> 00:00:05.000
Segment two
3
00:00:05.000 --> 00:00:10.500
Segment three
""")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"translation_text": "Segment one"},
            {"translation_text": "Segment two"},
            {"translation_text": "Segment three"},
        ]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translated_vtt = translate_subtitles(original_vtt)
            chunks = _parse_vtt_file(translated_vtt)

        assert len(chunks) == 3
        assert chunks[0].start_time == 0.5
        assert chunks[0].end_time == 2.75
        assert chunks[1].start_time == 2.75
        assert chunks[1].end_time == 5.0
        assert chunks[2].start_time == 5.0
        assert chunks[2].end_time == 10.5

    def test_handles_complex_vtt_format(self, tmp_path):
        """Test handling of complex VTT with various features."""
        original_vtt = tmp_path / "complex.vtt"
        original_vtt.write_text("""WEBVTT
Kind: captions
Language: dan

1
00:00:01.000 --> 00:00:04.000 align:start position:10%
(Dokumentar) Dette er en <i>dokumentar</i> film

2
00:00:05.000 --> 00:00:08.000
Intervju med <b>eksperten</b>
""")

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"translation_text": "This is a documentary film"},
            {"translation_text": "Interview with the expert"},
        ]

        with patch("but_with_subs.translation.pipeline", return_value=mock_pipeline):
            translated_vtt = translate_subtitles(original_vtt)
            chunks = _parse_vtt_file(translated_vtt)

        assert len(chunks) == 2
        assert chunks[0].speaker == "Dokumentar"
        assert "documentary" in chunks[0].text.lower() or "film" in chunks[0].text.lower()
