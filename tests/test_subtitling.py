"""Tests for the subtitling module.

This module contains comprehensive tests for the ``generate_subtitles`` function
and the ``_format_vtt_timestamp`` and ``_escape_vtt_text`` helpers, including
verification of VTT file creation, timestamp formatting, text escaping, progress
yielding, and error handling.
"""

import pathlib as pl
import tempfile as tf

from but_with_subs.subtitling import (
    _escape_vtt_text,
    _format_vtt_timestamp,
    _group_transcriptions,
    generate_subtitles,
)
from but_with_subs.data_models import Transcription

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transcription(
    start_time: float = 0.0, end_time: float = 1.0, text: str = "Hello world"
) -> Transcription:
    """Create a Transcription model with the given parameters.

    Args:
        start_time: Start time in seconds.
        end_time: End time in seconds.
        text: Transcribed text content.

    Returns:
        A Transcription instance.
    """
    return Transcription(start_time=start_time, end_time=end_time, text=text)


def _make_audio_file(suffix: str = ".wav", content: bytes | None = None) -> pl.Path:
    """Create a temporary audio file and return its path.

    Args:
        suffix: File extension for the temporary file.
        content: Optional bytes to write to the file.

    Returns:
        A pathlib.Path to the temporary file.
    """
    tmp = tf.NamedTemporaryFile(suffix=suffix, delete=False)
    if content is not None:
        tmp.write(content)
    tmp.close()
    return pl.Path(tmp.name)


# ---------------------------------------------------------------------------
# _format_vtt_timestamp tests
# ---------------------------------------------------------------------------


def test_format_vtt_timestamp_basic() -> None:
    """Test _format_vtt_timestamp formats a simple float correctly.

    Verifies that 0.0 seconds produces the expected ``00:00:00.000`` string.
    """
    result = _format_vtt_timestamp(seconds=0.0)

    assert result == "00:00:00.000"


def test_format_vtt_timestamp_rounds_milliseconds() -> None:
    """Test _format_vtt_timestamp rounds fractional milliseconds.

    Verifies that 1.2346 seconds rounds to ``00:00:01.235`` (nearest ms).
    """
    result = _format_vtt_timestamp(seconds=1.2346)

    assert result == "00:00:01.235"


def test_format_vtt_timestamp_with_minutes() -> None:
    """Test _format_vtt_timestamp handles minutes correctly.

    Verifies that 65.5 seconds produces ``00:01:05.500``.
    """
    result = _format_vtt_timestamp(seconds=65.5)

    assert result == "00:01:05.500"


def test_format_vtt_timestamp_with_hours() -> None:
    """Test _format_vtt_timestamp handles hours correctly.

    Verifies that 3661.123 seconds produces ``01:01:01.123``.
    """
    result = _format_vtt_timestamp(seconds=3661.123)

    assert result == "01:01:01.123"


def test_format_vtt_timestamp_boundary_rounding() -> None:
    """Test _format_vtt_timestamp rounds up at 0.5ms boundary.

    Verifies that 0.00051 seconds rounds up to ``00:00:00.001``.
    """
    result = _format_vtt_timestamp(seconds=0.00051)

    assert result == "00:00:00.001"


# ---------------------------------------------------------------------------
# _escape_vtt_text tests
# ---------------------------------------------------------------------------


def test_escape_vtt_text_unchanged() -> None:
    """Test _escape_vtt_text returns unchanged text without special chars.

    Verifies that plain text is returned as-is.
    """
    result = _escape_vtt_text(text="Hello world")

    assert result == "Hello world"


def test_escape_vtt_text_escapes_ampersand() -> None:
    """Test _escape_vtt_text escapes ``&`` to ``&amp;``.

    Verifies that an ampersand in the input text is replaced with
    the HTML entity ``&amp;``.
    """
    result = _escape_vtt_text(text="Tom & Jerry")

    assert result == "Tom &amp; Jerry"


def test_escape_vtt_text_escapes_angle_brackets() -> None:
    """Test _escape_vtt_text escapes ``<`` and ``>`` to HTML entities.

    Verifies that angle brackets are replaced with ``&lt;`` and ``&gt;``.
    """
    result = _escape_vtt_text(text="Use <div> tags")

    assert result == "Use &lt;div&gt; tags"


def test_escape_vtt_text_escapes_all_special_chars() -> None:
    """Test _escape_vtt_text escapes &, <, and > in combination.

    Verifies that all three special characters are escaped simultaneously.
    """
    result = _escape_vtt_text(text="A & B < C > D")

    assert result == "A &amp; B &lt; C &gt; D"


# ---------------------------------------------------------------------------
# generate_subtitles() tests
# ---------------------------------------------------------------------------


def test_generate_subtitles_creates_vtt_file() -> None:
    """Test generate_subtitles creates a .vtt file with correct content.

    Creates a single transcription and verifies that the output .vtt file
    exists at the expected path and contains the correct WEBVTT header,
    cue number, timestamps, and text.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=1.5, text="Hello world")
    ]

    generator = generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    list(generator)
    result_path = audio_path.with_suffix(".vtt")

    assert result_path.exists()
    assert result_path.suffix == ".vtt"
    assert result_path.stem == audio_path.stem

    content = result_path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT")
    assert "1" in content
    assert "00:00:00.000 --> 00:00:01.500" in content
    assert "Hello world" in content


def test_generate_subtitles_multiple_transcriptions() -> None:
    """Test generate_subtitles writes multiple groups with correct timestamps.

    Creates three sentence-level transcriptions (each ending with a period)
    so that each becomes its own group. Verifies all three appear as
    separate cues with correct numbering, timestamps, and text.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=1.0, text="First."),
        _make_transcription(start_time=1.5, end_time=3.0, text="Second."),
        _make_transcription(start_time=3.5, end_time=5.0, text="Third."),
    ]

    generator = generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    list(generator)
    result_path = audio_path.with_suffix(".vtt")

    content = result_path.read_text(encoding="utf-8")

    assert "1\n00:00:00.000 --> 00:00:01.000" in content
    assert "2\n00:00:01.500 --> 00:00:03.000" in content
    assert "3\n00:00:03.500 --> 00:00:05.000" in content
    assert "First." in content
    assert "Second." in content
    assert "Third." in content


def test_generate_subtitles_yields_progress() -> None:
    """Test generate_subtitles yields (current, total) progress tuples.

    Creates three sentence-level transcriptions (each ending with a period)
    and verifies that the generator yields exactly three tuples with
    incrementing current values and the correct total.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=1.0, text="One."),
        _make_transcription(start_time=1.0, end_time=2.0, text="Two."),
        _make_transcription(start_time=2.0, end_time=3.0, text="Three."),
    ]

    progress = list(
        generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    )

    assert len(progress) == 3
    assert progress[0] == (1, 3)
    assert progress[1] == (2, 3)
    assert progress[2] == (3, 3)


def test_generate_subtitles_empty_list_raises() -> None:
    """Test generate_subtitles raises ValueError for empty transcriptions list.

    Verifies that passing an empty list raises a ``ValueError`` with a
    message indicating the list must not be empty.
    """
    audio_path = _make_audio_file()

    try:
        list(generate_subtitles(transcriptions=[], audio_path=audio_path))
    except ValueError as e:
        assert "empty" in str(e).lower() or "must not be empty" in str(e).lower()
    else:
        assert False, "Expected ValueError to be raised"


def test_generate_subtitles_timestamp_format() -> None:
    """Test generate_subtitles produces correctly formatted VTT timestamps.

    Creates a transcription with a fractional start_time (1.2346) and
    verifies that the output VTT file contains the properly rounded
    millisecond timestamp ``00:00:01.235``.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=1.2346, end_time=2.0, text="Timestamp test")
    ]

    generator = generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    list(generator)
    result_path = audio_path.with_suffix(".vtt")

    content = result_path.read_text(encoding="utf-8")
    assert "00:00:01.235" in content


def test_generate_subtitles_special_characters_in_text() -> None:
    """Test generate_subtitles escapes HTML-like characters in VTT output.

    Creates a transcription with ``<``, ``>``, and ``&`` characters and
    verifies they are properly escaped in the generated VTT file.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=1.0, text="A & B < C > D")
    ]

    generator = generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    list(generator)
    result_path = audio_path.with_suffix(".vtt")

    content = result_path.read_text(encoding="utf-8")
    assert "&amp;" in content
    assert "&lt;" in content
    assert "&gt;" in content
    # Raw unescaped characters should NOT appear in the text line
    assert "A & B < C > D" not in content


def test_generate_subtitles_overwrites_existing_file() -> None:
    """Test generate_subtitles overwrites an existing .vtt file.

    Creates a .vtt file with old content, then runs generate_subtitles with
    different transcription data and verifies the file is overwritten with
    the new content.
    """
    audio_path = _make_audio_file()
    result_path = audio_path.with_suffix(".vtt")

    # First run: write initial content
    transcriptions_1 = [
        _make_transcription(start_time=0.0, end_time=1.0, text="Old content")
    ]
    generator_1 = generate_subtitles(
        transcriptions=transcriptions_1, audio_path=audio_path
    )
    list(generator_1)

    content_1 = result_path.read_text(encoding="utf-8")
    assert "Old content" in content_1
    assert "New content" not in content_1

    # Second run: overwrite with new content
    transcriptions_2 = [
        _make_transcription(start_time=0.0, end_time=1.0, text="New content")
    ]
    generator_2 = generate_subtitles(
        transcriptions=transcriptions_2, audio_path=audio_path
    )
    list(generator_2)

    content_2 = result_path.read_text(encoding="utf-8")
    assert "New content" in content_2
    assert "Old content" not in content_2


def test_generate_subtitles_groups_words_without_punctuation() -> None:
    """Test that consecutive words without sentence-ending punctuation are grouped.

    Two words without trailing punctuation should appear together in a
    single cue spanning from the first word's start to the last word's end.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=0.2, text="Hello"),
        _make_transcription(start_time=0.3, end_time=0.5, text="world"),
    ]

    generator = generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    list(generator)
    result_path = audio_path.with_suffix(".vtt")

    content = result_path.read_text(encoding="utf-8")

    assert "1\n00:00:00.000 --> 00:00:00.500\nHello world" in content


def test_generate_subtitles_punctuation_splits_groups() -> None:
    """Test that sentence-ending punctuation starts a new group.

    A word ending with a period should close the current group, so the
    following word starts a new cue.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=0.2, text="Hello"),
        _make_transcription(start_time=0.3, end_time=0.5, text="world."),
        _make_transcription(start_time=0.6, end_time=0.8, text="How"),
        _make_transcription(start_time=0.9, end_time=1.1, text="are"),
        _make_transcription(start_time=1.2, end_time=1.4, text="you"),
    ]

    generator = generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    list(generator)
    result_path = audio_path.with_suffix(".vtt")

    content = result_path.read_text(encoding="utf-8")

    assert "1\n00:00:00.000 --> 00:00:00.500\nHello world." in content
    assert "2\n00:00:00.600 --> 00:00:01.400\nHow are you" in content


def test_generate_subtitles_max_words_per_group() -> None:
    """Test that max_words_per_group splits long sequences.

    With max_words_per_group=2, three words without punctuation produce
    two cues: the first with 2 words and the second with 1.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=0.2, text="one"),
        _make_transcription(start_time=0.3, end_time=0.5, text="two"),
        _make_transcription(start_time=0.6, end_time=0.8, text="three"),
    ]

    generator = generate_subtitles(
        transcriptions=transcriptions, audio_path=audio_path, max_words_per_group=2
    )
    list(generator)
    result_path = audio_path.with_suffix(".vtt")

    content = result_path.read_text(encoding="utf-8")

    assert "1\n00:00:00.000 --> 00:00:00.500\none two" in content
    assert "2\n00:00:00.600 --> 00:00:00.800\nthree" in content


def test_generate_subtitles_vtt_header_and_cue_format() -> None:
    """Test generate_subtitles produces a valid WebVTT file format.

    Verifies the VTT file starts with the ``WEBVTT`` header, contains
    properly formatted cue blocks with numbering, timestamps, and text,
    and uses blank lines as separators between cues.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=0.5, text="First line."),
        _make_transcription(start_time=0.6, end_time=1.0, text="Second line."),
    ]

    generator = generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    list(generator)
    result_path = audio_path.with_suffix(".vtt")

    content = result_path.read_text(encoding="utf-8")

    # WEBVTT header
    assert content.startswith("WEBVTT")

    # Cue blocks: number, timestamp line, text, blank line
    lines = content.split("\n")

    # First line after header should be empty (blank line after WEBVTT)
    assert lines[0] == "WEBVTT"
    assert lines[1] == ""

    # Cue 1: number, timestamp, text, blank
    assert lines[2] == "1"
    assert "-->" in lines[3]
    assert "First line." in lines[4]
    assert lines[5] == ""

    # Cue 2: number, timestamp, text, blank
    assert lines[6] == "2"
    assert "-->" in lines[7]
    assert "Second line." in lines[8]
    assert lines[9] == ""


def test_generate_subtitles_cue_count_matches_groups() -> None:
    """Test that VTT cue count matches the number of word groups.

    Three words each ending with a period produce three groups and
    therefore three cues.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=0.2, text="One."),
        _make_transcription(start_time=0.5, end_time=0.7, text="Two."),
        _make_transcription(start_time=1.0, end_time=1.2, text="Three."),
    ]

    generator = generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    list(generator)
    result_path = audio_path.with_suffix(".vtt")

    content = result_path.read_text(encoding="utf-8")

    lines = content.split("\n")
    cue_numbers = [line for line in lines if line.strip().isdigit()]

    assert len(cue_numbers) == 3


def test_generate_subtitles_progress_yields_per_group() -> None:
    """Test that generate_subtitles yields one progress tuple per group.

    Four words forming two sentences (two groups) should yield exactly
    two progress tuples.
    """
    audio_path = _make_audio_file()
    transcriptions = [
        _make_transcription(start_time=0.0, end_time=0.2, text="Hello"),
        _make_transcription(start_time=0.3, end_time=0.5, text="world."),
        _make_transcription(start_time=0.6, end_time=0.8, text="Foo"),
        _make_transcription(start_time=0.9, end_time=1.1, text="Bar"),
    ]

    progress = list(
        generate_subtitles(transcriptions=transcriptions, audio_path=audio_path)
    )

    assert len(progress) == 2
    assert progress[0] == (1, 2)
    assert progress[1] == (2, 2)


# ---------------------------------------------------------------------------
# _group_transcriptions tests
# ---------------------------------------------------------------------------


def test_group_transcriptions_single_sentence() -> None:
    """Test that words without punctuation are grouped into one group."""
    transcriptions = [
        _make_transcription(text="Hello"),
        _make_transcription(text="world"),
    ]

    groups = _group_transcriptions(transcriptions, max_words_per_group=8)

    assert len(groups) == 1
    assert [t.text for t in groups[0]] == ["Hello", "world"]


def test_group_transcriptions_splits_on_punctuation() -> None:
    """Test that sentence-ending punctuation starts a new group."""
    transcriptions = [
        _make_transcription(text="Hello"),
        _make_transcription(text="world."),
        _make_transcription(text="Goodbye"),
    ]

    groups = _group_transcriptions(transcriptions, max_words_per_group=8)

    assert len(groups) == 2
    assert [t.text for t in groups[0]] == ["Hello", "world."]
    assert [t.text for t in groups[1]] == ["Goodbye"]


def test_group_transcriptions_splits_on_max_words() -> None:
    """Test that groups are split when max_words_per_group is reached."""
    transcriptions = [
        _make_transcription(text="a"),
        _make_transcription(text="b"),
        _make_transcription(text="c"),
    ]

    groups = _group_transcriptions(transcriptions, max_words_per_group=2)

    assert len(groups) == 2
    assert [t.text for t in groups[0]] == ["a", "b"]
    assert [t.text for t in groups[1]] == ["c"]


def test_group_transcriptions_empty_input() -> None:
    """Test that an empty list returns no groups."""
    groups = _group_transcriptions([], max_words_per_group=8)

    assert groups == []


def test_group_transcriptions_question_and_exclamation() -> None:
    """Test that ? and ! also split groups."""
    transcriptions = [
        _make_transcription(text="Really?"),
        _make_transcription(text="Yes!"),
        _make_transcription(text="Wow"),
    ]

    groups = _group_transcriptions(transcriptions, max_words_per_group=8)

    assert len(groups) == 3
    assert [t.text for t in groups[0]] == ["Really?"]
    assert [t.text for t in groups[1]] == ["Yes!"]
    assert [t.text for t in groups[2]] == ["Wow"]
