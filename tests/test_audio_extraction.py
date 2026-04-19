"""Tests for the audio extraction module.

This module contains comprehensive tests for the extract_audio function,
verifying ffmpeg command construction and output path handling.
"""

import pathlib
import unittest.mock as um

from but_with_subs.audio_extraction import extract_audio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video_path(name: str, suffix: str) -> pathlib.Path:
    """Create a real pathlib.Path for a video file.

    Args:
        name:
            The file name without extension.
        suffix:
            The file extension (e.g. ".mp4").

    Returns:
        A pathlib.Path for the video file.
    """
    return pathlib.Path(f"{name}{suffix}")


def _make_wav_path(name: str) -> pathlib.Path:
    """Create a real pathlib.Path for a WAV output file.

    Args:
        name:
            The file name without extension.

    Returns:
        A pathlib.Path for the WAV file.
    """
    return pathlib.Path(f"{name}.wav")


# ---------------------------------------------------------------------------
# extract_audio() function tests
# ---------------------------------------------------------------------------


def test_extract_audio_calls_ffmpeg_with_correct_command() -> None:
    """Test that extract_audio constructs and runs the correct ffmpeg command.

    Mocks subprocess.run and verifies the command arguments are correct.
    """
    video_path = _make_video_path("video", ".mp4")

    with um.patch("subprocess.run") as mock_run:
        extract_audio(video_path=video_path)

    mock_run.assert_called_once_with(
        args=["ffmpeg", "-i", str(video_path), "-vn", "video.wav"], check=True
    )


def test_extract_audio_returns_output_path() -> None:
    """Test that extract_audio returns the correct output WAV path.

    Verifies that the returned path has the correct suffix and parent.
    """
    video_path = _make_video_path("video", ".mp4")

    with um.patch("subprocess.run"):
        result = extract_audio(video_path=video_path)

    assert result.name == "video.wav"
    assert result.suffix == ".wav"


def test_extract_audio_output_path_same_directory() -> None:
    """Test that the output WAV file is in the same directory as input.

    Verifies that the parent directory of the output path matches the
    input video path's parent directory.
    """
    video_path = pathlib.Path("/data/video.mp4")

    with um.patch("subprocess.run"):
        result = extract_audio(video_path=video_path)

    assert result.parent == pathlib.Path("/data")


def test_extract_audio_with_nested_path() -> None:
    """Test extract_audio with a deeply nested video path.

    Verifies that the output WAV file is created in the same nested
    directory structure.
    """
    video_path = pathlib.Path("/data/movies/2024/deep_video.mkv")

    with um.patch("subprocess.run"):
        result = extract_audio(video_path=video_path)

    assert result.parent == pathlib.Path("/data/movies/2024")
    assert result.name == "deep_video.wav"


def test_extract_audio_replaces_extension() -> None:
    """Test that extract_audio replaces the video extension with .wav.

    Verifies that different video extensions (mp4, mkv, avi) are correctly
    replaced with .wav.
    """
    for ext in (".mp4", ".mkv", ".avi", ".webm"):
        video_path = pathlib.Path(f"movie{ext}")

        with um.patch("subprocess.run"):
            result = extract_audio(video_path=video_path)

        assert result.suffix == ".wav"


def test_extract_audio_with_special_characters() -> None:
    """Test extract_audio with special characters in the filename.

    Verifies that filenames with spaces, brackets, and other special
    characters are handled correctly.
    """
    video_path = pathlib.Path("/data/my movies/video [2024].mp4")

    with um.patch("subprocess.run"):
        result = extract_audio(video_path=video_path)

    assert result.name == "video [2024].wav"
    assert result.parent == pathlib.Path("/data/my movies")


def test_extract_audio_with_unicode_characters() -> None:
    """Test extract_audio with unicode characters in the filename.

    Verifies that filenames with unicode characters (e.g. accented
    letters, non-Latin scripts) are handled correctly.
    """
    video_path = pathlib.Path("/data/films/film_avec_é.mp4")

    with um.patch("subprocess.run"):
        result = extract_audio(video_path=video_path)

    assert result.name == "film_avec_é.wav"


def test_extract_audio_with_dot_in_name() -> None:
    """Test extract_audio with dots in the filename.

    Verifies that Path.with_suffix correctly replaces only the final
    extension, not intermediate dots.
    """
    video_path = pathlib.Path("/data/video.2024.mp4")

    with um.patch("subprocess.run"):
        result = extract_audio(video_path=video_path)

    assert result.name == "video.2024.wav"


def test_extract_audio_with_no_extension() -> None:
    """Test extract_audio with a file that has no extension.

    Verifies that the .wav suffix is appended to the filename.
    """
    video_path = pathlib.Path("/data/video")

    with um.patch("subprocess.run"):
        result = extract_audio(video_path=video_path)

    assert result.name == "video.wav"


def test_extract_audio_ffmpeg_command_uses_correct_flags() -> None:
    """Test that the ffmpeg command includes the -vn flag.

    Verifies that the -vn flag (disable video) is present in the
    command to ensure only audio is extracted.
    """
    video_path = _make_video_path("video", ".mp4")

    with um.patch("subprocess.run") as mock_run:
        extract_audio(video_path=video_path)

    call_kwargs = mock_run.call_args.kwargs
    assert "-vn" in call_kwargs["args"]


def test_extract_audio_ffmpeg_input_flag() -> None:
    """Test that the ffmpeg command includes the -i input flag.

    Verifies that the -i flag is correctly placed before the input path.
    """
    video_path = _make_video_path("video", ".mp4")

    with um.patch("subprocess.run") as mock_run:
        extract_audio(video_path=video_path)

    call_kwargs = mock_run.call_args.kwargs
    args = call_kwargs["args"]
    assert args.index("-i") == 1
    assert args[2] == str(video_path)


def test_extract_audio_ffmpeg_output_order() -> None:
    """Test that the output path is the last argument in the command.

    Verifies that the output WAV file path is positioned correctly
    as the final argument in the ffmpeg command.
    """
    video_path = _make_video_path("video", ".mp4")

    with um.patch("subprocess.run") as mock_run:
        extract_audio(video_path=video_path)

    call_kwargs = mock_run.call_args.kwargs
    args = call_kwargs["args"]
    assert args[-1] == "video.wav"
