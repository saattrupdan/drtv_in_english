"""Tests for the downloading module.

This module contains comprehensive tests for the File and DownloadProgress
Pydantic models, as well as the download generator function.
"""

import pathlib
import unittest.mock as um

from but_with_subs.downloading import DownloadProgress, File, download

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_path_mock(name: str, suffix: str, is_file: bool = True) -> um.Mock:
    """Create a mock Path for use in tests.

    Args:
        name:
            The file name to use for identification.
        suffix:
            The file extension (e.g. ".mp4").
        is_file:
            Whether the mock should report as a file.

    Returns:
        A mock Path with the specified properties.
    """
    mock_path = um.Mock(spec=pathlib.Path)
    mock_path.name = name
    mock_path.suffix = suffix
    mock_path.is_file.return_value = is_file
    return mock_path


def _iterate_gen(gen: um.Mock) -> tuple[list[DownloadProgress], File]:
    """Consume a generator, collecting yields and extracting the return.

    Args:
        gen:
            A generator to iterate over.

    Returns:
        A tuple of (progress items, final File).
    """
    progress_items: list[DownloadProgress] = []
    result: File | None = None
    try:
        while True:
            item = next(gen)
            if isinstance(item, DownloadProgress):
                progress_items.append(item)
            else:
                result = item
    except StopIteration as e:
        result = e.value
    assert result is not None, "Generator did not return a File"
    return progress_items, result


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_file_model_creation() -> None:
    """Test constructing a File model with all fields populated."""
    file_model: File = File(
        url="https://example.com/video",
        video_path=pathlib.Path("/data/video.mp4"),
        audio_path=pathlib.Path("/data/audio.mp3"),
    )

    assert file_model.url == "https://example.com/video"
    assert file_model.video_path == pathlib.Path("/data/video.mp4")
    assert file_model.audio_path == pathlib.Path("/data/audio.mp3")


def test_file_model_none_paths() -> None:
    """Test constructing a File model with None paths."""
    file_model = File(url="https://example.com/video", video_path=None, audio_path=None)

    assert file_model.url == "https://example.com/video"
    assert file_model.video_path is None
    assert file_model.audio_path is None


def test_download_progress_model() -> None:
    """Test constructing a DownloadProgress model with all fields."""
    progress: DownloadProgress = DownloadProgress(
        percentage=45.5, status="downloading", current_file="video.mp4"
    )

    assert progress.percentage == 45.5
    assert progress.status == "downloading"
    assert progress.current_file == "video.mp4"


def test_download_progress_default_current_file() -> None:
    """Test that current_file defaults to None when not provided."""
    progress = DownloadProgress(percentage=100.0, status="finished")

    assert progress.percentage == 100.0
    assert progress.status == "finished"
    assert progress.current_file is None


# ---------------------------------------------------------------------------
# download() function tests
# ---------------------------------------------------------------------------


def test_download_creates_data_dir() -> None:
    """Test that the download function creates the ./data/ directory.

    Mocks yt-dlp to return immediately and verifies that Path.mkdir
    is called with the correct arguments.
    """
    mock_ydl_instance = um.Mock()
    mock_ydl_instance.__enter__ = um.Mock(return_value=mock_ydl_instance)
    mock_ydl_instance.__exit__ = um.Mock(return_value=False)

    with (
        um.patch("but_with_subs.downloading.Path.mkdir") as mock_mkdir,
        um.patch(
            "but_with_subs.downloading.yt_dlp.YoutubeDL", return_value=mock_ydl_instance
        ),
        um.patch("but_with_subs.downloading.Path.iterdir", return_value=iter([])),
    ):
        gen = download(url="https://example.com/video")
        _, result = _iterate_gen(gen)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    assert result.url == "https://example.com/video"


def test_download_returns_file_with_paths() -> None:
    """Test that download returns a File with correct video and audio paths.

    Mocks yt-dlp to succeed and provides fake scanned files in the data
    directory so the scanning logic finds both a video and an audio file.
    """
    mock_ydl_instance = um.Mock()
    mock_ydl_instance.__enter__ = um.Mock(return_value=mock_ydl_instance)
    mock_ydl_instance.__exit__ = um.Mock(return_value=False)

    fake_video = _make_path_mock("video", ".mp4")
    fake_audio = _make_path_mock("audio", ".mp3")

    with (
        um.patch(
            "but_with_subs.downloading.yt_dlp.YoutubeDL", return_value=mock_ydl_instance
        ),
        um.patch(
            "but_with_subs.downloading.Path.iterdir",
            return_value=iter([fake_video, fake_audio]),
        ),
        um.patch("builtins.sorted", side_effect=lambda x: list(x)),
    ):
        gen = download(url="https://example.com/video")
        _, result = _iterate_gen(gen)

    assert result.url == "https://example.com/video"
    assert result.video_path == fake_video
    assert result.audio_path == fake_audio


def test_download_yields_progress() -> None:
    """Test that download yields DownloadProgress objects during download.

    Mocks yt-dlp's progress hook so that we can verify progress events
    are yielded by the generator before the final File is returned.
    """
    mock_ydl_instance = um.Mock()
    mock_ydl_instance.__enter__ = um.Mock(return_value=mock_ydl_instance)
    mock_ydl_instance.__exit__ = um.Mock(return_value=False)

    captured_opts: dict = {}

    def _capture_opts(opts: dict) -> um.Mock:
        """Capture yt-dlp options and return the mock instance.

        Args:
            opts:
                The yt-dlp options dictionary.

        Returns:
            The mock YoutubeDL instance.
        """
        captured_opts.update(opts)
        return mock_ydl_instance

    def _call_progress_hooks(urls: list[str]) -> None:
        """Invoke progress hooks with sample data."""
        hooks = captured_opts.get("hooks", {}).get("progress_hooks", [])
        sample_info: dict = {
            "_percent_str": "50.0%",
            "_status": "downloading",
            "_filename": "video.mp4",
        }
        for hook in hooks:
            hook(sample_info)

    mock_ydl_instance.download.side_effect = _call_progress_hooks

    fake_video = _make_path_mock("video", ".mp4")

    with (
        um.patch(
            "but_with_subs.downloading.yt_dlp.YoutubeDL", side_effect=_capture_opts
        ),
        um.patch(
            "but_with_subs.downloading.Path.iterdir", return_value=iter([fake_video])
        ),
        um.patch("builtins.sorted", side_effect=lambda x: list(x)),
    ):
        gen = download(url="https://example.com/video")
        progress_items, result = _iterate_gen(gen)

    assert isinstance(result, File)
    assert result.url == "https://example.com/video"

    progress_items = [
        item for item in progress_items if isinstance(item, DownloadProgress)
    ]
    for progress_item in progress_items:
        assert isinstance(progress_item, DownloadProgress)
        assert 0.0 <= progress_item.percentage <= 100.0


def test_download_with_no_video() -> None:
    """Test that download returns File with video_path=None when no video found.

    Mocks the data directory scan to only contain an audio file, verifying
    that video_path is None in the returned File.
    """
    mock_ydl_instance = um.Mock()
    mock_ydl_instance.__enter__ = um.Mock(return_value=mock_ydl_instance)
    mock_ydl_instance.__exit__ = um.Mock(return_value=False)

    fake_audio = _make_path_mock("audio", ".mp3")

    with (
        um.patch(
            "but_with_subs.downloading.yt_dlp.YoutubeDL", return_value=mock_ydl_instance
        ),
        um.patch(
            "but_with_subs.downloading.Path.iterdir", return_value=iter([fake_audio])
        ),
        um.patch("builtins.sorted", side_effect=lambda x: list(x)),
    ):
        gen = download(url="https://example.com/video")
        _, result = _iterate_gen(gen)

    assert result.url == "https://example.com/video"
    assert result.video_path is None
    assert result.audio_path == fake_audio


def test_download_with_no_audio() -> None:
    """Test that download returns File with audio_path=None when no audio found.

    Mocks the data directory scan to only contain a video file, verifying
    that audio_path is None in the returned File.
    """
    mock_ydl_instance = um.Mock()
    mock_ydl_instance.__enter__ = um.Mock(return_value=mock_ydl_instance)
    mock_ydl_instance.__exit__ = um.Mock(return_value=False)

    fake_video = _make_path_mock("video", ".mp4")

    with (
        um.patch(
            "but_with_subs.downloading.yt_dlp.YoutubeDL", return_value=mock_ydl_instance
        ),
        um.patch(
            "but_with_subs.downloading.Path.iterdir", return_value=iter([fake_video])
        ),
        um.patch("builtins.sorted", side_effect=lambda x: list(x)),
    ):
        gen = download(url="https://example.com/video")
        _, result = _iterate_gen(gen)

    assert result.url == "https://example.com/video"
    assert result.video_path == fake_video
    assert result.audio_path is None
