"""Tests for the downloading module."""

import pathlib
import unittest.mock as um

from danglish.data_models import DownloadProgress, File
from danglish.downloading import _noop_progress, _parse_progress_info, download


def _make_ydl_mock(
    requested_downloads: list[dict[str, str]] | None = None,
) -> um.MagicMock:
    mock = um.MagicMock()
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = False
    mock.extract_info.return_value = {"requested_downloads": requested_downloads or []}
    return mock


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_file_model_creation() -> None:
    """Test constructing a File model with all fields populated."""
    file_model = File(
        url="https://example.com/video",
        video_path=pathlib.Path("/data/video.mp4"),
        subtitles_path=pathlib.Path("/data/video.vtt"),
    )

    assert file_model.url == "https://example.com/video"
    assert file_model.video_path == pathlib.Path("/data/video.mp4")
    assert file_model.subtitles_path == pathlib.Path("/data/video.vtt")


def test_file_model_none_paths() -> None:
    """Test constructing a File model with None paths."""
    file_model = File(url="https://example.com/video", video_path=None)
    assert file_model.video_path is None
    assert file_model.subtitles_path is None


def test_download_progress_model() -> None:
    """Test constructing a DownloadProgress model with all fields."""
    progress = DownloadProgress(
        percentage=0.5, status="downloading", current_file="video.mp4"
    )
    assert progress.percentage == 0.5
    assert progress.status == "downloading"
    assert progress.current_file == "video.mp4"


def test_download_progress_default_current_file() -> None:
    """Test that current_file defaults to None when not provided."""
    progress = DownloadProgress(percentage=1.0, status="finished")
    assert progress.current_file is None


# ---------------------------------------------------------------------------
# download() function tests
# ---------------------------------------------------------------------------


def test_download_returns_file_with_paths(tmp_path: pathlib.Path) -> None:
    """Test that download returns a File with the correct video path."""
    mock_ydl = _make_ydl_mock(requested_downloads=[{"filepath": "/tmp/video.mp4"}])

    with (
        um.patch("danglish.downloading._resolve_episode_url", side_effect=lambda u: u),
        um.patch("danglish.downloading.yt_dlp.YoutubeDL", return_value=mock_ydl),
        um.patch("danglish.downloading.Path.mkdir"),
    ):
        result = download(url="https://example.com/video")

    assert result.url == "https://example.com/video"
    assert result.video_path == pathlib.Path("/tmp/video.mp4")
    assert result.subtitles_path is None


def test_download_with_no_video() -> None:
    """Test that download returns File with video_path=None when no video found."""
    mock_ydl = _make_ydl_mock(requested_downloads=[{"filepath": "/tmp/audio.mp3"}])

    with (
        um.patch("danglish.downloading._resolve_episode_url", side_effect=lambda u: u),
        um.patch("danglish.downloading.yt_dlp.YoutubeDL", return_value=mock_ydl),
        um.patch("danglish.downloading.Path.mkdir"),
    ):
        result = download(url="https://example.com/video")

    assert result.video_path is None


def test_download_progress_hook_wired() -> None:
    """Test that download passes a progress hook through to yt-dlp."""
    mock_ydl = _make_ydl_mock(requested_downloads=[{"filepath": "/tmp/video.mp4"}])

    with (
        um.patch("danglish.downloading._resolve_episode_url", side_effect=lambda u: u),
        um.patch(
            "danglish.downloading.yt_dlp.YoutubeDL", return_value=mock_ydl
        ) as mock_ctor,
        um.patch("danglish.downloading.Path.mkdir"),
    ):
        download(url="https://example.com/video", progress_hook=lambda _: None)

    ydl_opts = mock_ctor.call_args.args[0]
    assert "progress_hooks" in ydl_opts
    assert len(ydl_opts["progress_hooks"]) == 1


# ---------------------------------------------------------------------------
# _parse_progress_info() tests
# ---------------------------------------------------------------------------


def test_parse_progress_info_with_fragments() -> None:
    """Verify percentage = fragment_index / fragment_count."""
    updates: list[DownloadProgress] = []
    info = {
        "fragment_index": 5,
        "fragment_count": 10,
        "status": "downloading",
        "filename": "video.mp4",
    }
    _parse_progress_info(info=info, progress_hook=updates.append)
    assert updates[0].percentage == 0.5
    assert updates[0].status == "downloading"
    assert updates[0].current_file == "video.mp4"


def test_parse_progress_info_without_fragments() -> None:
    """Verify percentage = 0.0 when fragment info is missing."""
    updates: list[DownloadProgress] = []
    _parse_progress_info(
        info={"status": "downloading", "filename": "v.mp4"},
        progress_hook=updates.append,
    )
    assert updates[0].percentage == 0.0


def test_parse_progress_info_missing_status() -> None:
    """Verify status defaults to 'unknown' when missing."""
    updates: list[DownloadProgress] = []
    _parse_progress_info(
        info={"fragment_index": 3, "fragment_count": 6}, progress_hook=updates.append
    )
    assert updates[0].status == "unknown"


def test_noop_progress_callable() -> None:
    """The default no-op hook does not raise."""
    _noop_progress(DownloadProgress(percentage=0.5, status="downloading"))
