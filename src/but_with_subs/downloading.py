"""Downloading module for fetching video and audio from URLs.

This module provides Pydantic models for representing downloaded files
and download progress, along with a generator-based download function
that uses yt-dlp to fetch media from URLs.
"""

import collections.abc as c
import logging
from pathlib import Path

import yt_dlp
from pydantic import BaseModel

from .logging_config import log_once, logger


class File(BaseModel):
    """Model representing downloaded media files.

    Attributes:
        url:
            The original URL that was downloaded.
        video_path:
            Path to the downloaded video file, or None if not found.
        audio_path:
            Path to the downloaded audio file, or None if not found.
    """

    url: str
    video_path: Path | None
    audio_path: Path | None


class DownloadProgress(BaseModel):
    """Model representing download progress.

    Attributes:
        status:
            Human-readable status string (e.g., downloading, finished).
        current_file:
            Name of the file currently being downloaded, or None.
        percentage:
            Download progress as a float from 0.0 to 100.0.
    """

    status: str
    current_file: str | None = None
    percentage: float


def _parse_progress_info(
    info: dict, progress_hook: c.Callable[[DownloadProgress], None]
) -> str:
    """Parse yt-dlp progress info into a DownloadProgress model.

    Args:
        info:
            Progress dictionary from yt-dlp hook.
        progress_hook:
            A function to be called with progress updates.

    Returns:
        A DownloadProgress instance with extracted values.
    """
    # Build the DownloadProgress object
    fragment_index = info.get("fragment_index")
    fragment_count = info.get("fragment_count")
    if fragment_index is not None and fragment_count is not None:
        percentage = fragment_index / fragment_count
    else:
        percentage = 0.0
    status = info.get("status", "unknown")
    current_file = info.get("filename")
    progress = DownloadProgress(
        status=status, current_file=current_file, percentage=percentage
    )

    progress_hook(progress)
    return progress.model_dump_json()


def download(
    url: str, progress_hook: c.Callable[[DownloadProgress], None] = lambda _: None
) -> c.Generator[DownloadProgress, None, File]:
    """Download video and audio from a URL using yt-dlp.

    Args:
        url:
            The URL to download from.
        progress_hook:
            A function to be called with progress updates.

    Returns:
        A File model with the URLs and paths of the downloaded files.
    """
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts: dict = {
        "paths": dict(home="./data"),
        "format": "bestvideo*+bestaudio*",
        "noplaylist": True,
        "quiet": True,
        "consoletitle": False,
        "noprogress": True,
        "no_warnings": True,
        "progress_hooks": [
            lambda info: (
                log_once(
                    _parse_progress_info(info=info, progress_hook=progress_hook),
                    level=logging.INFO,
                )
            )
        ],
    }

    logger.info("Starting download from %s", url)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[no-untyped-call]
        ydl.download(url_list=[url])

    logger.info("Download completed, scanning for files in ./data/")

    video_path: Path | None = None
    audio_path: Path | None = None

    for item in sorted(data_dir.iterdir()):
        if item.is_file() and item.suffix in (".mp4", ".webm", ".mkv", ".avi", ".mov"):
            if video_path is None:
                video_path = item
        elif item.is_file() and item.suffix in (
            ".mp3",
            ".m4a",
            ".wav",
            ".flac",
            ".aac",
            ".ogg",
        ):
            if audio_path is None:
                audio_path = item

    logger.info("Download results - video: %s, audio: %s", video_path, audio_path)

    return File(url=url, video_path=video_path, audio_path=audio_path)
