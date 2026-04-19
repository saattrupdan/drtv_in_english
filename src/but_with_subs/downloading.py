"""Downloading module for fetching video and audio from URLs.

This module provides Pydantic models for representing downloaded files
and download progress, along with a generator-based download function
that uses yt-dlp to fetch media from URLs.
"""

import collections.abc as c
import queue
from pathlib import Path

import yt_dlp
from pydantic import BaseModel

from .logging_config import logger


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
        percentage:
            Download progress as a float from 0.0 to 100.0.
        status:
            Human-readable status string (e.g., downloading, finished).
        current_file:
            Name of the file currently being downloaded, or None.
    """

    percentage: float
    status: str
    current_file: str | None = None


def _parse_progress_info(
    info: dict, progress_queue: queue.Queue[DownloadProgress] | None = None
) -> DownloadProgress:
    """Parse yt-dlp progress info into a DownloadProgress model.

    Args:
        info:
            Progress dictionary from yt-dlp hook.
        progress_queue:
            Optional queue to put the progress update into.

    Returns:
        A DownloadProgress instance with extracted values.
    """
    percentage: float = 0.0
    if info.get("_percent_str") is not None:
        str_val = info["_percent_str"].strip().rstrip("%")
        try:
            percentage = float(str_val)
        except ValueError:
            percentage = 0.0
    elif info.get("_percent") is not None:
        percentage = float(info["_percent"])

    status = info.get("_status", "unknown")

    current_file: str | None = None
    if info.get("_filename") is not None:
        current_file = str(info["_filename"])

    progress = DownloadProgress(
        percentage=percentage, status=status, current_file=current_file
    )

    if progress_queue is not None:
        progress_queue.put(progress)

    return progress


def download(url: str) -> c.Generator[DownloadProgress, None, File]:
    """Download video and audio from a URL using yt-dlp.

    Args:
        url:
            The URL to download from.

    Yields:
        Progress updates as the download progresses.

    Returns:
        A File model with the URLs and paths of the downloaded files.
    """
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)

    progress_queue: queue.Queue[DownloadProgress] = queue.Queue()

    ydl_opts: dict = {
        "outtmpl": {
            "video": "./data/%(title)s.%(ext)s",
            "audio": "./data/%(title)s.%(ext)s",
        },
        "format": "bestvideo+bestaudio/best",
        "noplaylist": True,
        "hooks": {
            "progress_hooks": [
                lambda info: (
                    logger.info(
                        "Progress: %s", _parse_progress_info(info, progress_queue)
                    )
                    or _parse_progress_info(info, progress_queue)
                )
            ]
        },
    }

    logger.info("Starting download from %s", url)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

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

    while not progress_queue.empty():
        yield progress_queue.get()

    return File(url=url, video_path=video_path, audio_path=audio_path)
