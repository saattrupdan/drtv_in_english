"""Downloading module for fetching video and audio from URLs.

This module provides Pydantic models for representing downloaded files
and download progress, along with a generator-based download function
that uses yt-dlp to fetch media from URLs.
"""

import collections.abc as c
import typing as t
from pathlib import Path

import yt_dlp

from .constants import DATA_DIR
from .data_models import DownloadProgress, File
from .logging_config import logger


def _parse_progress_info(
    info: dict, progress_hook: c.Callable[[DownloadProgress], None]
) -> None:
    """Parse yt-dlp progress info into a DownloadProgress model."""
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


def _noop_progress(_: DownloadProgress) -> None:
    """No-op progress hook used as default when no callback is provided."""


def download(
    url: str, progress_hook: c.Callable[[DownloadProgress], None] = _noop_progress
) -> File:
    """Download video and audio from a URL using yt-dlp.

    Returns:
        File model with URLs and paths of downloaded files.
    """
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Priority order: real Danish tracks first, then auto-generated. yt-dlp
    # treats these as patterns and downloads every match, so we pick the
    # highest-priority result ourselves after extraction.
    subtitle_priority = [
        "da",
        "da_combined",
        "da-DK",
        "da_foreign",
        "a.da",
        "a.da-DK",
    ]

    ydl_opts: dict[str, t.Any] = {
        "paths": dict(home=DATA_DIR),
        "format": "bestvideo*+bestaudio*",
        "noplaylist": True,
        "quiet": True,
        "consoletitle": False,
        "noprogress": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": subtitle_priority,
        "subtitlesformat": "vtt",
        "progress_hooks": [
            lambda info: _parse_progress_info(info=info, progress_hook=progress_hook)
        ],
    }

    logger.info(f"Starting download from {url}")

    video_suffixes = (".mp4", ".webm", ".mkv", ".avi", ".mov")
    audio_suffixes = (".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[no-untyped-call]
        info = ydl.extract_info(url, download=True)  # type: ignore[no-untyped-call]

    downloaded_paths: list[Path] = []
    for entry in info.get("requested_downloads") or []:
        filepath = entry.get("filepath") or entry.get("_filename")
        if filepath:
            downloaded_paths.append(Path(filepath))
    if not downloaded_paths:
        downloaded_paths.append(Path(ydl.prepare_filename(info)))  # type: ignore[no-untyped-call]

    video_path: Path | None = None
    audio_path: Path | None = None
    for path in downloaded_paths:
        if video_path is None and path.suffix in video_suffixes:
            video_path = path
        elif audio_path is None and path.suffix in audio_suffixes:
            audio_path = path

    subtitles_path = _pick_subtitle(
        info=info, priority=subtitle_priority
    )

    logger.info(
        f"Download results - video: {video_path}, audio: {audio_path}, "
        f"subtitles: {subtitles_path}"
    )

    return File(
        url=url,
        video_path=video_path,
        audio_path=audio_path,
        subtitles_path=subtitles_path,
    )


def _pick_subtitle(info: dict, priority: list[str]) -> Path | None:
    """Return the highest-priority downloaded subtitle file, if any."""
    requested = info.get("requested_subtitles") or {}
    for lang in priority:
        entry = requested.get(lang)
        if not entry:
            continue
        filepath = entry.get("filepath")
        if filepath and Path(filepath).exists():
            return Path(filepath)
    return None
