"""Download a DR TV video + its Danish subtitles via yt-dlp.

For series URLs (``/drtv/serie/...``), yt-dlp returns a playlist; we pick
the first entry and download that episode only.
"""

import collections.abc as c
import typing as t
from pathlib import Path

import yt_dlp

from .constants import DATA_DIR
from .data_models import DownloadProgress, File
from .logging_config import logger

# Priority order: real Danish tracks first, then auto-generated. yt-dlp
# treats these as patterns and downloads every match, so we pick the
# highest-priority result ourselves after extraction.
SUBTITLE_PRIORITY = ["da", "da_combined", "da-DK", "da_foreign", "a.da", "a.da-DK"]

VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".avi", ".mov")


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
    progress_hook(
        DownloadProgress(
            status=status, current_file=current_file, percentage=percentage
        )
    )


def _noop_progress(_: DownloadProgress) -> None:
    """No-op progress hook used as default when no callback is provided."""


def _resolve_episode_url(url: str) -> str:
    """Return a single-episode URL.

    If ``url`` is a series page, yt-dlp returns a playlist; we pick the
    first entry's URL and recurse via that. Single episodes are returned
    unchanged.
    """
    probe_opts: dict[str, t.Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
    }
    with yt_dlp.YoutubeDL(probe_opts) as ydl:  # type: ignore[no-untyped-call]
        info = ydl.extract_info(url, download=False)  # type: ignore[no-untyped-call]

    if info is None:
        return url

    entries = info.get("entries")
    if not entries:
        return url

    first = next(iter(entries), None)
    if first is None:
        return url

    episode_url = first.get("url") or first.get("webpage_url")
    if not episode_url:
        return url

    logger.info(f"Series detected — picked first episode: {episode_url}")
    return episode_url


def download(
    url: str, progress_hook: c.Callable[[DownloadProgress], None] = _noop_progress
) -> File:
    """Download video + Danish subtitles for a DR URL.

    Returns:
        File with paths to the downloaded video and subtitles.
    """
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    url = _resolve_episode_url(url)

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
        "subtitleslangs": SUBTITLE_PRIORITY,
        "subtitlesformat": "vtt",
        "progress_hooks": [
            lambda info: _parse_progress_info(info=info, progress_hook=progress_hook)
        ],
    }

    logger.info(f"Starting download from {url}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[no-untyped-call]
        info = ydl.extract_info(url, download=True)  # type: ignore[no-untyped-call]

    downloaded_paths: list[Path] = []
    for entry in info.get("requested_downloads") or []:
        filepath = entry.get("filepath") or entry.get("_filename")
        if filepath:
            downloaded_paths.append(Path(filepath))
    if not downloaded_paths:
        downloaded_paths.append(Path(ydl.prepare_filename(info)))  # type: ignore[no-untyped-call]

    video_path: Path | None = next(
        (p for p in downloaded_paths if p.suffix in VIDEO_SUFFIXES), None
    )
    subtitles_path = _pick_subtitle(info=info, priority=SUBTITLE_PRIORITY)

    logger.info(f"Download results - video: {video_path}, subtitles: {subtitles_path}")

    return File(url=url, video_path=video_path, subtitles_path=subtitles_path)


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
