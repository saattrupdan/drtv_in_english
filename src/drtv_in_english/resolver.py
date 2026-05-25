"""Resolve a DR TV URL to its HLS playlist and subtitle URLs.

Uses yt-dlp's metadata extraction (``download=False``) so no media is
ever written to disk. The returned URLs are fetched on demand by the
HLS proxy and subtitle handler in :mod:`drtv_in_english.api`.
"""

import typing as t

import yt_dlp
from pydantic import BaseModel

from .logging_config import logger

SUBTITLE_PRIORITY = ["da", "da_combined", "da-DK", "da_foreign", "a.da", "a.da-DK"]


class ResolvedMedia(BaseModel):
    """HLS + subtitle endpoints for a single DR episode."""

    title: str
    hls_url: str
    hls_headers: dict[str, str]
    subtitle_url: str
    subtitle_headers: dict[str, str]


def resolve(url: str) -> ResolvedMedia:
    """Resolve ``url`` to an HLS master playlist and Danish subtitle URL.

    Series URLs are auto-resolved to their first episode (DR returns a
    playlist; we pick the first entry).

    Args:
        url:
            DR TV URL (episode, series, or film page).

    Returns:
        :class:`ResolvedMedia` containing playable URLs and the headers
        DR's CDN requires.

    Raises:
        ValueError:
            If yt-dlp cannot extract HLS or Danish subtitles for ``url``.
    """
    url = _resolve_episode_url(url)

    opts: dict[str, t.Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": SUBTITLE_PRIORITY,
        "subtitlesformat": "vtt",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore[no-untyped-call]
        info = ydl.extract_info(url, download=False)  # type: ignore[no-untyped-call]

    if info is None:
        raise ValueError(f"yt-dlp returned no info for {url}")

    hls_url, hls_headers = _pick_hls_format(info)
    subtitle_url, subtitle_headers = _pick_subtitle(info)
    title = info.get("title") or info.get("id") or "video"

    logger.info(f"Resolved {url} → HLS {hls_url[:80]}… subs {subtitle_url[:80]}…")

    return ResolvedMedia(
        title=title,
        hls_url=hls_url,
        hls_headers=hls_headers,
        subtitle_url=subtitle_url,
        subtitle_headers=subtitle_headers,
    )


def _resolve_episode_url(url: str) -> str:
    """Return a single-episode URL, resolving series pages to their first entry.

    Returns:
        The original URL if it points at a single episode, otherwise the
        first episode found in the playlist.
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


def _pick_hls_format(info: dict) -> tuple[str, dict[str, str]]:
    """Pick the best HLS-native format from a yt-dlp info dict.

    Returns:
        ``(url, headers)`` tuple. Headers are the request headers DR's
        CDN expects (User-Agent, Referer, …).

    Raises:
        ValueError:
            If the info dict contains no HLS-native format.
    """
    formats = info.get("formats") or []
    hls = [f for f in formats if f.get("protocol") == "m3u8_native"]
    if not hls:
        hls = [
            f for f in formats if isinstance(f.get("url"), str) and ".m3u8" in f["url"]
        ]
    if not hls:
        raise ValueError("No HLS format available for this DR video")

    best = max(hls, key=lambda f: (f.get("tbr") or 0, f.get("height") or 0))
    headers = dict(best.get("http_headers") or {})
    return best["url"], headers


def _pick_subtitle(info: dict) -> tuple[str, dict[str, str]]:
    """Pick the highest-priority Danish subtitle entry.

    Returns:
        ``(url, headers)`` for the chosen subtitle track.

    Raises:
        ValueError:
            If no Danish subtitles are listed for this video.
    """
    subs = info.get("requested_subtitles") or {}
    for lang in SUBTITLE_PRIORITY:
        entry = subs.get(lang)
        if not entry:
            continue
        sub_url = entry.get("url")
        if sub_url:
            headers = dict(entry.get("http_headers") or {})
            return sub_url, headers

    available = info.get("subtitles") or {}
    for lang in SUBTITLE_PRIORITY:
        tracks = available.get(lang) or []
        vtt = next((t for t in tracks if t.get("ext") == "vtt"), None) or next(
            iter(tracks), None
        )
        if vtt and vtt.get("url"):
            return vtt["url"], dict(vtt.get("http_headers") or {})

    raise ValueError("No Danish subtitles available for this video")
