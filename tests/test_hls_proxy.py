"""Tests for the HLS proxy module."""

from danglish.hls_proxy import HlsRegistry, is_playlist, rewrite_playlist


def test_registry_returns_same_token_for_same_url() -> None:
    """Re-registering an URL should return the existing token."""
    registry = HlsRegistry()
    token1 = registry.register("https://cdn.example.com/seg1.ts")
    token2 = registry.register("https://cdn.example.com/seg1.ts")
    assert token1 == token2


def test_registry_resolve_missing_token_returns_none() -> None:
    """Unknown tokens resolve to None — the proxy returns 404."""
    registry = HlsRegistry()
    assert registry.resolve("not-a-real-token") is None


def test_rewrite_master_playlist_rewrites_variants_and_uri_attrs() -> None:
    """Variant URIs (plain) and EXT-X-MEDIA URI="..." attributes are rewritten."""
    registry = HlsRegistry()
    body = (
        "#EXTM3U\n"
        "#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"a\",URI=\"audio.m3u8\"\n"
        "#EXT-X-STREAM-INF:BANDWIDTH=1000000\n"
        "variant1.m3u8\n"
    )
    out = rewrite_playlist(
        body=body,
        base_url="https://cdn.example.com/show/master.m3u8",
        registry=registry,
        proxy_prefix="/api/stream/JOB/p/",
    )
    # No upstream URLs remain
    assert "audio.m3u8" not in out
    assert "variant1.m3u8" not in out
    assert "/api/stream/JOB/p/" in out
    # Both upstream URLs got registered, resolved relative to base_url
    assert "https://cdn.example.com/show/audio.m3u8" in registry.urls.values()
    assert "https://cdn.example.com/show/variant1.m3u8" in registry.urls.values()


def test_rewrite_media_playlist_rewrites_segments_and_map() -> None:
    """Segment URIs and EXT-X-MAP URI attributes get rewritten."""
    registry = HlsRegistry()
    body = (
        "#EXTM3U\n"
        "#EXT-X-MAP:URI=\"init.mp4\"\n"
        "#EXTINF:6.0,\n"
        "seg0.m4s\n"
        "#EXTINF:6.0,\n"
        "seg1.m4s\n"
    )
    out = rewrite_playlist(
        body=body,
        base_url="https://cdn.example.com/show/variant.m3u8",
        registry=registry,
        proxy_prefix="/api/stream/JOB/p/",
    )
    assert out.count("/api/stream/JOB/p/") == 3
    assert "https://cdn.example.com/show/init.mp4" in registry.urls.values()
    assert "https://cdn.example.com/show/seg0.m4s" in registry.urls.values()


def test_is_playlist_detects_m3u8_extension_and_content_type() -> None:
    """A playlist is anything with .m3u8 or an Apple MPEG-URL content type."""
    assert is_playlist("https://x/master.m3u8", None)
    assert is_playlist("https://x/master.m3u8?token=foo", None)
    assert is_playlist("https://x/foo", "application/vnd.apple.mpegurl")
    assert not is_playlist("https://x/seg.ts", "video/mp2t")
