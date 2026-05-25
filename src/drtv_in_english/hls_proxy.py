"""Per-job HLS proxy.

DR's CDN segments require specific request headers (User-Agent, Referer)
that the browser will not send if it fetches them directly. The proxy
re-attaches those headers, and rewrites every URI inside playlists so
the browser always talks to us, never to DR directly.

URLs are never embedded in proxy URLs — each upstream URL is registered
under a random opaque token in the job's :class:`HlsRegistry`, and the
browser fetches ``/api/stream/<job>/p/<token>``. This makes SSRF
impossible: the proxy only follows URLs it has previously registered.
"""

import re
import secrets
import urllib.parse

import httpx
from pydantic import BaseModel


class HlsRegistry(BaseModel):
    """Mapping of opaque tokens to upstream URLs for one job.

    Attributes:
        urls:
            Token → upstream URL. Populated as playlists are parsed.
        headers:
            Request headers attached to every upstream fetch.
    """

    urls: dict[str, str] = {}
    headers: dict[str, str] = {}

    def register(self, url: str) -> str:
        """Add ``url`` to the registry and return its token.

        Returns:
            A short URL-safe token identifying ``url``.
        """
        for token, existing in self.urls.items():
            if existing == url:
                return token
        token = secrets.token_urlsafe(12)
        self.urls[token] = url
        return token

    def resolve(self, token: str) -> str | None:
        """Return the upstream URL for ``token``, or None if unknown.

        Returns:
            The previously-registered URL, or None.
        """
        return self.urls.get(token)


_URI_ATTR = re.compile(r'URI="([^"]+)"')


def rewrite_playlist(
    *, body: str, base_url: str, registry: HlsRegistry, proxy_prefix: str
) -> str:
    """Rewrite every URI in an m3u8 playlist to point at the proxy.

    Args:
        body:
            Raw playlist text.
        base_url:
            URL the playlist was fetched from — used to resolve relative
            URIs to absolute upstream URLs before registration.
        registry:
            Job's registry; new URLs are added here.
        proxy_prefix:
            Path prefix the browser uses to reach the proxy
            (e.g. ``/api/stream/<job>/p/``). Tokens are appended.

    Returns:
        Rewritten playlist text.
    """

    def _proxy(url: str) -> str:
        absolute = urllib.parse.urljoin(base_url, url)
        return proxy_prefix + registry.register(absolute)

    out_lines: list[str] = []
    for line in body.splitlines():
        if not line:
            out_lines.append(line)
            continue
        if line.startswith("#"):
            if 'URI="' in line:
                line = _URI_ATTR.sub(lambda m: f'URI="{_proxy(m.group(1))}"', line)
            out_lines.append(line)
        else:
            out_lines.append(_proxy(line.strip()))
    return "\n".join(out_lines) + "\n"


async def fetch_and_rewrite(
    *, client: httpx.AsyncClient, url: str, registry: HlsRegistry, proxy_prefix: str
) -> tuple[bytes, str]:
    """Fetch a playlist and rewrite its URIs.

    Returns:
        ``(rewritten_bytes, content_type)``.
    """
    response = await client.get(url, headers=registry.headers)
    response.raise_for_status()
    body = response.text
    rewritten = rewrite_playlist(
        body=body, base_url=url, registry=registry, proxy_prefix=proxy_prefix
    )
    return rewritten.encode("utf-8"), "application/vnd.apple.mpegurl"


def is_playlist(url: str, content_type: str | None) -> bool:
    """Return True if ``url`` or ``content_type`` looks like an HLS playlist.

    Returns:
        True for ``.m3u8`` URLs or ``application/vnd.apple.mpegurl``
        content types.
    """
    if url.split("?", 1)[0].endswith(".m3u8"):
        return True
    if content_type and "mpegurl" in content_type.lower():
        return True
    return False
