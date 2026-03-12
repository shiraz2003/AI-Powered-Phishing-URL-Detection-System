"""URL preprocessing: normalisation and cleaning."""

from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    """Return a lowercase, scheme-normalised copy of *url*.

    Steps applied:
    1. Strip leading/trailing whitespace.
    2. Force the scheme to lower-case.
    3. Force the netloc (host + port) to lower-case.
    4. Remove a trailing slash from the path when it is the only path
       component (e.g. ``https://example.com/`` → ``https://example.com``).

    Parameters
    ----------
    url:
        Raw URL string.
    """
    url = url.strip()
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") if parsed.path == "/" else parsed.path
    normalised = urlunparse((scheme, netloc, path, parsed.params, parsed.query, parsed.fragment))
    return normalised


def extract_domain(url: str) -> str:
    """Return the registered domain (netloc) from *url*.

    Parameters
    ----------
    url:
        A URL string (``http://…`` or ``https://…``).
    """
    return urlparse(url.strip()).netloc.lower()
