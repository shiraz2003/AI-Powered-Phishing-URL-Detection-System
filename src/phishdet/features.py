"""Feature extraction from URLs for phishing detection."""

import re
from urllib.parse import urlparse
from typing import Dict, List

import numpy as np
import pandas as pd

# Shortener services commonly abused in phishing campaigns
_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "buff.ly", "ift.tt", "is.gd", "clck.ru", "shorte.st",
}

# Keywords frequently found in phishing URLs
_SUSPICIOUS_KEYWORDS = [
    "secure", "login", "signin", "verify", "update", "account",
    "banking", "confirm", "paypal", "ebay", "amazon", "apple",
    "microsoft", "support", "alert", "urgent", "free", "winner",
    "click", "claim", "refund",
]

_IP_PATTERN = re.compile(
    r"(\b(?:\d{1,3}\.){3}\d{1,3}\b)"
)


def _has_ip_address(url: str) -> int:
    """Return 1 if the URL contains a bare IP address, else 0."""
    return int(bool(_IP_PATTERN.search(urlparse(url).netloc)))


def _count_subdomains(netloc: str) -> int:
    """Return the number of subdomains in *netloc* (excluding www)."""
    # Strip port if present
    host = netloc.split(":")[0]
    parts = host.split(".")
    # A plain domain has 2 parts (e.g. example.com)
    return max(0, len(parts) - 2)


def extract_features(url: str) -> Dict[str, float]:
    """Extract a fixed set of numeric features from a single *url*.

    Parameters
    ----------
    url:
        A URL string.

    Returns
    -------
    dict
        Feature name → numeric value mapping.
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query

    host = netloc.split(":")[0]

    url_length = len(url)
    dot_count = url.count(".")
    hyphen_count = url.count("-")
    at_count = url.count("@")
    slash_count = url.count("/")
    question_count = url.count("?")
    equals_count = url.count("=")
    underscore_count = url.count("_")
    digit_count = sum(c.isdigit() for c in url)
    special_char_count = sum(not c.isalnum() for c in url)

    has_https = int(parsed.scheme == "https")
    has_ip = _has_ip_address(url)
    subdomain_count = _count_subdomains(netloc)
    domain_length = len(host)
    path_length = len(path)
    has_query = int(bool(query))
    is_shortener = int(host in _SHORTENERS)
    double_slash_redirect = int("//" in path)

    # Suspicious keyword count
    url_lower = url.lower()
    suspicious_keyword_count = sum(kw in url_lower for kw in _SUSPICIOUS_KEYWORDS)

    return {
        "url_length": url_length,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "at_count": at_count,
        "slash_count": slash_count,
        "question_count": question_count,
        "equals_count": equals_count,
        "underscore_count": underscore_count,
        "digit_count": digit_count,
        "special_char_count": special_char_count,
        "has_https": has_https,
        "has_ip": has_ip,
        "subdomain_count": subdomain_count,
        "domain_length": domain_length,
        "path_length": path_length,
        "has_query": has_query,
        "is_shortener": is_shortener,
        "double_slash_redirect": double_slash_redirect,
        "suspicious_keyword_count": suspicious_keyword_count,
    }


def build_feature_matrix(urls: List[str]) -> pd.DataFrame:
    """Return a DataFrame of features for every URL in *urls*.

    Parameters
    ----------
    urls:
        List of URL strings.
    """
    records = [extract_features(u) for u in urls]
    return pd.DataFrame(records)


FEATURE_NAMES: List[str] = list(extract_features("http://example.com").keys())
