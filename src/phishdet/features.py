import ipaddress
import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import get_hostname_and_path, tokenize_url

SUSPICIOUS_KEYWORDS: List[str] = [
    "login",
    "verify",
    "secure",
    "update",
    "account",
    "bank",
    "confirm",
    "signin",
    "paypal",
    "ebay",
]

def tokens_to_string(tokens: List[str]) -> str:
    """Join tokens for TF-IDF input."""
    return " ".join(tokens)

def extract_numeric_features(urls: Iterable[str]) -> np.ndarray:
    """Extract numeric feature matrix for URLs."""
    feature_list: List[List[float]] = []
    for url in urls:
        raw = url.strip()
        url_norm = raw.lower()
        host, path = get_hostname_and_path(raw)

        url_length = len(url_norm)
        hostname_length = len(host)
        num_digits = sum(ch.isdigit() for ch in url_norm)
        num_dots = url_norm.count(".")
        num_hyphens = url_norm.count("-")
        num_special_chars = sum(
            ch in "@!$%^*(){}[]|\\'\"<>"
            for ch in url_norm
        )

        subdomains = host.split(".")[:-2] if len(host.split(".")) > 2 else []
        num_subdomains = len([s for s in subdomains if s])

        path_depth = len([p for p in path.split("/") if p])

        has_ip_address = _has_ip_in_host(host)

        suspicious_keyword_count = sum(
            url_norm.count(k) for k in SUSPICIOUS_KEYWORDS
        )

        feature_list.append(
            [
                float(url_length),
                float(hostname_length),
                float(num_digits),
                float(num_dots),
                float(num_hyphens),
                float(num_special_chars),
                float(num_subdomains),
                float(path_depth),
                float(has_ip_address),
                float(suspicious_keyword_count),
            ]
        )

    return np.asarray(feature_list, dtype=float)

def _has_ip_in_host(host: str) -> int:
    """Return 1 if host looks like an IP address, else 0."""
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        pass
    # check for dotted quad inside host
    ip_candidate = re.search(r"(\d{1,3}\.){3}\d{1,3}", host)
    if not ip_candidate:
        return 0
    try:
        ipaddress.ip_address(ip_candidate.group(0))
        return 1
    except ValueError:
        return 0

class FeatureBuilder:
    """Build combined TF-IDF + numeric feature matrices."""

    def __init__(
        self,
        max_features: int = 2000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            token_pattern=r"[^ ]+",
        )

    def fit_transform(self, urls: Iterable[str]) -> sparse.csr_matrix:
        tokens = [tokens_to_string(tokenize_url(u)) for u in urls]
        tfidf = self.vectorizer.fit_transform(tokens)
        numeric = extract_numeric_features(urls)
        numeric_sparse = sparse.csr_matrix(numeric)
        combined = sparse.hstack([tfidf, numeric_sparse]).tocsr()
        logger.info(
            f"Built feature matrix train with shape {combined.shape}\n"
        )
        return combined

    def transform(self, urls: Iterable[str]) -> sparse.csr_matrix:
        tokens = [tokens_to_string(tokenize_url(u)) for u in urls]
        tfidf = self.vectorizer.transform(tokens)
        numeric = extract_numeric_features(urls)
        numeric_sparse = sparse.csr_matrix(numeric)
        combined = sparse.hstack([tfidf, numeric_sparse]).tocsr()
        logger.info(
            f"Built feature matrix inference with shape {combined.shape}\n"
        )
        return combined
