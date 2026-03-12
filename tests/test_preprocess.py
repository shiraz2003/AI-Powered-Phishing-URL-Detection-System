"""Tests for phishdet.preprocess."""

import pytest

from phishdet.preprocess import extract_domain, normalize_url


class TestNormalizeUrl:
    def test_strips_whitespace(self):
        assert normalize_url("  https://Example.com  ") == "https://example.com"

    def test_lowercases_scheme(self):
        assert normalize_url("HTTP://example.com/path").startswith("http://")

    def test_lowercases_netloc(self):
        result = normalize_url("https://EXAMPLE.COM/page")
        from urllib.parse import urlparse
        assert urlparse(result).netloc == "example.com"

    def test_trailing_slash_removed_for_root(self):
        result = normalize_url("https://example.com/")
        assert not result.endswith("/")

    def test_non_root_path_preserved(self):
        url = "https://example.com/some/path"
        assert normalize_url(url) == url

    def test_query_string_preserved(self):
        url = "https://example.com/search?q=phishing"
        assert normalize_url(url) == url

    def test_already_normalised_unchanged(self):
        url = "https://example.com/page"
        assert normalize_url(url) == url


class TestExtractDomain:
    def test_simple_domain(self):
        assert extract_domain("https://example.com/path") == "example.com"

    def test_subdomain(self):
        assert extract_domain("https://www.example.com") == "www.example.com"

    def test_uppercase_lowercased(self):
        assert extract_domain("https://EXAMPLE.COM") == "example.com"

    def test_with_port(self):
        assert extract_domain("http://example.com:8080/page") == "example.com:8080"

    def test_ip_address(self):
        assert extract_domain("http://192.168.1.1/login") == "192.168.1.1"
