import pytest
from src.phishdet.preprocess import clean_url


def test_clean_url():
    assert clean_url(' http://example.com ') == 'http://example.com'
