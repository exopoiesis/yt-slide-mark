"""Tests for the lightweight punctuation module."""

import pytest

from yt_slide_mark.punctuation import punctuate_texts, supported_languages


def test_empty_input():
    assert punctuate_texts([]) == []


def test_basic_punctuation():
    result = punctuate_texts(["hello world this is a test"])
    assert len(result) == 1
    text = result[0]
    assert text[0].isupper()
    assert any(c in text for c in ".!?,")


def test_sentence_boundaries():
    result = punctuate_texts([
        "the quick brown fox jumps over the lazy dog it was a beautiful day"
    ])
    text = result[0]
    assert ". " in text or "? " in text or "! " in text


def test_question():
    result = punctuate_texts(["can you believe it works"])
    text = result[0]
    assert "?" in text


def test_multiple_texts():
    texts = [
        "hello world",
        "this is another sentence",
        "machine learning is great",
    ]
    results = punctuate_texts(texts)
    assert len(results) == 3
    for r in results:
        assert len(r) > 0
        assert r[0].isupper()


def test_preserves_meaning():
    result = punctuate_texts(["the cat sat on the mat"])
    text = result[0].lower().replace(",", "").replace(".", "")
    assert "the cat sat on the mat" in text


def test_language_parameter():
    """punctuate_texts accepts language parameter without error."""
    result = punctuate_texts(["hello world"], language="en")
    assert len(result) == 1
    assert result[0][0].isupper()


def test_supported_languages():
    langs = supported_languages()
    assert "en" in langs
    assert "ru" in langs
    assert "de" in langs
    assert len(langs) >= 40
