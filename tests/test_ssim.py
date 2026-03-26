"""Tests for the lightweight SSIM implementation in video.py."""

import numpy as np
import pytest

from yt_slide_mark.video import ssim


def test_identical_images():
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_inverted_images():
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    score = ssim(img, 255 - img)
    assert score < 0.1


def test_slightly_modified():
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    modified = img.copy()
    modified[20:40, 20:40] = 0
    score = ssim(img, modified)
    assert 0.7 < score < 1.0


def test_uniform_images():
    white = np.full((64, 64), 255, dtype=np.uint8)
    black = np.full((64, 64), 0, dtype=np.uint8)
    score = ssim(white, black)
    assert score < 0.1


def test_score_range():
    """SSIM should be in [-1, 1]."""
    for _ in range(5):
        a = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        b = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        score = ssim(a, b)
        assert -1.0 <= score <= 1.0
