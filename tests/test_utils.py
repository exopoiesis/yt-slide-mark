"""Tests for utility functions."""

from yt_slide_mark.utils import extract_video_id, sanitize_filename, format_timestamp


class TestExtractVideoId:
    def test_plain_id(self):
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_watch_url(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_invalid(self):
        assert extract_video_id("not-a-url") is None

    def test_url_with_params(self):
        assert extract_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ&t=42") == "dQw4w9WgXcQ"


class TestSanitizeFilename:
    def test_simple(self):
        assert sanitize_filename("Hello World") == "hello-world"

    def test_special_chars(self):
        result = sanitize_filename("What's New? (2024)")
        assert all(c.isalnum() or c in ".-_" for c in result)

    def test_ampersand(self):
        assert "and" in sanitize_filename("Tom & Jerry")

    def test_max_length(self):
        long = "A" * 200
        assert len(sanitize_filename(long)) <= 70

    def test_unicode(self):
        result = sanitize_filename("Ünïcödé Tëst")
        assert len(result) > 0


class TestFormatTimestamp:
    def test_seconds(self):
        assert format_timestamp(45) == "0:45"

    def test_minutes(self):
        assert format_timestamp(125) == "2:05"

    def test_hours(self):
        assert format_timestamp(3661) == "1:01:01"

    def test_zero(self):
        assert format_timestamp(0) == "0:00"
