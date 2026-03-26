"""Tests for Markdown generation."""

from yt_slide_mark.markdown_gen import generate_markdown
from yt_slide_mark.models import VideoInfo, SlideFrame, SlideWithText, TranscriptSegment


def test_basic_output():
    info = VideoInfo("abc123", "Test Video", "Author", "https://www.youtube.com/watch?v=abc123")
    slides = [
        SlideWithText(
            slide=SlideFrame(1, 0.0, "slides/slide_001.jpg"),
            segments=[TranscriptSegment("Hello world.", 0.0, 5.0)],
        ),
    ]
    md = generate_markdown(info, slides)
    assert "# Test Video" in md
    assert "**Author:** Author" in md
    assert "![Slide 1]" in md
    assert "Hello world." in md
    assert "&t=0" in md


def test_no_transcript():
    info = VideoInfo("abc123", "Test", "Author", "https://www.youtube.com/watch?v=abc123")
    slides = [SlideWithText(slide=SlideFrame(1, 0.0, "slides/slide_001.jpg"))]
    md = generate_markdown(info, slides)
    assert "no transcript" in md


def test_timestamp_link():
    info = VideoInfo("abc123", "Test", "Author", "https://www.youtube.com/watch?v=abc123")
    slides = [
        SlideWithText(slide=SlideFrame(1, 125.0, "slides/slide_001.jpg")),
    ]
    md = generate_markdown(info, slides)
    assert "2:05" in md
    assert "&t=125" in md
