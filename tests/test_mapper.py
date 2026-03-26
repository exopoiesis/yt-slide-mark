"""Tests for transcript-to-slide mapping."""

from yt_slide_mark.mapper import map_transcript_to_slides
from yt_slide_mark.models import SlideFrame, TranscriptSegment


def test_empty():
    assert map_transcript_to_slides([], []) == []


def test_single_slide():
    slides = [SlideFrame(1, 0.0, "s1.jpg")]
    segs = [TranscriptSegment("hello", 1.0, 2.0)]
    result = map_transcript_to_slides(slides, segs)
    assert len(result) == 1
    assert result[0].text == "hello"


def test_multiple_slides():
    slides = [
        SlideFrame(1, 0.0, "s1.jpg"),
        SlideFrame(2, 30.0, "s2.jpg"),
        SlideFrame(3, 60.0, "s3.jpg"),
    ]
    segs = [
        TranscriptSegment("intro", 5.0, 3.0),
        TranscriptSegment("middle", 35.0, 3.0),
        TranscriptSegment("end", 65.0, 3.0),
    ]
    result = map_transcript_to_slides(slides, segs)
    assert result[0].text == "intro"
    assert result[1].text == "middle"
    assert result[2].text == "end"


def test_segment_before_first_slide():
    slides = [SlideFrame(1, 10.0, "s1.jpg")]
    segs = [TranscriptSegment("before", 2.0, 1.0)]
    result = map_transcript_to_slides(slides, segs)
    # Segment before first slide gets assigned to first slide
    assert result[0].text == "before"


def test_multiple_segments_per_slide():
    slides = [SlideFrame(1, 0.0, "s1.jpg"), SlideFrame(2, 30.0, "s2.jpg")]
    segs = [
        TranscriptSegment("a", 1.0, 1.0),
        TranscriptSegment("b", 5.0, 1.0),
        TranscriptSegment("c", 10.0, 1.0),
    ]
    result = map_transcript_to_slides(slides, segs)
    assert result[0].text == "a b c"
    assert result[1].text == ""
