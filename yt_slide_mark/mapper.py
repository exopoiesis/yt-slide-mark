from .models import SlideFrame, SlideWithText, TranscriptSegment


def map_transcript_to_slides(
    slides: list[SlideFrame],
    segments: list[TranscriptSegment],
) -> list[SlideWithText]:
    """Assign each transcript segment to the slide visible at that time.

    Each segment is assigned to the last slide whose timestamp is <= segment.start.
    """
    if not slides:
        return []

    result = [SlideWithText(slide=s) for s in slides]

    for seg in segments:
        # Find the last slide that appeared before or at this segment's start
        target_idx = 0
        for i, s in enumerate(slides):
            if s.timestamp <= seg.start:
                target_idx = i
            else:
                break
        result[target_idx].segments.append(seg)

    return result
