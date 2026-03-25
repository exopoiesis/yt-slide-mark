import logging

from youtube_transcript_api import YouTubeTranscriptApi

from .models import TranscriptSegment

log = logging.getLogger(__name__)


def fetch_transcript(video_id: str, language: str = "en") -> list[TranscriptSegment]:
    """Fetch transcript with 4-level fallback. Returns segments with timestamps."""
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    fetched = None

    # 1. Manual transcript in target language
    try:
        fetched = transcript_list.find_manually_created_transcript([language]).fetch()
        log.info("Using manual transcript (%s)", language)
    except Exception:
        pass

    # 2. Auto-generated transcript in target language
    if fetched is None:
        try:
            fetched = transcript_list.find_generated_transcript([language]).fetch()
            log.info("Using auto-generated transcript (%s)", language)
        except Exception:
            pass

    # 3. Any transcript, translated to target language
    if fetched is None:
        try:
            available = list(transcript_list)
            if available:
                translated = available[0].translate(language)
                fetched = translated.fetch()
                log.info("Using translated transcript → %s", language)
        except Exception:
            pass

    # 4. Any transcript as-is
    if fetched is None:
        available = list(transcript_list)
        if available:
            fetched = available[0].fetch()
            log.info("Using transcript in original language")
        else:
            raise RuntimeError(f"No transcript found for video {video_id}")

    return _parse_snippets(fetched)


def _parse_snippets(fetched) -> list[TranscriptSegment]:
    segments = []
    for snippet in fetched:
        # Handle both dict-like and object-like access
        if hasattr(snippet, "text"):
            text = snippet.text
            start = snippet.start
            duration = snippet.duration
        else:
            text = snippet["text"]
            start = snippet["start"]
            duration = snippet["duration"]
        text = text.replace("\n", " ").strip()
        if text:
            segments.append(TranscriptSegment(text=text, start=start, duration=duration))
    return segments
