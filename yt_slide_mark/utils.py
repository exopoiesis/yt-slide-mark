import re
import unicodedata

import requests

from .models import VideoInfo


def extract_video_id(url_or_id: str) -> str | None:
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id
    pats = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})",
    ]
    for pat in pats:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    return None


def sanitize_filename(title: str) -> str:
    slug = (
        unicodedata.normalize("NFKD", title)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    slug = slug.replace("&", "and")
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^0-9A-Za-z._-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-_.").lower()
    return slug[:70]


def get_video_info(video_id: str) -> VideoInfo:
    url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
    try:
        data = requests.get(url, timeout=10).json()
    except requests.RequestException:
        data = {}
    return VideoInfo(
        video_id=video_id,
        title=data.get("title", "unknown-title"),
        author_name=data.get("author_name", "unknown-channel"),
        url=f"https://www.youtube.com/watch?v={video_id}",
    )


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
