from dataclasses import dataclass, field


@dataclass
class VideoInfo:
    video_id: str
    title: str
    author_name: str
    url: str


@dataclass
class TranscriptSegment:
    text: str
    start: float  # seconds
    duration: float


@dataclass
class SlideFrame:
    index: int  # 1-based slide number
    timestamp: float  # seconds into video
    image_path: str  # path to saved JPEG


@dataclass
class SlideWithText:
    slide: SlideFrame
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segments)
