import os

from .models import VideoInfo, SlideWithText
from .utils import format_timestamp


def generate_markdown(
    info: VideoInfo,
    slides_with_text: list[SlideWithText],
    slides_rel_dir: str = "slides",
) -> str:
    """Build the full Markdown document."""
    lines = [
        f"# {info.title}",
        f"**Author:** {info.author_name}",
        f"**Source:** [{info.url}]({info.url})",
        "",
        "---",
    ]

    for sw in slides_with_text:
        slide = sw.slide
        ts = format_timestamp(slide.timestamp)
        ts_seconds = int(slide.timestamp)
        yt_link = f"{info.url}&t={ts_seconds}"
        img_filename = os.path.basename(slide.image_path)
        img_rel = f"{slides_rel_dir}/{img_filename}"

        lines.append("")
        lines.append(f"## Slide {slide.index}")
        lines.append(f"![Slide {slide.index}]({img_rel})")
        lines.append(f"*Timestamp: [{ts}]({yt_link})*")
        lines.append("")

        text = sw.text.strip()
        if text:
            lines.append(text)
        else:
            lines.append("*(no transcript for this slide)*")

        lines.append("")
        lines.append("---")

    return "\n".join(lines) + "\n"


def save_output(markdown: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
