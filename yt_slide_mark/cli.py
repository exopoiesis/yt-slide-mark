import argparse
import logging
import os
import shutil
import sys
import tempfile

from .utils import extract_video_id, get_video_info, sanitize_filename
from .transcript import fetch_transcript
from .video import download_video, extract_unique_frames
from .mapper import map_transcript_to_slides
from .punctuation import punctuate_texts
from .markdown_gen import generate_markdown, save_output
from .region import Region, parse_region

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slide_scribe",
        description="Extract slides from a YouTube presentation and pair them with transcript text.",
    )
    p.add_argument("url", nargs="?", help="YouTube video URL or ID")
    p.add_argument("-b", "--batch", metavar="FILE",
                    help="Text file with YouTube URLs (one per line)")
    p.add_argument("-l", "--language", default="en", help="Transcript language (default: en)")
    p.add_argument("-o", "--output", default="./output", help="Output directory (default: ./output)")
    p.add_argument("--similarity", type=float, default=0.85,
                    help="SSIM threshold 0.0-1.0 (default: 0.85)")
    p.add_argument("--sample-interval", type=float, default=1.0,
                    help="Seconds between frame checks (default: 1.0)")
    p.add_argument("--cooldown", type=float, default=10.0,
                    help="Seconds to skip after detecting a new slide (default: 10.0)")
    p.add_argument("--include", action="append", metavar="REGION",
                    help="Only compare within this region (x1,y1-x2,y2). "
                         "Pixels or percents: 700,600-900,800 or 70%%,60%%-90%%,80%%. "
                         "Repeatable for multiple regions.")
    p.add_argument("--exclude", action="append", metavar="REGION",
                    help="Ignore this region during comparison (e.g. speaker area). "
                         "Same format as --include. Repeatable.")
    p.add_argument("--no-punctuate", action="store_true",
                    help="Disable punctuation restoration")
    p.add_argument("--keep-video", action="store_true",
                    help="Keep the downloaded video file")
    p.add_argument("-v", "--verbose", action="store_true",
                    help="Verbose logging")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def process_video(
    url: str,
    *,
    language: str,
    output_base: str,
    similarity: float,
    sample_interval: float,
    cooldown: float,
    no_punctuate: bool,
    keep_video: bool,
    include: list[Region] | None = None,
    exclude: list[Region] | None = None,
) -> str:
    """Process a single video. Returns the output directory path."""
    log = logging.getLogger(__name__)

    # 1. Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")
    log.info("Video ID: %s", video_id)

    # 2. Get video metadata
    info = get_video_info(video_id)
    log.info("Title: %s", info.title)
    log.info("Author: %s", info.author_name)

    slug = sanitize_filename(info.title)
    output_dir = os.path.join(output_base, slug)
    slides_dir = os.path.join(output_dir, "slides")

    # 3. Download video
    tmp_dir = tempfile.mkdtemp(prefix="slide_scribe_")
    try:
        video_path = download_video(video_id, tmp_dir)
        log.info("Downloaded: %s", video_path)

        # 4. Fetch transcript
        segments = fetch_transcript(video_id, language)
        log.info("Transcript: %d segments", len(segments))

        # 5. Extract unique frames
        slides = extract_unique_frames(
            video_path,
            slides_dir,
            similarity_threshold=similarity,
            sample_interval=sample_interval,
            cooldown=cooldown,
            include=include,
            exclude=exclude,
        )
        log.info("Extracted %d unique slides", len(slides))

        if not slides:
            raise RuntimeError("No slides extracted — video may not contain presentation content")

        # 6. Map transcript to slides
        slides_with_text = map_transcript_to_slides(slides, segments)

        # 7. Punctuate
        if not no_punctuate:
            log.info("Restoring punctuation…")
            raw_texts = [sw.text for sw in slides_with_text]
            punctuated = punctuate_texts(raw_texts)
            for sw, new_text in zip(slides_with_text, punctuated):
                if sw.segments:
                    from .models import TranscriptSegment
                    sw.segments = [TranscriptSegment(
                        text=new_text,
                        start=sw.segments[0].start,
                        duration=sw.segments[-1].start + sw.segments[-1].duration - sw.segments[0].start,
                    )]

        # 8. Generate Markdown
        md = generate_markdown(info, slides_with_text)
        md_path = os.path.join(output_dir, f"{slug}.md")
        save_output(md, md_path)
        log.info("Output: %s", md_path)

        # Optionally keep video
        if keep_video:
            kept = os.path.join(output_dir, os.path.basename(video_path))
            shutil.copy2(video_path, kept)
            log.info("Video saved: %s", kept)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_dir


def _read_batch_file(path: str) -> list[str]:
    """Read URLs from a text file, one per line. Skips blanks and #comments."""
    with open(path, encoding="utf-8") as f:
        urls = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    if not args.url and not args.batch:
        log.error("Provide a YouTube URL or use -b/--batch with a file")
        sys.exit(1)

    if args.include and args.exclude:
        log.error("--include and --exclude are mutually exclusive")
        sys.exit(1)

    # Parse region specs
    include_regions = None
    exclude_regions = None
    try:
        if args.include:
            include_regions = [parse_region(s) for s in args.include]
        if args.exclude:
            exclude_regions = [parse_region(s) for s in args.exclude]
    except ValueError as e:
        log.error("%s", e)
        sys.exit(1)

    # Build URL list: batch file or single URL
    if args.batch:
        if not os.path.isfile(args.batch):
            log.error("Batch file not found: %s", args.batch)
            sys.exit(1)
        urls = _read_batch_file(args.batch)
        if not urls:
            log.error("No URLs found in %s", args.batch)
            sys.exit(1)
        log.info("Batch: %d URLs from %s", len(urls), args.batch)
    else:
        urls = [args.url]

    total = len(urls)
    succeeded = []
    failed = []

    for i, url in enumerate(urls, 1):
        if total > 1:
            log.info("=== [%d/%d] %s ===", i, total, url)

        try:
            out = process_video(
                url,
                language=args.language,
                output_base=args.output,
                similarity=args.similarity,
                sample_interval=args.sample_interval,
                cooldown=args.cooldown,
                no_punctuate=args.no_punctuate,
                keep_video=args.keep_video,
                include=include_regions,
                exclude=exclude_regions,
            )
            succeeded.append((url, out))
        except Exception as e:
            log.error("Failed: %s — %s", url, e)
            failed.append((url, str(e)))
            if total == 1:
                sys.exit(1)

    # Summary
    print()
    if total == 1:
        print(f"Done! Output saved to: {succeeded[0][1]}")
    else:
        print(f"Batch complete: {len(succeeded)}/{total} succeeded")
        for url, out in succeeded:
            print(f"  OK  {out}")
        for url, err in failed:
            print(f"  FAIL {url} — {err}")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
