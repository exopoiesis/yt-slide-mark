# yt-slide-mark — Project Context

## What This Is

CLI tool that takes a YouTube presentation video and produces a Markdown file with unique slide screenshots paired with transcript text. Run via `python cli.py`.

## Project Structure

Flat layout — all .py files in repo root, no package subfolder. Direct imports (e.g. `from models import ...`).

```
cli.py             — argparse CLI entry point, process_video(), batch mode via -b FILE
models.py          — dataclasses: VideoInfo, TranscriptSegment, SlideFrame, SlideWithText
utils.py           — extract_video_id, sanitize_filename (70 char limit), get_video_info (noembed), format_timestamp
transcript.py      — fetch_transcript: 4-level fallback (manual → auto → translated → any)
video.py           — download_video (yt-dlp, video-only mp4) + extract_unique_frames (OpenCV + SSIM)
region.py          — ROI parsing (x1,y1-x2,y2 pixels/percents) + mask builder for SSIM
mapper.py          — map_transcript_to_slides: assigns transcript segments to slides by timestamp
punctuation.py     — punctuators pcs_en wrapper, lazy-loaded ONNX model
markdown_gen.py    — generate_markdown + save_output
```

## Key Technical Decisions

- **youtube-transcript-api v1.x** — uses instance API: `api = YouTubeTranscriptApi()` then `api.list(video_id)`. NOT class methods.
- **yt-dlp video-only download** — avoids ffmpeg merge requirement. Format: `bestvideo[height<=720][ext=mp4]/best[...]/best`
- **SSIM frame comparison** — compares against *last accepted* frame (not previous checked). Prevents drift on gradual transitions.
- **sample_interval (default 1.0s)** — check 1 frame per second, regardless of video fps. Step = `int(fps * interval)` frames.
- **cooldown (default 10.0s)** — after detecting new slide, skip ahead. Reduces noise from animations.
- **sanitize_filename caps at 70 chars** — Windows long path prevention.
- **slides.tsv** — diagnostic log written to slides/ dir with columns: slide, timestamp, time, gap_sec, ssim. Useful for tuning parameters.
- **Progress indicator** — stderr, updates every 50 comparisons: `pos/total (pct%) — N slides found`

## CLI

```
python cli.py <url>           # single video
python cli.py -b urls.txt     # batch from file (one URL per line, # comments ok)

Options:
  -l, --language          default: en
  -o, --output            default: ./output
  --similarity            SSIM threshold, default: 0.85
  --sample-interval       seconds between checks, default: 1.0
  --cooldown              seconds to skip after new slide, default: 10.0
  --include REGION        only compare within this region (repeatable)
  --exclude REGION        ignore this region for comparison (repeatable)
  --no-punctuate          skip punctuation restoration
  --keep-video            keep downloaded mp4
  -v, --verbose

Region format: x1,y1-x2,y2 (diagonal corners). Pixels or percents:
  --exclude 0,400-200,720        exclude bottom-left speaker area (pixels)
  --include 25%,0%-100%,100%     only compare right 75% of frame (percents)
  --include and --exclude are mutually exclusive. Both are repeatable.
```

## Output

```
output/<slug>/
  <slug>.md
  slides/
    slide_001.jpg ... slide_NNN.jpg
    slides.tsv          # diagnostic log
```

## Dependencies

youtube-transcript-api>=1.0.0, yt-dlp, opencv-python-headless, scikit-image, numpy, punctuators, requests

## Common Issues

- If OpenCV can't open video: likely yt-dlp produced split files (no ffmpeg). Fix: use video-only format selector.
- youtube-transcript-api v0.x vs v1.x: v1.x uses `YouTubeTranscriptApi()` instance, not class methods. `list_transcripts()` → `list()`.
- Windows path length: slug capped at 70, output paths stay under ~150 chars.
