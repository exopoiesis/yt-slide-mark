# yt-slide-mark

Extract slides from YouTube presentations and pair them with transcript text.

Takes a YouTube video URL, downloads the video, extracts unique slide frames using SSIM comparison, fetches the transcript with timestamps, maps text to slides, restores punctuation, and generates a Markdown document with embedded slide images and clickable YouTube timestamps.

## Installation

```bash
pip install yt-slide-mark
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install yt-slide-mark
```

## Usage

```bash
# Single video
yt-slide-mark "https://youtube.com/watch?v=VIDEO_ID"

# Batch — from a file with URLs
yt-slide-mark -b urls.txt

# Without installing (via uvx)
uvx yt-slide-mark "https://youtube.com/watch?v=VIDEO_ID"

# Or via python -m
python -m yt_slide_mark "https://youtube.com/watch?v=VIDEO_ID"
```

### Options

```
yt-slide-mark <url> [-b FILE]
  -b, --batch FILE        Text file with URLs (one per line)
  -l, --language          Transcript language (default: en)
  -o, --output            Output directory (default: ./output)
  --similarity            SSIM threshold 0.0-1.0 (default: 0.85)
  --sample-interval       Seconds between frame checks (default: 1.0)
  --cooldown              Seconds to skip after new slide detected (default: 10.0)
  --include REGION        Only compare within this region (repeatable)
  --exclude REGION        Ignore this region during comparison (repeatable)
  --no-punctuate          Disable punctuation restoration
  --keep-video            Keep the downloaded video file
  -v, --verbose           Verbose logging
```

### Region of interest (--include / --exclude)

Lecture videos often show the speaker alongside slides, causing excessive slide detections. Use `--include` or `--exclude` to limit which part of the frame is used for SSIM comparison. Full frames are still saved as slide images.

**Format:** `x1,y1-x2,y2` — two diagonal corners of a rectangle. Values can be pixels or percents:

```
--exclude 0,400-200,720          # pixels: exclude bottom-left speaker area
--include 25%,0%-100%,100%       # percents: only compare right 75% of frame
```

Multiple regions can be specified by repeating the flag:

```
--exclude 0,80%-25%,100% --exclude 75%,80%-100%,100%
```

`--include` and `--exclude` are mutually exclusive.

### Batch file format

One URL per line. Blank lines and `#` comments are ignored:

```
# lectures.txt
https://youtube.com/watch?v=VIDEO1
https://youtube.com/watch?v=VIDEO2

# this one is optional
https://youtube.com/watch?v=VIDEO3
```

### Examples

```bash
# Basic usage
yt-slide-mark "https://youtube.com/watch?v=dQw4w9WgXcQ"

# Custom output directory, verbose
yt-slide-mark "https://youtube.com/watch?v=dQw4w9WgXcQ" -o ./my_notes -v

# Skip punctuation, keep video, lower similarity threshold
yt-slide-mark "https://youtube.com/watch?v=dQw4w9WgXcQ" --no-punctuate --keep-video --similarity 0.80

# Faster sampling for long lectures, wider cooldown
yt-slide-mark "https://youtube.com/watch?v=dQw4w9WgXcQ" --sample-interval 2.0 --cooldown 15.0

# Exclude speaker area in bottom-left corner (pixels)
yt-slide-mark "https://youtube.com/watch?v=dQw4w9WgXcQ" --exclude 0,500-250,720

# Only compare the slide area (right 75% of frame, percents)
yt-slide-mark "https://youtube.com/watch?v=dQw4w9WgXcQ" --include 25%,0%-100%,100%

# Batch: process a list of videos from file
yt-slide-mark -b lectures.txt -o ./lectures -v
```

## Output

```
output/video-title/
├── video-title.md
└── slides/
    ├── slide_001.jpg
    ├── slide_002.jpg
    └── ...
```

The Markdown file contains:
- Video title, author, and source link
- Each unique slide as a JPEG image
- Clickable YouTube timestamps linking to the exact moment
- Transcript text mapped to each slide with restored punctuation

In batch mode, each video gets its own subfolder. A summary is printed at the end showing successes and failures; one failed video does not stop the rest.

## Pipeline

```
YouTube URL
  → extract video ID
  → get video info (noembed.com API)
  → download video (yt-dlp, 720p)
  → fetch transcript (youtube-transcript-api, 4-level fallback)
  → extract unique frames (OpenCV + SSIM)
  → map transcript to slides (by timestamps)
  → restore punctuation (punctuators pcs_en, ONNX)
  → generate Markdown
```

## Dependencies

~200 MB total (vs ~900 MB in v1.0.0 — torch, scipy, scikit-image eliminated):
- **youtube-transcript-api** — transcript fetching with fallbacks
- **yt-dlp** — video downloading
- **opencv-python-headless** — frame extraction + SSIM
- **onnxruntime** — punctuation model inference
- **sentencepiece** — tokenization for punctuation
- **numpy** — numerical operations
- **requests** — video metadata + one-time model download

The 200 MB ONNX punctuation model is downloaded automatically on first run and cached in `%LOCALAPPDATA%/yt-slide-mark` (Windows) or `~/.cache/yt-slide-mark` (Linux/macOS).

## License

MIT
