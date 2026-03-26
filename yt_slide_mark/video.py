import logging
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from .models import SlideFrame
from .region import Region, build_roi_mask, apply_roi_mask


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute mean SSIM between two grayscale uint8 images using OpenCV+numpy.

    Equivalent to skimage.metrics.structural_similarity with default params.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    a = img1.astype(np.float64)
    b = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(np.mean(num / den))
from .utils import format_timestamp

log = logging.getLogger(__name__)


def download_video(video_id: str, output_dir: str) -> str:
    """Download 720p MP4 via yt-dlp. Returns path to downloaded file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    if os.path.exists(output_path):
        log.info("Video already downloaded: %s", output_path)
        return output_path

    # Prefer video-only (we only need frames) to avoid ffmpeg merge requirement.
    # Fallback to any single-file format if video-only isn't available.
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]/best[height<=720]/best",
        "-o", output_path,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    log.info("Downloading video %s …", video_id)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # yt-dlp may use a different extension; find the actual file
    if not os.path.exists(output_path):
        for f in os.listdir(output_dir):
            if f.startswith(video_id):
                output_path = os.path.join(output_dir, f)
                break

    return output_path


def extract_unique_frames(
    video_path: str,
    slides_dir: str,
    similarity_threshold: float = 0.85,
    sample_interval: float = 1.0,
    cooldown: float = 10.0,
    jpeg_quality: int = 85,
    include: list[Region] | None = None,
    exclude: list[Region] | None = None,
) -> list[SlideFrame]:
    """Extract visually unique frames using SSIM comparison.

    Compares each sampled frame against the *last accepted* frame
    to avoid missing slides during gradual transitions.

    Args:
        sample_interval: seconds between frame checks (default: 1.0 = one check/sec)
        cooldown: seconds to skip after accepting a new slide (default: 10.0)
        include: only compare pixels inside these regions
        exclude: ignore pixels inside these regions (e.g. speaker area)
    """
    os.makedirs(slides_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    log.info("Video: %.1f fps, %d frames (%.0fs)", fps, total_frames, duration)

    # Convert time intervals to frame counts
    step = max(1, int(fps * sample_interval))
    cooldown_frames = max(step, int(fps * cooldown))

    log.info("Sampling every %d frames (%.1fs), cooldown %d frames (%.1fs)",
             step, sample_interval, cooldown_frames, cooldown)

    # Build ROI mask from include/exclude regions
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    roi_mask = build_roi_mask(frame_h, frame_w, include=include, exclude=exclude)
    if roi_mask is not None:
        roi_pct = roi_mask.sum() / roi_mask.size * 100
        log.info("ROI mask: %.0f%% of frame used for comparison", roi_pct)

    slides: list[SlideFrame] = []
    last_accepted_gray = None
    frame_idx = 0
    comparisons = 0
    prev_timestamp = 0.0

    # slides.tsv — diagnostic log for tuning parameters
    tsv_path = os.path.join(slides_dir, "slides.tsv")
    tsv = open(tsv_path, "w", encoding="utf-8")
    tsv.write("slide\ttimestamp\ttime\tgap_sec\tssim\n")

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        comparisons += 1
        score = None

        if last_accepted_gray is None:
            is_new = True
        else:
            h = min(gray.shape[0], last_accepted_gray.shape[0])
            w = min(gray.shape[1], last_accepted_gray.shape[1])
            a = cv2.resize(last_accepted_gray, (w, h))
            b = cv2.resize(gray, (w, h))
            if roi_mask is not None:
                a = apply_roi_mask(a, roi_mask)
                b = apply_roi_mask(b, roi_mask)
            score = ssim(a, b)
            is_new = score < similarity_threshold

        if is_new:
            slide_num = len(slides) + 1
            timestamp = frame_idx / fps
            gap = timestamp - prev_timestamp if slides else 0.0
            filename = f"slide_{slide_num:03d}.jpg"
            filepath = os.path.join(slides_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            slides.append(SlideFrame(
                index=slide_num,
                timestamp=timestamp,
                image_path=filepath,
            ))
            last_accepted_gray = gray
            prev_timestamp = timestamp

            ts_str = format_timestamp(timestamp)
            ssim_str = f"{score:.3f}" if score is not None else "-"
            tsv.write(f"{slide_num}\t{timestamp:.1f}\t{ts_str}\t{gap:.1f}\t{ssim_str}\n")
            log.debug("Slide %d at %s (gap %.0fs, ssim=%s)", slide_num, ts_str, gap, ssim_str)

            # Skip ahead — next slide won't appear for at least `cooldown` seconds
            frame_idx += cooldown_frames
        else:
            frame_idx += step

        # Progress (every 50 comparisons)
        if comparisons % 50 == 0:
            pct = frame_idx / total_frames * 100
            pos = format_timestamp(frame_idx / fps)
            total_ts = format_timestamp(duration)
            sys.stderr.write(f"\r  {pos}/{total_ts} ({pct:.0f}%) — {len(slides)} slides found")
            sys.stderr.flush()

    # Clear progress line
    sys.stderr.write("\r" + " " * 60 + "\r")
    sys.stderr.flush()

    tsv.close()
    cap.release()
    log.info("Extracted %d unique slides (%d comparisons)", len(slides), comparisons)
    log.info("Slide log: %s", tsv_path)
    return slides
