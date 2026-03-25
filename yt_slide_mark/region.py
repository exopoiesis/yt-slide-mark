"""Region of interest (ROI) parsing and masking for SSIM comparison."""

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Region:
    """Rectangle defined by two diagonal corners.

    Values are either pixels (int) or percentages (float 0.0-1.0).
    """
    x1: float
    y1: float
    x2: float
    y2: float
    is_percent: bool = False


def parse_region(spec: str) -> Region:
    """Parse region spec like '700,600-900,800' or '70%,60%-90%,80%'.

    Format: x1,y1-x2,y2  where each value is pixels (int) or percent (N%).
    All four values must be the same type (all pixels or all percent).
    """
    m = re.match(
        r'^\s*(\d+%?)\s*,\s*(\d+%?)\s*-\s*(\d+%?)\s*,\s*(\d+%?)\s*$',
        spec,
    )
    if not m:
        raise ValueError(
            f"Invalid region format: '{spec}'. "
            f"Expected: x1,y1-x2,y2 (e.g. 700,600-900,800 or 70%,60%-90%,80%)"
        )

    parts = [m.group(i) for i in range(1, 5)]
    has_pct = [p.endswith('%') for p in parts]

    if any(has_pct) and not all(has_pct):
        raise ValueError(
            f"Mix of pixel and percent values in region: '{spec}'. "
            f"Use all pixels (700,600-900,800) or all percents (70%,60%-90%,80%)"
        )

    is_percent = all(has_pct)

    if is_percent:
        vals = [int(p.rstrip('%')) / 100.0 for p in parts]
        for v in vals:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"Percent value out of 0-100 range in: '{spec}'")
    else:
        vals = [int(p) for p in parts]
        for v in vals:
            if v < 0:
                raise ValueError(f"Negative pixel value in: '{spec}'")

    # Normalize so x1<=x2, y1<=y2
    x1, y1, x2, y2 = vals
    return Region(
        x1=min(x1, x2), y1=min(y1, y2),
        x2=max(x1, x2), y2=max(y1, y2),
        is_percent=is_percent,
    )


def _resolve(region: Region, height: int, width: int) -> tuple[int, int, int, int]:
    """Convert region to pixel coordinates (y1, y2, x1, x2) clamped to frame."""
    if region.is_percent:
        x1 = int(region.x1 * width)
        y1 = int(region.y1 * height)
        x2 = int(region.x2 * width)
        y2 = int(region.y2 * height)
    else:
        x1, y1, x2, y2 = int(region.x1), int(region.y1), int(region.x2), int(region.y2)

    # Clamp to frame bounds
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return y1, y2, x1, x2


def build_roi_mask(
    height: int,
    width: int,
    include: list[Region] | None = None,
    exclude: list[Region] | None = None,
) -> np.ndarray | None:
    """Build a boolean mask (True = use for SSIM) from include/exclude regions.

    Returns None if no regions specified (use full frame).
    """
    if not include and not exclude:
        return None

    if include:
        # Start with all-False, set True inside each include region
        mask = np.zeros((height, width), dtype=bool)
        for r in include:
            y1, y2, x1, x2 = _resolve(r, height, width)
            mask[y1:y2, x1:x2] = True
    else:
        # Start with all-True, set False inside each exclude region
        mask = np.ones((height, width), dtype=bool)
        for r in exclude:
            y1, y2, x1, x2 = _resolve(r, height, width)
            mask[y1:y2, x1:x2] = False

    return mask


def apply_roi_mask(gray: np.ndarray, mask: np.ndarray, fill: int = 128) -> np.ndarray:
    """Apply ROI mask: pixels outside ROI are set to `fill` value.

    Both frames get the same fill, so masked-out areas contribute zero
    difference to SSIM.
    """
    result = gray.copy()
    result[~mask] = fill
    return result
