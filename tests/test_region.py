"""Tests for ROI region parsing and masking."""

import numpy as np
import pytest

from yt_slide_mark.region import parse_region, build_roi_mask, apply_roi_mask, Region


class TestParseRegion:
    def test_pixels(self):
        r = parse_region("100,200-300,400")
        assert r == Region(100, 200, 300, 400, is_percent=False)

    def test_percents(self):
        r = parse_region("10%,20%-30%,40%")
        assert r == Region(0.1, 0.2, 0.3, 0.4, is_percent=True)

    def test_auto_normalize(self):
        r = parse_region("300,400-100,200")
        assert r.x1 == 100 and r.y1 == 200

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_region("garbage")

    def test_mixed_types(self):
        with pytest.raises(ValueError):
            parse_region("100,200%-300,400")

    def test_spaces(self):
        r = parse_region("  100 , 200 - 300 , 400  ")
        assert r == Region(100, 200, 300, 400, is_percent=False)


class TestBuildRoiMask:
    def test_no_regions(self):
        assert build_roi_mask(100, 100) is None

    def test_include(self):
        r = Region(10, 10, 50, 50, is_percent=False)
        mask = build_roi_mask(100, 100, include=[r])
        assert mask.shape == (100, 100)
        assert mask[20, 20] is np.True_
        assert mask[0, 0] is np.False_

    def test_exclude(self):
        r = Region(10, 10, 50, 50, is_percent=False)
        mask = build_roi_mask(100, 100, exclude=[r])
        assert mask[0, 0] is np.True_
        assert mask[20, 20] is np.False_

    def test_percent_regions(self):
        r = Region(0.0, 0.0, 0.5, 0.5, is_percent=True)
        mask = build_roi_mask(100, 200, include=[r])
        # 50% of 200=100 wide, 50% of 100=50 tall
        assert mask[10, 10] is np.True_
        assert mask[60, 110] is np.False_


class TestApplyRoiMask:
    def test_masked_pixels(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=bool)
        mask[5:, :] = False
        result = apply_roi_mask(gray, mask, fill=128)
        assert result[0, 0] == 0
        assert result[7, 7] == 128
