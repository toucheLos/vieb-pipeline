"""`vieb.feat.streams`: mask features carried between keyframes."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from vieb.feat import streams                                        # noqa: E402
from vieb.pixel import sam                                           # noqa: E402


def _moving_ellipse(T=25, every=6, vx=2.0):
    crops = {}
    for t in range(0, T, every):
        yy, xx = np.mgrid[:40, :80]
        c = ((xx - 40) / 30.0) ** 2 + ((yy - 20) / 10.0) ** 2 <= 1
        crops[t] = (int(round(100 + vx * t)) - 40, 80, c)
    z = sam.pack_crops(crops)
    z["every"] = np.int64(every)
    return z


def test_mask_stream_speed_area_and_elongation():
    z = _moving_ellipse()
    f, v = streams.mask_stream(z, 25, bl=100.0, fps=30.0)
    assert v[1:].all() and not v[0]            # frame 0 has no speed yet
    assert np.nanmedian(f[v, 2]) == pytest.approx(2.0 * 30 / 100, rel=0.1)
    assert np.nanmedian(f[v, 0]) == pytest.approx(np.pi * 30 * 10 / 1e4, rel=0.1)
    assert np.nanmedian(f[v, 1]) == pytest.approx(3.0, rel=0.1)
    assert np.nanmax(np.abs(f[v, 3])) < 0.5


def test_head_coverage_needs_both_ears_on_the_mask():
    z = _moving_ellipse(T=7)
    pose = np.zeros((7, 7, 2))
    pose[:, :, 0] = 100.0
    pose[:, :, 1] = 100.0
    pose[:, 1] = (100.0, 160.0)              # right ear far off
    cov, dfn = streams.head_coverage(z, pose, bl=100.0)
    assert dfn[:7].all() and not cov[:7].any()
    pose[:, 1] = (100.0, 100.0)
    cov, _ = streams.head_coverage(z, pose, bl=100.0)
    assert cov[0]
