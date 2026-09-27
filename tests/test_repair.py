"""`vieb.clean.repair`: REPAIR_PREREGISTRATION §1."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from vieb.clean import repair as rp                                  # noqa: E402
from vieb.pixel import sam                                           # noqa: E402


def _masks(T=13, every=6, cx0=100.0, vx=1.0):
    """A disc of radius 20 moving right by vx px/frame, masked every 6th frame."""
    crops = {}
    for t in range(0, T, every):
        cx = cx0 + vx * t
        yy, xx = np.mgrid[:60, :60]
        c = ((xx - 30) ** 2 + (yy - 30) ** 2) <= 20 ** 2
        crops[t] = (int(round(cx)) - 30, 70, c)
    z = sam.pack_crops(crops)
    z["every"] = np.int64(every)
    return z


def test_distance_is_zero_inside_and_grows_outside_and_carries_with_motion():
    d = rp.MaskDistance(_masks(), pad_px=40)
    assert d.at(0, [[100.0, 100.0]])[0] == 0.0
    assert d.at(0, [[100.0, 130.0]])[0] == pytest.approx(10.0, abs=1.5)
    # frame 3: the disc has moved ~3 px; a point at its new centre is inside
    assert d.at(3, [[103.0, 100.0]])[0] == 0.0


def test_no_mask_where_a_bracket_is_missing():
    z = _masks(T=13)
    d = rp.MaskDistance(z, pad_px=40)
    assert d.has(0) and d.has(5) and d.has(12)
    assert not d.has(13)                       # no keyframe at 18
    assert np.isnan(d.at(13, [[100.0, 100.0]])).all()


def test_veto_takes_keypoints_off_the_animal_and_keeps_near_edge_ones():
    d = rp.MaskDistance(_masks(), pad_px=60)
    T = 13
    pose = np.zeros((T, 3, 2))
    pose[:, 0] = [100.0, 100.0]                # centre: inside
    pose[:, 1] = [100.0, 125.0]                # 5 px past the edge: kept
    pose[:, 2] = [100.0, 160.0]                # 40 px off: vetoed at bl=100
    for t in range(T):
        pose[t, :, 0] += t
    vet, has = rp.veto(pose, d, bl=100.0)
    assert has.all()
    assert not vet[:, 0].any() and not vet[:, 1].any() and vet[:, 2].all()


def test_hold_short_keeps_long_and_edge_runs_missing_only():
    m = np.zeros((20, 1), dtype=bool)
    m[3:5, 0] = True                           # short, bracketed: held
    m[8:13, 0] = True                          # 5 frames: stays missing
    m[18:, 0] = True                           # touches the end: stays
    keep = rp.hold_short(m)
    assert not keep[3:5].any() and keep[8:13].all() and keep[18:].all()


def test_repair_without_masks_is_disposition_plus_median():
    T = 60
    rng = np.random.default_rng(0)
    pose = np.cumsum(rng.normal(0, 1, (T, 2, 2)), axis=0) + 200
    miss = np.zeros((T, 2), dtype=bool)
    miss[10:12, 0] = True
    miss[30:40, 1] = True
    out = rp.repair(pose, miss, None, bl=100.0, fps=30.0, use_veto=False)
    assert not out["vetoed"].any() and not out["has_mask"].any()
    assert np.isfinite(out["pose"][10:12, 0]).all()
    assert np.isnan(out["pose"][30:40, 1]).all()
    assert out["pose"].shape == pose.shape


def test_unsmoothed_is_the_series_repair_filters():
    """MOSEQ §1: same holding and same long gaps as REPAIR, without the median."""
    from vieb.clean import arms

    T = 80
    rng = np.random.default_rng(1)
    pose = np.cumsum(rng.normal(0, 1, (T, 3, 2)), axis=0) + 200
    miss = np.zeros((T, 3), dtype=bool)
    miss[10:12, 0] = True                      # short: held
    miss[30:40, 1] = True                      # long: NaN
    vet = np.zeros((T, 3), dtype=bool)
    vet[50, 2] = True                          # a vetoed keypoint is held too
    raw = rp.unsmoothed(pose, miss, vet)
    assert np.isfinite(raw[10:12, 0]).all() and np.isnan(raw[30:40, 1]).all()
    assert np.isfinite(raw[50, 2]).all() and not np.allclose(raw[50, 2], pose[50, 2])
    ok = ~np.isnan(pose[:, :, 0]) & ~miss & ~vet
    assert np.allclose(raw[ok], pose[ok])      # present values untouched
    out = rp.repair(pose, miss | vet, None, bl=100.0, fps=30.0, use_veto=False)
    held, stays = rp.held_and_stays(pose, miss, vet)
    med = arms.rolling_median(held, None, 30.0, window_s=rp.MEDIAN_S)
    med[stays] = np.nan
    assert np.allclose(out["pose"], med, equal_nan=True)


def test_moseq_input_zero_confidence_where_missing_and_surrogates_share_it():
    from vieb.moseq import inputs

    T, K = 200, 7
    rng = np.random.default_rng(2)
    pose = np.cumsum(rng.normal(0, 1, (T, K, 2)), axis=0) + 200
    miss = np.zeros((T, K), dtype=bool)
    miss[20:30, 2] = True                      # long gap: conf 0
    miss[40:42, 3] = True                      # short gap: held, conf kept
    vet = np.zeros((T, K), dtype=bool)
    vet[60, 1] = True                          # vetoed: conf 0 even though held
    ab = np.zeros(T, dtype=bool)
    ab[100:110] = True                         # abstained frames: all missing
    conf = np.full((T, K), 0.9)
    xy, cf = inputs.real(pose, miss, vet, ab, conf)
    assert np.isfinite(xy).all()
    assert (cf[20:30, 2] == 0).all() and (cf[40:42, 3] == 0.9).all()
    assert cf[60, 1] == 0 and (cf[100:110] == 0).all()
    for kind in ("white", "phase"):
        s, sc = inputs.surrogate(xy, cf, kind, np.random.default_rng(0))
        assert np.isfinite(s).all() and np.array_equal(sc, cf)
        assert not np.allclose(s, xy)
