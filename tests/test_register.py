"""Keypoint-free crop registration. `results/STABILISE_PREREGISTRATION.md`.

The scene is the one Amendment 1 was decided on: a textured ellipse, roughly a
mouse's size in this corpus, over a textured background at 640 x 480.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vieb.pixel import head as hd                                    # noqa: E402
from vieb.pixel import register as rg                                # noqa: E402
from vieb.tok import ego as tego                                     # noqa: E402

H, W = 480, 640
C0 = (320.0, 240.0)
A_PX, B_PX = 55.0, 22.0
I2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _scene(seed: int = 1):
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.uniform(60, 120, (H, W)), (0, 0), 3)
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (H, W)), (0, 0), 1.5) * 0.5 + 150
    yy, xx = np.mgrid[:H, :W]
    body = (((xx - C0[0]) / A_PX) ** 2 + ((yy - C0[1]) / B_PX) ** 2) <= 1
    return bg, np.where(body, tex, 0.0), body.astype(np.float64)


def _pose0() -> np.ndarray:
    """Seven keypoints on the ellipse, nose to the RIGHT (+x)."""
    p = np.zeros((7, 2))
    cx, cy = C0
    p[tego.NOSE] = (cx + 50, cy)
    p[tego.LEFT_EAR] = (cx + 35, cy - 8)
    p[tego.RIGHT_EAR] = (cx + 35, cy + 8)
    p[tego.CENTER] = (cx, cy)
    p[tego.LEFT_HIP] = (cx - 30, cy - 10)
    p[tego.RIGHT_HIP] = (cx - 30, cy + 10)
    p[tego.TAIL_BASE] = (cx - 50, cy)
    return p


def _render(bg, layer, msk, M) -> np.ndarray:
    L = cv2.warpAffine(layer, M, (W, H))
    A = cv2.warpAffine(msk, M, (W, H)) > 0.5
    return np.clip(np.round(np.where(A, L, bg)), 0, 255).astype(np.uint8)


def _rot(ang_deg, dx, dy):
    m = cv2.getRotationMatrix2D(C0, float(ang_deg), 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    return m


def _sequence(transforms, jitter_px: float = 0.0, seed: int = 0):
    bg, layer, msk = _scene()
    rng = np.random.default_rng(seed)
    frames, poses = [], []
    for M in transforms:
        frames.append(_render(bg, layer, msk, M))
        q = rg.apply(M, _pose0())
        poses.append(q + rng.normal(0.0, jitter_px, q.shape))
    return bg, frames, np.asarray(poses)


def _scan(frames, pose, bg, win=10):
    bgb = rg.blur(bg.astype(np.uint8))
    return rg.scan_register(lambda: iter(frames), pose, bg=bgb, cutoff=15.0,
                            radii_px=[0.6 * 100.0], dilate_px=100.0, win=win,
                            body_length_px=100.0)


# ---- mask geometry ----------------------------------------------------------

def test_mask_pose_recovers_centroid_and_axis():
    yy, xx = np.mgrid[:H, :W]
    for ang in (0.0, 30.0, -60.0):
        t = np.radians(ang)
        u = (xx - 300) * np.cos(t) + (yy - 200) * np.sin(t)
        v = -(xx - 300) * np.sin(t) + (yy - 200) * np.cos(t)
        m = (u / 60.0) ** 2 + (v / 20.0) ** 2 <= 1
        cx, cy, th, area = rg.mask_pose(m)
        assert abs(cx - 300) < 0.5 and abs(cy - 200) < 0.5
        d = np.degrees(np.angle(np.exp(2j * (th - t)))) / 2
        assert abs(d) < 0.5
        assert area == pytest.approx(np.pi * 60 * 20, rel=0.02)


def test_continuity_resolves_the_half_turn_without_a_sign():
    prev = np.radians(80.0)
    # The moment axis reports -95 deg; +85 is the same axis and 5 deg away.
    got = rg.continuous(np.radians(-95.0), prev)
    assert np.degrees(got) == pytest.approx(85.0)
    assert rg.continuous(0.3, None) == 0.3


def test_animal_mask_finds_the_animal_and_nothing_else():
    bg, layer, msk = _scene()
    f = _render(bg, layer, msk, I2)
    m = rg.animal_mask(rg.blur(f), rg.blur(bg.astype(np.uint8)), 15.0)
    assert m is not None
    truth = msk > 0.5
    inter = (m & truth).sum() / (m | truth).sum()
    assert inter > 0.8


# ---- Amendment 1: ECC recovers rotation the log-polar map could not --------

@pytest.mark.parametrize("ang,dx,dy", [(4, 0, 0), (-6, 2, 1), (10, -4, 3),
                                       (0, 3, -2), (0.5, 0.5, 0)])
def test_rigid_between_recovers_synthetic_rotation(ang, dx, dy):
    bg, layer, msk = _scene()
    a = rg.blur(_render(bg, layer, msk, I2))
    b = rg.blur(_render(bg, layer, msk, rg.invert(_rot(ang, dx, dy))))
    y0, y1, x0, x1 = 190, 291, 240, 401
    got = rg.rigid_between(a[y0:y1, x0:x1], b[y0:y1, x0:x1])
    assert got is not None
    # The recovered warp must register b onto a about as well as the TRUE warp
    # does. Rendering (uint8 rounding, a jagged mask edge, interpolation) leaves
    # a floor no registration can beat, so the oracle is the yardstick.
    inside = msk > 0.5
    reg = rg.warp_inverse(b, rg.to_frame(got[3], x0, y0), (H, W))
    true = rg.warp(b, _rot(ang, dx, dy), (H, W))
    after = np.abs(reg - a)[inside].mean()
    oracle = np.abs(true - a)[inside].mean()
    assert after <= 1.10 * oracle + 0.05


def test_rigid_between_angle_is_the_true_rotation_magnitude():
    bg, layer, msk = _scene()
    a = rg.blur(_render(bg, layer, msk, I2))
    for ang in (4.0, -6.0, 10.0):
        b = rg.blur(_render(bg, layer, msk, rg.invert(_rot(ang, 0, 0))))
        got = rg.rigid_between(a[190:291, 240:401], b[190:291, 240:401])
        assert got is not None
        assert abs(abs(got[0]) - abs(ang)) < 0.1


def test_rigid_between_is_exactly_the_identity_on_identical_crops():
    bg, layer, msk = _scene()
    a = rg.blur(_render(bg, layer, msk, I2))[190:291, 240:401]
    got = rg.rigid_between(a, a.copy())
    assert got is not None
    assert got[:3] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("h,w", [(101, 161), (128, 160), (120, 180)])
def test_rigid_between_never_writes_into_the_frame(h, w):
    """OpenCV's phaseCorrelate mutates its inputs at DFT-optimal sizes; a crop is
    a view of the frame the next pair is differenced against."""
    bg, layer, msk = _scene()
    a = rg.blur(_render(bg, layer, msk, I2))
    b = rg.blur(_render(bg, layer, msk, rg.invert(_rot(3, 1, 0))))
    a0, b0 = a.copy(), b.copy()
    rg.rigid_between(a[180:180 + h, 240:240 + w], b[180:180 + h, 240:240 + w])
    assert np.array_equal(a, a0) and np.array_equal(b, b0)


# ---- §2 gate 1: duplicate frames give exactly zero, for B and P ------------

def test_duplicate_frames_are_exactly_zero_for_B_and_P_but_not_K():
    Ms = [I2] * 12
    bg, frames, pose = _sequence(Ms, jitter_px=1.5)
    s = _scan(frames, pose, bg)
    assert s["identical"][1:].all()
    r = 60.0
    for arm in ("B", "P"):
        v = s[f"{arm}|head|{r}"][1:]
        assert np.isfinite(v).all()
        assert np.max(v) <= 1e-6, arm
    # K is handed jittered poses of identical images: it reports motion.
    assert np.nanmax(s[f"K|head|{r}"][1:]) > 0.1


# ---- §5-like: a rigidly moving animal, registered without keypoints --------

def test_rigid_motion_leaves_little_residual_in_B_and_P():
    steps = [_rot(1.0 * k, 1.5 * k, -0.7 * k) for k in range(12)]
    bg, frames, pose = _sequence(steps, jitter_px=1.5)
    s = _scan(frames, pose, bg)
    r = 60.0
    med = {a: float(np.nanmedian(s[f"{a}|head|{r}"][1:])) for a in rg.ARMS}
    # The unregistered difference inside the animal, for scale.
    _, _, msk = _scene()
    raw = float(np.median([
        np.abs(rg.blur(frames[t]) - rg.blur(frames[t - 1]))[
            cv2.warpAffine(msk, steps[t - 1], (W, H)) > 0.5].mean()
        for t in range(1, len(frames))]))
    assert med["P"] < med["K"]
    assert med["P"] < 0.5 * raw
    assert np.isfinite(med["B"])


# ---- §7: K is the incumbent, to the bit ------------------------------------

def test_K_reproduces_scan_head_exactly(tmp_path):
    steps = [_rot(0.5 * k, 1.0 * k, 0.3 * k) for k in range(15)]
    bg, frames, pose = _sequence(steps, jitter_px=1.0)
    path = os.path.join(tmp_path, "s.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (W, H))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()
    radii = [40.0, 60.0]
    ref = hd.scan_head(path, pose, radii_px=radii, dilate_px=100.0)

    def open_frames():
        cap = cv2.VideoCapture(path)
        try:
            while True:
                ok, fr = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        finally:
            cap.release()

    bgb = rg.blur(bg.astype(np.uint8))
    s = rg.scan_register(open_frames, pose, bg=bgb, cutoff=15.0,
                         radii_px=radii, dilate_px=100.0, win=5,
                         body_length_px=100.0)
    for r in radii:
        a = ref[f"energy_ego|{r}"]
        b = s[f"K|head|{r}"] - s["arena"]
        assert a.shape == b.shape
        both = np.isfinite(a)
        assert (both == np.isfinite(b)).all()
        assert np.max(np.abs(a[both] - b[both])) <= 1e-9
        c = ref[f"energy_ego_hip|{r}"]
        d = s[f"K|hip|{r}"] - s["arena"]
        ok = np.isfinite(c)
        assert np.max(np.abs(c[ok] - d[ok])) <= 1e-9


# ---- STABILISE 2 §1: the background the animal is not in --------------------

def test_masked_median_background_drops_an_animal_that_sits_still():
    """An animal parked for most of the samples stays in a plain median and is
    removed by the masked one."""
    bg, layer, msk = _scene()
    parked = [I2] * 70
    walk = [_rot(0, dx, 0) for dx in np.linspace(-250, 250, 31)]
    Ms = parked + walk
    frames = [rg.blur(_render(bg, layer, msk, M)) for M in Ms]
    poses = [rg.apply(M, _pose0()) for M in Ms]
    plain = rg.median_background(frames)
    masked, und = rg.masked_median_background(frames, poses, dilate_px=20.0,
                                              min_samples=10)
    inside = msk > 0.5
    truth = rg.blur(bg.astype(np.uint8))
    assert np.abs(plain - truth)[inside].mean() > 20.0
    assert np.abs(masked - truth)[inside].mean() < 1.0
    assert und == 0.0


def test_masked_median_background_reports_pixels_it_never_saw():
    bg, layer, msk = _scene()
    frames = [rg.blur(_render(bg, layer, msk, I2))] * 20
    poses = [_pose0()] * 20
    _, und = rg.masked_median_background(frames, poses, dilate_px=20.0,
                                         min_samples=10)
    assert und > 0.0
