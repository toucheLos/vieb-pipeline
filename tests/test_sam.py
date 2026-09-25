"""`vieb.pixel.sam`: the keyframe track, with the segmenter stood in.

The stand-in returns the TRUE ellipse mask of the synthetic scene, so the track
logic -- prompting, refusal, re-seeding, interpolation, gate 0 -- is tested
without the model.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tests import test_register as T                               # noqa: E402
from vieb.pixel import register as rg, sam                           # noqa: E402
from vieb.tok import ego as tego                                     # noqa: E402


def _truth_predictor(masks, ious=None, calls=None):
    def predict(rgb, box):
        t = int(rgb[0, 0, 0])            # the frame index is stamped in pixel 0
        if calls is not None:
            calls.append((t, np.asarray(box).copy()))
        return masks[t], (1.0 if ious is None else ious[t])
    return predict


def _seq(n, step=1.0):
    Ms = [T._rot(0.5 * k * step, 1.2 * k * step, -0.4 * k * step) for k in range(n)]
    bg, layer, msk = T._scene()
    frames, masks, poses = [], [], []
    for t, M in enumerate(Ms):
        g = T._render(bg, layer, msk, M)
        rgb = np.repeat(g[:, :, None], 3, axis=2)
        rgb[0, 0, :] = t
        frames.append(rgb)
        masks.append(cv2.warpAffine(msk, M, (T.W, T.H)) > 0.5)
        poses.append(rg.apply(M, T._pose0()))
    return frames, masks, np.asarray(poses), Ms


def test_keyframes_every_third_and_seeded_once():
    frames, masks, pose, _ = _seq(13)
    calls = []
    tr = sam.sam_track(lambda: iter(frames), pose,
                       _truth_predictor(masks, calls=calls), body_length_px=100.0)
    assert [c[0] for c in calls] == [0, 3, 6, 9, 12]
    assert tr["key_seeded"].tolist() == [True, False, False, False, False]
    assert tr["ok"][:13].all()


def test_interpolated_pose_tracks_the_true_centroid():
    frames, masks, pose, _ = _seq(13)
    tr = sam.sam_track(lambda: iter(frames), pose, _truth_predictor(masks),
                       body_length_px=100.0)
    for t in (1, 4, 8):
        cx, cy, _, _ = rg.mask_pose(masks[t])
        assert abs(tr["cx"][t] - cx) < 1.0 and abs(tr["cy"][t] - cy) < 1.0


def test_a_low_iou_keyframe_refuses_its_brackets_and_forces_a_reseed():
    frames, masks, pose, _ = _seq(13)
    ious = [1.0] * 13
    ious[6] = 0.5
    calls = []
    tr = sam.sam_track(lambda: iter(frames), pose,
                       _truth_predictor(masks, ious, calls), body_length_px=100.0)
    assert tr["key_seeded"].tolist() == [True, False, False, True, False]
    assert not tr["ok"][3:9].any()          # both spans touching keyframe 6
    assert tr["ok"][:3].all() and tr["ok"][9:13].all()


def test_gate0_counts_a_disc_centre_inside_the_mask():
    frames, masks, pose, _ = _seq(7)
    tr = sam.sam_track(lambda: iter(frames), pose, _truth_predictor(masks),
                       body_length_px=100.0)
    inside_c = np.array([[tr["cx"][t], tr["cy"][t]] for t in range(7)])
    assert sam.on_animal(tr, inside_c) == (3, 3)
    outside_c = inside_c + 200.0
    assert sam.on_animal(tr, outside_c) == (0, 3)


def test_scan_track_runs_on_a_sam_track_and_B_P_stay_exact_on_duplicates():
    frames, masks, pose, _ = _seq(1)
    frames = [frames[0]] * 10
    masks = [masks[0]] * 10
    pose = np.repeat(pose[:1], 10, axis=0)
    # re-stamp so the stand-in finds the mask
    fr = []
    for t, f in enumerate(frames):
        g = f.copy()
        g[0, 0, :] = 0
        fr.append(g)
    tr = sam.sam_track(lambda: iter(fr), pose, _truth_predictor(masks),
                       body_length_px=100.0)
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in fr]
    s = rg.scan_track(lambda: iter(grey), pose, tr, radii_px=[60.0],
                      dilate_px=100.0, win=5)
    for arm in ("B", "P", "I"):
        v = s[f"{arm}|head|60.0"][1:]
        assert np.isfinite(v).all() and np.max(v) <= 1e-6


def test_keypoint_mode_prompts_every_keyframe_from_the_pose():
    frames, masks, pose, _ = _seq(10)
    calls = []
    tr = sam.sam_track(lambda: iter(frames), pose,
                       _truth_predictor(masks, calls=calls),
                       body_length_px=100.0, prompt_mode="keypoint")
    assert tr["key_seeded"].all()
    for t, box in calls:
        exp = sam.keypoint_box(pose[t], 25.0)
        assert np.allclose(box, exp)


def test_keypoint_mode_refuses_a_mask_of_some_other_object():
    frames, masks, pose, _ = _seq(10)
    wall = np.zeros_like(masks[0])
    wall[:40, :] = True                      # a bright strip far from the animal
    bad = list(masks)
    bad[3] = wall
    tr = sam.sam_track(lambda: iter(frames), pose, _truth_predictor(bad),
                       body_length_px=100.0, prompt_mode="keypoint")
    assert not tr["key_accepted"][1]
    assert not tr["ok"][0:6].any() and tr["ok"][6:10].all()


def test_mask_at_carries_the_keyframe_mask_by_the_centroid_shift():
    frames, masks, pose, _ = _seq(7)
    tr = sam.sam_track(lambda: iter(frames), pose, _truth_predictor(masks),
                       body_length_px=100.0, prompt_mode="keypoint")
    m = sam.mask_at(tr, 4, dilate_px=0.0)
    truth = masks[4]
    iou = (m & truth).sum() / (m | truth).sum()
    assert iou > 0.85


def test_masked_arm_follows_a_moving_animal_over_a_textured_floor():
    """STABILISE 5 Amendment 1 (D26), pinned: over a high-contrast bar floor,
    ECC restricted to the animal's ERODED masks on both frames follows it."""
    bg, layer, msk = T._scene()
    yy, xx = np.mgrid[:T.H, :T.W]
    floor = np.where((xx // 6) % 2 == 0, 230.0, 20.0)       # a bar grid
    steps = [T._rot(0.8 * k, 2.5 * k, -1.0 * k) for k in range(10)]
    frames, masks, poses = [], [], []
    for t, M in enumerate(steps):
        L = cv2.warpAffine(layer, M, (T.W, T.H))
        A = cv2.warpAffine(msk, M, (T.W, T.H)) > 0.5
        g = np.clip(np.round(np.where(A, L, floor)), 0, 255).astype(np.uint8)
        rgb = np.repeat(g[:, :, None], 3, axis=2)
        rgb[0, 0, :] = t
        frames.append(rgb)
        masks.append(A)
        poses.append(rg.apply(M, T._pose0()))
    pose = np.asarray(poses)
    tr = sam.sam_track(lambda: iter(frames), pose, _truth_predictor(masks),
                       body_length_px=100.0, prompt_mode="keypoint")
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    # 12 px: inside the body. A 0.6 bl disc reaches onto the floor, where a
    # correctly registered ANIMAL leaves the static bars misaligned by
    # construction -- which is why gate 4 is scored at an interior disc.
    s = rg.scan_track(lambda: iter(grey), pose, tr, radii_px=[12.0],
                      dilate_px=100.0, win=10,
                      mask_fn=lambda t: sam.mask_at(tr, t, erode_px=4.0))
    c = pose[:, tego.CENTER]
    f = {}
    for arm in ("P", "M"):
        W = s[f"{arm}|W"]
        num = den = 0.0
        for t in range(1, len(grey)):
            if not np.isfinite(W[t]).all():
                continue
            d_arm = rg.apply(W[t], c[t - 1])[0] - c[t - 1]
            d_kp = c[t] - c[t - 1]
            num += float(d_arm @ d_arm)
            den += float(d_kp @ d_arm)
        f[arm] = num / den if den else float("nan")
    assert 0.9 < f["M"] < 1.1, f
    assert np.nanmedian(s["M|head|12.0"][1:]) < np.nanmedian(s["P|head|12.0"][1:])


def test_animal_region_scoring_removes_the_floor_a_disc_would_include():
    """STABILISE 7: on a bar floor, a correctly registered animal leaves the
    floor misaligned inside a big disc; scored on the animal's own pixels the
    residual is small, and far below the full-disc residual."""
    bg, layer, msk = T._scene()
    yy, xx = np.mgrid[:T.H, :T.W]
    floor = np.where((xx // 6) % 2 == 0, 230.0, 20.0)
    steps = [T._rot(0.8 * k, 2.5 * k, -1.0 * k) for k in range(8)]
    frames, masks, poses = [], [], []
    for t, M in enumerate(steps):
        L = cv2.warpAffine(layer, M, (T.W, T.H))
        A = cv2.warpAffine(msk, M, (T.W, T.H)) > 0.5
        g = np.clip(np.round(np.where(A, L, floor)), 0, 255).astype(np.uint8)
        rgb = np.repeat(g[:, :, None], 3, axis=2)
        rgb[0, 0, :] = t
        frames.append(rgb)
        masks.append(A)
        poses.append(rg.apply(M, T._pose0()))
    pose = np.asarray(poses)
    tr = sam.sam_track(lambda: iter(frames), pose, _truth_predictor(masks),
                       body_length_px=100.0, prompt_mode="keypoint")
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    mf = lambda t: sam.mask_at(tr, t, erode_px=4.0)          # noqa: E731
    s = rg.scan_track(lambda: iter(grey), pose, tr, radii_px=[60.0],
                      dilate_px=100.0, win=8, mask_fn=mf, best_start=True,
                      region_fn=mf)
    full = np.nanmedian(s["N|head|60.0"][1:])
    anim = np.nanmedian(s["N|head|60.0|a"][1:])
    assert np.isfinite(anim) and anim < 0.25 * full
    assert np.isfinite(np.nanmedian(s["K|head|60.0|a"][1:]))
