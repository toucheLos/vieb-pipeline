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
