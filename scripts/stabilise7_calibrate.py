"""STABILISE 7, before its registration: calibrate gates 3 and 4 on the ANIMAL'S pixels.

    sbatch jobs/stabilise7_calibrate.slurm       # GPU: SAM masks only
    python3 scripts/stabilise7_calibrate.py --phase combine

`STABILISE6.md` §2: gate 3's 0.6 bl disc reaches the bar floor, which a fit
to the animal's own sub-pixel motion moves. STABILISE 7 scores gates 3-5 on
the disc INTERSECTED with SAM's eroded mask of the animal. The bars for that
region are set here, **from the incumbent alone** -- no candidate arm is
computed -- exactly as `stabilise3_calibrate.py` set STABILISE 3's:

* **Gate 3:** on census immobile frames (first 60 pairs per recording), K's
  in-disc |delta| on animal pixels against the IDENTITY's on animal pixels.
  For a still animal the identity is the perfect registration. Ratio of means
  per animal.
* **Gate 4:** each planted moving trajectory scanned as K (real still-window
  jitter) and K0 (perfect poses), against the true-transform oracle, in the
  0.2 bl disc on animal pixels.

SAM is prompted from the padded keypoint box (STABILISE 4 §1) and its masks
are eroded 0.04 bl (STABILISE 5 Amendment 1), as the candidates will be.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot                                      # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.render import video as vid                                # noqa: E402
from recur.util import frames, log, write_json                       # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.pixel import head as hd, motion as mo, register as rg, sam  # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stabilise6 as s6                                              # noqa: E402

s3, s2, st = s6.s3, s6.s2, s6.st
G3_PAIRS = 60
G3_BL = 0.6
G4_BL = 0.2


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "stabilise7_cal")


def _eroded(m, bl: float):
    import cv2

    r = int(round(s6.ERODE_BL * bl))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    e = cv2.erode(np.asarray(m, dtype=np.uint8), ker) > 0
    return e if e.any() else None


def _sam_mask(predict, grey, pose_frame, bl):
    box = sam.keypoint_box(pose_frame, sam.PAD_BL * bl)
    if box is None:
        return None
    m, iou = predict(s3._as_rgb([grey])[0], box)
    if iou < sam.IOU_MIN or not np.asarray(m).any():
        return None
    return _eroded(m, bl)


def _gate3_cal(video, pose, immobile, bl, w, predict) -> dict:
    t_im = np.flatnonzero(immobile)
    t_im = t_im[(t_im >= 1) & (t_im < pose.shape[0])][:G3_PAIRS]
    r = G3_BL * bl
    num = den = 0.0
    n = 0
    for t in t_im:
        fr = st._frames_at(video, [t - 1, t])
        if len(fr) != 2:
            continue
        a, b = rg.blur(fr[0]), rg.blur(fr[1])
        shape = a.shape
        m0 = _sam_mask(predict, fr[0], pose[t - 1], bl)
        m1 = _sam_mask(predict, fr[1], pose[t], bl)
        pt = pose[t]
        if m0 is None or m1 is None or not np.isfinite(pt).all():
            continue
        # K, in its body frame, on the animal carried there by K's own warp.
        wk0 = hd._ego_warp(a, pose[t - 1], shape)
        wk1 = hd._ego_warp(b, pt, shape)
        dmk = hd._fixed_disc(pt, hd.SKULL, r, shape)
        if wk0 is None or wk1 is None or dmk is None:
            continue
        mk = rg.frame_matrix(float(pt[tego.ORIGIN, 0]), float(pt[tego.ORIGIN, 1]),
                             float(tego.heading(pt[None])[0]), shape)
        regk = rg.warp(m1.astype(np.float64), mk, shape) > 0.5
        # The identity, in t-1's image coordinates, on the animal at t-1.
        k = t // w
        seg = pose[k * w:(k + 1) * w][:, list(hd.SKULL)].reshape(-1, 2)
        c = np.nanmedian(seg, axis=0)
        dmi = rg.disc(float(c[0]), float(c[1]), r, shape)
        if dmi is None or not (dmk & regk).any() or not (dmi & m0).any():
            continue
        num += float(np.abs(wk1 - wk0)[dmk & regk].mean())
        den += float(np.abs(b - a)[dmi & m0].mean())
        n += 1
    return {"g3_num": num, "g3_den": den, "g3_n": n}


def _gate4_cal(video, pose, speed, immobile, bg_raw, bl, fps, w, predict) -> dict:
    inp = st._plant_inputs(pose, speed, immobile, w)
    if inp is None:
        return {}
    src = st._frames_at(video, [inp["src"]])
    if not src:
        return {}
    sg, sp = src[0], pose[inp["src"]]
    box = sam.keypoint_box(sp, sam.PAD_BL * bl)
    if box is None:
        return {}
    m_raw, iou = predict(s3._as_rgb([sg])[0], box)
    if iou < sam.IOU_MIN or not np.asarray(m_raw).any():
        return {}
    m_src = _eroded(m_raw, bl)
    if m_src is None:
        return {}
    r = G4_BL * bl
    out: dict = {}
    for label, jit in (("K", inp["jit_body"]),
                       ("K0", np.zeros_like(inp["jit_body"]))):
        fr, poses, Ms = st._plant_frames(sg, sp, bg_raw, m_raw, bl,
                                         {**inp, "jit_body": jit}, moving=True,
                                         hz=None, amp=None, fps=fps)
        rgbs = s3._as_rgb(fr)
        tr = sam.sam_track(lambda: iter(rgbs), poses, predict,
                           body_length_px=bl, prompt_mode=s6.PROMPT)
        s = rg.scan_track(lambda: iter(fr), poses, tr, radii_px=[r],
                          dilate_px=bl * mo.DILATE_BODY_LENGTHS, win=len(fr),
                          region_fn=lambda t, tr=tr: sam.mask_at(
                              tr, t, erode_px=s6.ERODE_BL * bl))
        o = st._oracle(fr, Ms, sp, [r], region=m_src)
        kv = np.asarray(s[f"K|head|{r}|a"])[1:]
        ov = np.asarray(o[f"O|head|{r}|a"])[1:]
        if np.isfinite(kv).any() and np.isfinite(ov).any() and np.nanmedian(ov) > 0:
            out[label] = float(np.nanmedian(kv) / np.nanmedian(ov))
            out["oracle"] = float(np.nanmedian(ov))
    return out


def shard(a) -> int:
    s3._check_checkpoint()
    gg = st._load("grooming_gate")
    pp = st._load("pixel_pilot")
    predict = sam.sam_predictor(s3.CHECKPOINT)
    fps = spine.fps()
    w = int(frames(st.WIN_S, fps))
    os.makedirs(os.path.join(work_dir(), "rec"), exist_ok=True)
    rows = [r for r in st.manifest()["recordings"] if r.get("in_pilot")]
    mine = rows[a.shard::a.of]
    if a.limit:
        mine = mine[:a.limit]
    for n, r in enumerate(mine, 1):
        rid = r["recording_id"]
        out = os.path.join(work_dir(), "rec", f"{rid}.npz")
        if os.path.exists(out) and not a.force:
            continue
        pix = st._pixel(rid)
        if pix is None or not bool(pix["usable"]):
            continue
        pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
        bl = float(np.nanmedian(tego.body_length(pose)))
        video = vid.video_path(rid)
        nf = min(int(pix["n_frames"]), pose.shape[0])
        idx = np.unique(np.round(np.linspace(0, nf - 1, s2.BG_SAMPLES)).astype(int))
        raw = st._frames_at(video, idx)
        bg_raw, _ = rg.masked_median_background(
            [np.asarray(f, dtype=np.float64) for f in raw],
            [pose[i] for i in idx[:len(raw)]], dilate_px=bl * mo.DILATE_BODY_LENGTHS,
            min_samples=s2.MIN_SAMPLES)
        sp_ = gg._speed(r["animal"], rid)
        speed = np.asarray(sp_, dtype=np.float64) if sp_ is not None else np.zeros(0)
        still = pp._still_frames(r["animal"], rid, int(pix["n_frames"]))
        immobile = (st._immobile(pix, still) if still is not None
                    else np.zeros(0, dtype=bool))
        got: dict = {}
        if immobile.size:
            got.update(_gate3_cal(video, pose, immobile, bl, w, predict))
            if speed.size:
                got.update(_gate4_cal(video, pose, speed, immobile, bg_raw, bl,
                                      fps, w, predict))
        np.savez_compressed(out, **got)
        log(f"  {n}/{len(mine)} {rid}: {got}")
    return 0


def combine(a) -> int:
    meta = {r["recording_id"]: r for r in st.manifest()["recordings"]
            if r.get("in_pilot")}
    g4: dict[str, dict[str, list[float]]] = {}
    g3: dict[str, list[float]] = {}
    for rid, m in meta.items():
        p = os.path.join(work_dir(), "rec", f"{rid}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=False) as z:
            d = {k: float(z[k]) for k in z.files}
        an = m["animal"]
        for k in ("K", "K0", "oracle"):
            if k in d and np.isfinite(d[k]):
                g4.setdefault(k, {}).setdefault(an, []).append(d[k])
        if d.get("g3_n", 0) > 0 and d.get("g3_den", 0) > 0:
            x = g3.setdefault(an, [0.0, 0.0, 0.0])
            x[0] += d["g3_num"]
            x[1] += d["g3_den"]
            x[2] += d["g3_n"]
    res: dict = {}
    for k, per in g4.items():
        vals = [float(np.mean(v)) for v in per.values()]
        res[f"gate4_{k}"] = boot.animal_interval(vals, list(per), how="mean",
                                                 n_boot=st.N_BOOT, seed=st.SEED)
        b = res[f"gate4_{k}"]
        log(f"  gate4 {k}: {b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]")
    per3 = {an: v[0] / v[1] for an, v in g3.items()
            if v[2] >= s2.MIN_IMMOBILE_FRAMES and v[1] > 0}
    if len(per3) >= 2:
        res["gate3_K_over_identity"] = boot.animal_interval(
            list(per3.values()), list(per3), how="mean", n_boot=st.N_BOOT,
            seed=st.SEED)
        res["gate3_n_animals"] = len(per3)
        b = res["gate3_K_over_identity"]
        log(f"  gate3 K/identity: {b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
            f" over {len(per3)} animals")
    out = a.out or config.PATHS.result("stabilise7_calibration.json")
    write_json({**provenance.header(anchors.LUNA, stage="stabilise7_calibration",
                                    unverified="a calibration of the incumbent"),
                "inherited_digest": spine.digest(), "results": res}, out)
    log(f"  wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("combine",), default=None)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.phase == "combine":
        return combine(a)
    if a.shard is None:
        raise SystemExit("pass --shard I --of N, or --phase combine")
    return shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
