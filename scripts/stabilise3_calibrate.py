"""STABILISE 3, before its registration: calibrate gates 3 and 4 on REAL data.

    sbatch jobs/stabilise3_calibrate.slurm
    python3 scripts/stabilise3_calibrate.py --phase combine

`DEVIATIONS.md` D24: STABILISE 2's gate 4 bar was calibrated on a synthetic
scene and did not transfer, and its gate 3 averaged ratios over a near-zero
denominator. This script measures, on the real sample, what those gates can
separate, using ONLY the incumbent: no candidate arm is computed here, so the
bars it informs are set before any candidate exists.

* **Gate 4:** each planted trajectory is scanned twice, K with its real
  still-window jitter and K0 with perfect poses, against the true-transform
  oracle. This is done at the headline disc (0.6 bl) and at a small interior disc
  (0.2 bl), where paste edges cannot enter.
* **Gate 3:** on census immobile frames, K's in-disc |delta| against the
  IDENTITY difference in an image-coordinate disc at the window-median skull
  position. For a still animal the identity is the perfect registration.
  Ratio of means per animal, not a mean of ratios.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot                                      # noqa: E402
from recur.render import video as vid                                # noqa: E402
from recur.util import frames, log, write_json                       # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.pixel import head as hd, register as rg                    # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stabilise2 as s2                                              # noqa: E402

st = s2.st
RADII_CAL = (0.2, 0.6)


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "stabilise3_cal")


def _gate4_cal(video, pose, speed, immobile, bg_raw, bg_blur, cutoff, bl, fps,
               w) -> dict:
    inp = st._plant_inputs(pose, speed, immobile, w)
    if inp is None:
        return {}
    src = st._frames_at(video, [inp["src"]])
    if not src:
        return {}
    out: dict = {}
    saved = st.HEADLINE_BL
    try:
        for rad in RADII_CAL:
            st.HEADLINE_BL = rad
            r = bl * rad
            for label, jit in (("K", inp["jit_body"]),
                               ("K0", np.zeros_like(inp["jit_body"]))):
                s = st._plant_scan(src[0], pose[inp["src"]], bg_raw, bg_blur,
                                   cutoff, bl, {**inp, "jit_body": jit},
                                   moving=True, hz=None, amp=None, fps=fps,
                                   oracle=True)
                k = np.asarray(s.get(f"K|head|{r}", np.zeros(0)))[1:]
                o = np.asarray(s.get(f"O|head|{r}", np.zeros(0)))[1:]
                if k.size and np.isfinite(o).any() and np.nanmedian(o) > 0:
                    out[f"{label}|{rad:g}"] = float(np.nanmedian(k)
                                                    / np.nanmedian(o))
                    out[f"oracle|{rad:g}"] = float(np.nanmedian(o))
    finally:
        st.HEADLINE_BL = saved
    return out


def _gate3_cal(video, pose, immobile, kz: dict, bl: float, w: int) -> dict:
    """K vs identity on immobile frames, ratio of MEANS within the recording."""
    t_im = np.flatnonzero(immobile)
    t_im = t_im[(t_im >= 1) & (t_im < pose.shape[0])]
    if t_im.size == 0:
        return {}
    r = bl * st.HEADLINE_BL
    kser = np.asarray(kz["K__head__0.6"], dtype=np.float64)
    num, den, n = 0.0, 0.0, 0
    for t in t_im:
        k = t // w
        seg = pose[k * w:(k + 1) * w][:, list(hd.SKULL)]
        c = np.nanmedian(seg.reshape(-1, 2) if seg.size else np.full((1, 2), np.nan),
                         axis=0)
        if not np.isfinite(c).all() or t >= kser.size or not np.isfinite(kser[t]):
            continue
        fr = st._frames_at(video, [t - 1, t])
        if len(fr) != 2:
            continue
        a, b = rg.blur(fr[0]), rg.blur(fr[1])
        dm = rg.disc(float(c[0]), float(c[1]), r, a.shape)
        if dm is None:
            continue
        num += float(kser[t])
        den += float(np.abs(b - a)[dm].mean())
        n += 1
    return {"g3_num": num, "g3_den": den, "g3_n": n}


def shard(a) -> int:
    gg = st._load("grooming_gate")
    pp = st._load("pixel_pilot")
    fps = spine.fps()
    w = int(frames(st.WIN_S, fps))
    os.makedirs(os.path.join(work_dir(), "rec"), exist_ok=True)
    rows = [r for r in st.manifest()["recordings"] if r.get("in_pilot")]
    mine = rows[a.shard::a.of]
    for n, r in enumerate(mine, 1):
        rid = r["recording_id"]
        out = os.path.join(work_dir(), "rec", f"{rid}.npz")
        if os.path.exists(out) and not a.force:
            continue
        pix = st._pixel(rid)
        kpath = os.path.join(config.REPO, "work", "stabilise", "rec", f"{rid}.npz")
        if pix is None or not bool(pix["usable"]) or not os.path.exists(kpath):
            continue
        with np.load(kpath, allow_pickle=False) as z:
            kz = {k: z[k] for k in ("K__head__0.6",)}
        cutv = dict(zip([str(k) for k in pix["cutoff_keys"]],
                        [float(v) for v in pix["cutoff_values"]]))
        pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
        bl = float(np.nanmedian(tego.body_length(pose)))
        video = vid.video_path(rid)
        nf = min(int(pix["n_frames"]), pose.shape[0])
        idx = np.unique(np.round(np.linspace(0, nf - 1, s2.BG_SAMPLES)).astype(int))
        raw = st._frames_at(video, idx)
        idx = idx[:len(raw)]
        dil = bl * st.mo.DILATE_BODY_LENGTHS
        bg_blur, _ = rg.masked_median_background(
            [rg.blur(f) for f in raw], [pose[i] for i in idx], dilate_px=dil,
            min_samples=s2.MIN_SAMPLES)
        bg_raw, _ = rg.masked_median_background(
            [np.asarray(f, dtype=np.float64) for f in raw],
            [pose[i] for i in idx], dilate_px=dil, min_samples=s2.MIN_SAMPLES)
        sp = gg._speed(r["animal"], rid)
        speed = np.asarray(sp, dtype=np.float64) if sp is not None else np.zeros(0)
        still = pp._still_frames(r["animal"], rid, int(pix["n_frames"]))
        immobile = (st._immobile(pix, still) if still is not None
                    else np.zeros(0, dtype=bool))
        got = {}
        if speed.size and immobile.size:
            got.update(_gate4_cal(video, pose, speed, immobile, bg_raw, bg_blur,
                                  cutv["m2"], bl, fps, w))
            got.update(_gate3_cal(video, pose, immobile, kz, bl, w))
        np.savez_compressed(out, **{k.replace("|", "__"): v
                                    for k, v in got.items()})
        log(f"  {n}/{len(mine)} {rid}: {got}")
    return 0


def combine(a) -> int:
    meta = {r["recording_id"]: r for r in st.manifest()["recordings"]
            if r.get("in_pilot")}
    by: dict[str, dict[str, list[float]]] = {}
    g3: dict[str, list[float]] = {}
    for rid, m in meta.items():
        p = os.path.join(work_dir(), "rec", f"{rid}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=False) as z:
            d = {k: float(z[k]) for k in z.files}
        an = m["animal"]
        for k, v in d.items():
            if k.startswith(("K__", "K0__", "oracle__")) and np.isfinite(v):
                by.setdefault(k, {}).setdefault(an, []).append(v)
        if d.get("g3_n", 0) > 0 and d["g3_den"] > 0:
            g3.setdefault(an, []).extend([d["g3_num"], d["g3_den"], d["g3_n"]])
    res: dict = {}
    for k, per in sorted(by.items()):
        units = [float(np.mean(v)) for v in per.values()]
        allv = np.concatenate([np.asarray(v) for v in per.values()])
        res[k] = {"animal_mean": boot.animal_interval(
            units, list(per), how="mean", n_boot=st.N_BOOT, seed=st.SEED),
            "recording_pct_5_50_95": np.percentile(allv, [5, 50, 95]).tolist(),
            "n_recordings": int(allv.size)}
        log(f"  {k}: {res[k]['animal_mean']['point']:.3f} "
            f"[{res[k]['animal_mean']['lo']:.3f}, {res[k]['animal_mean']['hi']:.3f}]"
            f"  recordings 5/50/95 {np.round(res[k]['recording_pct_5_50_95'], 3)}")
    ratios, who = [], []
    for an, v in g3.items():
        x = np.asarray(v).reshape(-1, 3)
        if x[:, 2].sum() >= s2.MIN_IMMOBILE_FRAMES:
            ratios.append(float(x[:, 0].sum() / x[:, 1].sum()))
            who.append(an)
    if len(ratios) >= 2:
        res["gate3_K_over_identity"] = {
            "animal_mean": boot.animal_interval(ratios, who, how="mean",
                                                n_boot=st.N_BOOT, seed=st.SEED),
            "animal_values_5_50_95": np.percentile(ratios, [5, 50, 95]).tolist(),
            "n_animals": len(ratios)}
        g = res["gate3_K_over_identity"]["animal_mean"]
        log(f"  gate3 K/identity: {g['point']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}] "
            f"over {len(ratios)} animals")
    out = a.out or config.PATHS.result("stabilise3_calibration.json")
    write_json({**anchors.header(anchors.LUNA, stage="stabilise3_calibration",
                                 unverified="a calibration of the incumbent"),
                "inherited_digest": spine.digest(), "results": res}, out)
    log(f"  wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("combine",), default=None)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=12)
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
