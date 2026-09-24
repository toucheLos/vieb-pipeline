"""STABILISE 4: SAM prompted from the located animal; SP only; jitter planted.

    sbatch jobs/stabilise4.slurm                 # 12 GPU shards
    python3 scripts/stabilise4.py --phase combine

READ results/STABILISE4_PREREGISTRATION.md FIRST. It changes three things in
STABILISE 3 and nothing else, so this script loads `stabilise3.py` and reuses
its gates 0, 1, 3, 4 and 5 unchanged:

* §1: `sam_track(..., prompt_mode="keypoint")`.
* §2: the arms are K and SP.
* §3: gate 2 is a planted-jitter response, K as its positive control.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot                                      # noqa: E402
from recur.read import Read                                          # noqa: E402
from recur.render import video as vid                                # noqa: E402
from recur.util import frames, log, write_json                       # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.pixel import motion as mo, register as rg, sam             # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stabilise3 as s3                                              # noqa: E402

s2, st = s3.s2, s3.st
ARMS = ("K", "SP")
RENAME = {"K": "K", "P": "SP", "I": "I"}
PROMPT = "keypoint"
#: §3.
JIT_WINDOWS = 10
JIT_SIGMA_PX = 2.0
JIT_BAND = (0.90, 1.10)
JIT_CONTROL = 1.25


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "stabilise4")


st.work_dir = work_dir
s3.STAGE = "stabilise4"
st.MIN_IMMOBILE_FRAMES = s2.MIN_IMMOBILE_FRAMES


def _block(video: str, a: int, n: int) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video)
    out = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(a))
        for _ in range(n):
            ok, fr = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    return out


def _jitter_plant(video, pose, speed, bl, w, predict) -> dict:
    """§3: the first 10 still windows, each run with and without planted jitter."""
    n = min(speed.size, pose.shape[0])
    k = n // w
    if k < 4:
        return {}
    S = speed[:k * w].reshape(k, w)
    ok = np.isfinite(S).all(axis=1)
    if ok.sum() < 4:
        return {}
    ms = S.mean(axis=1)
    still = np.flatnonzero(ok & (ms <= np.percentile(ms[ok], st.STILL_PCT)))
    still = still[:JIT_WINDOWS]
    rng = np.random.default_rng(st.SEED)
    r = bl * st.HEADLINE_BL
    sums = {f"{arm}__{run}": 0.0 for arm in ARMS for run in ("orig", "jit")}
    n_frames = 0
    for i in still:
        a = int(i) * w
        grey = _block(video, a, w)
        if len(grey) < w:
            continue
        rgbs = s3._as_rgb(grey)
        p0 = pose[a:a + w]
        pj = p0 + rng.normal(0.0, JIT_SIGMA_PX, p0.shape)
        got = {}
        for run, pw in (("orig", p0), ("jit", pj)):
            tr = sam.sam_track(lambda: iter(rgbs), pw, predict,
                               body_length_px=bl, prompt_mode=PROMPT)
            s = rg.scan_track(lambda: iter(grey), pw, tr, radii_px=[r],
                              dilate_px=bl * mo.DILATE_BODY_LENGTHS, win=w)
            got[run] = {arm: np.asarray(s[f"{src}|head|{r}"])
                        for src, arm in RENAME.items() if arm in ARMS}
        for arm in ARMS:
            both = np.isfinite(got["orig"][arm]) & np.isfinite(got["jit"][arm])
            sums[f"{arm}__orig"] += float(got["orig"][arm][both].sum())
            sums[f"{arm}__jit"] += float(got["jit"][arm][both].sum())
        n_frames += w
    return {f"jit__{k_}": v for k_, v in sums.items()} | {"jit__frames": n_frames}


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
    log(f"  shard {a.shard}/{a.of}: {len(mine)} recordings")
    for n, r in enumerate(mine, 1):
        rid = r["recording_id"]
        out = os.path.join(work_dir(), "rec", f"{rid}.npz")
        if os.path.exists(out) and not a.force:
            continue
        pix = st._pixel(rid)
        if pix is None or not bool(pix["usable"]):
            log(f"  {n}/{len(mine)} {rid}: no usable pixel scan, skipped")
            continue
        pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
        bl = float(np.nanmedian(tego.body_length(pose)))
        dil = bl * mo.DILATE_BODY_LENGTHS
        video = vid.video_path(rid)
        tr = sam.sam_track(s3._rgb(video), pose, predict, body_length_px=bl,
                           prompt_mode=PROMPT)
        s = rg.scan_track(st._open(video), pose, tr,
                          radii_px=[bl * x for x in st.RADII_BL],
                          dilate_px=dil, win=w)
        got = {f"{RENAME[arm]}__{reg}__{st._rkey(x)}": s[f"{arm}|{reg}|{bl * x}"]
               for arm in RENAME for reg, _ in rg.REGIONS for x in st.RADII_BL}
        inside, scored = sam.on_animal(tr, s["centre_head_img"])
        nf = min(int(pix["n_frames"]), pose.shape[0])
        idx = np.unique(np.round(np.linspace(0, nf - 1, s2.BG_SAMPLES)).astype(int))
        raw = st._frames_at(video, idx)
        idx = idx[:len(raw)]
        bg_raw, _ = rg.masked_median_background(
            [np.asarray(f, dtype=np.float64) for f in raw],
            [pose[i] for i in idx], dilate_px=dil, min_samples=s2.MIN_SAMPLES)
        sp_ = gg._speed(r["animal"], rid)
        speed = (np.asarray(sp_, dtype=np.float64) if sp_ is not None
                 else np.zeros(0))
        still = pp._still_frames(r["animal"], rid, int(pix["n_frames"]))
        immobile = (st._immobile(pix, still) if still is not None
                    else np.zeros(0, dtype=bool))
        # §1 applies inside the planted sequences too.
        plants = (s3._plants3(video, pose, speed, immobile, bg_raw, bl, fps, w,
                              predict, prompt_mode=PROMPT)
                  if speed.size and immobile.size else {"plant_ok": False})
        jit = _jitter_plant(video, pose, speed, bl, w, predict) if speed.size else {}
        nan_frac = float(1.0 - np.mean(tr["ok"])) if tr["n"] else 1.0
        np.savez_compressed(
            out, body_length_px=bl, win=w, arena=s["arena"],
            identical=s["identical"], mask_area=s["mask_area"],
            mask_ok=s["mask_ok"], p_estimate=s["p_estimate"],
            nan_frac=nan_frac, key_iou=tr["key_iou"],
            key_accepted=tr["key_accepted"],
            gate0_inside=inside, gate0_scored=scored,
            speed=speed, immobile=immobile, **got, **plants, **jit)
        log(f"  {n}/{len(mine)} {rid}: refused {nan_frac:.3f}, accepted "
            f"{int(tr['key_accepted'].sum())}/{tr['key_accepted'].size}, "
            f"on-animal {inside}/{scored}, jitter K "
            f"{jit.get('jit__K__jit', np.nan) / max(1e-12, jit.get('jit__K__orig', np.nan)):.3f} "
            f"SP {jit.get('jit__SP__jit', np.nan) / max(1e-12, jit.get('jit__SP__orig', np.nan)):.3f}, "
            f"plants {'ok' if plants.get('plant_ok') else 'REFUSED'}")
    return 0


# ---- combine -------------------------------------------------------------------

def _jit_units(recs, meta, arm, refused) -> dict[str, list[float]]:
    acc: dict[str, list[float]] = {}
    for rid, z in recs.items():
        if rid in refused or f"jit__{arm}__orig" not in z:
            continue
        x = acc.setdefault(meta[rid]["animal"], [0.0, 0.0])
        x[0] += float(z[f"jit__{arm}__jit"])
        x[1] += float(z[f"jit__{arm}__orig"])
    return {a: [v[0] / v[1]] for a, v in acc.items() if v[1] > 0}


def _gate2(recs, meta, refused_sp) -> dict[str, Read]:
    out: dict[str, Read] = {}
    ku = _jit_units(recs, meta, "K", set())
    obj = {"dataset": "luna", "arm": "stabilise4|K|gate2_jitter_control"}
    if len(ku) < st.MIN_ANIMALS:
        out["K"] = Read("NOT_A_RESULT", obj, f"{len(ku)} animals",
                        n_effective=max(1, len(ku)))
        out["SP"] = out["K"]
        return out
    kb = boot.animal_interval([v[0] for v in ku.values()], list(ku), how="mean",
                              n_boot=st.N_BOOT, seed=st.SEED)
    powered = float(kb["lo"]) >= JIT_CONTROL
    out["K"] = Read("NOT_A_RESULT", obj,
                    (f"positive control: planted sigma={JIT_SIGMA_PX:g} px jitter "
                     f"raises the incumbent's head energy "
                     f"{kb['point']:.3f}x [{kb['lo']:.3f}, {kb['hi']:.3f}]; "
                     f"{'powered' if powered else 'NOT powered'} (lower bound "
                     f"{'>=' if powered else '<'} {JIT_CONTROL})"),
                    n_effective=len(ku), detail={"interval": kb,
                                                 "powered": powered})
    obj = {"dataset": "luna", "arm": "stabilise4|SP|gate2_jitter"}
    su = _jit_units(recs, meta, "SP", refused_sp)
    if not powered:
        out["SP"] = Read("NOT_A_RESULT", obj,
                         "the positive control is not powered", n_effective=1)
        return out
    if len(su) < st.MIN_ANIMALS:
        out["SP"] = Read("NOT_A_RESULT", obj, f"{len(su)} animals",
                         n_effective=max(1, len(su)))
        return out
    b = boot.animal_interval([v[0] for v in su.values()], list(su), how="mean",
                             n_boot=st.N_BOOT, seed=st.SEED)
    ok = float(b["lo"]) >= JIT_BAND[0] and float(b["hi"]) <= JIT_BAND[1]
    out["SP"] = Read("PASS" if ok else "FAIL", obj,
                     (f"planted jitter changes SP's head energy "
                      f"{b['point']:.3f}x [{b['lo']:.3f}, {b['hi']:.3f}] against "
                      f"K's {kb['point']:.3f}x; the bar is the whole interval "
                      f"within [{JIT_BAND[0]}, {JIT_BAND[1]}]"),
                     n_effective=len(su), detail={"interval": b})
    return out


def combine(a) -> int:
    fps = spine.fps()
    w = int(frames(st.WIN_S, fps))
    meta = {r["recording_id"]: r for r in st.manifest()["recordings"]
            if r.get("in_pilot")}
    recs = {rid: z for rid in meta if (z := st._rec(rid)) is not None}
    log(f"  {len(recs)} of {len(meta)} recordings scanned")
    out = a.out or config.PATHS.result("stabilise4.json")
    reads: dict = {}
    kc = st._k_check(recs)
    reads["k_check"] = kc
    if not kc["ok"]:
        log(f"  §7 STOP: K is not the incumbent: {kc}")
        write_json({"stage": "stabilise4", "stopped": "§7", "k_check": kc}, out)
        return 1
    log(f"  §7 K reproduces the incumbent: max |diff| {kc['max_abs_diff']:.2e}")
    ref = {arm: s3._refused3(recs, arm) for arm in ARMS}
    g2 = _gate2(recs, meta, ref["SP"])
    for arm in ARMS:
        det = {"refused_recordings": len(ref[arm]),
               "refused_share": len(ref[arm]) / max(1, len(recs))}
        reads[arm] = {"detail": det,
                      "gate0": s3._gate0(recs, meta, arm, ref[arm]).to_dict(),
                      "gate1": st._gate1(recs, meta, arm, ref[arm]).to_dict(),
                      "gate2": g2[arm].to_dict(),
                      "gate3": s3._gate3(recs, meta, arm, ref[arm]).to_dict(),
                      "gate4": s3._gate4(recs, meta, arm, ref[arm]).to_dict(),
                      "gate5_curve": s2._gate5(recs, arm, ref[arm], fps),
                      "coupling_with_speed": st._coupling(recs, arm, w)}
    for arm in ARMS:
        r = reads[arm]
        vs = [r[g]["verdict"] for g in ("gate0", "gate1", "gate2", "gate3",
                                        "gate4")]
        share = r["detail"]["refused_share"]
        obj = {"dataset": "luna", "arm": f"stabilise4|{arm}"}
        if arm == "K":
            v, why = ("FAIL" if "FAIL" in vs else "NOT_A_RESULT",
                      "the incumbent; gate 2 is its positive control")
        elif share > rg.MAX_REFUSED_FRAC:
            v, why = "NOT_A_RESULT", (f"{100 * share:.1f}% of recordings "
                                      f"refused, against "
                                      f"{100 * rg.MAX_REFUSED_FRAC:.0f}%")
        elif "FAIL" in vs:
            v, why = "FAIL", f"gates 0-4: {vs}"
        elif all(x == "PASS" for x in vs):
            v, why = "PASS", ("gates 0-4 all pass; the arm is ELIGIBLE for a "
                              "separately registered stage, nothing more")
        else:
            v, why = "NOT_A_RESULT", f"gates 0-4: {vs}"
        r["verdict"] = Read(v, obj, why, n_effective=len(recs)).to_dict()
        log(f"  {arm}: {v} -- {why}")
    write_json({**anchors.header(anchors.LUNA, stage="stabilise4",
                                 unverified="a registration instrument"),
                "inherited_digest": spine.digest(),
                "registration": "results/STABILISE4_PREREGISTRATION.md",
                "checkpoint_sha256": s3.CKPT_SHA256,
                "seed": st.SEED, "fps": fps, "win_frames": w,
                "reads": reads}, out)
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
