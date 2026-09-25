"""STABILISE 6: masked ECC from the better of two starts (SM6), and a chained keypoint gate.

    sbatch jobs/stabilise6.slurm                 # 12 GPU shards
    python3 scripts/stabilise6.py --phase combine

READ results/STABILISE6_PREREGISTRATION.md FIRST. It is STABILISE 5 with two
changes, and this file is a copy of `stabilise5.py` edited only there:

* §1: arm SM6, `scan_track(..., best_start=True)` -- masked ECC from the
  identity and from SAM's centroid shift, the higher ECC correlation kept.
  SM (STABILISE 5's arm) and SP are carried beside it.
* §2: gate 6 chained over k = 10 pairs; beta AND f both in [0.90, 1.10].
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
from recur.read import Read                                          # noqa: E402
from recur.render import video as vid                                # noqa: E402
from recur.util import frames, log, write_json                       # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.pixel import motion as mo, register as rg, sam             # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stabilise4 as s4                                              # noqa: E402

s3, s2, st = s4.s3, s4.s2, s4.st
ARMS = ("K", "SP", "SM", "SM6")
RENAME = {"K": "K", "P": "SP", "M": "SM", "N": "SM6", "I": "I"}
PROMPT = "keypoint"
#: Amendment 1.
ERODE_BL = 0.04
#: §2, gate 6.
G6_PCT = 75.0
G6_BAND = (0.90, 1.10)
G6_K = 10
G6_MIN_CHAINS = 20


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "stabilise6")


st.work_dir = work_dir
st.MIN_IMMOBILE_FRAMES = s2.MIN_IMMOBILE_FRAMES
s3.STAGE = "stabilise6"


def _mfn(tr: dict, bl: float):
    return lambda t: sam.mask_at(tr, t, erode_px=ERODE_BL * bl)


def _jitter_plant(video, pose, speed, bl, w, predict) -> dict:
    """STABILISE 4 §3, unchanged, with SM run beside SP and K."""
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
    still = still[:s4.JIT_WINDOWS]
    rng = np.random.default_rng(st.SEED)
    r = bl * st.HEADLINE_BL
    sums = {f"{arm}__{run}": 0.0 for arm in ARMS for run in ("orig", "jit")}
    for i in still:
        a = int(i) * w
        grey = s4._block(video, a, w)
        if len(grey) < w:
            continue
        rgbs = s3._as_rgb(grey)
        p0 = pose[a:a + w]
        pj = p0 + rng.normal(0.0, s4.JIT_SIGMA_PX, p0.shape)
        got = {}
        for run, pw in (("orig", p0), ("jit", pj)):
            tr = sam.sam_track(lambda: iter(rgbs), pw, predict,
                               body_length_px=bl, prompt_mode=PROMPT)
            s = rg.scan_track(lambda: iter(grey), pw, tr, radii_px=[r],
                              dilate_px=bl * mo.DILATE_BODY_LENGTHS, win=w,
                              mask_fn=_mfn(tr, bl), best_start=True)
            got[run] = {arm: np.asarray(s[f"{src}|head|{r}"])
                        for src, arm in RENAME.items() if arm in ARMS}
        for arm in ARMS:
            both = np.isfinite(got["orig"][arm]) & np.isfinite(got["jit"][arm])
            sums[f"{arm}__orig"] += float(got["orig"][arm][both].sum())
            sums[f"{arm}__jit"] += float(got["jit"][arm][both].sum())
    return {f"jit__{k_}": v for k_, v in sums.items()}


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
        d = spine.clean(rid)
        pose = np.asarray(d["pose"], dtype=np.float64)
        bl = float(np.nanmedian(tego.body_length(pose)))
        dil = bl * mo.DILATE_BODY_LENGTHS
        video = vid.video_path(rid)
        tr = sam.sam_track(s3._rgb(video), pose, predict, body_length_px=bl,
                           prompt_mode=PROMPT)
        s = rg.scan_track(st._open(video), pose, tr,
                          radii_px=[bl * x for x in st.RADII_BL],
                          dilate_px=dil, win=w, mask_fn=_mfn(tr, bl),
                          best_start=True)
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
        plants = (s3._plants3(video, pose, speed, immobile, bg_raw, bl, fps, w,
                              predict, prompt_mode=PROMPT, rename=RENAME,
                              mask_erode_bl=ERODE_BL, best_start=True)
                  if speed.size and immobile.size else {"plant_ok": False})
        jit = _jitter_plant(video, pose, speed, bl, w, predict) if speed.size else {}
        # Gate 6's reference: CENTER, and whether it was measured (not filled).
        ci = tego.CENTER
        center = pose[:, ci]
        center_ok = (np.isfinite(center).all(axis=1)
                     & ~np.asarray(d["missing"], dtype=bool)[:, ci]
                     & ~np.asarray(d["interpolated"], dtype=bool)[:, ci])
        nan_frac = float(1.0 - np.mean(tr["ok"])) if tr["n"] else 1.0
        np.savez_compressed(
            out, body_length_px=bl, win=w, arena=s["arena"],
            identical=s["identical"], mask_area=s["mask_area"],
            mask_ok=s["mask_ok"], nan_frac=nan_frac, key_iou=tr["key_iou"],
            key_accepted=tr["key_accepted"],
            gate0_inside=inside, gate0_scored=scored,
            speed=speed, immobile=immobile,
            W_SP=s["P|W"], W_SM=s["M|W"], W_SM6=s["N|W"],
            center=center, center_ok=center_ok,
            **got, **plants, **jit)
        f6 = {arm: _f_one({"W": s[f"{src}|W"], "center": center,
                           "center_ok": center_ok, "speed": speed})
              for src, arm in (("P", "SP"), ("M", "SM"), ("N", "SM6"))}
        log(f"  {n}/{len(mine)} {rid}: refused {nan_frac:.3f}, on-animal "
            f"{inside}/{scored}, chained f SP {f6['SP'][0]:.3f} SM "
            f"{f6['SM'][0]:.3f} SM6 {f6['SM6'][0]:.3f} ({f6['SM6'][1]} chains), "
            f"jitter SM6 "
            f"{jit.get('jit__SM6__jit', np.nan) / max(1e-12, jit.get('jit__SM6__orig', np.nan)):.3f}, "
            f"plants {'ok' if plants.get('plant_ok') else 'REFUSED'}")
    return 0


# ---- gate 6, chained (§2) ---------------------------------------------------

def _h(W: np.ndarray) -> np.ndarray:
    return np.vstack([W, [0.0, 0.0, 1.0]])


def _chains(W, center, center_ok, speed, k: int = G6_K):
    """(sum|d_arm|^2, sum d_kp.d_arm, sum|d_kp|^2, n_chains), §2's population."""
    W = np.asarray(W, dtype=np.float64)
    n = min(W.shape[0], center.shape[0], speed.size)
    sp = np.asarray(speed[:n], dtype=np.float64)
    fin = np.isfinite(sp)
    if fin.sum() < 4:
        return 0.0, 0.0, 0.0, 0
    thr = float(np.percentile(sp[fin], G6_PCT))
    good = np.isfinite(W[:n]).all(axis=(1, 2))
    num = den = kk = 0.0
    m = 0
    for t in range(k, n, k):                       # non-overlapping chains
        if not (center_ok[t] and center_ok[t - k]):
            continue
        if not good[t - k + 1:t + 1].all():
            continue
        seg = sp[t - k + 1:t + 1]
        if not np.isfinite(seg).all() or float(seg.mean()) <= thr:
            continue
        M = np.eye(3)
        for j in range(t - k + 1, t + 1):
            M = _h(W[j]) @ M
        c = center[t - k]
        d_arm = M[:2, :2] @ c + M[:2, 2] - c
        d_kp = center[t] - c
        num += float(d_arm @ d_arm)
        den += float(d_kp @ d_arm)
        kk += float(d_kp @ d_kp)
        m += 1
    return num, den, kk, m


def _f_one(z) -> tuple[float, int]:
    num, den, kk, m = _chains(z["W"], np.asarray(z["center"]),
                              np.asarray(z["center_ok"], dtype=bool),
                              np.asarray(z["speed"]))
    return (num / den if den else float("nan")), m


def _gate6(recs, meta, arm, refused) -> Read:
    obj = {"dataset": "luna", "arm": f"stabilise6|{arm}|gate6_chained"}
    acc: dict[str, list[float]] = {}
    for rid, z in recs.items():
        if rid in refused:
            continue
        num, den, kk, m = _chains(z[f"W_{arm}"], np.asarray(z["center"]),
                                  np.asarray(z["center_ok"], dtype=bool),
                                  np.asarray(z["speed"]))
        x = acc.setdefault(meta[rid]["animal"], [0.0, 0.0, 0.0, 0])
        x[0] += num
        x[1] += den
        x[2] += kk
        x[3] += m
    keep = {a: v for a, v in acc.items()
            if v[3] >= G6_MIN_CHAINS and v[1] > 0 and v[2] > 0}
    if len(keep) < st.MIN_ANIMALS:
        return Read("NOT_A_RESULT", obj,
                    f"{len(keep)} animals with >= {G6_MIN_CHAINS} chains, "
                    f"against {st.MIN_ANIMALS}", n_effective=max(1, len(keep)))
    f = {a: v[0] / v[1] for a, v in keep.items()}
    beta = {a: v[1] / v[2] for a, v in keep.items()}
    bf = boot.animal_interval(list(f.values()), list(f), how="mean",
                              n_boot=st.N_BOOT, seed=st.SEED)
    bb = boot.animal_interval(list(beta.values()), list(beta), how="mean",
                              n_boot=st.N_BOOT, seed=st.SEED)
    ok = all(float(x["lo"]) >= G6_BAND[0] and float(x["hi"]) <= G6_BAND[1]
             for x in (bf, bb))
    return Read("PASS" if ok else "FAIL", obj,
                (f"over {G6_K}-frame chains on real fast frames, the "
                 f"alignment's body motion against the keypoints' is bracketed "
                 f"beta {bb['point']:.3f} [{bb['lo']:.3f}, {bb['hi']:.3f}] to f "
                 f"{bf['point']:.3f} [{bf['lo']:.3f}, {bf['hi']:.3f}] over "
                 f"{len(keep)} animals; the bar is both intervals within "
                 f"[{G6_BAND[0]}, {G6_BAND[1]}]"),
                n_effective=len(keep),
                detail={"interval": bf, "beta": bb, "k": G6_K,
                        "n_chains": int(sum(v[3] for v in keep.values()))})


def _gate2(recs, meta, refused) -> dict[str, Read]:
    """STABILISE 4 §3, unchanged, for SP and SM, K as the positive control."""
    out: dict[str, Read] = {}
    ku = s4._jit_units(recs, meta, "K", set())
    obj = {"dataset": "luna", "arm": "stabilise6|K|gate2_jitter_control"}
    if len(ku) < st.MIN_ANIMALS:
        rd = Read("NOT_A_RESULT", obj, f"{len(ku)} animals",
                  n_effective=max(1, len(ku)))
        return {arm: rd for arm in ARMS}
    kb = boot.animal_interval([v[0] for v in ku.values()], list(ku), how="mean",
                              n_boot=st.N_BOOT, seed=st.SEED)
    powered = float(kb["lo"]) >= s4.JIT_CONTROL
    out["K"] = Read("NOT_A_RESULT", obj,
                    (f"positive control: planted jitter raises the incumbent's "
                     f"head energy {kb['point']:.3f}x [{kb['lo']:.3f}, "
                     f"{kb['hi']:.3f}]; {'powered' if powered else 'NOT powered'}"),
                    n_effective=len(ku), detail={"interval": kb,
                                                 "powered": powered})
    for arm in ("SP", "SM", "SM6"):
        obj = {"dataset": "luna", "arm": f"stabilise6|{arm}|gate2_jitter"}
        su = s4._jit_units(recs, meta, arm, refused[arm])
        if not powered or len(su) < st.MIN_ANIMALS:
            out[arm] = Read("NOT_A_RESULT", obj,
                            "control not powered" if not powered
                            else f"{len(su)} animals", n_effective=1)
            continue
        b = boot.animal_interval([v[0] for v in su.values()], list(su),
                                 how="mean", n_boot=st.N_BOOT, seed=st.SEED)
        ok = (float(b["lo"]) >= s4.JIT_BAND[0]
              and float(b["hi"]) <= s4.JIT_BAND[1])
        out[arm] = Read("PASS" if ok else "FAIL", obj,
                        (f"planted jitter changes {arm}'s head energy "
                         f"{b['point']:.3f}x [{b['lo']:.3f}, {b['hi']:.3f}]; "
                         f"the bar is the whole interval within "
                         f"[{s4.JIT_BAND[0]}, {s4.JIT_BAND[1]}]"),
                        n_effective=len(su), detail={"interval": b})
    return out


def combine(a) -> int:
    fps = spine.fps()
    w = int(frames(st.WIN_S, fps))
    meta = {r["recording_id"]: r for r in st.manifest()["recordings"]
            if r.get("in_pilot")}
    recs = {rid: z for rid in meta if (z := st._rec(rid)) is not None}
    log(f"  {len(recs)} of {len(meta)} recordings scanned")
    out = a.out or config.PATHS.result("stabilise6.json")
    reads: dict = {}
    kc = st._k_check(recs)
    reads["k_check"] = kc
    if not kc["ok"]:
        log(f"  §7 STOP: K is not the incumbent: {kc}")
        write_json({"stage": "stabilise6", "stopped": "§7", "k_check": kc}, out)
        return 1
    log(f"  §7 K reproduces the incumbent: max |diff| {kc['max_abs_diff']:.2e}")
    ref = {arm: s3._refused3(recs, arm) for arm in ARMS}
    g2 = _gate2(recs, meta, ref)
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
        if arm != "K":
            reads[arm]["gate6"] = _gate6(recs, meta, arm, ref[arm]).to_dict()
    for arm in ARMS:
        r = reads[arm]
        gates = ["gate0", "gate1", "gate2", "gate3", "gate4"] + (
            ["gate6"] if arm != "K" else [])
        vs = [r[g]["verdict"] for g in gates]
        share = r["detail"]["refused_share"]
        obj = {"dataset": "luna", "arm": f"stabilise6|{arm}"}
        if arm == "K":
            v, why = ("FAIL" if "FAIL" in vs else "NOT_A_RESULT",
                      "the incumbent; gate 2 is its positive control and gate 6 "
                      "is circular for it")
        elif share > rg.MAX_REFUSED_FRAC:
            v, why = "NOT_A_RESULT", (f"{100 * share:.1f}% of recordings "
                                      f"refused, against "
                                      f"{100 * rg.MAX_REFUSED_FRAC:.0f}%")
        elif "FAIL" in vs:
            v, why = "FAIL", f"gates {', '.join(gates)}: {vs}"
        elif all(x == "PASS" for x in vs):
            v, why = "PASS", ("every gate passes; the arm is ELIGIBLE for a "
                              "separately registered stage, nothing more")
        else:
            v, why = "NOT_A_RESULT", f"gates {', '.join(gates)}: {vs}"
        r["verdict"] = Read(v, obj, why, n_effective=len(recs)).to_dict()
        log(f"  {arm}: {v} -- {why}")
    write_json({**provenance.header(anchors.LUNA, stage="stabilise6",
                                    unverified="a registration instrument"),
                "inherited_digest": spine.digest(),
                "registration": "results/STABILISE6_PREREGISTRATION.md",
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
