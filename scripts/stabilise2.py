"""STABILISE 2: the same arms against a background the animal is not in.

    sbatch jobs/stabilise2.slurm
    python3 scripts/stabilise2.py --phase combine

READ results/STABILISE2_PREREGISTRATION.md FIRST. Everything it does not change
is `scripts/stabilise.py`, loaded and reused here rather than copied, so the
two stages cannot drift apart silently. The three changes:

* §1: B2's background is `register.masked_median_background` over 301 frames,
  refused past 1% undefined pixels. P2 is P placed by B2's mask. K is K.
* §2 gate 3: the per-animal floor is 20 immobile frames, not 50.
* §2 gate 4: the arm over a true-transform ORACLE, bar 1.25. Gate 5 is scored
  on the moving plants only, against each arm's and region's own null.
"""
from __future__ import annotations

import argparse
import importlib.util
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
from vieb.pixel import head as hd, motion as mo, register as rg      # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402


def _load_stabilise():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stabilise.py")
    spec = importlib.util.spec_from_file_location("stabilise_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["stabilise"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


st = _load_stabilise()

#: §1.
BG_SAMPLES = 301
MIN_SAMPLES = 10
MAX_UNDEFINED = 0.01
#: §2.
MIN_IMMOBILE_FRAMES = 20
ORACLE_BAR = 1.25


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "stabilise2")


# The reused combine helpers find their files and floors through these names.
st.work_dir = work_dir
st.MIN_IMMOBILE_FRAMES = MIN_IMMOBILE_FRAMES


def _plants2(video, pose, speed, immobile, bg_raw, bg_blur, cutoff, bl, fps,
             w) -> dict:
    """§2: the moving plants only, each with its oracle."""
    inp = st._plant_inputs(pose, speed, immobile, w)
    if inp is None:
        return {"plant_ok": False}
    src = st._frames_at(video, [inp["src"]])
    if not src:
        return {"plant_ok": False}
    sg, sp = src[0], pose[inp["src"]]
    r = bl * st.HEADLINE_BL
    out: dict = {"plant_ok": True, "plant_src": inp["src"],
                 "plant_moving_block": inp["moving_block"]}
    cells = [(None, None)] + [(hz, amp) for hz in st.PLANT_HZ
                              for amp in st.PLANT_AMP_PX]
    for hz, amp in cells:
        s = st._plant_scan(sg, sp, bg_raw, bg_blur, cutoff, bl, inp,
                           moving=True, hz=hz, amp=amp, fps=fps, oracle=True)
        tag = "base" if hz is None else f"{hz:g}__{amp:g}"
        for arm in rg.ARMS + ("O",):
            for reg in ("head", "hip"):
                v = s.get(f"{arm}|{reg}|{r}")
                out[f"p__{tag}__{arm}__{reg}"] = (np.asarray(v) if v is not None
                                                  else np.zeros(0))
    return out


def shard(a) -> int:
    gg = st._load("grooming_gate")
    pp = st._load("pixel_pilot")
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
        cutv = dict(zip([str(k) for k in pix["cutoff_keys"]],
                        [float(v) for v in pix["cutoff_values"]]))
        cutoff = cutv["m2"]
        pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
        bl = float(np.nanmedian(tego.body_length(pose)))
        dil = bl * mo.DILATE_BODY_LENGTHS
        video = vid.video_path(rid)
        nf = min(int(pix["n_frames"]), pose.shape[0])
        idx = np.unique(np.round(np.linspace(0, max(0, nf - 1),
                                             BG_SAMPLES)).astype(int))
        raw = st._frames_at(video, idx)
        idx = idx[:len(raw)]
        bg_blur, undefined = rg.masked_median_background(
            [rg.blur(f) for f in raw], [pose[i] for i in idx],
            dilate_px=dil, min_samples=MIN_SAMPLES)
        # The composite background for §5's paste: the same masked median, raw.
        bg_raw, _ = rg.masked_median_background(
            [np.asarray(f, dtype=np.float64) for f in raw],
            [pose[i] for i in idx], dilate_px=dil, min_samples=MIN_SAMPLES)
        s = rg.scan_register(st._open(video), pose, bg=bg_blur, cutoff=cutoff,
                             radii_px=[bl * x for x in st.RADII_BL],
                             dilate_px=dil, win=w, body_length_px=bl)
        got = {f"{arm}__{reg}__{st._rkey(x)}": s[f"{arm}|{reg}|{bl * x}"]
               for arm in rg.ARMS for reg, _ in rg.REGIONS
               for x in st.RADII_BL}
        sp = gg._speed(r["animal"], rid)
        speed = (np.asarray(sp, dtype=np.float64) if sp is not None
                 else np.zeros(0))
        still = pp._still_frames(r["animal"], rid, int(pix["n_frames"]))
        immobile = (st._immobile(pix, still) if still is not None
                    else np.zeros(0, dtype=bool))
        plants = (_plants2(video, pose, speed, immobile, bg_raw, bg_blur,
                           cutoff, bl, fps, w)
                  if speed.size and immobile.size else {"plant_ok": False})
        np.savez_compressed(
            out, body_length_px=bl, cutoff=cutoff, win=w,
            bg_undefined=undefined, arena=s["arena"],
            identical=s["identical"], mask_area=s["mask_area"],
            mask_ok=s["mask_ok"], mask_theta=s["mask_theta"],
            p_estimate=s["p_estimate"], nan_frac=s["nan_frac"],
            speed=speed, immobile=immobile, **got, **plants)
        log(f"  {n}/{len(mine)} {rid}: bg undefined {undefined:.4f}, mask NaN "
            f"{s['nan_frac']:.3f}, mask/bl^2 "
            f"{np.median(s['mask_area'][s['mask_area'] > 0]) / bl ** 2:.3f}, "
            f"plants {'ok' if plants.get('plant_ok') else 'REFUSED'}")
    return 0


# ---- combine ----------------------------------------------------------------

def _refused2(recs: dict, arm: str) -> set[str]:
    if arm == "K":
        return set()
    return {rid for rid, z in recs.items()
            if float(z["nan_frac"]) > rg.MAX_NAN_FRAC
            or float(z["bg_undefined"]) > MAX_UNDEFINED}


def _gate4(recs, meta, arm, refused) -> Read:
    obj = {"dataset": "luna", "arm": f"stabilise2|{arm}|gate4_oracle"}
    vals: dict[str, list[float]] = {}
    for rid, z in recs.items():
        if rid in refused or not bool(z.get("plant_ok", False)):
            continue
        v = np.asarray(z.get(f"p__base__{arm}__head", np.zeros(0)))[1:]
        o = np.asarray(z.get("p__base__O__head", np.zeros(0)))[1:]
        if v.size == 0 or not np.isfinite(v).any() or not np.isfinite(o).any():
            continue
        den = float(np.nanmedian(o))
        if den <= 0:
            continue
        vals.setdefault(meta[rid]["animal"], []).append(
            float(np.nanmedian(v)) / den)
    per = {a: v for a, v in vals.items() if v}
    if len(per) < st.MIN_ANIMALS:
        return Read("NOT_A_RESULT", obj,
                    f"{len(per)} animals with a scored plant, against "
                    f"{st.MIN_ANIMALS}", n_effective=max(1, len(per)))
    units = [float(np.mean(v)) for v in per.values()]
    b = boot.animal_interval(units, list(per), how="mean", n_boot=st.N_BOOT,
                             seed=st.SEED)
    ok = float(b["hi"]) <= ORACLE_BAR
    return Read("PASS" if ok else "FAIL", obj,
                (f"planted-trajectory residual is {b['point']:.3f}x "
                 f"[{b['lo']:.3f}, {b['hi']:.3f}] a perfect registration's over "
                 f"{len(per)} animals; the bar is an upper bound <= "
                 f"{ORACLE_BAR}"),
                n_effective=len(per), detail={"interval": b})


def _gate5(recs, arm, refused, fps) -> dict:
    def pe(v):
        v = np.asarray(v, dtype=np.float64)[1:]
        if v.size < 8 or not np.isfinite(v).all():
            return np.nan
        return float(hd.peak_excess(v, fps=fps))
    ok = [z for rid, z in recs.items()
          if rid not in refused and bool(z.get("plant_ok", False))]
    curve: dict = {}
    thr: dict[str, float] = {}
    for reg in ("head", "hip"):
        null = np.asarray([x for z in ok
                           if np.isfinite(x := pe(z.get(f"p__base__{arm}__{reg}",
                                                        np.zeros(0))))])
        if null.size < 20:
            return {"refused": f"{null.size} unplanted {reg} windows"}
        thr[reg] = float(np.percentile(null, st.DETECT_PCT))
        curve[f"threshold_{reg}"] = thr[reg]
        curve[f"n_null_{reg}"] = int(null.size)
    for hz in st.PLANT_HZ:
        for amp in st.PLANT_AMP_PX:
            row: dict = {}
            for reg in ("head", "hip"):
                xs = [pe(z.get(f"p__{hz:g}__{amp:g}__{arm}__{reg}", np.zeros(0)))
                      for z in ok]
                xs = [x for x in xs if np.isfinite(x)]
                row[reg] = (float(np.mean(np.asarray(xs) > thr[reg]))
                            if xs else None)
                row[f"n_{reg}"] = len(xs)
            row["head_minus_hip"] = (row["head"] - row["hip"]
                                     if row["head"] is not None
                                     and row["hip"] is not None else None)
            curve[f"{hz:g}Hz|{amp:g}px"] = row
    return curve


def combine(a) -> int:
    gj = st._load("grooming_jitter")
    fps = spine.fps()
    w = int(frames(st.WIN_S, fps))
    meta = {r["recording_id"]: r for r in st.manifest()["recordings"]
            if r.get("in_pilot")}
    recs = {rid: z for rid in meta if (z := st._rec(rid)) is not None}
    log(f"  {len(recs)} of {len(meta)} recordings scanned")
    reads: dict = {}
    kc = st._k_check(recs)
    reads["k_check"] = kc
    out = a.out or config.PATHS.result("stabilise2.json")
    if not kc["ok"]:
        log(f"  §7 STOP: K is not the incumbent: {kc}")
        write_json({"stage": "stabilise2", "stopped": "§7", "k_check": kc}, out)
        return 1
    log(f"  §7 K reproduces the incumbent: max |diff| {kc['max_abs_diff']:.2e}")
    cache: dict = {}

    def jit_of(rid):
        return np.asarray(gj.skull_jitter(rid, cache), dtype=np.float64)

    g2: dict = {}
    for arm in rg.ARMS:
        ref = _refused2(recs, arm)
        und = [float(z["bg_undefined"]) for z in recs.values()]
        det = {"refused_recordings": len(ref),
               "refused_share": len(ref) / max(1, len(recs)),
               "bg_undefined_max": max(und) if und else None}
        g2[arm] = st._gate2(recs, meta, arm, ref, jit_of, w)
        reads[arm] = {"detail": det,
                      "gate1": st._gate1(recs, meta, arm, ref).to_dict(),
                      "gate3": st._gate3(recs, meta, arm, ref).to_dict(),
                      "gate4": _gate4(recs, meta, arm, ref).to_dict(),
                      "gate5_curve": _gate5(recs, arm, ref, fps),
                      "coupling_with_speed": st._coupling(recs, arm, w)}
    k_rd, k_det = g2["K"]
    for arm in rg.ARMS:
        rd, det = g2[arm]
        obj = {"dataset": "luna", "arm": f"stabilise2|{arm}|gate2_jitter"}
        if rd is None:
            b = det["interval"]
            if arm == "K":
                rd = Read("NOT_A_RESULT", obj,
                          (f"the incumbent's own jitter coupling on still "
                           f"windows: {b['point']:+.3f} [{b['lo']:+.3f}, "
                           f"{b['hi']:+.3f}] -- the reference, not a gate"),
                          n_effective=det["n_animals"], detail=det)
            elif k_rd is not None:
                rd = Read("NOT_A_RESULT", obj,
                          "K's reference interval is itself refused",
                          n_effective=det["n_animals"], detail=det)
            else:
                kb = k_det["interval"]
                ok = (float(b["hi"]) < st.JITTER_BAR
                      and float(b["hi"]) < float(kb["lo"]))
                rd = Read("PASS" if ok else "FAIL", obj,
                          (f"jitter -> head energy, speed held fixed: "
                           f"{b['point']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}] "
                           f"against K's {kb['point']:+.3f} [{kb['lo']:+.3f}, "
                           f"{kb['hi']:+.3f}]; the bar is an upper bound < "
                           f"{st.JITTER_BAR} AND below K's lower bound"),
                          n_effective=det["n_animals"], detail=det)
        reads[arm]["gate2"] = rd.to_dict()
    for arm in rg.ARMS:
        r = reads[arm]
        vs = [r[g]["verdict"] for g in ("gate1", "gate2", "gate3", "gate4")]
        ref_share = r["detail"]["refused_share"]
        obj = {"dataset": "luna", "arm": f"stabilise2|{arm}"}
        if arm != "K" and ref_share > rg.MAX_REFUSED_FRAC:
            v, why = "NOT_A_RESULT", (f"{100 * ref_share:.1f}% of recordings "
                                      f"refused, against "
                                      f"{100 * rg.MAX_REFUSED_FRAC:.0f}%")
        elif arm == "K":
            v, why = ("FAIL" if "FAIL" in vs else "NOT_A_RESULT",
                      "the incumbent; gate 2 is its reference, so it is scored "
                      "on gates 1, 3 and 4 only")
        elif "FAIL" in vs:
            v, why = "FAIL", f"gates 1-4: {vs}"
        elif all(x == "PASS" for x in vs):
            v, why = "PASS", ("gates 1-4 all pass; the arm is ELIGIBLE for a "
                              "separately registered stage, nothing more")
        else:
            v, why = "NOT_A_RESULT", f"gates 1-4: {vs}"
        r["verdict"] = Read(v, obj, why, n_effective=len(recs)).to_dict()
        log(f"  {arm}: {v} -- {why}")
    write_json({**provenance.header(anchors.LUNA, stage="stabilise2",
                                 unverified="a registration instrument"),
                "inherited_digest": spine.digest(),
                "registration": "results/STABILISE2_PREREGISTRATION.md",
                "seed": st.SEED, "fps": fps, "win_frames": w,
                "headline_radius_bl": st.HEADLINE_BL, "reads": reads}, out)
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
