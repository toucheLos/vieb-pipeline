"""STABILISE: can crop registration read no keypoints, and does it then pass?

    sbatch jobs/stabilise.slurm                  # 12 shards: scans + plants
    python3 scripts/stabilise.py --phase combine

READ results/STABILISE_PREREGISTRATION.md FIRST, with its Amendment 1 (D22).

This is an INSTRUMENT stage. It reads no spectrum as a behaviour and names no
behaviour. Its output is a verdict per registration arm -- K (the incumbent pose
warp), B (background mask), P (phase-correlation-initialised ECC) -- on four
gates: duplicate frames (§2), jitter decoupling (§3), the immobility floor (§4)
and a planted rigid trajectory (§5). §6's planted local motion is a curve with
no verdict.

## Why keypoints still appear here

§0: keypoints may LOCATE, never REGISTER. They choose the disc (a window-median,
inside `register.scan_register`), define the still population by speed (§3),
generate the planted trajectory and the jitter K is fed (§5). None of them moves
a B or P frame.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
from vieb.pixel import head as hd, motion as mo, register as rg      # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

SEED = 0
N_BOOT = 2000
RADII_BL = (0.4, 0.6, 0.8)
HEADLINE_BL = 0.6
WIN_S = 2.0
#: §3: the still population, by speed alone.
STILL_PCT = 25.0
#: §5: the trajectory source.
MOVING_PCT = 75.0
#: §2 gate 1.
DUP_TOL = 1e-6
#: §3 gate 2.
JITTER_BAR = 0.10
MIN_STILL_WINDOWS = 30
#: §4 gate 3 and §5 gate 4.
FLOOR_BAR = 1.10
MIN_IMMOBILE_FRAMES = 50
#: §8.
MIN_ANIMALS = 20
#: §5: the paste is cut with B's mask dilated by this many body lengths.
PASTE_DILATE_BL = 0.1
#: §6.
PATCH_BL = 0.15
PLANT_HZ = (4.0, 6.0)
PLANT_AMP_PX = (0.25, 0.5, 1.0, 2.0, 4.0)
DETECT_PCT = 95.0
#: §7.
K_TOL = 1e-9


def _load(name: str):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "stabilise")


def manifest() -> dict:
    with open(os.path.join(config.REPO, "work", "pixel", "manifest.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def _pixel(rid: str) -> dict | None:
    p = os.path.join(config.REPO, "work", "pixel", "rec", f"{rid}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _rkey(rad_bl: float) -> str:
    return f"{rad_bl:g}"


# ---- decoding ---------------------------------------------------------------

def _open(video: str):
    import cv2

    def gen():
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {video}")
        try:
            while True:
                ok, fr = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        finally:
            cap.release()
    return gen


def _frames_at(video: str, idx) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video)
    out = []
    try:
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    return out


# ---- §2-§4 census populations ----------------------------------------------

def _immobile(pix: dict, still: np.ndarray) -> np.ndarray:
    """§4: genuine immobility at TRI_MARGINS = 0 -- `pixel_pilot`'s own rule."""
    M = np.asarray(pix["motion__m2"], dtype=np.float64)
    A = np.asarray(pix["arena__m2"], dtype=np.float64)
    na = np.asarray(pix["n_arena"], dtype=np.float64)
    npx = float(pix["n_pixels"])
    n = min(M.size, still.size)
    animal_px = np.maximum(npx - na[:n], 1.0)
    expect = A[:n] * (animal_px / np.maximum(na[:n], 1.0))
    dup = np.asarray(pix["identical"], dtype=bool)[:n] & still[:n]
    rest = still[:n] & ~dup
    drop = rest & ((M[:n] - A[:n]) > expect)
    return rest & ~drop


# ---- §5-§6 plants -----------------------------------------------------------

def _blocks(x: np.ndarray, w: int) -> np.ndarray:
    k = x.size // w
    return x[:k * w].reshape(k, w)


def _rot2(h: float) -> np.ndarray:
    return np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])


def _plant_inputs(pose: np.ndarray, speed: np.ndarray, immobile: np.ndarray,
                  w: int):
    """The source frame, the real trajectory and the real still-window jitter."""
    src = np.flatnonzero(immobile & np.isfinite(pose).all(axis=(1, 2))[
        :immobile.size])
    if src.size == 0:
        return None
    s0 = int(src[0])                              # §5.1: first eligible
    n = min(speed.size, pose.shape[0])
    S = _blocks(speed[:n], w)
    allok = _blocks(np.isfinite(pose[:n]).all(axis=(1, 2)).astype(np.float64),
                    w).min(axis=1) > 0
    ms = np.nanmean(S, axis=1)
    good = allok & np.isfinite(ms)
    if good.sum() < 4:
        return None
    hi = np.flatnonzero(good & (ms > np.percentile(ms[good], MOVING_PCT)))
    lo = np.flatnonzero(good & (ms <= np.percentile(ms[good], STILL_PCT)))
    if hi.size == 0 or lo.size == 0:
        return None
    a = int(hi[0]) * w
    tr = pose[a:a + w + 1] if a + w + 1 <= pose.shape[0] else None
    if tr is None or not np.isfinite(tr).all():
        return None
    hd_ = tego.heading(tr)
    dh = np.unwrap(hd_) - np.unwrap(hd_)[0]
    dc = tr[:, tego.CENTER] - tr[0, tego.CENTER]
    b = int(lo[0]) * w
    st = pose[b:b + w + 1]
    if st.shape[0] < w + 1 or not np.isfinite(st).all():
        return None
    med = np.median(st, axis=0)
    h_still = float(np.median(np.unwrap(tego.heading(st))))
    jit_body = np.einsum("ij,tkj->tki", _rot2(-h_still), st - med)
    return {"src": s0, "dh": dh, "dc": dc, "jit_body": jit_body,
            "moving_block": int(hi[0]), "still_block": int(lo[0])}


def _plant_frames(src_grey: np.ndarray, src_pose: np.ndarray,
                  bg_raw: np.ndarray, m0: np.ndarray, bl: float, inp: dict, *,
                  moving: bool, hz: float | None, amp: float | None,
                  fps: float) -> tuple[list, np.ndarray, list]:
    """§5/§6: the planted frames, their pose track for K and their TRUE
    transforms. `m0` is the source frame's animal mask; the paste is it,
    dilated by `PASTE_DILATE_BL`."""
    import cv2

    shape = src_grey.shape
    k = int(max(1, round(PASTE_DILATE_BL * bl)))
    paste = cv2.dilate(np.asarray(m0).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (2 * k + 1, 2 * k + 1))) > 0
    layer = src_grey.astype(np.float64)
    h_src = float(tego.heading(src_pose[None])[0])
    axis = np.array([np.cos(h_src), np.sin(h_src)])
    skull = src_pose[list(hd.SKULL)].mean(axis=0)
    patch = rg.disc(float(skull[0]), float(skull[1]), PATCH_BL * bl, shape)
    centre = tuple(float(v) for v in src_pose[tego.CENTER])
    n = inp["dh"].size
    frames_, poses, Ms = [], [], []
    for t in range(n):
        L = layer
        if hz is not None and amp is not None and patch is not None:
            d = amp * np.sin(2 * np.pi * hz * t / fps) * axis
            sh = cv2.warpAffine(layer, np.float64([[1, 0, d[0]], [0, 1, d[1]]]),
                                (shape[1], shape[0]),
                                borderMode=cv2.BORDER_REFLECT)
            L = np.where(patch, sh, layer)
        if moving:
            M = cv2.getRotationMatrix2D(centre, -float(np.degrees(inp["dh"][t])),
                                        1.0)
            M[:, 2] += inp["dc"][t]
        else:
            M = np.float64([[1, 0, 0], [0, 1, 0]])
        Lt = cv2.warpAffine(L, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
        Pt = cv2.warpAffine(paste.astype(np.float64), M,
                            (shape[1], shape[0])) > 0.5
        frames_.append(np.clip(np.round(np.where(Pt, Lt, bg_raw)), 0, 255)
                       .astype(np.uint8))
        Ms.append(np.asarray(M, dtype=np.float64))
        q = rg.apply(M, src_pose)
        j = inp["jit_body"][t % inp["jit_body"].shape[0]]
        q = q + (np.asarray(M)[:, :2] @ (_rot2(h_src) @ j.T)).T
        poses.append(q)
    return frames_, np.asarray(poses), Ms


def _oracle(frames_: list, Ms: list, src_pose: np.ndarray, radii_px) -> dict:
    """STABILISE 2 §2: each frame carried back to the source by its TRUE inverse
    transform, then differenced, discs at the source frame's own centroids."""
    shape = frames_[0].shape
    n = len(frames_)
    skull = src_pose[list(hd.SKULL)].mean(axis=0)
    hip = src_pose[list(hd.HIPS)].mean(axis=0)
    discs = {(k, r): rg.disc(float(c[0]), float(c[1]), r, shape)
             for r in radii_px for k, c in (("head", skull), ("hip", hip))}
    ser = {key: np.full(n, np.nan) for key in discs}
    prev = None
    for t in range(n):
        u = rg.warp(rg.blur(frames_[t]), rg.invert(Ms[t]), shape)
        if prev is not None:
            d = np.abs(u - prev)
            for key, dm in discs.items():
                if dm is not None:
                    ser[key][t] = float(d[dm].mean())
        prev = u
    return {f"O|{k}|{r}": v for (k, r), v in ser.items()}


def _plant_scan(src_grey: np.ndarray, src_pose: np.ndarray, bg_raw: np.ndarray,
                bg_blur: np.ndarray, cutoff: float, bl: float, inp: dict, *,
                moving: bool, hz: float | None, amp: float | None,
                fps: float, oracle: bool = False) -> dict:
    """§5/§6: one planted 61-frame sequence, scanned by every arm.

    `oracle` (STABILISE 2 §2) adds ``"O|head|r"`` / ``"O|hip|r"``. Off by
    default, so STABILISE's run is unchanged.
    """
    m0 = rg.animal_mask(rg.blur(src_grey), bg_blur, cutoff)
    if m0 is None:
        return {}
    frames_, poses, Ms = _plant_frames(src_grey, src_pose, bg_raw, m0, bl, inp,
                                       moving=moving, hz=hz, amp=amp, fps=fps)
    n = len(frames_)
    out = rg.scan_register(lambda: iter(frames_), poses,
                           bg=bg_blur, cutoff=cutoff,
                           radii_px=[bl * HEADLINE_BL],
                           dilate_px=bl * mo.DILATE_BODY_LENGTHS, win=n,
                           body_length_px=bl)
    if oracle:
        out.update(_oracle(frames_, Ms, src_pose, [bl * HEADLINE_BL]))
    return out


def _plants(video, pose, speed, immobile, bg_raw, bg_blur, cutoff, bl, fps,
            w) -> dict:
    inp = _plant_inputs(pose, speed, immobile, w)
    if inp is None:
        return {"plant_ok": False}
    src = _frames_at(video, [inp["src"]])
    if not src:
        return {"plant_ok": False}
    sg, sp = src[0], pose[inp["src"]]
    r = bl * HEADLINE_BL
    out: dict = {"plant_ok": True, "plant_src": inp["src"],
                 "plant_moving_block": inp["moving_block"],
                 "plant_still_block": inp["still_block"]}
    for cond in ("still", "moving"):
        base = _plant_scan(sg, sp, bg_raw, bg_blur, cutoff, bl, inp,
                           moving=(cond == "moving"), hz=None, amp=None,
                           fps=fps)
        for arm in rg.ARMS:
            for reg in ("head", "hip"):
                v = base.get(f"{arm}|{reg}|{r}")
                out[f"g4|{cond}|{arm}|{reg}"] = (np.asarray(v) if v is not None
                                                 else np.zeros(0))
        for hz in PLANT_HZ:
            for amp in PLANT_AMP_PX:
                s = _plant_scan(sg, sp, bg_raw, bg_blur, cutoff, bl, inp,
                                moving=(cond == "moving"), hz=hz, amp=amp,
                                fps=fps)
                for arm in rg.ARMS:
                    for reg in ("head", "hip"):
                        v = s.get(f"{arm}|{reg}|{r}")
                        out[f"g5|{cond}|{hz:g}|{amp:g}|{arm}|{reg}"] = (
                            np.asarray(v) if v is not None else np.zeros(0))
    return out


# ---- the shard ---------------------------------------------------------------

def shard(a) -> int:
    gg = _load("grooming_gate")
    pp = _load("pixel_pilot")
    fps = spine.fps()
    w = int(frames(WIN_S, fps))
    os.makedirs(os.path.join(work_dir(), "rec"), exist_ok=True)
    rows = [r for r in manifest()["recordings"] if r.get("in_pilot")]
    mine = rows[a.shard::a.of]
    if a.limit:
        mine = mine[:a.limit]
    log(f"  shard {a.shard}/{a.of}: {len(mine)} recordings")
    for n, r in enumerate(mine, 1):
        rid = r["recording_id"]
        out = os.path.join(work_dir(), "rec", f"{rid}.npz")
        if os.path.exists(out) and not a.force:
            continue
        pix = _pixel(rid)
        if pix is None or not bool(pix["usable"]):
            log(f"  {n}/{len(mine)} {rid}: no usable pixel scan, skipped")
            continue
        cutv = dict(zip([str(k) for k in pix["cutoff_keys"]],
                        [float(v) for v in pix["cutoff_values"]]))
        cutoff = cutv["m2"]
        d = spine.clean(rid)
        pose = np.asarray(d["pose"], dtype=np.float64)
        bl = float(np.nanmedian(tego.body_length(pose)))
        video = vid.video_path(rid)
        nf = int(pix["n_frames"])
        idx = np.unique(np.round(np.linspace(0, max(0, nf - 1),
                                             rg.BG_FRAMES)).astype(int))
        raw = _frames_at(video, idx)
        bg_raw = np.median(np.stack(raw).astype(np.float64), axis=0)
        bg_blur = rg.median_background([rg.blur(f) for f in raw])
        s = rg.scan_register(_open(video), pose, bg=bg_blur, cutoff=cutoff,
                             radii_px=[bl * x for x in RADII_BL],
                             dilate_px=bl * mo.DILATE_BODY_LENGTHS, win=w,
                             body_length_px=bl)
        got: dict = {}
        for arm in rg.ARMS:
            for reg, _ in rg.REGIONS:
                for x in RADII_BL:
                    got[f"{arm}|{reg}|{_rkey(x)}"] = s[f"{arm}|{reg}|{bl * x}"]
        sp = gg._speed(r["animal"], rid)
        speed = (np.asarray(sp, dtype=np.float64) if sp is not None
                 else np.zeros(0))
        still = pp._still_frames(r["animal"], rid, nf)
        immobile = (_immobile(pix, still) if still is not None
                    else np.zeros(0, dtype=bool))
        plants = (_plants(video, pose, speed, immobile, bg_raw, bg_blur,
                          cutoff, bl, fps, w)
                  if speed.size and immobile.size else {"plant_ok": False})
        np.savez_compressed(
            out, body_length_px=bl, cutoff=cutoff, win=w,
            arena=s["arena"], identical=s["identical"],
            mask_area=s["mask_area"], mask_ok=s["mask_ok"],
            mask_theta=s["mask_theta"], p_estimate=s["p_estimate"],
            nan_frac=s["nan_frac"], speed=speed, immobile=immobile,
            **{k.replace("|", "__"): v for k, v in got.items()},
            **{k.replace("|", "__"): v for k, v in plants.items()})
        log(f"  {n}/{len(mine)} {rid}: mask NaN {s['nan_frac']:.3f}, "
            f"plants {'ok' if plants.get('plant_ok') else 'REFUSED'}")
    return 0


# ---- combine -----------------------------------------------------------------

def _rec(rid: str) -> dict | None:
    p = os.path.join(work_dir(), "rec", f"{rid}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _series(z: dict, arm: str, reg: str, rad: float = HEADLINE_BL) -> np.ndarray:
    return np.asarray(z[f"{arm}__{reg}__{_rkey(rad)}"], dtype=np.float64)


def _k_check(recs: dict) -> dict:
    """§7: K is the incumbent. The stored grooming series are float32, so the
    comparison is made on float32 casts -- equality there to 1e-9 is exact."""
    worst, n = 0.0, 0
    for rid, z in recs.items():
        p = os.path.join(config.REPO, "work", "grooming", "rec", f"{rid}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=False) as g:
            for x in RADII_BL:
                ref = np.asarray(g[f"energy_ego__{x:g}"], dtype=np.float32)
                mine = (_series(z, "K", "head", x)
                        - np.asarray(z["arena"])).astype(np.float32)
                m = min(ref.size, mine.size)
                a_, b_ = ref[:m], mine[:m]
                if not (np.isfinite(a_) == np.isfinite(b_)).all():
                    return {"ok": False, "reason": f"NaN pattern differs {rid}"}
                f = np.isfinite(a_)
                if f.any():
                    worst = max(worst, float(np.max(np.abs(
                        a_[f].astype(np.float64) - b_[f].astype(np.float64)))))
        n += 1
    return {"ok": n > 0 and worst <= K_TOL, "max_abs_diff": worst,
            "n_recordings": n, "tolerance": K_TOL,
            "note": "compared as float32, the precision the grooming run stored"}


def _refused(recs: dict, arm: str) -> set[str]:
    if arm == "K":
        return set()
    return {rid for rid, z in recs.items()
            if float(z["nan_frac"]) > rg.MAX_NAN_FRAC}


def _gate1(recs, meta, arm, refused) -> Read:
    obj = {"dataset": "luna", "arm": f"stabilise|{arm}|gate1_duplicates"}
    viol, tot = 0, 0
    for rid, z in recs.items():
        if rid in refused:
            continue
        dup = np.asarray(z["identical"], dtype=bool)
        for reg, _ in rg.REGIONS:
            for x in RADII_BL:
                v = _series(z, arm, reg, x)[:dup.size][dup]
                v = v[np.isfinite(v)]
                tot += v.size
                viol += int((v > DUP_TOL).sum())
    share = viol / tot if tot else np.nan
    det = {"n_values": tot, "n_nonzero": viol, "share_nonzero": share}
    if tot == 0:
        return Read("NOT_A_RESULT", obj, "no duplicate frames scored",
                    n_effective=1, detail=det)
    ok = viol == 0
    return Read("PASS" if ok else "FAIL", obj,
                (f"{viol:,} of {tot:,} in-disc values on duplicate video frames "
                 f"are non-zero ({100 * share:.2f}%); the bar is none"),
                n_effective=tot, detail=det)


def _gate2(recs, meta, arm, refused, jit_of, w) -> tuple[Read | None, dict]:
    """(refusal, {}) or (None, detail): K's interval is needed to read the rest."""
    from scipy import stats as st

    obj = {"dataset": "luna", "arm": f"stabilise|{arm}|gate2_jitter"}
    per: dict[str, list] = {}
    for rid, z in recs.items():
        if rid in refused:
            continue
        sp = np.asarray(z["speed"], dtype=np.float64)
        e = _series(z, arm, "head") - np.asarray(z["arena"])
        j = jit_of(rid)
        n = min(sp.size, e.size, j.size)
        k = n // w
        if k < 4:
            continue
        S = sp[:k * w].reshape(k, w)
        E = e[:k * w].reshape(k, w)
        J = j[:k * w].reshape(k, w)
        okS = np.isfinite(S).all(axis=1)
        ms = S.mean(axis=1)
        if okS.sum() < 4:
            continue
        still = okS & (ms <= np.percentile(ms[okS], STILL_PCT))
        okE = np.isfinite(E).all(axis=1)
        jj = np.array([np.nanstd(r) if np.isfinite(r).sum() >= 2 else np.nan
                       for r in J])
        use = still & okE & np.isfinite(jj)
        a = meta[rid]["animal"]
        for i in np.flatnonzero(use):
            per.setdefault(a, []).append((ms[i], E[i].mean(), jj[i]))
    corr, who = [], []
    for a, rows in per.items():
        if len(rows) < MIN_STILL_WINDOWS:
            continue
        x = np.asarray(rows)
        sr, er, jr = (st.rankdata(x[:, 0]), st.rankdata(x[:, 1]),
                      st.rankdata(x[:, 2]))
        rj = jr - np.polyval(np.polyfit(sr, jr, 1), sr)
        re = er - np.polyval(np.polyfit(sr, er, 1), sr)
        if np.std(rj) == 0 or np.std(re) == 0:
            continue
        corr.append(float(np.corrcoef(rj, re)[0, 1]))
        who.append(a)
    if len(corr) < MIN_ANIMALS:
        return Read("NOT_A_RESULT", obj,
                    f"{len(corr)} animals with >= {MIN_STILL_WINDOWS} still "
                    f"windows, against {MIN_ANIMALS}",
                    n_effective=max(1, len(corr))), {}
    b = boot.animal_interval(corr, who, how="mean", n_boot=N_BOOT, seed=SEED)
    return None, {"interval": b, "n_animals": len(corr),
                  "n_windows": int(sum(len(v) for v in per.values()))}


def _ratio_gate(values: dict[str, list[float]], obj: dict, what: str,
                min_n: int) -> Read:
    per = {a: v for a, v in values.items() if len(v) >= min_n}
    if len(per) < MIN_ANIMALS:
        return Read("NOT_A_RESULT", obj,
                    f"{len(per)} animals with >= {min_n} {what}, against "
                    f"{MIN_ANIMALS}", n_effective=max(1, len(per)))
    units = [float(np.mean(v)) for v in per.values()]
    b = boot.animal_interval(units, list(per), how="mean", n_boot=N_BOOT,
                             seed=SEED)
    ok = float(b["hi"]) <= FLOOR_BAR
    return Read("PASS" if ok else "FAIL", obj,
                (f"registered in-disc motion is {b['point']:.3f}x "
                 f"[{b['lo']:.3f}, {b['hi']:.3f}] the arena's own noise "
                 f"over {len(per)} animals; the bar is an upper bound <= "
                 f"{FLOOR_BAR}"),
                n_effective=len(per), detail={"interval": b})


def _gate3(recs, meta, arm, refused) -> Read:
    obj = {"dataset": "luna", "arm": f"stabilise|{arm}|gate3_immobility"}
    vals: dict[str, list[float]] = {}
    for rid, z in recs.items():
        if rid in refused:
            continue
        im = np.asarray(z["immobile"], dtype=bool)
        e = _series(z, arm, "head")
        ar = np.asarray(z["arena"], dtype=np.float64)
        n = min(im.size, e.size, ar.size)
        r = e[:n][im[:n]] / ar[:n][im[:n]]
        r = r[np.isfinite(r)]
        vals.setdefault(meta[rid]["animal"], []).extend(r.tolist())
    return _ratio_gate(vals, obj, "immobile frames", MIN_IMMOBILE_FRAMES)


def _gate4(recs, meta, arm, refused) -> Read:
    obj = {"dataset": "luna", "arm": f"stabilise|{arm}|gate4_plant"}
    vals: dict[str, list[float]] = {}
    for rid, z in recs.items():
        if rid in refused or not bool(z.get("plant_ok", False)):
            continue
        v = np.asarray(z.get(f"g4__moving__{arm}__head", np.zeros(0)))[1:]
        floor = float(np.nanmedian(z["arena"]))
        if v.size == 0 or not np.isfinite(v).any() or floor <= 0:
            continue
        vals.setdefault(meta[rid]["animal"], []).append(
            float(np.nanmean(v)) / floor)
    return _ratio_gate(vals, obj, "planted recordings", 1)


def _gate5(recs, arm, fps) -> dict:
    """§6: recall against the arm's own unplanted 95th percentile. No verdict."""
    def pe(v):
        v = np.asarray(v, dtype=np.float64)[1:]
        if v.size < 8 or not np.isfinite(v).all():
            return np.nan
        return float(hd.peak_excess(v, fps=fps))
    null = []
    for z in recs.values():
        if not bool(z.get("plant_ok", False)):
            continue
        for cond in ("still", "moving"):
            null.append(pe(z.get(f"g4__{cond}__{arm}__head", np.zeros(0))))
    null_a = np.asarray([v for v in null if np.isfinite(v)])
    if null_a.size < 20:
        return {"refused": f"{null_a.size} unplanted windows"}
    thr = float(np.percentile(null_a, DETECT_PCT))
    curve: dict = {"threshold": thr, "n_null": int(null_a.size)}
    for cond in ("still", "moving"):
        for hz in PLANT_HZ:
            for amp in PLANT_AMP_PX:
                row = {}
                for reg in ("head", "hip"):
                    xs = [pe(z.get(f"g5__{cond}__{hz:g}__{amp:g}__{arm}__{reg}",
                                   np.zeros(0)))
                          for z in recs.values()
                          if bool(z.get("plant_ok", False))]
                    xs = [x for x in xs if np.isfinite(x)]
                    row[reg] = {"recall": (float(np.mean(np.asarray(xs) > thr))
                                           if xs else None), "n": len(xs)}
                curve[f"{cond}|{hz:g}Hz|{amp:g}px"] = row
    return curve


def _coupling(recs, arm, w) -> dict:
    """Reported, never gated: window energy vs keypoint speed, per recording."""
    out = {}
    for x in RADII_BL:
        for reg in ("head", "hip"):
            cs = []
            for z in recs.values():
                sp = np.asarray(z["speed"], dtype=np.float64)
                e = _series(z, arm, reg, x) - np.asarray(z["arena"])
                n = min(sp.size, e.size)
                k = n // w
                if k < 4:
                    continue
                S = sp[:k * w].reshape(k, w).mean(axis=1)
                E = e[:k * w].reshape(k, w).mean(axis=1)
                ok = np.isfinite(S) & np.isfinite(E)
                if ok.sum() >= 4 and np.std(E[ok]) > 0:
                    cs.append(float(np.corrcoef(S[ok], E[ok])[0, 1]))
            out[f"{reg}|{x:g}"] = float(np.median(cs)) if cs else None
    return out


def combine(a) -> int:
    gj = _load("grooming_jitter")
    fps = spine.fps()
    w = int(frames(WIN_S, fps))
    meta = {r["recording_id"]: r for r in manifest()["recordings"]
            if r.get("in_pilot")}
    recs = {rid: z for rid in meta if (z := _rec(rid)) is not None}
    log(f"  {len(recs)} of {len(meta)} recordings scanned")
    reads: dict = {}
    kc = _k_check(recs)
    reads["k_check"] = kc
    if not kc["ok"]:
        log(f"  §7 STOP: K is not the incumbent: {kc}")
        write_json({"stage": "stabilise", "stopped": "§7", "k_check": kc},
                   a.out or config.PATHS.result("stabilise.json"))
        return 1
    log(f"  §7 K reproduces the incumbent: max |diff| {kc['max_abs_diff']:.2e}")
    cache: dict = {}

    def jit_of(rid):
        return np.asarray(gj.skull_jitter(rid, cache), dtype=np.float64)

    g2: dict = {}
    for arm in rg.ARMS:
        ref = _refused(recs, arm)
        det: dict = {"refused_recordings": len(ref),
                     "refused_share": len(ref) / max(1, len(recs))}
        g2[arm] = _gate2(recs, meta, arm, ref, jit_of, w)
        reads[arm] = {"detail": det,
                      "gate1": _gate1(recs, meta, arm, ref).to_dict(),
                      "gate3": _gate3(recs, meta, arm, ref).to_dict(),
                      "gate4": _gate4(recs, meta, arm, ref).to_dict(),
                      "gate5_curve": _gate5(recs, arm, fps),
                      "coupling_with_speed": _coupling(recs, arm, w)}
    # §3 needs K's interval before any other arm's can be read against it.
    k_rd, k_det = g2["K"]
    for arm in rg.ARMS:
        rd, det = g2[arm]
        obj = {"dataset": "luna", "arm": f"stabilise|{arm}|gate2_jitter"}
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
                ok = float(b["hi"]) < JITTER_BAR and float(b["hi"]) < float(kb["lo"])
                rd = Read("PASS" if ok else "FAIL", obj,
                          (f"jitter -> head energy, speed held fixed: "
                           f"{b['point']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}] "
                           f"against K's {kb['point']:+.3f} [{kb['lo']:+.3f}, "
                           f"{kb['hi']:+.3f}]; the bar is an upper bound < "
                           f"{JITTER_BAR} AND below K's lower bound"),
                          n_effective=det["n_animals"], detail=det)
        reads[arm]["gate2"] = rd.to_dict()
    for arm in rg.ARMS:
        r = reads[arm]
        vs = [r[g]["verdict"] for g in ("gate1", "gate2", "gate3", "gate4")]
        ref_share = r["detail"]["refused_share"]
        obj = {"dataset": "luna", "arm": f"stabilise|{arm}"}
        if arm != "K" and ref_share > rg.MAX_REFUSED_FRAC:
            v, why = "NOT_A_RESULT", (f"{100 * ref_share:.1f}% of recordings "
                                      f"refused, against {100 * rg.MAX_REFUSED_FRAC:.0f}%")
        elif arm == "K":
            v, why = ("FAIL" if "FAIL" in vs else "NOT_A_RESULT",
                      "the incumbent; gate 2 is its reference, so it is scored "
                      "on gates 1, 3 and 4 only")
        elif "FAIL" in vs:
            v, why = "FAIL", f"gates 1-4: {vs}"
        elif all(x == "PASS" for x in vs):
            v, why = "PASS", ("gates 1-4 all pass; the arm is ELIGIBLE for a "
                              "separately registered stage, nothing more (§10.6)")
        else:
            v, why = "NOT_A_RESULT", f"gates 1-4: {vs}"
        r["verdict"] = Read(v, obj, why, n_effective=len(recs)).to_dict()
        log(f"  {arm}: {v} -- {why}")
    out = a.out or config.PATHS.result("stabilise.json")
    write_json({**anchors.header(anchors.LUNA, stage="stabilise",
                                 unverified="a registration instrument"),
                "inherited_digest": spine.digest(),
                "registration": "results/STABILISE_PREREGISTRATION.md",
                "deviation": "DEVIATIONS.md D22",
                "seed": SEED, "fps": fps, "win_frames": w,
                "headline_radius_bl": HEADLINE_BL, "reads": reads}, out)
    log(f"  wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("combine",), default=None)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=12)
    p.add_argument("--limit", type=int, default=0,
                   help="smoke run: only the first N recordings of the shard")
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
