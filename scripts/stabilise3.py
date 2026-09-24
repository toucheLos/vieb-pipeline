"""STABILISE 3: the same registrations on a SAM mask track, gates calibrated on real data.

    sbatch jobs/stabilise3.slurm                 # 12 GPU shards
    python3 scripts/stabilise3.py --phase combine

READ results/STABILISE3_PREREGISTRATION.md FIRST, and
results/stabilise3_calibration.json, which set its bars before this existed.

Arms: K (the incumbent, §7-checked), SB (moment warp on SAM's interpolated mask
pose), SP (phase-correlation-initialised ECC in SAM's interpolated box). The
track is `vieb.pixel.sam.sam_track`; the differencing is
`register.scan_track`, the same pass 2 every earlier stage used.
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
import stabilise2 as s2                                              # noqa: E402

st = s2.st
CHECKPOINT = os.environ.get("VIEB_SAM_CKPT",
                            "/home/tul26194/models/sam/sam_vit_b_01ec64.pth")
CKPT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
ARMS = ("K", "SB", "SP")
#: scan_track's arm names, and what they are here.
RENAME = {"K": "K", "B": "SB", "P": "SP", "I": "I"}
#: §3.
GATE4_BL = 0.2
BAND = (0.80, 1.25)
GATE0_BAR = 0.90


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "stabilise3")


st.work_dir = work_dir
st.MIN_IMMOBILE_FRAMES = s2.MIN_IMMOBILE_FRAMES


def _rgb(video: str):
    import cv2

    def gen():
        cap = cv2.VideoCapture(video)
        try:
            while True:
                ok, fr = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
    return gen


def _as_rgb(greys):
    return [np.repeat(np.asarray(g)[:, :, None], 3, axis=2) for g in greys]


def _plants3(video, pose, speed, immobile, bg_raw, bl, fps, w, predict,
             prompt_mode: str = "propagate") -> dict:
    """§3: the moving plants, cut with SAM's source mask, tracked by SAM."""
    inp = st._plant_inputs(pose, speed, immobile, w)
    if inp is None:
        return {"plant_ok": False}
    src = st._frames_at(video, [inp["src"]])
    if not src:
        return {"plant_ok": False}
    sg, sp = src[0], pose[inp["src"]]
    box = sam.keypoint_box(sp, sam.PAD_BL * bl)
    if box is None:
        return {"plant_ok": False}
    m0, iou = predict(_as_rgb([sg])[0], box)
    if iou < sam.IOU_MIN or not np.asarray(m0).any():
        return {"plant_ok": False, "plant_src_iou": float(iou)}
    r4, r5 = bl * GATE4_BL, bl * st.HEADLINE_BL
    out: dict = {"plant_ok": True, "plant_src": inp["src"],
                 "plant_src_iou": float(iou)}
    cells = [(None, None)] + [(hz, amp) for hz in st.PLANT_HZ
                              for amp in st.PLANT_AMP_PX]
    for hz, amp in cells:
        fr, poses, Ms = st._plant_frames(sg, sp, bg_raw, m0, bl, inp,
                                         moving=True, hz=hz, amp=amp, fps=fps)
        rgbs = _as_rgb(fr)
        tr = sam.sam_track(lambda: iter(rgbs), poses, predict,
                           body_length_px=bl, prompt_mode=prompt_mode)
        s = rg.scan_track(lambda: iter(fr), poses, tr, radii_px=[r4, r5],
                          dilate_px=bl * mo.DILATE_BODY_LENGTHS, win=len(fr))
        s.update(st._oracle(fr, Ms, sp, [r4, r5]))
        tag = "base" if hz is None else f"{hz:g}__{amp:g}"
        for src_arm, arm in list(RENAME.items()) + [("O", "O")]:
            for reg in ("head", "hip"):
                v = s.get(f"{src_arm}|{reg}|{r5}")
                out[f"p__{tag}__{arm}__{reg}"] = (np.asarray(v) if v is not None
                                                  else np.zeros(0))
                if tag == "base":
                    v4 = s.get(f"{src_arm}|{reg}|{r4}")
                    out[f"g4__{arm}__{reg}"] = (np.asarray(v4) if v4 is not None
                                                else np.zeros(0))
        if tag == "base":
            out["plant_track_ok"] = float(np.mean(tr["ok"]))
    return out


def _check_checkpoint() -> None:
    import hashlib

    h = hashlib.sha256()
    with open(CHECKPOINT, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != CKPT_SHA256:
        raise SystemExit(f"SAM checkpoint hash {h.hexdigest()} is not the "
                         f"registered {CKPT_SHA256}")


def shard(a) -> int:
    _check_checkpoint()
    gg = st._load("grooming_gate")
    pp = st._load("pixel_pilot")
    predict = sam.sam_predictor(CHECKPOINT)
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
        tr = sam.sam_track(_rgb(video), pose, predict, body_length_px=bl)
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
        plants = (_plants3(video, pose, speed, immobile, bg_raw, bl, fps, w,
                           predict)
                  if speed.size and immobile.size else {"plant_ok": False})
        nan_frac = float(1.0 - np.mean(tr["ok"])) if tr["n"] else 1.0
        np.savez_compressed(
            out, body_length_px=bl, win=w, arena=s["arena"],
            identical=s["identical"], mask_area=s["mask_area"],
            mask_ok=s["mask_ok"], p_estimate=s["p_estimate"],
            nan_frac=nan_frac, key_iou=tr["key_iou"],
            key_accepted=tr["key_accepted"], key_seeded=tr["key_seeded"],
            gate0_inside=inside, gate0_scored=scored,
            speed=speed, immobile=immobile, **got, **plants)
        log(f"  {n}/{len(mine)} {rid}: refused {nan_frac:.3f}, seeds "
            f"{int(tr['key_seeded'].sum())}, iou median "
            f"{np.nanmedian(tr['key_iou']):.3f}, mask/bl^2 "
            f"{np.median(tr['area'][tr['ok']]) / bl ** 2 if tr['ok'].any() else float('nan'):.3f}, "
            f"on-animal {inside}/{scored}, plants "
            f"{'ok' if plants.get('plant_ok') else 'REFUSED'}")
    return 0


# ---- combine ------------------------------------------------------------------

def _refused3(recs: dict, arm: str) -> set[str]:
    if arm == "K":
        return set()
    return {rid for rid, z in recs.items()
            if float(z["nan_frac"]) > rg.MAX_NAN_FRAC}


def _band_read(units: dict[str, list[float]], obj: dict, what: str) -> Read:
    per = {a: v for a, v in units.items() if v}
    if len(per) < st.MIN_ANIMALS:
        return Read("NOT_A_RESULT", obj,
                    f"{len(per)} animals with {what}, against {st.MIN_ANIMALS}",
                    n_effective=max(1, len(per)))
    vals = [float(np.mean(v)) for v in per.values()]
    b = boot.animal_interval(vals, list(per), how="mean", n_boot=st.N_BOOT,
                             seed=st.SEED)
    ok = float(b["lo"]) >= BAND[0] and float(b["hi"]) <= BAND[1]
    return Read("PASS" if ok else "FAIL", obj,
                (f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] of a perfect "
                 f"registration over {len(per)} animals; the bar is the whole "
                 f"interval within [{BAND[0]}, {BAND[1]}]"),
                n_effective=len(per), detail={"interval": b})


def _gate0(recs, meta, arm, refused) -> Read:
    obj = {"dataset": "luna", "arm": f"stabilise3|{arm}|gate0_on_animal"}
    ins: dict[str, list[int]] = {}
    for rid, z in recs.items():
        if rid in refused:
            continue
        a = meta[rid]["animal"]
        x = ins.setdefault(a, [0, 0])
        x[0] += int(z["gate0_inside"])
        x[1] += int(z["gate0_scored"])
    per = {a: v[0] / v[1] for a, v in ins.items() if v[1] > 0}
    if len(per) < st.MIN_ANIMALS:
        return Read("NOT_A_RESULT", obj, f"{len(per)} animals scored",
                    n_effective=max(1, len(per)))
    b = boot.animal_interval(list(per.values()), list(per), how="mean",
                             n_boot=st.N_BOOT, seed=st.SEED)
    ok = float(b["lo"]) >= GATE0_BAR
    return Read("PASS" if ok else "FAIL", obj,
                (f"the head-disc centre lies inside SAM's mask on "
                 f"{100 * b['point']:.1f}% [{100 * b['lo']:.1f}, "
                 f"{100 * b['hi']:.1f}] of accepted keyframes over {len(per)} "
                 f"animals; the bar is a lower bound >= {100 * GATE0_BAR:.0f}%"),
                n_effective=len(per), detail={"interval": b})


def _gate3(recs, meta, arm, refused) -> Read:
    """§3: ratio of MEANS against the identity, on immobile frames."""
    obj = {"dataset": "luna", "arm": f"stabilise3|{arm}|gate3_identity"}
    acc: dict[str, list[float]] = {}
    for rid, z in recs.items():
        if rid in refused:
            continue
        im = np.asarray(z["immobile"], dtype=bool)
        e = st._series(z, arm, "head")
        i = st._series(z, "I", "head")
        n = min(im.size, e.size, i.size)
        m = im[:n] & np.isfinite(e[:n]) & np.isfinite(i[:n])
        x = acc.setdefault(meta[rid]["animal"], [0.0, 0.0, 0.0])
        x[0] += float(e[:n][m].sum())
        x[1] += float(i[:n][m].sum())
        x[2] += float(m.sum())
    units = {a: [v[0] / v[1]] for a, v in acc.items()
             if v[2] >= s2.MIN_IMMOBILE_FRAMES and v[1] > 0}
    return _band_read(units, obj, f">= {s2.MIN_IMMOBILE_FRAMES} immobile frames")


def _gate4(recs, meta, arm, refused) -> Read:
    obj = {"dataset": "luna", "arm": f"stabilise3|{arm}|gate4_oracle"}
    units: dict[str, list[float]] = {}
    for rid, z in recs.items():
        if rid in refused or not bool(z.get("plant_ok", False)):
            continue
        v = np.asarray(z.get(f"g4__{arm}__head", np.zeros(0)))[1:]
        o = np.asarray(z.get("g4__O__head", np.zeros(0)))[1:]
        if not (v.size and np.isfinite(v).any() and np.isfinite(o).any()):
            continue
        den = float(np.nanmedian(o))
        if den > 0:
            units.setdefault(meta[rid]["animal"], []).append(
                float(np.nanmedian(v)) / den)
    return _band_read(units, obj, "a scored plant")


def combine(a) -> int:
    gj = st._load("grooming_jitter")
    fps = spine.fps()
    w = int(frames(st.WIN_S, fps))
    meta = {r["recording_id"]: r for r in st.manifest()["recordings"]
            if r.get("in_pilot")}
    recs = {rid: z for rid in meta if (z := st._rec(rid)) is not None}
    log(f"  {len(recs)} of {len(meta)} recordings scanned")
    out = a.out or config.PATHS.result("stabilise3.json")
    reads: dict = {}
    kc = st._k_check(recs)
    reads["k_check"] = kc
    if not kc["ok"]:
        log(f"  §7 STOP: K is not the incumbent: {kc}")
        write_json({"stage": "stabilise3", "stopped": "§7", "k_check": kc}, out)
        return 1
    log(f"  §7 K reproduces the incumbent: max |diff| {kc['max_abs_diff']:.2e}")
    cache: dict = {}

    def jit_of(rid):
        return np.asarray(gj.skull_jitter(rid, cache), dtype=np.float64)

    g2: dict = {}
    for arm in ARMS:
        ref = _refused3(recs, arm)
        det = {"refused_recordings": len(ref),
               "refused_share": len(ref) / max(1, len(recs))}
        g2[arm] = st._gate2(recs, meta, arm, ref, jit_of, w)
        reads[arm] = {"detail": det,
                      "gate0": _gate0(recs, meta, arm, ref).to_dict(),
                      "gate1": st._gate1(recs, meta, arm, ref).to_dict(),
                      "gate3": _gate3(recs, meta, arm, ref).to_dict(),
                      "gate4": _gate4(recs, meta, arm, ref).to_dict(),
                      "gate5_curve": s2._gate5(recs, arm, ref, fps),
                      "coupling_with_speed": st._coupling(recs, arm, w)}
    k_rd, k_det = g2["K"]
    for arm in ARMS:
        rd, det = g2[arm]
        obj = {"dataset": "luna", "arm": f"stabilise3|{arm}|gate2_jitter"}
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
    for arm in ARMS:
        r = reads[arm]
        gates = ("gate0", "gate1", "gate2", "gate3", "gate4")
        vs = [r[g]["verdict"] for g in gates]
        ref_share = r["detail"]["refused_share"]
        obj = {"dataset": "luna", "arm": f"stabilise3|{arm}"}
        if arm != "K" and ref_share > rg.MAX_REFUSED_FRAC:
            v, why = "NOT_A_RESULT", (f"{100 * ref_share:.1f}% of recordings "
                                      f"refused, against "
                                      f"{100 * rg.MAX_REFUSED_FRAC:.0f}%")
        elif arm == "K":
            v, why = ("FAIL" if "FAIL" in vs else "NOT_A_RESULT",
                      "the incumbent; gate 2 is its reference")
        elif "FAIL" in vs:
            v, why = "FAIL", f"gates 0-4: {vs}"
        elif all(x == "PASS" for x in vs):
            v, why = "PASS", ("gates 0-4 all pass; the arm is ELIGIBLE for a "
                              "separately registered stage, nothing more")
        else:
            v, why = "NOT_A_RESULT", f"gates 0-4: {vs}"
        r["verdict"] = Read(v, obj, why, n_effective=len(recs)).to_dict()
        log(f"  {arm}: {v} -- {why}")
    write_json({**anchors.header(anchors.LUNA, stage="stabilise3",
                                 unverified="a registration instrument"),
                "inherited_digest": spine.digest(),
                "registration": "results/STABILISE3_PREREGISTRATION.md",
                "checkpoint_sha256": CKPT_SHA256,
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
