"""Step 3: does head-region motion carry a 3-8 Hz rhythm keypoints cannot see?

    sbatch jobs/grooming.slurm
    python3 scripts/grooming_clips.py                  # 30 + 15, blind panel
    python3 scripts/grooming_gate.py --phase combine

READ results/GROOMING_PREREGISTRATION.md FIRST, and `DEVIATIONS.md` D19, which
corrects §4's claim about the statistic before any of it was used.

## The circularity, and why this script cannot resolve it alone

Candidates are selected on LOW body speed AND HIGH head-region motion, because
there are no paw keypoints and no labels. Testing head-region motion on windows
selected for head-region motion would guarantee its own answer. §3 and §4 are the
two fixes and **both are required**:

* `scripts/grooming_clips.py` renders 30 candidates and 15 speed-matched controls, shuffled
  under opaque ids, for confirmation BY EYE. Below a 50% confirmation rate §9.4
  forbids reading the spectral statistic at all.
* the statistic is `peak_excess`, which is **exactly** invariant to the
  multiplicative rescaling that amplitude selection performs -- verified to six
  decimals over a 1000x range -- rather than an amplitude.

D19 records what that invariance does NOT cover: `peak_excess` is sensitive to
the background SLOPE, so the fitted slope is reported for both arms and a
difference in it is reported beside the headline.
"""
from __future__ import annotations

import argparse
import glob
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
from vieb.pixel import head as hd, motion as mo                      # noqa: E402
from vieb.seg import controls as co                                  # noqa: E402
from vieb.tok import config, ego as tego, quantize as qz             # noqa: E402

SEED = 0
N_BOOT = 2000
#: §6 refusals.
MIN_ANIMALS = 20
MIN_CANDIDATES = 500
MIN_CONFIRMED = 0.50
#: §7, the registered grid. Headline is the emphasised centre of each.
RADII_BL = (0.4, 0.6, 0.8)
WINDOWS_S = (1.5, 2.0, 3.0)
STILL_PCTS = (10.0, 25.0, 40.0)
HEADLINE = (0.6, 2.0, 25.0)
#: §3, the eye-confirmation panel.
N_CLIPS_CANDIDATE = 30
N_CLIPS_CONTROL = 15
MIN_CLIP_ANIMALS = 15


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "grooming")


def manifest() -> dict:
    with open(os.path.join(config.REPO, "work", "pixel", "manifest.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def shard(a) -> int:
    """§1: one sequential decode per recording, at every registered radius."""
    os.makedirs(os.path.join(work_dir(), "rec"), exist_ok=True)
    rows = [r for r in manifest()["recordings"] if r.get("in_pilot")]
    mine = rows[a.shard::a.of]
    log(f"  shard {a.shard}/{a.of}: {len(mine)} recordings")
    for n, r in enumerate(mine, 1):
        rid = r["recording_id"]
        out = os.path.join(work_dir(), "rec", f"{rid}.npz")
        if os.path.exists(out) and not a.force:
            continue
        d = spine.clean(rid)
        pose = np.asarray(d["pose"], dtype=np.float64)
        bl = float(np.nanmedian(tego.body_length(pose)))
        got: dict = {}
        for rad in RADII_BL:
            s = hd.scan_head(vid.video_path(rid), pose, radius_px=bl * rad,
                             dilate_px=bl * mo.DILATE_BODY_LENGTHS)
            got[f"energy__{rad:g}"] = s["energy"].astype(np.float32)
        sp = _speed(r["animal"], rid)
        np.savez_compressed(out, body_length_px=bl,
                            speed=(sp if sp is not None
                                   else np.zeros(0, dtype=np.float32)),
                            **got)
        log(f"  {n}/{len(mine)} {rid}")
    return 0


def _speed(animal: str, rid: str):
    p = os.path.join(config.REPO, "work", "ego", f"raw__bodylen__{animal}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        rids = [str(v) for v in z["recording_ids"]]
        if rid not in rids:
            return None
        i = rids.index(rid)
        b = np.asarray(z["bounds"], dtype=np.int64)
        return qz.speed(np.asarray(z["X"], dtype=np.float64)[b[i]:b[i + 1]]
                        ).astype(np.float32)


def _windows(rid: str, rad: float, win_s: float, still_pct: float,
             fps: float) -> list[dict] | None:
    """§2: non-overlapping windows, and which of them are candidates."""
    p = os.path.join(work_dir(), "rec", f"{rid}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        key = f"energy__{rad:g}"
        if key not in z.files:
            return None
        e = np.asarray(z[key], dtype=np.float64)
        sp = np.asarray(z["speed"], dtype=np.float64)
    if sp.size == 0:
        return None
    n = min(e.size, sp.size)
    w = int(frames(win_s, fps))
    if w < 8 or n < w:
        return None
    k = n // w
    E = e[:k * w].reshape(k, w)
    S = sp[:k * w].reshape(k, w)
    ok = np.isfinite(E).all(axis=1) & np.isfinite(S).all(axis=1)
    if int(ok.sum()) < 4:
        return None
    me, ms = E.mean(axis=1), S.mean(axis=1)
    # Percentiles from THIS recording's usable windows, so a recording with a
    # noisier camera or a slower animal is thresholded against itself.
    t_still = float(np.percentile(ms[ok], still_pct))
    t_head = float(np.percentile(me[ok], hd.HEAD_PCT))
    rows = []
    for i in range(k):
        if not ok[i]:
            continue
        rows.append({"rid": rid, "i": int(i), "a": int(i * w),
                     "b": int((i + 1) * w),
                     "speed": float(ms[i]), "energy": float(me[i]),
                     "candidate": bool(ms[i] <= t_still and me[i] >= t_head),
                     "series": E[i]})
    return rows


def _collect(rad: float, win_s: float, still_pct: float, fps: float) -> dict:
    man = manifest()
    rows: list[dict] = []
    by_rid = {r["recording_id"]: r for r in man["recordings"]
              if r.get("in_pilot")}
    for rid, meta in by_rid.items():
        got = _windows(rid, rad, win_s, still_pct, fps)
        if not got:
            continue
        for r in got:
            st = hd.spectrum_stats(r["series"], fps=fps)
            if not np.isfinite(st["peak_excess"]):
                continue
            rows.append({**{k: v for k, v in r.items() if k != "series"},
                         **st, "animal": meta["animal"],
                         "context": meta["context"], "day": meta["day"]})
    cand = np.asarray([r["candidate"] for r in rows], dtype=bool)
    animal = np.asarray([r["animal"] for r in rows])
    feats = np.column_stack([np.log(np.maximum(
        [r["speed"] for r in rows], 1e-9))])
    partners = co.matched_partners(cand, animal, feats,
                                   pool=np.isfinite(feats).all(axis=1) & ~cand)
    return {"rows": rows, "cand": cand, "animal": animal, "feats": feats,
            "partners": partners}


def _arm_read(got: dict, obj: dict, *, headline: bool) -> tuple[Read, dict]:
    rows, cand, partners = got["rows"], got["cand"], got["partners"]
    t_idx = np.flatnonzero(cand)
    ok = partners >= 0
    n_cand = int(cand.sum())
    if n_cand < MIN_CANDIDATES:
        return Read("NOT_A_RESULT", obj,
                    f"{n_cand} candidate windows, against the registered "
                    f"minimum of {MIN_CANDIDATES}",
                    n_effective=max(1, n_cand)), {}
    bal = co.balance_read(got["feats"][t_idx[ok]], got["feats"][partners[ok]],
                          names=("log_speed_bl_s",), scored_object=obj,
                          n_effective=int(ok.sum()))
    if bal.verdict != "PASS":
        return Read("FAIL", obj,
                    f"the speed-matched control is not balanced: {bal.reason}. "
                    f"§5 refuses; re-drawing for balance is forbidden",
                    n_effective=int(ok.sum())), {"balance": bal.to_dict()}
    who_t = [rows[int(i)]["animal"] for i in t_idx[ok]]
    who_c = [rows[int(i)]["animal"] for i in partners[ok]]
    n_an = len(set(who_t) | set(who_c))
    if n_an < MIN_ANIMALS:
        return Read("NOT_A_RESULT", obj,
                    f"{n_an} animals contribute a matched pair, against the "
                    f"registered minimum of {MIN_ANIMALS}",
                    n_effective=max(1, n_an)), {"balance": bal.to_dict()}
    det: dict = {"balance": bal.to_dict(), "n_candidates": n_cand,
                 "n_pairs": int(ok.sum()), "n_animals": n_an}
    for stat in ("peak_excess", "band_share", "slope", "excess_low",
                 "excess_high"):
        tv = [rows[int(i)][stat] for i in t_idx[ok]]
        cv = [rows[int(i)][stat] for i in partners[ok]]
        det[stat] = {
            "candidate": dict(boot.animal_interval(tv, who_t, how="mean",
                                                   n_boot=N_BOOT, seed=SEED)),
            "control": dict(boot.animal_interval(cv, who_c, how="mean",
                                                 n_boot=N_BOOT, seed=SEED))}
    pe = det["peak_excess"]
    clears = float(pe["candidate"]["lo"]) > float(pe["control"]["hi"])
    # D19: peak_excess rises with a steeper background, so a slope difference
    # between the arms can masquerade as a peak. Reported beside the verdict.
    sl = det["slope"]
    slope_differs = (float(sl["candidate"]["lo"]) > float(sl["control"]["hi"])
                     or float(sl["candidate"]["hi"])
                     < float(sl["control"]["lo"]))
    det["slope_differs"] = slope_differs
    # §5: if the excess rises monotonically with amplitude, the invariance claim
    # is false and the arm is INCONCLUSIVE whatever the contrast says.
    e = np.asarray([rows[int(i)]["energy"] for i in t_idx[ok]])
    x = np.asarray([rows[int(i)]["peak_excess"] for i in t_idx[ok]])
    dec = np.clip(np.searchsorted(np.percentile(e, np.arange(10, 100, 10)),
                                  e), 0, 9)
    means = [float(np.mean(x[dec == d])) if (dec == d).any() else float("nan")
             for d in range(10)]
    fin = [m for m in means if np.isfinite(m)]
    mono = len(fin) >= 5 and all(b >= a for a, b in zip(fin, fin[1:]))
    det["amplitude_deciles"] = means
    det["monotone_in_amplitude"] = mono
    v = "INCONCLUSIVE" if mono else ("PASS" if clears else "FAIL")
    return Read(v, obj,
                (f"3-8 Hz peak_excess {'CLEARS' if clears else 'does NOT clear'} "
                 f"the speed-matched control: candidates "
                 f"{pe['candidate']['point']:+.4f} "
                 f"[{pe['candidate']['lo']:+.4f}, {pe['candidate']['hi']:+.4f}] "
                 f"against {pe['control']['point']:+.4f} "
                 f"[{pe['control']['lo']:+.4f}, {pe['control']['hi']:+.4f}] over "
                 f"{n_an} animals and {int(ok.sum())} matched pairs, against a "
                 f"peak-free null of +0.06 (D19, NOT zero). Sub-bands 3-6 Hz "
                 f"{det['excess_low']['candidate']['point']:+.4f} and 6-8 Hz "
                 f"{det['excess_high']['candidate']['point']:+.4f}"
                 + (". INCONCLUSIVE: peak_excess rises monotonically across "
                    "amplitude deciles, so §4's invariance claim fails on this "
                    "data and the selection cannot be ruled out" if mono else "")
                 + (". NOTE: the background slopes DIFFER between arms "
                    f"({sl['candidate']['point']:+.3f} against "
                    f"{sl['control']['point']:+.3f}), and D19 shows a steeper "
                    f"background inflates peak_excess" if slope_differs else "")
                 + ("" if headline else " [sweep arm, not the headline]")),
                n_effective=n_an, detail=det), det


def combine(a) -> int:
    fps = spine.fps()
    reads: dict = {}
    verdicts: dict[str, str] = {}
    obj0 = {"dataset": "luna", "arm": "grooming_gate", "split": "fit"}
    for rad in RADII_BL:
        for win_s in WINDOWS_S:
            for pct in STILL_PCTS:
                name = f"r{rad:g}_w{win_s:g}_p{pct:g}"
                headline = (rad, win_s, pct) == HEADLINE
                got = _collect(rad, win_s, pct, fps)
                rd, _det = _arm_read(got, {**obj0, "arm": f"gate|{name}",
                                           "radius_bl": rad, "win_s": win_s,
                                           "still_pct": pct},
                                     headline=headline)
                reads[f"gate|{name}"] = rd.to_dict()
                verdicts[name] = rd.verdict
                if headline:
                    log("  HEADLINE " + rd.line())
                else:
                    log(f"    {name}: {rd.verdict}")
    head = verdicts[f"r{HEADLINE[0]:g}_w{HEADLINE[1]:g}_p{HEADLINE[2]:g}"]
    others = [v for k, v in verdicts.items()
              if k != f"r{HEADLINE[0]:g}_w{HEADLINE[1]:g}_p{HEADLINE[2]:g}"]
    if any(v != head for v in others):
        reads["gate"] = Read(
            "GRID_LIMITED", {**obj0, "arm": "gate"},
            (f"the headline verdict {head} does not hold across the registered "
             f"grid: {sum(v != head for v in others)} of {len(others)} other "
             f"arms disagree. §7 makes that GRID_LIMITED, not {head}"),
            n_effective=len(verdicts)).to_dict()
        log("  " + Read("GRID_LIMITED", {**obj0, "arm": "gate"},
                        "see gate", n_effective=len(verdicts)).line())
    out = a.out or config.PATHS.result("grooming_gate.json")
    write_json({**anchors.header(anchors.LUNA, stage="grooming_gate",
                                 unverified="a detected candidate set"),
                "inherited_digest": spine.digest(),
                "registration": "results/GROOMING_PREREGISTRATION.md",
                "deviation": "DEVIATIONS.md D19",
                "seed": SEED, "fps": fps, "headline": list(HEADLINE),
                "confirmation": _confirmation(),
                "reads": reads}, out)
    log(f"  wrote {out}")
    return 0


def _confirmation() -> dict:
    """§3's eye-confirmation rate, if the panel has been scored; else None."""
    p = os.path.join(work_dir(), "confirmation.json")
    if not os.path.exists(p):
        return {"scored": False,
                "note": ("§3's panel has not been confirmed by eye. §9.4 "
                         "forbids reading the spectral gate as an answer about "
                         "GROOMING until it is; the statistic still stands as a "
                         "statement about the detected windows, whatever they "
                         "contain")}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


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
