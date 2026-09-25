"""Post-hoc diagnostic: is the grooming detector firing on KEYPOINT JITTER?

    python3 scripts/grooming_jitter.py

NOT_A_RESULT by construction. It scores no hypothesis registered in
`GROOMING_PREREGISTRATION.md`; it asks what the detector was firing on, after
`work/grooming/confirmation.json` recorded that **0 of 30 candidates are
grooming**. Category as `DEVIATIONS.md` D10: a post-hoc diagnostic, labelled.

## The claim under test, and why it is testable rather than merely plausible

The rater's reading was that DLC keypoints move in micro-increments and that
this alone can produce a "grooming" detection. That is a statement about
tracking noise, and tracking noise has a motion-independent measurement in this
corpus already: **the skull triangle is rigid.** `left_ear-right_ear`,
`left_ear-nose` and `right_ear-nose` cannot change length no matter what the
animal does, so within-window variance in those three lengths is jitter and
nothing else -- the same logic `vieb/seg/jitter.py` uses to calibrate the
keypoint noise floor.

So the mechanism predicts something specific and falsifiable: **candidate
windows should carry MORE skull jitter than their speed-matched partners**, and
head-region energy should track jitter even with speed held fixed.

## Why the prediction is not circular

Selection used keypoint SPEED and head-region PIXEL energy. It never saw a bone
length. The control is matched on speed, so the comparison holds fixed the one
selection variable that jitter could otherwise be confounded with, and skull
jitter is free to come out equal.
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
from recur.util import log, write_json                               # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.qc import bones as bn                                      # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

SEED = 0
N_BOOT = 2000


def _gg():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "grooming_gate.py")
    spec = importlib.util.spec_from_file_location("gg_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["grooming_gate"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def skull_jitter(rid: str, cache: dict) -> np.ndarray:
    """``(T,)`` per-frame |skull bone length - its median|, in body lengths.

    Averaged over the three skull bones, NaN where either endpoint is missing
    or interpolated. The reference is this recording's own median, so a
    per-recording tracking offset cannot masquerade as noise.
    """
    if rid in cache:
        return cache[rid]
    d = spine.clean(rid)
    p = np.asarray(d["pose"], dtype=np.float64)
    bad = (np.asarray(d["missing"], dtype=bool)
           | np.asarray(d["interpolated"], dtype=bool))
    ell = float(np.nanmedian(tego.body_length(p)))
    lens = bn.bone_lengths(p, bn.SKULL) / ell
    dev = np.full(lens.shape, np.nan)
    for m, (i, j) in enumerate(bn.SKULL):
        ok = ~bad[:, i] & ~bad[:, j] & np.isfinite(lens[:, m])
        if int(ok.sum()) < 2:
            continue
        dev[ok, m] = np.abs(lens[ok, m] - float(np.median(lens[ok, m])))
    out = np.nanmean(dev, axis=1)
    if len(cache) > 40:
        cache.pop(next(iter(cache)))
    cache[rid] = out
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    gg = _gg()
    fps = spine.fps()
    rad, win_s, pct = gg.HEADLINE
    got = gg._collect(rad, win_s, pct, fps, "energy_ego")
    rows, cand, partners = got["rows"], got["cand"], got["partners"]
    t_idx = np.flatnonzero(cand)
    ok = partners >= 0
    log(f"  {int(cand.sum())} candidates, {int(ok.sum())} matched pairs")

    cache: dict = {}
    jit = np.full(len(rows), np.nan)
    for n, r in enumerate(rows):
        j = skull_jitter(r["rid"], cache)
        seg = j[int(r["a"]):int(r["b"])]
        if seg.size and np.isfinite(seg).any():
            # SD within the window, not the mean: a constant offset is a
            # calibration error, the FLUCTUATION is the jitter.
            jit[n] = float(np.nanstd(seg))
        if (n + 1) % 5000 == 0:
            log(f"    {n + 1}/{len(rows)} windows")

    obj = {"dataset": "luna", "arm": "grooming_jitter", "split": "fit",
           "signal": "energy_ego", "radius_bl": rad, "win_s": win_s,
           "still_pct": pct}
    reads: dict = {}

    ti = [int(i) for i in t_idx[ok] if np.isfinite(jit[int(i)])]
    ci = [int(i) for i in partners[ok] if np.isfinite(jit[int(i)])]
    who_t = [rows[i]["animal"] for i in ti]
    who_c = [rows[i]["animal"] for i in ci]
    bt = boot.animal_interval([jit[i] for i in ti], who_t, how="mean",
                              n_boot=N_BOOT, seed=SEED)
    bc = boot.animal_interval([jit[i] for i in ci], who_c, how="mean",
                              n_boot=N_BOOT, seed=SEED)
    higher = float(bt["lo"]) > float(bc["hi"])
    ratio = float(bt["point"]) / float(bc["point"]) if bc["point"] else np.nan
    rd = Read("NOT_A_RESULT", {**obj, "arm": "skull_jitter"},
              (f"DIAGNOSTIC: candidate windows carry "
               f"{'MORE' if higher else 'no more'} skull jitter than their "
               f"speed-matched partners -- {bt['point']:.5f} "
               f"[{bt['lo']:.5f}, {bt['hi']:.5f}] against {bc['point']:.5f} "
               f"[{bc['lo']:.5f}, {bc['hi']:.5f}] body lengths, "
               f"{ratio:.2f}x, {'non-overlapping' if higher else 'overlapping'}, "
               f"over {len(set(who_t) | set(who_c))} animals and {len(ti)} pairs. "
               f"The skull is rigid, so this is tracking noise and not motion. "
               f"Scores no registered hypothesis"),
              n_effective=len(set(who_t) | set(who_c)),
              detail={"candidate": dict(bt), "control": dict(bc),
                      "ratio": ratio, "n_pairs": len(ti),
                      "candidates_carry_more_jitter": higher})
    reads["skull_jitter"] = rd.to_dict()
    log("  " + rd.line())

    # Does jitter predict head energy with SPEED HELD FIXED? Partial Spearman
    # by residualising both on speed rank -- the selection used speed, so a raw
    # correlation would be reporting the selection back.
    from scipy import stats as st
    m = np.isfinite(jit) & np.asarray(
        [np.isfinite(r["energy"]) and np.isfinite(r["speed"]) for r in rows])
    jr = st.rankdata(jit[m])
    er = st.rankdata([rows[i]["energy"] for i in np.flatnonzero(m)])
    sr = st.rankdata([rows[i]["speed"] for i in np.flatnonzero(m)])
    rj = jr - np.polyval(np.polyfit(sr, jr, 1), sr)
    re = er - np.polyval(np.polyfit(sr, er, 1), sr)
    partial = float(np.corrcoef(rj, re)[0, 1])
    raw = float(st.spearmanr(jit[m], [rows[i]["energy"]
                                      for i in np.flatnonzero(m)]).statistic)
    pr = Read("NOT_A_RESULT", {**obj, "arm": "jitter_predicts_energy"},
              (f"DIAGNOSTIC: skull jitter and head-region energy correlate "
               f"{raw:+.3f} (Spearman) over {int(m.sum()):,} windows, and "
               f"{partial:+.3f} with SPEED HELD FIXED by rank residualisation. "
               f"Speed is a selection variable, so the partial figure is the "
               f"one that is not reporting the selection back. Scores no "
               f"registered hypothesis"),
              n_effective=int(m.sum()),
              detail={"spearman": raw, "partial_given_speed": partial,
                      "n_windows": int(m.sum())})
    reads["jitter_predicts_energy"] = pr.to_dict()
    log("  " + pr.line())

    out = a.out or config.PATHS.result("grooming_jitter.json")
    write_json({**provenance.header(anchors.LUNA, stage="grooming_jitter",
                                 unverified="a post-hoc instrument diagnostic"),
                "inherited_digest": spine.digest(),
                "diagnostic": "NOT a result; see DEVIATIONS.md D10",
                "prompted_by": ("0 of 30 candidates confirmed as grooming; the "
                                "rater's reading was that DLC micro-jitter "
                                "alone can produce the detection"),
                "reads": reads}, out)
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
