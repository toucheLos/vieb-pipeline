"""Stage 2: do the boundaries survive halving the frame rate?

    python3 scripts/subsample.py [--limit N]

READ results/VALIDATION_PREREGISTRATION.md §2 FIRST.

The cheapest available test that a boundary is in the animal rather than in the
sampling grid, and **the first frame-rate sweep this programme has run**.
Everything temporal derives from `recur.util.frames(seconds, fps)` at use time,
so passing 15 fps re-derives every window in SECONDS and changes nothing else.

A boundary found at 15 fps sits at original frame `2i`. Survival is matched
one-to-one at each stream's own registered band -- +/-2 configuration, +/-15
dynamics -- because Stage 0 measured them to localise differently.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits                # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import frames, log, write_json                        # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.seg import annot as an, breaks as bk, dynplant as dp        # noqa: E402
from vieb.seg import twostream as ts                                  # noqa: E402
from vieb.tok import config                                           # noqa: E402

SEED = 0
SPLIT = "tune"
POSE_ARM = "raw"
GROUP = "shape"
FACTOR = 2
STRIDE = 2
SURVIVAL_MIN = 0.5


def ego_path(tag: str) -> str:
    return os.path.join(config.REPO, "work", "ego",
                        f"{POSE_ARM}__bodylen__{tag}.npz")


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.PATHS.tok_dir, "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or config.PATHS.result("subsample.json")

    fps = spine.fps()
    half = fps / FACTOR
    sd = basis_sd()
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    w_full, w_half = frames(dp.WIN_S, fps), frames(dp.WIN_S, half)
    log(f"  {fps:g} -> {half:g} fps; dynamics window {w_full} -> {w_half} "
        f"frames (both {dp.WIN_S:g} s)")

    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))
    by_animal: dict[str, list[str]] = {}
    for rid in spine.recording_ids():
        tag = lab.animal_tag(rid)
        if split_of.get(tag) == SPLIT:
            by_animal.setdefault(tag, []).append(rid)
    tags = sorted(by_animal)
    if a.limit:
        tags = tags[:a.limit]

    rows: list[dict] = []
    for n, tag in enumerate(tags, 1):
        with np.load(ego_path(tag), allow_pickle=False) as z:
            X = np.asarray(z["X"], dtype=np.float64) / sd[None, :]
            bounds = z["bounds"].astype(np.int64)
            valid = z["valid"].astype(bool)
        ab_p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
        with np.load(ab_p, allow_pickle=False) as z:
            abstain = z["abstain"].astype(bool) | ~valid
        got = {k: {"full": [], "half": []} for k in ("config", "dyn")}
        secs = {"full": 0.0, "half": 0.0}
        for r in range(bounds.size - 1):
            lo, hi = int(bounds[r]), int(bounds[r + 1])
            if hi - lo < 8 * w_full:
                continue
            blk = abstain[lo:hi]
            sub, sblk = X[lo:hi][::FACTOR], blk[::FACTOR]
            got["config"]["full"].append(
                ts.config_peaks(X[lo:hi], idx=idx, fps=fps, k_mad=bk.K_MAD,
                                blocked=blk) + lo)
            got["config"]["half"].append(
                ts.config_peaks(sub, idx=idx, fps=half, k_mad=bk.K_MAD,
                                blocked=sblk) * FACTOR + lo)
            got["dyn"]["full"].append(
                ts.dyn_peaks(X[lo:hi][:, idx], fps=fps, win=w_full,
                             blocked=blk, stride=STRIDE) + lo)
            got["dyn"]["half"].append(
                ts.dyn_peaks(sub[:, idx], fps=half, win=w_half, blocked=sblk,
                             stride=1) * FACTOR + lo)
            h = max(2, int(round(bk.DERIV_SEC * fps)))
            secs["full"] += float((~blk[h:hi - lo - h]).sum()) / fps
            secs["half"] += float((~sblk).sum()) / half
        rec: dict = {"animal": tag}
        for k, tol in (("config", ts.CONFIG_TOL), ("dyn", ts.DYN_TOL)):
            f = np.concatenate(got[k]["full"]) if got[k]["full"] \
                else np.zeros(0, np.int64)
            hv = np.concatenate(got[k]["half"]) if got[k]["half"] \
                else np.zeros(0, np.int64)
            m = an.match(np.sort(f).tolist(), np.sort(hv).tolist(), tol)
            rec[f"{k}_full"] = int(f.size)
            rec[f"{k}_half"] = int(hv.size)
            rec[f"{k}_survival"] = (len(m) / f.size) if f.size else float("nan")
            rec[f"{k}_rate_full"] = f.size / max(secs["full"], 1e-9)
            rec[f"{k}_rate_half"] = hv.size / max(secs["half"], 1e-9)
        rows.append(rec)
        if n % 10 == 0 or n == len(tags):
            log(f"  {n}/{len(tags)}")

    obj = {"dataset": "luna", "arm": "subsample", "split": SPLIT,
           "pose_arm": POSE_ARM, "group": GROUP, "factor": FACTOR}
    tagl = [r["animal"] for r in rows]
    reads: dict = {}
    summ: dict = {}
    for k in ("config", "dyn"):
        s = [r[f"{k}_survival"] for r in rows]
        ok = [i for i, v in enumerate(s) if np.isfinite(v)]
        ci = boot.animal_interval([s[i] for i in ok], [tagl[i] for i in ok],
                                  how="mean", n_boot=ts.N_BOOT, seed=SEED)
        rf = boot.animal_interval([r[f"{k}_rate_full"] for r in rows], tagl,
                                  how="mean", n_boot=ts.N_BOOT, seed=SEED)
        rh = boot.animal_interval([r[f"{k}_rate_half"] for r in rows], tagl,
                                  how="mean", n_boot=ts.N_BOOT, seed=SEED)
        summ[k] = {"survival": dict(ci), "rate_full": dict(rf),
                   "rate_half": dict(rh),
                   "tol": ts.CONFIG_TOL if k == "config" else ts.DYN_TOL}
        log(f"  {k:<7} survival {ci['point']:.4f} [{ci['lo']:.4f}, "
            f"{ci['hi']:.4f}]  rate {rf['point']:.4f} -> {rh['point']:.4f}/s")

    c = summ["config"]["survival"]
    rd = Read("PASS" if float(c["lo"]) > SURVIVAL_MIN else "FAIL", obj,
              (f"{'the frozen detector survives' if float(c['lo']) > SURVIVAL_MIN else 'THE FROZEN DETECTOR DOES NOT SURVIVE'} "
               f"halving the frame rate: {c['point']:.4f} [{c['lo']:.4f}, "
               f"{c['hi']:.4f}] of its boundaries are re-found at 15 fps "
               f"within ±{ts.CONFIG_TOL}, against a registered floor of "
               f"{SURVIVAL_MIN}. The corpus's power is concentrated below 3 Hz, "
               f"well inside the new 7.5 Hz Nyquist"),
              n_effective=len(rows), detail=summ["config"])
    reads["config_survival"] = rd.to_dict()
    log("  " + rd.line())

    d = summ["dyn"]["survival"]
    lower = float(d["hi"]) < float(c["lo"])
    rd2 = Read("PASS" if lower else "FAIL", {**obj, "arm": "dyn_survival"},
               (f"the dynamics stream survives {'LESS' if lower else 'no less'} "
                f"than the configuration stream: {d['point']:.4f} "
                f"[{d['lo']:.4f}, {d['hi']:.4f}] against {c['point']:.4f} "
                f"[{c['lo']:.4f}, {c['hi']:.4f}], and it is matched at the "
                f"wider ±{ts.DYN_TOL} band. At a jitter share of 0.9665 that "
                f"is what a boundary set made mostly of noise should do"),
               n_effective=len(rows), detail=summ["dyn"])
    reads["dyn_survival"] = rd2.to_dict()
    log("  " + rd2.line())

    rose = {k: float(summ[k]["rate_half"]["point"])
            > float(summ[k]["rate_full"]["hi"]) for k in summ}
    rd3 = Read("PASS" if not any(rose.values()) else "FAIL",
               {**obj, "arm": "rate_stability"},
               (f"boundary rate at 15 fps against 30: config "
                f"{summ['config']['rate_full']['point']:.4f} -> "
                f"{summ['config']['rate_half']['point']:.4f}/s, dyn "
                f"{summ['dyn']['rate_full']['point']:.4f} -> "
                f"{summ['dyn']['rate_half']['point']:.4f}/s. "
                f"{'Neither rose' if not any(rose.values()) else 'A RATE ROSE'} "
                f"when half the data was discarded"),
               n_effective=len(rows), detail={"rose": rose})
    reads["rate_stability"] = rd3.to_dict()
    log("  " + rd3.line())

    write_json({**provenance.header(anchors.LUNA, stage="subsample",
                                 unverified="an instrument probe on tune"),
                "inherited_digest": spine.digest(),
                "registration": "results/VALIDATION_PREREGISTRATION.md",
                "split": SPLIT, "n_animals": len(rows), "seed": SEED,
                "fps_full": fps, "fps_half": half, "factor": FACTOR,
                "survival_floor": SURVIVAL_MIN,
                "summary": summ, "reads": reads, "rows": rows}, out)
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
