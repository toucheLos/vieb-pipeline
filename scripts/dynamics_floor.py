"""Stage 0: the noise colour, the descriptor floors, and the dynamics probe.

    python3 scripts/dynamics_floor.py [--limit N]

READ results/DYNAMICS_PREREGISTRATION.md FIRST.

Three phases, in the order the registration fixes:

1. **model** -- does a noise generator built from the frozen calibration
   reproduce the MEASURED colour? Mechanical, and it gates everything: a
   descriptor floor computed under the wrong colour is not a measurement.
2. **floor** -- the descriptors' null distribution under that noise, beside the
   corpus's own, so every number downstream has something to be read against.
3. **probe** -- order-controlled dynamics plants, scored at the registered +/-2
   band, with the localisation offset reported beside the recall.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits                # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import frames, log, write_json                        # noqa: E402
from vieb import seeds                                                # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.seg import breaks as bk, descriptors as ds                  # noqa: E402
from vieb.seg import dynplant as dp, noise as nz                      # noqa: E402
from vieb.tok import config                                           # noqa: E402

SEED = 0
SPLIT = "tune"
POSE_ARM = "raw"
GROUP = "shape"
STRIDE = 2
#: Windows drawn per animal for the descriptor floor.
N_WINDOWS = 40
#: Planted instances per (kind, level) cell.
N_PROBE = 240


def ego_path(tag: str) -> str:
    return os.path.join(config.REPO, "work", "ego",
                        f"{POSE_ARM}__bodylen__{tag}.npz")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or config.PATHS.result("dynamics_floor.json")

    fps = spine.fps()
    win = frames(dp.WIN_S, fps)
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    cal = nz.calibration()
    obj = {"dataset": "luna", "arm": "dynamics_floor", "split": SPLIT,
           "pose_arm": POSE_ARM, "group": GROUP, "win_s": dp.WIN_S,
           "alpha": dp.ALPHA}
    reads: dict = {}

    # ---- phase 1: the noise model ------------------------------------
    rng = np.random.default_rng(seeds.stable_seed(SEED, "model"))
    sample = nz.draw(rng, 60_000, 7, fps=fps, cal=cal)
    v = nz.validate(sample, fps=fps, cal=cal)
    mr = nz.model_read(v, scored_object={**obj, "arm": "noise_model"},
                       n_effective=1)
    reads["noise_model"] = mr.to_dict()
    log("  " + mr.line())
    if mr.verdict != "PASS":
        write_json({**anchors.header(anchors.LUNA, stage="dynamics_floor",
                                     unverified="an instrument probe on tune"),
                    "inherited_digest": spine.digest(),
                    "registration": "results/DYNAMICS_PREREGISTRATION.md",
                    "reads": reads, "noise_validation": v}, out)
        log(f"  wrote {out}")
        return 1

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
    log(f"  {len(tags)} {SPLIT} animals, win = {win} frames")

    # ---- phase 2: descriptor floors ----------------------------------
    null: dict[str, list[float]] = {}
    corp: dict[str, list[float]] = {}
    for n, tag in enumerate(tags, 1):
        with np.load(ego_path(tag), allow_pickle=False) as z:
            X = np.asarray(z["X"], dtype=np.float64)[:, idx]
            bounds = z["bounds"].astype(np.int64)
            ell = float(z["ell_a"])
        r = np.random.default_rng(seeds.stable_seed(SEED, f"floor|{tag}"))
        for _ in range(N_WINDOWS):
            lo = int(r.integers(0, max(1, X.shape[0] - 2 * win)))
            rec = ds.describe(X[lo:lo + 2 * win], fps=fps)
            for k, val in rec.items():
                corp.setdefault(k, []).append(float(val))
            nse = nz.draw(r, 2 * win, 7, fps=fps, cal=cal).reshape(2 * win,
                                                                   14) / ell
            rec = ds.describe(nse, fps=fps)
            for k, val in rec.items():
                null.setdefault(k, []).append(float(val))
        if n % 10 == 0 or n == len(tags):
            log(f"  floor {n}/{len(tags)}")
    fr = dp.floor_read(null, scored_object={**obj, "arm": "descriptor_floor"},
                       n_effective=len(tags))
    reads["descriptor_floor"] = fr.to_dict()
    log("  " + fr.line())

    bands = [f"band_{b[0]}" for b in ds.BANDS]
    sep: dict[str, dict] = {}
    for k in ["pole_radius", "pole_frequency_hz"] + bands:
        cv = np.asarray(corp.get(k, []), dtype=np.float64)
        nv = np.asarray(null.get(k, []), dtype=np.float64)
        cv, nv = cv[np.isfinite(cv)], nv[np.isfinite(nv)]
        if cv.size and nv.size:
            sep[k] = {"corpus_median": float(np.median(cv)),
                      "null_median": float(np.median(nv)),
                      "null_p95": float(np.quantile(nv, 0.95)),
                      "corpus_above_null_p95": float(
                          np.mean(cv > np.quantile(nv, 0.95)))}
    log("  descriptor      corpus med   null med   null p95   corpus>p95")
    for k, s in sep.items():
        log("  %-18s %9.4f %9.4f %9.4f %9.3f"
            % (k, s["corpus_median"], s["null_median"], s["null_p95"],
               s["corpus_above_null_p95"]))

    # ---- phase 3: the dynamics probe ---------------------------------
    ladders = {"frequency": dp.DFREQ_HZ, "damping": dp.RADII,
               "amplitude": (0.5, 2.0, 5.0)}
    amp_bl = 0.3
    slice_len = 2 * dp.SPAN * win
    probe: dict = {}
    for kind, levels in ladders.items():
        for lv in levels:
            rows: list[dict] = []
            offs: list[float] = []
            for tag in tags:
                with np.load(ego_path(tag), allow_pickle=False) as z:
                    X = np.asarray(z["X"], dtype=np.float64)[:, idx]
                r = np.random.default_rng(
                    seeds.stable_seed(SEED, f"probe|{kind}|{lv}|{tag}"))
                hits = tot = n_pk = n_fr = 0
                per = max(1, N_PROBE // max(len(tags), 1))
                for _ in range(per):
                    lo = int(r.integers(0, max(1, X.shape[0] - slice_len)))
                    base = X[lo:lo + slice_len].copy()
                    if base.shape[0] < slice_len:
                        continue
                    c0 = slice_len // 2
                    y = dp.plant_into(base, [c0], kind=kind, level=float(lv),
                                      fps=fps, win=win, amp_bl=amp_bl,
                                      idx=range(len(idx)), rng=r)
                    pk = dp.peaks_of(y, fps=fps, win=win, alpha=dp.ALPHA,
                                     stride=STRIDE)
                    tot += 1
                    n_pk += int(pk.size)
                    n_fr += slice_len
                    if pk.size:
                        d = pk - c0
                        j = int(np.argmin(np.abs(d)))
                        offs.append(float(d[j]))
                        hits += int(abs(int(d[j])) <= 2)
                if tot:
                    rows.append({"animal": tag, "recall": hits / tot,
                                 "chance": min(1.0, n_pk * 5.0 / max(n_fr, 1))})
            rd = dp.recovery_read(rows, kind=kind, level=float(lv),
                                  n_instances=sum(1 for _ in rows) * max(
                                      1, N_PROBE // max(len(tags), 1)),
                                  scored_object={**obj, "kind": kind},
                                  n_effective=len(tags), seed=SEED)
            o = np.asarray(offs, dtype=np.float64)
            probe[f"{kind}|{lv:g}"] = {
                "read": rd.to_dict(),
                "offset_median": float(np.median(o)) if o.size else float("nan"),
                "offset_p05": float(np.quantile(o, 0.05)) if o.size else np.nan,
                "offset_p95": float(np.quantile(o, 0.95)) if o.size else np.nan,
                "within_30_frames": float(np.mean(np.abs(o) <= 30))
                if o.size else float("nan"), "n_offsets": int(o.size)}
            log("  %-10s %-6g recall %.3f  chance %.3f  offset med %+6.1f  "
                "|off|<=30 %.3f" % (
                    kind, lv,
                    rd.detail.get("recall", {}).get("point", float("nan")),
                    rd.detail.get("chance", {}).get("point", float("nan")),
                    probe[f"{kind}|{lv:g}"]["offset_median"],
                    probe[f"{kind}|{lv:g}"]["within_30_frames"]))

    write_json({**anchors.header(anchors.LUNA, stage="dynamics_floor",
                                 unverified="an instrument probe on tune"),
                "inherited_digest": spine.digest(),
                "registration": "results/DYNAMICS_PREREGISTRATION.md",
                "split": SPLIT, "n_animals": len(tags), "seed": SEED,
                "fps": fps, "win_frames": win, "stride": STRIDE,
                "slice_len": slice_len, "amp_bl": amp_bl,
                "f_c_hz": float(cal["f_c_hz"]),
                "noise_validation": v, "descriptor_separation": sep,
                "probe": probe, "reads": reads}, out)
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
