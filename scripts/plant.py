"""At what order and magnitude does the detector recover a discontinuity?

    python3 scripts/plant.py [--limit N]

READ results/PLANT_PREREGISTRATION.md FIRST.

The first accuracy measurement in this programme. Everything else on disk is a
comparison against a null; this is a recall, against a chance level, on a target
whose order is controlled.

Two arms: the F3 `raw` corpus and `white` -- i.i.d. noise at each recording's
own mean and SD. The `white` arm discharges the negative control
`DETECTOR_PREREGISTRATION.md:84-90` has required since Step B and which has
never run, because no cell ever cleared the gate that would have triggered it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab, splits                      # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import frames, log, write_json                        # noqa: E402
from vieb import seeds                                                # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.seg import breaks as bk, floor as fl, plant as pl           # noqa: E402
from vieb.tok import config                                           # noqa: E402

SEED = 0
SPLIT = "tune"
POSE_ARM = "raw"
GROUP = "shape"
ARMS = ("corpus", "white")
#: Plant extent, in seconds. Longer than the detector's own 0.133 s half-window
#: so the onset is resolvable in principle rather than by luck.
PLANT_S = 0.25
PER_RECORDING = 10


def ego_path(arm: str, tag: str) -> str:
    name = f"{POSE_ARM}__bodylen__{tag}.npz"
    if arm == "corpus":
        return os.path.join(config.REPO, "work", "ego", name)
    return os.path.join(config.REPO, "work", "surrogate", arm, "ego", name)


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.PATHS.tok_dir, "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


def sigma_bar() -> float:
    """The calibration's confidence-weighted mean amplitude, body lengths."""
    with open(config.PATHS.result("noise_floor.json"), encoding="utf-8") as fh:
        t = json.load(fh)["calibration"]
    sg = np.asarray(t["sigma_bl"], dtype=np.float64)
    ct = np.asarray(t["counts"], dtype=np.float64)
    ok = np.isfinite(sg)
    return float(np.sum(sg[ok] * ct[ok]) / np.sum(ct[ok]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or config.PATHS.result("plant.json")

    fps = spine.fps()
    sd = basis_sd()
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    w = frames(PLANT_S, fps)
    guard = bk.guard_frames(fps, deriv_sec=bk.DERIV_SEC)
    sig = sigma_bar()
    log(f"  sigma_bar = {sig:.6f} body lengths, w = {w} frames, "
        f"ladder {pl.AMPS} x sigma")

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
    log(f"  {len(tags)} {SPLIT} animals")

    # per arm -> order -> amp -> list of {animal, recall, chance}
    got: dict = {arm: {o: {amp: [] for amp in pl.AMPS} for o in pl.ORDERS}
                 for arm in ARMS}
    counts: dict = {arm: {o: {amp: 0 for amp in pl.AMPS} for o in pl.ORDERS}
                    for arm in ARMS}

    for n, tag in enumerate(tags, 1):
        ab_p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
        for arm in ARMS:
            with np.load(ego_path(arm, tag), allow_pickle=False) as z:
                X = np.asarray(z["X"], dtype=np.float64)
                bounds = z["bounds"].astype(np.int64)
                valid = z["valid"].astype(bool)
            with np.load(ab_p, allow_pickle=False) as z:
                abstain = z["abstain"].astype(bool) | ~valid
            rng = np.random.default_rng(
                seeds.stable_seed(SEED, f"plant|{arm}|{tag}"))
            starts = pl.place(rng, bounds=bounds, abstain=abstain, w=w,
                              per_recording=PER_RECORDING, guard=guard)
            if starts.size == 0:
                continue
            for order in pl.ORDERS:
                for amp in pl.AMPS:
                    r2 = np.random.default_rng(
                        seeds.stable_seed(SEED, f"dir|{arm}|{tag}|{order}"))
                    Y = pl.apply_plant(X, starts, order=order,
                                       amp=amp * sig, w=w, idx=idx, rng=r2)
                    Ys = Y / sd[None, :]
                    hits = tot = n_pk = n_fr = any_hit = 0
                    for r in range(bounds.size - 1):
                        lo, hi = int(bounds[r]), int(bounds[r + 1])
                        mine = starts[(starts >= lo) & (starts < hi)]
                        if mine.size == 0:
                            continue
                        pk = fl.peaks_of(Ys[lo:hi], idx=idx, fps=fps,
                                         k_mad=bk.K_MAD,
                                         blocked=abstain[lo:hi])
                        h, t = pl.recovered(pk + lo, mine)
                        ha, _ = pl.recovered_anywhere(pk + lo, mine,
                                                      extent=pl.INSTANCE_SPANS * w)
                        hits += h
                        any_hit += ha
                        tot += t
                        n_pk += int(pk.size)
                        n_fr += hi - lo
                    if tot == 0:
                        continue
                    got[arm][order][amp].append(
                        {"animal": tag, "recall": hits / tot,
                         "recall_anywhere": any_hit / tot,
                         "chance": pl.chance_of(n_pk, n_fr)})
                    counts[arm][order][amp] += tot
        if n % 10 == 0 or n == len(tags):
            log(f"  {n}/{len(tags)} animals")

    obj = {"dataset": "luna", "arm": "plant", "split": SPLIT,
           "pose_arm": POSE_ARM, "group": GROUP, "k_mad": bk.K_MAD,
           "deriv_sec": bk.DERIV_SEC, "degree": bk.DEGREE}
    reads: dict = {}
    recall50: dict = {arm: {o: {} for o in pl.ORDERS} for arm in ARMS}
    anywhere: dict = {arm: {o: {} for o in pl.ORDERS} for arm in ARMS}
    for arm in ARMS:
        for order in pl.ORDERS:
            for amp in pl.AMPS:
                rows = got[arm][order][amp]
                rd = pl.recovery_read(
                    rows, order=order, amp=amp,
                    n_instances=counts[arm][order][amp],
                    scored_object={**obj, "null": arm}, n_effective=len(tags),
                    seed=SEED)
                reads[f"{arm}|o{order}|a{amp:g}"] = rd.to_dict()
                if rd.detail.get("recall"):
                    recall50[arm][order][amp] = float(
                        rd.detail["recall"]["point"])
                    anywhere[arm][order][amp] = float(np.mean(
                        [r["recall_anywhere"] for r in rows]))
        log(f"  --- {arm}: recall AT ONSET (+/-2) | ANYWHERE in extent ---")
        for order in pl.ORDERS:
            log("  order %d: " % order + "  ".join(
                f"{amp:g}s={recall50[arm][order].get(amp, float('nan')):.3f}"
                f"|{anywhere[arm][order].get(amp, float('nan')):.3f}"
                for amp in pl.AMPS))

    mono = pl.monotone_read(recall50["corpus"], scored_object=obj,
                            n_effective=len(tags))
    reads["monotone"] = mono.to_dict()
    log("  " + mono.line())

    orr = pl.ordering_read(recall50["corpus"], scored_object=obj,
                           n_effective=len(tags))
    reads["ordering"] = orr.to_dict()
    log("  " + orr.line())

    above = [a_ for a_ in pl.AMPS if a_ > 4.0]
    c_flat = {f"o{o}|a{a_:g}": recall50["corpus"][o][a_]
              for o in pl.ORDERS for a_ in above if a_ in recall50["corpus"][o]}
    w_flat = {f"o{o}|a{a_:g}": recall50["white"][o][a_]
              for o in pl.ORDERS for a_ in above if a_ in recall50["white"][o]}
    nc = pl.negative_control_read(c_flat, w_flat,
                                  scored_object={**obj, "null": "white"},
                                  n_effective=len(tags))
    reads["negative_control"] = nc.to_dict()
    log("  " + nc.line())

    write_json({**provenance.header(anchors.LUNA, stage="plant",
                                 unverified="an instrument probe on tune"),
                "inherited_digest": spine.digest(),
                "registration": "results/PLANT_PREREGISTRATION.md",
                "split": SPLIT, "n_animals": len(tags), "seed": SEED,
                "group": GROUP, "k_mad": bk.K_MAD, "deriv_sec": bk.DERIV_SEC,
                "degree": bk.DEGREE, "plant_s": PLANT_S, "w_frames": w,
                "per_recording": PER_RECORDING, "tol": pl.TOL,
                "sigma_bar_bl": sig, "amps_sigma": list(pl.AMPS),
                "orders": list(pl.ORDERS), "arms": list(ARMS),
                "n_instances": {arm: {str(o): {str(k): v for k, v in
                                               counts[arm][o].items()}
                                      for o in pl.ORDERS} for arm in ARMS},
                "recall": {arm: {str(o): {str(k): v for k, v in
                                          recall50[arm][o].items()}
                                 for o in pl.ORDERS} for arm in ARMS},
                "recall_anywhere": {arm: {str(o): {str(k): v for k, v in
                                                   anywhere[arm][o].items()}
                                          for o in pl.ORDERS}
                                    for arm in ARMS},
                "reads": reads}, out)
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
