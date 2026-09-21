"""Do the detector's boundaries survive a change of cleaning arm?

    python3 scripts/arm_concordance.py --shard I --of N     # one slice
    python3 scripts/arm_concordance.py --combine

READ results/NOISEFLOOR_PREREGISTRATION.md §6 FIRST.

## The prediction that needs no floor

`viterbi` reassigns **0.31%** of keypoint-frames (`CLEANING.md`). If changing
three keypoint-frames in a thousand moves most of the boundaries, the boundaries
are sitting on noise -- and that inference needs no noise floor, no human and no
surrogate. Registered prediction 4: concordance at ±2 between `raw` and
`viterbi` is **below 0.5**.

## Which arms, and which are excluded by name

`raw`, `viterbi`, `disposition` -- the three `F3_PREPROCESSING_FREEZE.md`
carries. **`wiener` and `butterworth` are excluded**: a low-pass filter
manufactures exactly the smoothness whose breaks this detects. They are also
`STORED_ARMS`, read off disk and not applicable to an array, so they could not
be run here in any case.

## The matcher

`annot.match` / `annot.prf` -- greedy, nearest-first, one-to-one. Written and
tested for rater-rater comparison and exactly right here.
`scripts/breaks.py:165 _agreement` is NOT used: it counts unmatched hits, which
inflates agreement precisely where a detector is noisiest.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits                # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import log, write_json                                # noqa: E402
from vieb.clean import arms as clean_arms                             # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.seg import annot as an, breaks as bk, floor as fl           # noqa: E402
from vieb.tok import config, ego                                      # noqa: E402

SEED = 0
SPLIT = "tune"
POSE_ARM = "raw"
GROUP = "shape"
#: The three F3-carried arms. Wiener and butterworth are excluded by name.
ARMS: tuple[str, ...] = ("raw", "viterbi", "disposition")
#: Registered prediction 4: raw-vs-viterbi at ±2 below this means the
#: boundaries move with the cleaner.
PRED4_LIMIT = 0.5


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "arm_concordance")


def ego_path(tag: str) -> str:
    return os.path.join(config.REPO, "work", "ego",
                        f"{POSE_ARM}__bodylen__{tag}.npz")


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.PATHS.tok_dir, "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


def _injection():
    """`disposition_arm` from `scripts/injection.py`, imported not copied.

    A second implementation that drifted by one constant would benchmark an
    arm that was never published.
    """
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "injection.py")
    spec = importlib.util.spec_from_file_location("inj_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["injection"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def tune_animals() -> dict[str, list[str]]:
    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))
    out: dict[str, list[str]] = {}
    for rid in spine.recording_ids():
        tag = lab.animal_tag(rid)
        if split_of.get(tag) == SPLIT:
            out.setdefault(tag, []).append(rid)
    return out


def shard(a) -> int:
    os.makedirs(work_dir(), exist_ok=True)
    fps = spine.fps()
    sd = basis_sd()
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    inj = _injection()
    by_animal = tune_animals()
    tags = sorted(by_animal)[a.shard::a.of]
    log(f"  shard {a.shard}/{a.of}: {len(tags)} animals")

    rows: list[dict] = []
    for n, tag in enumerate(tags, 1):
        with np.load(ego_path(tag), allow_pickle=False) as z:
            ell = float(z["ell_a"])
            bounds = z["bounds"].astype(np.int64)
            valid = z["valid"].astype(bool)
        ab_p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
        with np.load(ab_p, allow_pickle=False) as z:
            abstain = z["abstain"].astype(bool) | ~valid

        peaks: dict[str, list[np.ndarray]] = {k: [] for k in ARMS}
        for r, rid in enumerate(by_animal[tag]):
            lo, hi = int(bounds[r]), int(bounds[r + 1])
            d = spine.clean(rid)
            held = clean_arms.held_array(d["pose_unfiltered"].astype(float),
                                         d["missing"].astype(bool))
            conf = np.asarray(d["conf"], dtype=np.float64)
            usable = spine.representation(rid)["usable"].astype(bool)
            for arm in ARMS:
                if arm == "raw":
                    pose = held
                elif arm == "viterbi":
                    pose = clean_arms.apply("viterbi", held, conf, fps)
                else:
                    pose = inj.disposition_arm(held, conf)
                X, _v = ego.transform(pose, ell, fps, usable=usable)
                pk = fl.peaks_of(X / sd[None, :], idx=idx, fps=fps,
                                 k_mad=bk.K_MAD, blocked=abstain[lo:hi])
                peaks[arm].append(np.asarray(pk, dtype=np.int64) + lo)

        cat = {k: (np.concatenate(v) if v else np.zeros(0, np.int64))
               for k, v in peaks.items()}
        for i, one in enumerate(ARMS):
            for two in ARMS[i + 1:]:
                for tol in an.TOLERANCES:
                    got = an.prf(cat[one].tolist(), cat[two].tolist(), tol)
                    rows.append({"animal": tag, "a": one, "b": two,
                                 "tol": int(tol), "n_a": int(cat[one].size),
                                 "n_b": int(cat[two].size), **got})
        # How many of `raw`'s boundaries survive BOTH other arms at +/-2.
        surv = 0
        for m in an.match(cat["raw"].tolist(), cat["viterbi"].tolist(), 2):
            ok = any(x == m[0] for x, _ in
                     an.match(cat["raw"].tolist(),
                              cat["disposition"].tolist(), 2))
            surv += int(ok)
        rows.append({"animal": tag, "a": "raw", "b": "both_others",
                     "tol": 2, "n_a": int(cat["raw"].size),
                     "n_b": surv, "survival": (surv / cat["raw"].size)
                     if cat["raw"].size else float("nan")})
        log(f"  {n}/{len(tags)} {tag}: raw {cat['raw'].size} peaks, "
            f"viterbi {cat['viterbi'].size}, disposition "
            f"{cat['disposition'].size}, {surv} survive both")

    out = os.path.join(work_dir(), f"shard_{a.shard:03d}.json")
    write_json({"shard": a.shard, "of": a.of, "n_animals": len(tags),
                "rows": rows}, out)
    log(f"  wrote {out}")
    return 0


def combine(a) -> int:
    paths = sorted(glob.glob(os.path.join(work_dir(), "shard_*.json")))
    if not paths:
        raise SystemExit("no shards; run --shard first")
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows.extend(json.load(fh)["rows"])
    tags = sorted({r["animal"] for r in rows})
    obj = {"dataset": "luna", "arm": "arm_concordance", "split": SPLIT,
           "pose_arm": POSE_ARM, "group": GROUP, "k_mad": bk.K_MAD,
           "deriv_sec": bk.DERIV_SEC, "degree": bk.DEGREE}
    reads: dict = {}
    table: dict = {}
    for one, two in (("raw", "viterbi"), ("raw", "disposition"),
                     ("viterbi", "disposition")):
        for tol in an.TOLERANCES:
            sel = [r for r in rows if r["a"] == one and r["b"] == two
                   and r["tol"] == tol]
            if not sel:
                continue
            ci = boot.animal_interval([float(r["f1"]) for r in sel],
                                      [str(r["animal"]) for r in sel],
                                      how="mean", n_boot=fl.N_BOOT, seed=SEED)
            table[f"{one}|{two}|{tol}"] = dict(ci)

    key = table.get("raw|viterbi|2")
    surv = [r for r in rows if r["b"] == "both_others"]
    sci = boot.animal_interval([float(r["survival"]) for r in surv],
                               [str(r["animal"]) for r in surv], how="mean",
                               n_boot=fl.N_BOOT, seed=SEED) if surv else None
    if key is None:
        rd = Read("NOT_A_RESULT", obj, "no raw-vs-viterbi rows at ±2",
                  n_effective=len(tags))
    elif float(key["hi"]) < PRED4_LIMIT:
        rd = Read(
            "PASS", obj,
            (f"THE BOUNDARIES MOVE WITH THE CLEANER: raw-vs-viterbi "
             f"concordance at ±2 is {key['point']:.4f} [{key['lo']:.4f}, "
             f"{key['hi']:.4f}], below the registered {PRED4_LIMIT} limit, "
             f"even though viterbi reassigns 0.31% of keypoint-frames. "
             f"Changing three keypoint-frames in a thousand moves most of the "
             f"boundaries, which needs no floor and no surrogate to read"),
            n_effective=len(tags), detail={"f1": dict(key), "survival": sci})
    elif float(key["lo"]) > 0.9:
        rd = Read(
            "FAIL", obj,
            (f"the boundaries are ROBUST to the cleaner: raw-vs-viterbi "
             f"concordance at ±2 is {key['point']:.4f} [{key['lo']:.4f}, "
             f"{key['hi']:.4f}], above 0.9. Registered prediction 4 is "
             f"refuted and the noise reading is substantially weakened"),
            n_effective=len(tags), detail={"f1": dict(key), "survival": sci})
    else:
        rd = Read(
            "INCONCLUSIVE", obj,
            (f"raw-vs-viterbi concordance at ±2 is {key['point']:.4f} "
             f"[{key['lo']:.4f}, {key['hi']:.4f}], between the registered "
             f"{PRED4_LIMIT} limit and the 0.9 robustness bar. The boundaries "
             f"are neither clearly noise nor clearly robust to the cleaner"),
            n_effective=len(tags), detail={"f1": dict(key), "survival": sci})
    reads["prediction4"] = rd.to_dict()
    log("  " + rd.line())

    out = a.out or config.PATHS.result("arm_concordance.json")
    write_json({**anchors.header(anchors.LUNA, stage="arm_concordance",
                                 unverified="an instrument probe on tune"),
                "inherited_digest": spine.digest(),
                "registration": "results/NOISEFLOOR_PREREGISTRATION.md",
                "split": SPLIT, "arms": list(ARMS), "n_animals": len(tags),
                "excluded_arms": ["wiener", "butterworth"],
                "tolerances": list(an.TOLERANCES),
                "concordance": table,
                "raw_survives_both": dict(sci) if sci else None,
                "reads": reads, "rows": rows}, out)
    log(f"  wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=12)
    p.add_argument("--combine", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.combine:
        return combine(a)
    if a.shard is None:
        raise SystemExit("pass --shard I --of N, or --combine")
    return shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
