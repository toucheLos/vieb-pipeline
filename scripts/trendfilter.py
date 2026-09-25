"""A detector with an explicit error model, scored on the same plant.

    python3 scripts/trendfilter.py --phase sweep --shard I --of N
    python3 scripts/trendfilter.py --phase sweep --combine
    python3 scripts/trendfilter.py --phase plant --shard I --of N
    python3 scripts/trendfilter.py --phase plant --combine

READ results/TRENDFILTER_PREREGISTRATION.md FIRST.

Phase `sweep` walks the alpha path per recording with warm starts, on the
corpus and on a jitter-only arm, and picks lambda* as the smallest alpha whose
jitter share clears the registered 5% bar. Phase `plant` re-runs
`scripts/plant.py`'s harness at lambda*, unchanged, so the two detectors are
compared on one instrument rather than each on its own probe.

**Convergence is carried through to the verdict.** A recovery number computed
from a solver that did not converge is not a measurement, so every cell records
its convergence rate and `lambda_read` refuses a lambda* that lands in a
region the solver could not reach.
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

from recur import anchors, boot, labels as lab, splits                # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import frames, log, write_json                        # noqa: E402
from vieb import seeds                                                # noqa: E402
from vieb.clean import arms as clean_arms                             # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.seg import breaks as bk, floor as fl, jitter as jit         # noqa: E402
from vieb.seg import plant as pl, trendfilter as tf                   # noqa: E402
from vieb.tok import config, ego                                      # noqa: E402

SEED = 0
SPLIT = "tune"
POSE_ARM = "raw"
GROUP = "shape"
PLANT_S = 0.25
PER_RECORDING = 10


def work_dir(phase: str) -> str:
    return os.path.join(config.REPO, "work", f"trendfilter_{phase}")


def ego_path(arm: str, tag: str) -> str:
    name = f"{POSE_ARM}__bodylen__{tag}.npz"
    if arm == "corpus":
        return os.path.join(config.REPO, "work", "ego", name)
    return os.path.join(config.REPO, "work", "surrogate", arm, "ego", name)


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.PATHS.tok_dir, "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


def calibration() -> dict:
    with open(config.PATHS.result("noise_floor.json"), encoding="utf-8") as fh:
        return dict(json.load(fh)["calibration"])


def sigma_bar(t: dict) -> float:
    sg = np.asarray(t["sigma_bl"], dtype=np.float64)
    ct = np.asarray(t["counts"], dtype=np.float64)
    ok = np.isfinite(sg)
    return float(np.sum(sg[ok] * ct[ok]) / np.sum(ct[ok]))


def tune_animals() -> dict[str, list[str]]:
    split_of = splits.split_of_animal(
        splits.load(spine.sf("results/splits.json")))
    out: dict[str, list[str]] = {}
    for rid in spine.recording_ids():
        tag = lab.animal_tag(rid)
        if split_of.get(tag) == SPLIT:
            out.setdefault(tag, []).append(rid)
    return out


def weights_for(tag: str, rids: list[str], table: dict,
                n: int) -> np.ndarray:
    """`w_t = 1/sigma^2(c_t)`, normalised to mean 1 so `alpha` keeps meaning.

    The error model. A low-confidence frame pulls the fit less hard, which is
    the principled form of "a low-confidence boundary is reported as
    uncertain".
    """
    w = np.ones(n, dtype=np.float64)
    off = 0
    for rid in rids:
        d = spine.clean(rid)
        c = np.asarray(d["conf"], dtype=np.float64).min(axis=1)
        s = jit.sigma_of(c, table)
        k = c.size
        w[off:off + k] = 1.0 / np.maximum(s, 1e-6) ** 2
        off += k
    return w / float(np.mean(w))


def sweep(a) -> int:
    os.makedirs(work_dir("sweep"), exist_ok=True)
    fps = spine.fps()
    sd = basis_sd()
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    table = calibration()
    by_animal = tune_animals()
    tags = sorted(by_animal)[a.shard::a.of]
    log(f"  sweep shard {a.shard}/{a.of}: {len(tags)} animals")

    rows: list[dict] = []
    for n, tag in enumerate(tags, 1):
        with np.load(ego_path("corpus", tag), allow_pickle=False) as z:
            X = np.asarray(z["X"], dtype=np.float64)
            bounds = z["bounds"].astype(np.int64)
            valid = z["valid"].astype(bool)
            ell = float(z["ell_a"])
        ab_p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
        with np.load(ab_p, allow_pickle=False) as z:
            abstain = z["abstain"].astype(bool) | ~valid
        w = weights_for(tag, by_animal[tag], table, X.shape[0])

        # the jitter-only arm: a constant pose carrying only measured noise
        rng = np.random.default_rng(seeds.stable_seed(SEED, f"tf|{tag}"))
        parts = []
        for r, rid in enumerate(by_animal[tag]):
            lo, hi = int(bounds[r]), int(bounds[r + 1])
            d = spine.clean(rid)
            held = clean_arms.held_array(d["pose_unfiltered"].astype(float),
                                         d["missing"].astype(bool))
            usable = spine.representation(rid)["usable"].astype(bool)
            base = fl.static_pose(held, usable)
            pose = base + jit.draw(rng, d["conf"], table) * ell
            Xi, _v = ego.transform(pose, ell, fps, usable=usable)
            parts.append(Xi)
        J = np.concatenate(parts, axis=0)

        for arm, arr in (("corpus", X), ("jitter", J)):
            As = arr / sd[None, :]
            warm: dict = {}
            for alpha in tf.ALPHAS:
                n_pk = n_fr = n_conv = n_rec = 0
                for r in range(bounds.size - 1):
                    lo, hi = int(bounds[r]), int(bounds[r + 1])
                    if hi - lo < 16:
                        continue
                    sub = As[lo:hi][:, idx]
                    lam = alpha * tf.scale_of(sub)
                    if not np.isfinite(lam) or lam <= 0:
                        continue
                    got = tf.fit(sub, w[lo:hi], lam,
                                 warm=warm.get((arm, r)))
                    warm[(arm, r)] = got
                    k = tf.knots(got["z"])
                    k = k[(k >= 0) & (k < hi - lo)]
                    k = k[~abstain[lo:hi][k]]
                    n_pk += int(k.size)
                    n_fr += int((~abstain[lo:hi]).sum())
                    n_conv += int(bool(got["converged"]))
                    n_rec += 1
                if n_rec == 0:
                    continue
                rows.append({"animal": tag, "arm": arm, "alpha": float(alpha),
                             "rate_per_s": n_pk / max(n_fr / fps, 1e-9),
                             "converged_frac": n_conv / n_rec,
                             "n_recordings": n_rec})
        log(f"  {n}/{len(tags)} {tag}")

    out = os.path.join(work_dir("sweep"), f"shard_{a.shard:03d}.json")
    write_json({"shard": a.shard, "of": a.of, "rows": rows}, out)
    log(f"  wrote {out}")
    return 0


def sweep_combine(a) -> int:
    paths = sorted(glob.glob(os.path.join(work_dir("sweep"),
                                          "shard_*.json")))
    if not paths:
        raise SystemExit("no sweep shards")
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows.extend(json.load(fh)["rows"])
    tags = sorted({r["animal"] for r in rows})
    obj = {"dataset": "luna", "arm": "trendfilter", "split": SPLIT,
           "pose_arm": POSE_ARM, "group": GROUP, "estimator": "l1_tf_d3_group"}
    per: dict = {}
    conv: dict = {}
    for alpha in tf.ALPHAS:
        c = [r for r in rows if r["arm"] == "corpus" and r["alpha"] == alpha]
        j = [r for r in rows if r["arm"] == "jitter" and r["alpha"] == alpha]
        if not c or not j:
            continue
        ci = boot.animal_interval([r["rate_per_s"] for r in c],
                                  [r["animal"] for r in c], how="mean",
                                  n_boot=fl.N_BOOT, seed=SEED)
        ji = boot.animal_interval([r["rate_per_s"] for r in j],
                                  [r["animal"] for r in j], how="mean",
                                  n_boot=fl.N_BOOT, seed=SEED)
        per[alpha] = {"corpus": ci["point"], "jitter": ji["point"],
                      "jitter_hi": ji["hi"], "corpus_ci": dict(ci),
                      "jitter_ci": dict(ji)}
        conv[f"{alpha:g}"] = float(np.mean(
            [r["converged_frac"] for r in c + j]))
        log(f"  alpha={alpha:<6g} corpus {ci['point']:.4f}/s  jitter "
            f"{ji['point']:.4f}/s  share "
            f"{ji['hi'] / max(ci['point'], 1e-9):.4f}  converged "
            f"{conv[f'{alpha:g}']:.2f}")

    rd = tf.lambda_read(per, scored_object=obj, n_effective=len(tags))
    star = rd.detail.get("lambda_star_alpha")
    if star is not None and conv.get(f"{star:g}", 0.0) < 0.95:
        rd = Read("NOT_A_RESULT", obj,
                  (f"lambda* would be alpha = {star:g}, but the solver "
                   f"converged on only {conv[f'{star:g}']:.0%} of recordings "
                   f"there. A boundary set from an unconverged solve is not a "
                   f"measurement, so the selection is refused rather than "
                   f"reported. The registration forbids tuning the iteration "
                   f"budget against an outcome, and raising it until the "
                   f"number appears is exactly that"),
                  n_effective=len(tags),
                  detail={**rd.detail, "converged_by_alpha": conv})
    else:
        rd.detail["converged_by_alpha"] = conv
    log("  " + rd.line())

    out = a.out or config.PATHS.result("trendfilter_sweep.json")
    write_json({**provenance.header(anchors.LUNA, stage="trendfilter_sweep",
                                 unverified="an instrument probe on tune"),
                "inherited_digest": spine.digest(),
                "registration": "results/TRENDFILTER_PREREGISTRATION.md",
                "split": SPLIT, "n_animals": len(tags),
                "alphas": list(tf.ALPHAS), "scale": "median||D3 Y||",
                "max_iters": tf.MAX_ITERS,
                "by_alpha": {f"{k:g}": v for k, v in per.items()},
                "converged_by_alpha": conv,
                "reads": {"lambda": rd.to_dict()}, "rows": rows}, out)
    log(f"  wrote {out}")
    return 0 if rd.verdict == "PASS" else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="sweep", choices=("sweep", "plant"))
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=12)
    p.add_argument("--combine", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.phase == "sweep":
        return sweep_combine(a) if a.combine else sweep(a)
    raise SystemExit("phase 'plant' runs only after the sweep selects lambda*")


if __name__ == "__main__":
    raise SystemExit(main())
