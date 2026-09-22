"""Registered prediction 2: does the published floor survive the right colour?

    python3 scripts/colour_check.py [--limit N]

READ results/DYNAMICS_PREREGISTRATION.md §6 FIRST.

`NOISEFLOOR.md` measured the frozen detector's boundary rate on a constant pose
carrying **white** noise at sigma-bar = 0.0618 bl, and got **0.2599/s**. The
frozen calibration says the real residual is not white -- lag-1 +0.5817 over
267 ms. This re-runs the identical arm with the identical detector and the
**measured colour**, and asks how far the floor moves.

The registration predicts **less than 25%**, on the grounds that `PLANT.md`
measured near-invariance to amplitude and the threshold is scale-adaptive. A
larger move means the published 0.2599/s is colour-dependent and has to be
restated rather than left standing.
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
from recur.read import Read                                           # noqa: E402
from recur.util import log, write_json                                # noqa: E402
from vieb import seeds                                                # noqa: E402
from vieb.clean import arms as clean_arms                             # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.seg import breaks as bk, floor as fl, jitter as jit         # noqa: E402
from vieb.seg import noise as nz                                      # noqa: E402
from vieb.tok import config, ego                                      # noqa: E402

SEED = 0
SPLIT = "tune"
POSE_ARM = "raw"
GROUP = "shape"
#: The registered tolerance on the move.
MOVE_TOL = 0.25


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
    out = a.out or config.PATHS.result("colour_check.json")

    fps = spine.fps()
    sd = basis_sd()
    idx = list(bk.CHANNEL_GROUPS[GROUP])
    cal = nz.calibration()
    tbl = json.load(open(config.PATHS.result("noise_floor.json"),
                         encoding="utf-8"))
    published = float(tbl["rates"]["jitter1.0"]["ci"]["point"])
    jit_tbl = tbl["calibration"]

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
    log(f"  {len(tags)} {SPLIT} animals; published white floor "
        f"{published:.4f}/s")

    per: dict[str, list[float]] = {"white": [], "coloured": []}
    who: list[str] = []
    for n, tag in enumerate(tags, 1):
        with np.load(ego_path(tag), allow_pickle=False) as z:
            bounds = z["bounds"].astype(np.int64)
            valid = z["valid"].astype(bool)
            ell = float(z["ell_a"])
        ab_p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
        with np.load(ab_p, allow_pickle=False) as z:
            abstain = z["abstain"].astype(bool) | ~valid
        rng = np.random.default_rng(seeds.stable_seed(SEED, f"colour|{tag}"))
        parts: dict[str, list[np.ndarray]] = {"white": [], "coloured": []}
        for r, rid in enumerate(by_animal[tag]):
            d = spine.clean(rid)
            held = clean_arms.held_array(d["pose_unfiltered"].astype(float),
                                         d["missing"].astype(bool))
            usable = spine.representation(rid)["usable"].astype(bool)
            base = fl.static_pose(held, usable)
            for arm in ("white", "coloured"):
                if arm == "white":
                    pose = base + jit.draw(rng, d["conf"], jit_tbl) * ell
                else:
                    pose = base + nz.draw(rng, base.shape[0], base.shape[1],
                                          fps=fps, cal=cal)
                Xi, _v = ego.transform(pose, ell, fps, usable=usable)
                parts[arm].append(Xi / sd[None, :])
        for arm in ("white", "coloured"):
            Xi = np.concatenate(parts[arm], axis=0)
            g = fl.rate_of(Xi, bounds, abstain, idx=idx, fps=fps,
                           k_mad=bk.K_MAD)
            per[arm].append(float(g["rate_per_s"]))
        who.append(tag)
        if n % 10 == 0 or n == len(tags):
            log(f"  {n}/{len(tags)}")

    obj = {"dataset": "luna", "arm": "colour_check", "split": SPLIT,
           "pose_arm": POSE_ARM, "group": GROUP, "k_mad": bk.K_MAD}
    ci = {k: dict(boot.animal_interval(v, who, how="mean", n_boot=fl.N_BOOT,
                                       seed=SEED)) for k, v in per.items()}
    w, c = float(ci["white"]["point"]), float(ci["coloured"]["point"])
    move = abs(c - w) / max(w, 1e-12)
    detail = {"white": ci["white"], "coloured": ci["coloured"],
              "published_white_floor": published, "relative_move": move,
              "tolerance": MOVE_TOL}
    if move < MOVE_TOL:
        rd = Read("PASS", obj,
                  (f"the floor SURVIVES the correct noise colour: "
                   f"{c:.4f} [{ci['coloured']['lo']:.4f}, "
                   f"{ci['coloured']['hi']:.4f}]/s under measured-colour noise "
                   f"against {w:.4f}/s under white, a move of {move:.1%} below "
                   f"the registered {MOVE_TOL:.0%}. NOISEFLOOR.md's "
                   f"{published:.4f}/s stands, and the reason it stands is the "
                   f"scale-adaptive threshold, not luck"),
                  n_effective=len(tags), detail=detail)
    else:
        rd = Read("FAIL", obj,
                  (f"THE FLOOR IS COLOUR-DEPENDENT: {c:.4f} "
                   f"[{ci['coloured']['lo']:.4f}, {ci['coloured']['hi']:.4f}]/s "
                   f"under measured-colour noise against {w:.4f}/s under white, "
                   f"a move of {move:.1%} above the registered {MOVE_TOL:.0%}. "
                   f"NOISEFLOOR.md's published {published:.4f}/s was measured "
                   f"under the wrong colour and must be restated"),
                  n_effective=len(tags), detail=detail)
    log("  " + rd.line())

    write_json({**anchors.header(anchors.LUNA, stage="colour_check",
                                 unverified="an instrument probe on tune"),
                "inherited_digest": spine.digest(),
                "registration": "results/DYNAMICS_PREREGISTRATION.md",
                "split": SPLIT, "n_animals": len(tags), "seed": SEED,
                "rates": ci, "published_white_floor": published,
                "relative_move": move,
                "reads": {"colour": rd.to_dict()}}, out)
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
