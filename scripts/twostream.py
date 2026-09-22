"""Stage 1: configuration and dynamics as two streams, combined and labelled.

    python3 scripts/twostream.py --shard I --of N
    python3 scripts/twostream.py --combine

READ results/TWOSTREAM_PREREGISTRATION.md FIRST.

Three arms per animal:

* `config14`  the frozen detector on the 14 `shape` channels -- the unchanged
              baseline, so the added channel's contribution is visible;
* `config15`  the same plus `triage.body_extension`;
* `dyn`       a two-window contrast on `pole_radius` alone.

plus `dyn_noise`, the dynamics stream on a constant pose carrying
measured-colour tracking noise, which is the floor its rate is read against.
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
from recur.audit import triage                                        # noqa: E402
from recur.read import Read                                           # noqa: E402
from recur.util import frames, log, write_json                        # noqa: E402
from vieb import seeds                                                # noqa: E402
from vieb.clean import arms as clean_arms                             # noqa: E402
from vieb.io import spine                                             # noqa: E402
from vieb.qc import concentration as cc                               # noqa: E402
from vieb.seg import breaks as bk, dynplant as dp, floor as fl        # noqa: E402
from vieb.seg import noise as nz, twostream as ts                     # noqa: E402
from vieb.tok import config, ego                                      # noqa: E402

SEED = 0
SPLIT = "tune"
POSE_ARM = "raw"
GROUP = "shape"
STRIDE = 2
CENTRE = 3
#: Fixed scale for `body_extension`, so the added channel is not per-recording
#: adaptive. It is already normalised by the recording's own median, so this is
#: one constant putting it on the same footing as the standardised shape
#: channels, computed once and recorded in the result.
#:
#: Measured at 0.1518 over 20 recordings: `body_extension` has median 1.0000
#: by construction and SD 0.1518 about it.
EXT_SD = 0.1518


def work_dir() -> str:
    return os.path.join(config.REPO, "work", "twostream")


def ego_path(tag: str) -> str:
    return os.path.join(config.REPO, "work", "ego",
                        f"{POSE_ARM}__bodylen__{tag}.npz")


def basis_sd() -> np.ndarray:
    with open(os.path.join(config.PATHS.tok_dir, "basis.json"),
              encoding="utf-8") as fh:
        return np.asarray(json.load(fh)["sd_used"], dtype=np.float64)


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
    win = frames(dp.WIN_S, fps)
    cal = nz.calibration()
    by_animal = tune_animals()
    tags = sorted(by_animal)[a.shard::a.of]
    log(f"  shard {a.shard}/{a.of}: {len(tags)} animals, win={win}")

    rows: list[dict] = []
    for n, tag in enumerate(tags, 1):
        with np.load(ego_path(tag), allow_pickle=False) as z:
            X = np.asarray(z["X"], dtype=np.float64)
            bounds = z["bounds"].astype(np.int64)
            valid = z["valid"].astype(bool)
            ell = float(z["ell_a"])
        ab_p = os.path.join(config.PATHS.tok_dir, "abstain", f"{tag}.npz")
        with np.load(ab_p, allow_pickle=False) as z:
            abstain = z["abstain"].astype(bool) | ~valid

        ext = np.full(X.shape[0], np.nan)
        edge = np.full(X.shape[0], np.nan)
        noisy: list[np.ndarray] = []
        rng = np.random.default_rng(seeds.stable_seed(SEED, f"ts|{tag}"))
        for r, rid in enumerate(by_animal[tag]):
            lo, hi = int(bounds[r]), int(bounds[r + 1])
            d = spine.clean(rid)
            held = clean_arms.held_array(d["pose_unfiltered"].astype(float),
                                         d["missing"].astype(bool))
            ext[lo:hi] = triage.body_extension(held)
            edge[lo:hi] = cc.edgeness(held[:, CENTRE])
            usable = spine.representation(rid)["usable"].astype(bool)
            base = fl.static_pose(held, usable)
            pose = base + nz.draw(rng, base.shape[0], base.shape[1], fps=fps,
                                  cal=cal)
            Xi, _v = ego.transform(pose, ell, fps, usable=usable)
            noisy.append(Xi / sd[None, :])
        J = np.concatenate(noisy, axis=0)

        Xs = X / sd[None, :]
        X15 = np.column_stack([Xs[:, idx],
                               np.nan_to_num(ext, nan=1.0) / EXT_SD])
        h = max(2, int(round(bk.DERIV_SEC * fps)))
        e_edges = np.quantile(edge[np.isfinite(edge)], [1 / 3, 2 / 3]) \
            if np.isfinite(edge).any() else np.array([0.0, 0.0])

        acc: dict[str, list[np.ndarray]] = {k: [] for k in
                                            ("config14", "config15", "dyn",
                                             "dyn_noise")}
        n_elig = 0
        tercile_pk = {k: np.zeros(3) for k in ("config15", "dyn")}
        tercile_fr = np.zeros(3)
        for r in range(bounds.size - 1):
            lo, hi = int(bounds[r]), int(bounds[r + 1])
            if hi - lo < 4 * win:
                continue
            blk = abstain[lo:hi]
            p14 = ts.config_peaks(Xs[lo:hi], idx=idx, fps=fps,
                                  k_mad=bk.K_MAD, blocked=blk)
            p15 = ts.config_peaks(X15[lo:hi], idx=range(X15.shape[1]),
                                  fps=fps, k_mad=bk.K_MAD, blocked=blk)
            pdy = ts.dyn_peaks(Xs[lo:hi][:, idx], fps=fps, win=win,
                               blocked=blk, stride=STRIDE)
            pdn = ts.dyn_peaks(J[lo:hi][:, idx], fps=fps, win=win,
                               blocked=blk, stride=STRIDE)
            for k, v in (("config14", p14), ("config15", p15),
                         ("dyn", pdy), ("dyn_noise", pdn)):
                acc[k].append(np.asarray(v, dtype=np.int64) + lo)
            n_elig += int((~blk[h:hi - lo - h]).sum())
            te = np.clip(np.searchsorted(e_edges, edge[lo:hi]), 0, 2)
            for t in range(3):
                tercile_fr[t] += int(((te == t) & ~blk).sum())
                for k, v in (("config15", p15), ("dyn", pdy)):
                    if v.size:
                        tercile_pk[k][t] += int((te[v] == t).sum())
        cat = {k: (np.concatenate(v) if v else np.zeros(0, np.int64))
               for k, v in acc.items()}
        lab_ = ts.label_boundaries(cat["config15"], cat["dyn"])
        secs = max(n_elig / fps, 1e-9)
        rows.append({"animal": tag, "eligible_seconds": secs,
                     "n_config14": int(cat["config14"].size),
                     "n_config": int(lab_["n_config"]),
                     "n_dyn": int(lab_["n_dyn"]),
                     "n_both": int(lab_["n_both"]),
                     "n_dyn_only": int(lab_["n_dyn_only"]),
                     "n_config_only": int(lab_["n_config_only"]),
                     "rate_config14": cat["config14"].size / secs,
                     "rate_config15": cat["config15"].size / secs,
                     "rate_dyn": cat["dyn"].size / secs,
                     "rate_dyn_noise": cat["dyn_noise"].size / secs,
                     "tercile_frames": tercile_fr.tolist(),
                     "tercile_config15": tercile_pk["config15"].tolist(),
                     "tercile_dyn": tercile_pk["dyn"].tolist()})
        log(f"  {n}/{len(tags)} {tag}: cfg15 {lab_['n_config']} dyn "
            f"{lab_['n_dyn']} both {lab_['n_both']} dyn_only "
            f"{lab_['n_dyn_only']}")

    out = os.path.join(work_dir(), f"shard_{a.shard:03d}.json")
    write_json({"shard": a.shard, "of": a.of, "rows": rows}, out)
    log(f"  wrote {out}")
    return 0


def combine(a) -> int:
    paths = sorted(glob.glob(os.path.join(work_dir(), "shard_*.json")))
    if not paths:
        raise SystemExit("no shards")
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows.extend(json.load(fh)["rows"])
    tags = [r["animal"] for r in rows]
    obj = {"dataset": "luna", "arm": "twostream", "split": SPLIT,
           "pose_arm": POSE_ARM, "group": GROUP, "k_mad": bk.K_MAD,
           "win_s": dp.WIN_S, "dyn_tol": ts.DYN_TOL}
    reads: dict = {}
    rates = {k: dict(boot.animal_interval([r[f"rate_{k}"] for r in rows], tags,
                                          how="mean", n_boot=ts.N_BOOT,
                                          seed=SEED))
             for k in ("config14", "config15", "dyn", "dyn_noise")}
    for k, v in rates.items():
        log(f"  {k:<12} {v['point']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]/s")

    sp = ts.split_read(rows, scored_object=obj, n_effective=len(tags),
                       seed=SEED)
    reads["split"] = sp.to_dict()
    log("  " + sp.line())

    with open(config.PATHS.result("noise_floor.json"), encoding="utf-8") as fh:
        nf = json.load(fh)
    incumbent = (float(nf["rates"]["jitter1.0"]["ci"]["point"])
                 / float(nf["rates"]["corpus"]["ci"]["point"]))
    sep = ts.separation_read(rates["dyn"], rates["dyn_noise"],
                             incumbent=incumbent,
                             scored_object={**obj, "arm": "dyn_vs_floor"},
                             n_effective=len(tags))
    reads["separation"] = sep.to_dict()
    log("  " + sep.line())

    by_t: dict[str, dict[str, float]] = {}
    for k in ("config15", "dyn"):
        fr = np.sum([r["tercile_frames"] for r in rows], axis=0)
        pk = np.sum([r[f"tercile_{k}"] for r in rows], axis=0)
        by_t[k] = {f"t{i}": float(pk[i] / max(fr[i], 1) * 30.0)
                   for i in range(3)}
    loc = ts.location_read(by_t, scored_object={**obj, "arm": "location"},
                           n_effective=len(tags))
    reads["location"] = loc.to_dict()
    log("  " + loc.line())

    d14, d15 = rates["config14"]["point"], rates["config15"]["point"]
    move = abs(d15 - d14) / max(d14, 1e-12)
    ext = Read("PASS" if move < 0.20 else "FAIL", {**obj, "arm": "extension"},
               (f"adding body_extension moves the configuration rate from "
                f"{d14:.4f}/s to {d15:.4f}/s, {move:.1%} "
                f"{'within' if move < 0.20 else 'ABOVE'} the registered 20%. "
                f"A channel that moved it more would be dominating rather "
                f"than contributing"), n_effective=len(tags),
               detail={"rate_14": d14, "rate_15": d15, "relative_move": move})
    reads["extension"] = ext.to_dict()
    log("  " + ext.line())

    out = a.out or config.PATHS.result("twostream.json")
    write_json({**anchors.header(anchors.LUNA, stage="twostream",
                                 unverified="an instrument probe on tune"),
                "inherited_digest": spine.digest(),
                "registration": "results/TWOSTREAM_PREREGISTRATION.md",
                "split": SPLIT, "n_animals": len(tags), "seed": SEED,
                "ext_sd": EXT_SD, "stride": STRIDE,
                "config_tol": ts.CONFIG_TOL, "dyn_tol": ts.DYN_TOL,
                "rates": rates, "by_tercile": by_t,
                "incumbent_jitter_share": incumbent,
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
