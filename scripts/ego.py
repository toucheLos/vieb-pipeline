"""Stage 1. The egocentric transform over the corpus, and the A4 parity gate.

    python3 scripts/ego.py --write-grid
    sbatch --array=0-29%12 jobs/ego.slurm --pose-arm raw --scale-arm bodylen
    python3 scripts/ego.py --combine --pose-arm raw --scale-arm bodylen

Sharded by **animal**, because `ell_a` is the animal's own median body length
pooled over all of its recordings. Not per recording -- per-recording
standardisation is what silently disarmed a control on this project once already.

## Two independent arm axes, and they were once one name

**`--scale-arm`** is about `ell_a`: `bodylen` divides by the animal's own body
length, `unitlen` sets it to 1. It was written because the brief's premise about
the identity leak is wrong -- 5.88 nats is the PREVIOUS instrument's number,
measured with a shuffle correction, where shapeflow's direct probe reads
0.833 / 0.730 at gamma = 0. `EGO.md` settled it: `bodylen` takes animal
identification from 15.0% to 6.9% and session from 17.9% to 1.4%.

**`--pose-arm`** is about which coordinates the transform consumes, and it is the
axis this script originally did not have. `wiener` is `clean["pose"]`, which
`shapeflow/results/clean.json` records as `filter.default`; `raw` is
`held_array(pose_unfiltered, missing)`, which `F3_PREPROCESSING_FREEZE.md` §1
names as a carried arm and calls "the floor".

**`unitlen` used to be called `raw`.** That is why shard names now carry both
axes: `raw__103.npz` meant *no body-length normalisation, on Wiener pose*, and
adding a pose arm named `raw` would have collided with it silently. Shards are
`<pose_arm>__<scale_arm>__<tag>.npz`; the 596 two-part files already on disk are
the pre-freeze set and are left where they are.

## Why the pose arm matters here more than anywhere else

Wiener is a low-pass filter, and the stage this feeds measures memory depth --
how far back the past predicts. A low-pass filter manufactures exactly that. So
the carried arm is not a preference; it is the difference between measuring the
corpus and measuring the filter.
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

from recur import anchors, labels as lab, splits                  # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.audit import leak                                      # noqa: E402
from recur.geom import represent as rep                           # noqa: E402
from vieb.clean import arms as clean_arms                         # noqa: E402
from vieb.io import spine                                        # noqa: E402
from recur.qc import swap                                         # noqa: E402
from vieb.tok import config, ego, parity, reversal               # noqa: E402
from recur.util import describe, log, peak_rss_gb, write_json     # noqa: E402

#: How `ell_a` is applied. `unitlen` was called `raw` before the pose axis
#: existed; the old name is gone rather than aliased, because an alias would let
#: `--arm raw` keep running and mean something different from what it used to.
SCALE_ARMS = ("bodylen", "unitlen")

#: Which coordinates the transform consumes.
#:
#: NOT `config.POSE_ARMS`, and the difference is load-bearing. That tuple's
#: `unfiltered` is bare `pose_unfiltered`, which is what `bones.py` measures on
#: so that its 2x2 against shapeflow's own `bone_flagged` is like-for-like.
#: F3's `raw` arm is `held_array(pose_unfiltered, missing)` -- the gap policy's
#: output, which is what the incumbent filter actually received. Reusing
#: `POSE_KEY` here would have silently dropped the hold.
POSE_ARMS = ("raw", "wiener")

#: Written into every result document, so a reader never has to go and find out
#: which coordinates a number was computed on. This is the provenance that was
#: missing when the ego stage was first run.
POSE_ARM_NOTE = {
    "raw": ("held_array(pose_unfiltered, missing) -- the gap policy's output, "
            "no smoother of any kind. F3_PREPROCESSING_FREEZE.md SS1 carries "
            "this arm and calls it the floor"),
    "wiener": ("clean['pose'], which shapeflow/results/clean.json records as "
               "filter.default = wiener. F3 SS1 lists it as not benchmarkable "
               "and therefore NOT carried onto the MDL branch"),
}

#: Contiguous windows per recording for the leak probe. Matches `scripts/audit.py`
#: so the two numbers are on the same footing.
N_WINDOWS = 8
MIN_WINDOW = 50


def pose_for(rid: str, pose_arm: str) -> np.ndarray:
    """The coordinates one arm consumes, as float64.

    One function, called by BOTH the shard and the regressor arm of `combine`.
    They used to be two `spine.clean(rid)["pose"]` reads in different places,
    which is how an egocentric representation built on one array could have been
    scored against kinematics computed from another.
    """
    d = spine.clean(rid)
    if pose_arm == "wiener":
        return d["pose"].astype(np.float64)
    if pose_arm == "raw":
        return clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                     d["missing"].astype(bool))
    raise SystemExit(f"unknown pose arm {pose_arm!r}; one of {POSE_ARMS}")


def shard_path(pose_arm: str, scale_arm: str, tag: str) -> str:
    return os.path.join(config.PATHS.ego_dir,
                        f"{pose_arm}__{scale_arm}__{tag}.npz")


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def shard(args, tag: str) -> int:
    paths = config.PATHS
    os.makedirs(paths.ego_dir, exist_ok=True)
    fps = spine.fps()
    mine = animals_of(spine.recording_ids())[tag]

    poses, usable, lengths = [], [], []
    for rid in mine:
        r = spine.representation(rid)
        poses.append(pose_for(rid, args.pose_arm))
        usable.append(r["usable"].astype(bool))
        lengths.append(int(poses[-1].shape[0]))

    # One number per animal, over every recording it has, on usable frames only.
    ell = 1.0 if args.scale_arm == "unitlen" else ego.ell_a(poses, usable)
    if not np.isfinite(ell) or ell <= 0:
        raise SystemExit(f"[{tag}] ell_a = {ell!r}; refusing to divide by it")

    xs, vs, rows = [], [], []
    for i, rid in enumerate(mine):
        x, valid = ego.transform(poses[i], ell, fps, usable=usable[i])
        speed, omega = rep.raw_kinematics(poses[i], fps, axis=ego.AXIS,
                                          causal=True)
        rows.append({
            "recording_id": rid, "animal": tag, "n_frames": lengths[i],
            "frac_valid": float(valid.mean()),
            "exact": parity.exact_scores(x, ell, fps, speed, omega, valid=valid),
            "channel_sd": np.nanstd(np.where(valid[:, None], x, np.nan),
                                    axis=0).tolist(),
        })
        xs.append(x.astype(np.float32))
        vs.append(valid)

    X = np.concatenate(xs)
    valid = np.concatenate(vs)
    bounds = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez_compressed(
        shard_path(args.pose_arm, args.scale_arm, tag),
        X=X, valid=valid, bounds=bounds, recording_ids=np.array(mine),
        ell_a=np.array(ell), arm=np.array(args.scale_arm),
        scale_arm=np.array(args.scale_arm), pose_arm=np.array(args.pose_arm),
        channels=np.array(ego.CHANNELS),
        rank=np.array(json.dumps(ego.rank_read(X[valid][:200000]))),
        rows_json=np.array(json.dumps(rows)),
        inherited_digest=np.array(spine.digest()))
    log(f"[{tag}] pose={args.pose_arm} scale={args.scale_arm} "
        f"ell_a={ell:.3f} valid={valid.mean():.4f} "
        f"peak_rss={peak_rss_gb():.2f} GB")
    return 0


def _window_rows(X, valid, bounds, rids, fps):
    """One probe row per contiguous window: per-channel mean and SD.

    A representation is continuous, so the probe sees summary statistics rather
    than the histogram a labeller's probe would use. Held out **by window within
    recording**, which is what asks whether the representation carries session
    identity rather than whether two windows of one session resemble each other.
    """
    feats, animals, sessions = [], [], []
    for r in range(len(rids)):
        lo, hi = int(bounds[r]), int(bounds[r + 1])
        edges = np.linspace(lo, hi, N_WINDOWS + 1).astype(int)
        for w in range(N_WINDOWS):
            a, b = edges[w], edges[w + 1]
            m = valid[a:b]
            if int(m.sum()) < MIN_WINDOW:
                continue
            blk = X[a:b][m].astype(np.float64)
            feats.append(np.concatenate([blk.mean(axis=0), blk.std(axis=0)]))
            animals.append(lab.animal_tag(str(rids[r])))
            sessions.append(str(rids[r]))
    return (np.asarray(feats), animals, sessions)


def combine(args) -> int:
    paths = config.PATHS
    anchor = anchors.LUNA
    pattern = os.path.join(paths.ego_dir,
                           f"{args.pose_arm}__{args.scale_arm}__*.npz")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise SystemExit(f"no shards matching {pattern}")

    rows, ells, ranks = [], {}, []
    feats, animals, sessions = [], [], []
    rng = np.random.default_rng(0)
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            rows += json.loads(str(z["rows_json"]))
            tag = os.path.basename(path).split("__")[2][:-4]
            ells[tag] = float(z["ell_a"])
            ranks.append(json.loads(str(z["rank"])))
            f, a, s = _window_rows(z["X"], z["valid"], z["bounds"],
                                   z["recording_ids"], spine.fps())
            if len(f):
                feats.append(f)
                animals += a
                sessions += s
    feats = np.concatenate(feats) if feats else np.zeros((0, 2 * ego.N_DIMS))
    log(f"{len(shards)} shards, {len(rows)} recordings, {feats.shape[0]} windows")

    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    obj = {"dataset": "luna", "arm": f"ego_{args.pose_arm}_{args.scale_arm}",
           "pose_arm": args.pose_arm, "scale_arm": args.scale_arm,
           "split": "all", "dim": ego.N_DIMS}

    # ---- A4, on the report split -----------------------------------------
    report = [r for r in rows if of.get(r["animal"]) == "report"]
    n_frames = np.array([r["exact"]["n_frames"] for r in report], dtype=np.float64)
    scores = {"exact": {
        "speed": float(np.average([r["exact"]["speed"] for r in report],
                                  weights=n_frames)),
        "angular": float(np.average([r["exact"]["angular"] for r in report],
                                    weights=n_frames)),
        "n_frames": int(n_frames.sum()),
        "n_recordings": len(report)}}

    if args.regressor:
        sub = rng.permutation(len(shards))[:args.regressor_shards]
        xs, vs, sp, om, gr = [], [], [], [], []
        for i in sub:
            with np.load(shards[i], allow_pickle=False) as z:
                b, rids = z["bounds"], z["recording_ids"]
                for r in range(len(rids)):
                    lo, hi = int(b[r]), int(b[r + 1])
                    pose = pose_for(str(rids[r]), args.pose_arm)
                    s_, o_ = rep.raw_kinematics(pose, spine.fps(),
                                                axis=ego.AXIS, causal=True)
                    xs.append(z["X"][lo:hi].astype(np.float64))
                    vs.append(z["valid"][lo:hi])
                    sp.append(s_)
                    om.append(o_)
                    gr += [lab.animal_tag(str(rids[r]))] * (hi - lo)
        X = np.concatenate(xs)
        V, S, O = np.concatenate(vs), np.concatenate(sp), np.concatenate(om)
        scores["windowed"] = parity.windowed_scores(X, S, O, gr, k=5, valid=V)
        scores["pose_only_single_frame"] = parity.windowed_scores(
            X, S, O, gr, k=1, valid=V, pose_only=True)

    reads = {"a4": parity.a4_read(scores, scored_object=obj,
                                  n_effective=len({r["animal"] for r in report}))}

    # ---- the identity leak, the whole reason there are two arms ----------
    for kind, ids in (("animal", animals), ("session", sessions)):
        reads[f"identity_leak_{kind}"] = leak.leak_read(
            feats, ids, kind=kind, scored_object={**obj, "probe": "window"},
            n_effective=len(set(ids)))

    rv = reversal.audit(fps=spine.fps())
    # This assertion exists because the check it makes had already failed once.
    # The repo split rewrote this import to `recur.geom.reversal`, which is the
    # inherited SEVEN checks with none of the three SE(2) twist checks
    # `vieb/tok/reversal.py` was written to add -- and the audit still said
    # PASS, because seven passing checks do pass. `n_checks_inherited` exists
    # only on the composed audit, so its presence is what proves which one ran.
    if "n_checks_inherited" not in rv or int(rv["n_checks"]) < 10:
        raise SystemExit(
            f"the reversal audit ran with {rv.get('n_checks')} checks and "
            f"{'no' if 'n_checks_inherited' not in rv else 'an'} inherited "
            f"count: this is recur's bare audit, not vieb.tok.reversal's "
            f"composition, so the twist channels went unchecked")
    if int(rv["n_failed"]):
        raise SystemExit(f"reversal audit failed {rv['n_failed']} checks")
    doc = {
        **provenance.header(anchor,
                         stage=f"ego_{args.pose_arm}_{args.scale_arm}",
                         observed={
                             "n_recordings": len(rows),
                             "n_frames": sum(r["n_frames"] for r in rows),
                             "fps": spine.fps()}),
        "inherited_digest": spine.digest(),
        "preprocessing_freeze": "F3",
        "reads": {k: v.to_dict() for k, v in reads.items()},
        "arm": args.scale_arm,
        "scale_arm": args.scale_arm,
        "pose_arm": args.pose_arm,
        "pose_arm_note": POSE_ARM_NOTE[args.pose_arm],
        "dims": {"total": ego.N_DIMS, "pose": ego.N_POSE, "twist": ego.N_TWIST,
                 "channels": list(ego.CHANNELS),
                 "unreconciled": ("the brief says '28 dims stay 28 dims'; at 7 "
                                  "keypoints the two formulas give 14 + 3 = 17 "
                                  "and 28 is not reconstructible from them. "
                                  "Recorded rather than papered over with "
                                  "invented channels")},
        "scaling": ego.SCALING,
        "scaling_note": ("the brief applies f/ell_a to the whole twist; that is "
                         "dimensionally wrong for omega and makes a large "
                         "mouse's turning read systematically slower, so "
                         "rotation takes f alone"),
        "rank": {"expected": ego.POSE_RANK,
                 "observed": describe([r["rank"] for r in ranks if r.get("rank")]),
                 "why": ("SE(2) removal costs 3 of 14 pose dimensions: two to "
                         "the origin keypoint and one to the aligned axis")},
        "ell_a": {"per_animal": ells, "summary": describe(list(ells.values()))},
        "frac_valid": describe([r["frac_valid"] for r in rows]),
        "a4_scores": scores,
        "reversal_audit": rv,
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    for name, rd in reads.items():
        log(f"{name}: {rd.line()}")
    log(f"reversal audit: {rv['verdict']} ({rv['n_checks']} checks)")
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("ego")
    with open(path, "w") as fh:
        fh.write("\n".join(tags) + "\n")
    log(f"wrote {path}: {len(tags)} animals")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scale-arm", choices=SCALE_ARMS, default="bodylen")
    p.add_argument("--pose-arm", choices=POSE_ARMS, default="raw")
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--regressor", action="store_true",
                   help="run the reported (never gated) regressor arms")
    p.add_argument("--regressor-shards", type=int, default=6)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result(
            f"ego_{a.pose_arm}_{a.scale_arm}.json")
        return combine(a)

    swap.check_order(anchors.LUNA.keypoints)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        path = config.PATHS.grid("ego")
        with open(path) as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")

    for tag in tags:
        out = shard_path(a.pose_arm, a.scale_arm, tag)
        if not a.force and os.path.exists(out):
            log(f"[{tag}] {a.pose_arm}/{a.scale_arm} shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
