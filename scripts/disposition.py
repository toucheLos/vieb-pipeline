"""Phase D. Correct, abstain, or down-weight — and the gate that can fail.

    python3 scripts/disposition.py --write-grid
    sbatch --array=0-29%15 jobs/disposition.slurm
    python3 scripts/disposition.py --combine

The configuration is fixed in `results/DISPOSITION_PREREGISTRATION.md`, committed
before this ran on anything. Q1 is not re-scored here.
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

from recur import anchors, boot, labels as lab, splits             # noqa: E402
from recur.util import describe, log, peak_rss_gb, write_json      # noqa: E402
from vieb.clean import arms as clean_arms                          # noqa: E402
from vieb.io import spine                                          # noqa: E402
from recur.qc.swap import runs_of
from vieb.audit import separable as sep                            # noqa: E402
from vieb.qc import bones, disposition as dp, lobo                 # noqa: E402
from vieb.tok import ego                                           # noqa: E402
from vieb.tok import config                                        # noqa: E402

EPS = 0.10
PAIRS = bones.pair_indices(7)
#: Indices into PAIRS of the bones the corrector is allowed to constrain.
#: The constrained group and nothing else -- so the TRUNK it is evaluated
#: on is genuinely unseen, which the gate's reason string asserts.
CONSTRAIN = [PAIRS.index(b) for b in bones.SKULL]


def animals_of(ids) -> dict:
    out: dict = {}
    for rid in ids:
        out.setdefault(lab.animal_tag(rid), []).append(rid)
    return out


def shard(args, tag: str) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "disposition")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    mine = animals_of(spine.recording_ids())[tag]

    held_by_rec, conf_by_rec, lengths = {}, {}, []
    for rid in mine:
        d = spine.clean(rid)
        held = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                     d["missing"].astype(bool))
        held_by_rec[rid] = held
        conf_by_rec[rid] = d["conf"].astype(np.float64)
        lengths.append(bones.bone_lengths(held, PAIRS))

    # Per-animal rigid pairs and SCALE-FREE reference lengths for all 21 pairs.
    # Scale-free because the per-frame common scale has SD 0.155 log units
    # against 0.178 for the geometry itself: a constraint in raw pixels spends
    # most of its tolerance on a nuisance parameter.
    pooled = np.concatenate(lengths)
    del lengths
    logs = np.log(np.where(pooled > 0, pooled, np.nan))
    keep = bones.rigid_pairs(logs, PAIRS)
    scale_pooled = np.nanmedian(logs[:, keep], axis=1)
    sf = pooled / np.exp(scale_pooled)[:, None]
    l_hat_sf = np.array([bones.reference_length(sf[:, m], EPS)["l_hat"]
                         for m in range(len(PAIRS))])
    del pooled, logs, sf

    # Group references, in raw px, FIXED: the same yardstick before and after.
    stacked = np.concatenate([held_by_rec[r] for r in mine])
    l_skull = np.array([bones.reference_length(
        bones.metric_lengths(stacked, bones.SKULL, "raw", pairs=PAIRS,
                             keep=keep)[:, m], EPS)["l_hat"] for m in range(3)])
    l_trunk = np.array([bones.reference_length(
        bones.metric_lengths(stacked, bones.TRUNK, "raw", pairs=PAIRS,
                             keep=keep)[:, m], EPS)["l_hat"]
        for m in range(len(bones.TRUNK))])
    del stacked

    rows, poses, abst_all, corr_all, rel_all, lens = [], [], [], [], [], []
    for rid in mine:
        held = held_by_rec[rid]
        conf = conf_by_rec[rid]
        T = held.shape[0]
        Lsk = bones.metric_lengths(held, bones.SKULL, "raw", pairs=PAIRS, keep=keep)
        viol = bones.violations(Lsk, l_skull, EPS)
        mask = bones.frame_mask(viol)
        correctable, abstain = dp.envelope(mask)
        susp = dp.suspects(viol, bones.SKULL, conf)

        raw_logs = np.log(np.where(bones.bone_lengths(held, PAIRS) > 0,
                                   bones.bone_lengths(held, PAIRS), np.nan))
        out = held.copy()
        n_moved, moved_px, n_no_scale = 0, [], 0
        n_no_donor = n_boundary = n_unconverged = 0
        corrected = np.zeros(T, dtype=bool)
        for t in np.flatnonzero(correctable):
            s = int(susp[t])
            if s < 0:
                abstain[t] = True          # no single suspect -> abstain
                continue
            touching = set(dp.pairs_touching(s, PAIRS))
            keep_s = [m for m in keep if m not in touching]
            if len(keep_s) < dp.MIN_SCALE_PAIRS:
                # `common_log_scale` would silently return 1.0 here and revert
                # the arm to raw pixels. Abstain instead of correcting blind.
                abstain[t] = True
                n_no_scale += 1
                continue
            # The target: the suspect carried forward from the nearest clean
            # frame of THIS recording by the similarity the other six keypoints
            # fix. A run against a recording boundary has no donor -- abstain
            # rather than reach across a seam.
            donor = dp.donor_frame(mask, t)
            tgt = None if donor < 0 else dp.predict(out, t, donor, s)
            if tgt is None:
                abstain[t] = True
                n_no_donor += 1
                continue
            scale_t = float(np.exp(np.nanmedian(raw_logs[t, keep_s])))
            frame, info = dp.project(out, t, s, pairs=PAIRS,
                                     constrain=CONSTRAIN,
                                     l_hat_scaled=l_hat_sf * scale_t, eps=EPS,
                                     target=tgt)
            if not info["converged"]:
                # A frame still violating after the projection is not corrected,
                # and saying it is would be the claim, not the measurement.
                abstain[t] = True
                n_unconverged += 1
                continue
            out[t] = frame
            if info["landed_on_boundary"]:
                n_boundary += 1
            if info["moved"]:
                n_moved += 1
                corrected[t] = True
                moved_px.append(info["displacement_px"])

        # THE GATE: trunk violations, fixed yardstick, before and after.
        before = lobo.held_out_rate(held, group=bones.TRUNK, pairs=PAIRS,
                                    keep=keep, l_hat=l_trunk, eps=EPS)
        after = lobo.held_out_rate(out, group=bones.TRUNK, pairs=PAIRS,
                                   keep=keep, l_hat=l_trunk, eps=EPS)
        skull_after = float(bones.frame_mask(bones.violations(
            bones.metric_lengths(out, bones.SKULL, "raw", pairs=PAIRS, keep=keep),
            l_skull, EPS)).mean())

        rel = dp.reliability(conf, mask, corrected, abstain)
        rows.append({
            "recording_id": rid, "animal": tag, "n_frames": int(T),
            "violation_rate": float(mask.mean()),
            "frac_correctable": float(correctable.mean()),
            "frac_corrected": float(corrected.mean()),
            "frac_abstained": float(abstain.mean()),
            "n_abstain_runs": int(len(runs_of(abstain))),
            "n_no_scale": int(n_no_scale),
            "n_no_donor": int(n_no_donor),
            "n_boundary": int(n_boundary),
            "n_unconverged": int(n_unconverged),
            "n_corrected": int(n_moved),
            "median_move_px": float(np.median(moved_px)) if moved_px else 0.0,
            "trunk_before": before, "trunk_after": after,
            "skull_before": float(mask.mean()), "skull_after": skull_after,
            "mean_reliability": float(rel.mean()),
        })
        poses.append(out.astype(np.float32))
        abst_all.append(abstain)
        corr_all.append(corrected)
        rel_all.append(rel.astype(np.float32))
        lens.append(T)
        log(f"[{tag}] {rid[-28:]}  corrected {corrected.mean():.4%}  "
            f"abstained {abstain.mean():.4%}  trunk {before:.4%}->{after:.4%}")

    np.savez_compressed(
        os.path.join(out_dir, f"{tag}.npz"),
        recording_ids=np.array(mine),
        bounds=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        pose=np.concatenate(poses), abstain=np.concatenate(abst_all),
        corrected=np.concatenate(corr_all),
        reliability=np.concatenate(rel_all),
        rows_json=np.array(json.dumps(rows)),
        inherited_digest=np.array(spine.digest()))
    log(f"[{tag}] peak_rss={peak_rss_gb():.2f} GB")
    return 0


#: Clean rows kept per animal. The corrected class is ~0.4% of frames, so the
#: probe is balanced by subsampling the majority anyway -- this only bounds what
#: has to be held in memory before that happens.
CLEAN_PER_ANIMAL = 4000


def gather(out_dir: str, tags: list[str], fps: float, seed: int = 0,
           *, source: str = "after") -> tuple:
    """Egocentric rows for the probe: every corrected frame, some clean ones.

    `clean` is a frame the disposition neither moved nor flagged. Abstained
    frames are excluded from BOTH classes -- they are violating geometry left in
    place, so putting them in the negative class would ask the probe to separate
    corrected frames from violating ones rather than from ordinary ones.

    `source="before"` takes the SAME rows, labels and groups from the array the
    corrector was handed instead of the one it produced. That is the control the
    "after" probe cannot do without: corrected frames are SELECTED for violating,
    so they are unusual geometry whether or not anything was done to them, and a
    probe that separates them may be reading the selection rather than the
    correction. Only the increment from before to after is the corrector's.
    """
    rng = np.random.default_rng(seed)
    feats, ys, groups = [], [], []
    for tag in tags:
        path = os.path.join(out_dir, f"{tag}.npz")
        with np.load(path, allow_pickle=False) as z:
            if "corrected" not in z.files:
                raise SystemExit(f"{path} predates the corrected mask; re-run it")
            rids = [str(v) for v in z["recording_ids"]]
            bounds = z["bounds"]
            pose = z["pose"].astype(np.float64)
            corr = z["corrected"].astype(bool)
            abst = z["abstain"].astype(bool)
        if source == "before":
            pose = np.concatenate([
                clean_arms.held_array(
                    spine.clean(r)["pose_unfiltered"].astype(np.float64),
                    spine.clean(r)["missing"].astype(bool)) for r in rids])
        span = [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(len(rids))]
        ell = ego.ell_a([pose[sl] for sl in span])
        if not np.isfinite(ell) or ell <= 0:
            continue
        rows_a, y_a, g_a = [], [], []
        for rid, sl in zip(rids, span):
            x, valid = ego.transform(pose[sl], ell, fps)
            g = np.array(sep.blocks(rid, x.shape[0], fps=fps))
            y = corr[sl]
            keep_row = valid & (y | ~abst[sl])
            rows_a.append(x[keep_row])
            y_a.append(y[keep_row])
            g_a.append(g[keep_row])
        x_a = np.concatenate(rows_a)
        ya = np.concatenate(y_a)
        ga = np.concatenate(g_a)
        neg = np.flatnonzero(~ya)
        if neg.size > CLEAN_PER_ANIMAL:
            neg = rng.permutation(neg)[:CLEAN_PER_ANIMAL]
        idx = np.sort(np.concatenate([np.flatnonzero(ya), neg]))
        feats.append(x_a[idx])
        ys.append(ya[idx])
        groups.append(ga[idx])
    return (np.concatenate(feats), np.concatenate(ys),
            np.concatenate(groups).tolist())


def combine(args) -> int:
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "disposition")
    shards = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}")
    rows: list = []
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            rows += json.loads(str(z["rows_json"]))
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    sel = [r for r in rows if of.get(r["animal"]) == args.split]
    log(f"{len(shards)} shards, {len(rows)} recordings, {len(sel)} on {args.split}")

    an = [r["animal"] for r in sel]
    w = [r["n_frames"] for r in sel]
    ci = lambda k: boot.animal_interval([r[k] for r in sel], an, weights=w,
                                        how="wmean")
    trunk_before = float(np.average([r["trunk_before"] for r in sel], weights=w))
    trunk_after = float(np.average([r["trunk_after"] for r in sel], weights=w))
    obj = {"dataset": "luna", "arm": "disposition", "split": args.split,
           "eps": EPS}
    gate = lobo.lobo_read(trunk_before, trunk_after, constrained="skull",
                          evaluated="trunk", scored_object=obj,
                          n_effective=len({r["animal"] for r in sel}))

    tags = sorted({r["animal"] for r in sel})
    x, y, g = gather(out_dir, tags, spine.fps())
    probe = sep.separability(x, y, g)
    atom = sep.separable_read(probe, scored_object={**obj, "probe": "block"},
                              n_effective=len(tags))
    # The control: same rows, same labels, the array BEFORE the correction.
    xb, yb, gb = gather(out_dir, tags, spine.fps(), source="before")
    probe_before = sep.separability(xb, yb, gb)
    del xb, yb, gb

    doc = {
        **anchors.header(anchors.LUNA, stage="disposition",
                         unverified="scored on one split; the anchor counts all"),
        "inherited_digest": spine.digest(),
        "preregistration": "results/DISPOSITION_PREREGISTRATION.md",
        "reads": {"leave_one_bone_out": gate.to_dict(),
                  "separable": atom.to_dict()},
        "separable_before_correction": {
            **probe_before,
            "why": ("the same rows, labels and groups on the array the corrector "
                    "was HANDED. Corrected frames are selected for violating, so "
                    "they are unusual geometry before anything touches them; only "
                    "the rise from this number to the probe's is the corrector's "
                    "own signature")},
        "split": args.split, "eps": EPS,
        "envelope": {"max_correct_frames": dp.MAX_CORRECT_FRAMES,
                     "merge_gap_frames": dp.MERGE_GAP_FRAMES,
                     "min_scale_pairs": dp.MIN_SCALE_PAIRS},
        "domain": {"frac_corrected": ci("frac_corrected"),
                   "frac_abstained": ci("frac_abstained"),
                   "frac_correctable": ci("frac_correctable"),
                   "median_move_px": describe([r["median_move_px"] for r in sel]),
                   "abstain_runs_per_recording": describe(
                       [r["n_abstain_runs"] for r in sel]),
                   "frames_abstained_for_want_of_scale": int(
                       sum(r["n_no_scale"] for r in sel)),
                   "frames_abstained_for_want_of_a_donor": int(
                       sum(r["n_no_donor"] for r in sel)),
                   "frames_abstained_for_non_convergence": int(
                       sum(r["n_unconverged"] for r in sel))},
        "atom": {
            "n_corrected": int(sum(r["n_corrected"] for r in sel)),
            "n_landing_on_the_constraint_surface": int(
                sum(r["n_boundary"] for r in sel)),
            "frac_landing_on_the_constraint_surface": float(
                sum(r["n_boundary"] for r in sel)
                / max(1, sum(r["n_corrected"] for r in sel))),
            "why": ("a frame whose target was infeasible is projected, and lands "
                    "exactly on a codimension-1 surface. The first run put 100% "
                    "of corrections there; this is the count that says whether "
                    "the interior target fixed it")},
        "constrained_group_do_not_cite": {
            "skull_before": ci("skull_before"), "skull_after": ci("skull_after"),
            "why": ("a corrector that enforces skull constraints scoring well on "
                    "skull violations is measuring its own premise; recorded so "
                    "the circularity is visible, never as evidence")},
        "held_out_group": {"trunk_before": trunk_before,
                           "trunk_after": trunk_after,
                           "per_animal": ci("trunk_after"),
                           "constrained_pairs": [list(PAIRS[m]) for m in CONSTRAIN],
                           "leak": ("none: the corrector constrains the SKULL "
                                    "pairs and nothing else, so no TRUNK bone "
                                    "appears in its constraint set. Amendment 1 "
                                    "records the first run, where it did")},
        "reliability": ci("mean_reliability"),
        "q1_rescored": False,
        "n_animals": len(shards),
        "peak_rss_gb": peak_rss_gb(),
    }
    write_json(doc, args.out)
    log(gate.line())
    log(atom.line())
    log(f"  same probe BEFORE the correction: "
        f"{probe_before['balanced_accuracy']:.3f} balanced accuracy "
        f"(AUC {probe_before['auc']:.3f})")
    log(f"  corrected {doc['domain']['frac_corrected']['point']:.4%}  "
        f"abstained {doc['domain']['frac_abstained']['point']:.4%}")
    log(f"wrote {args.out}")
    return 0


def write_grid(args) -> int:
    tags = sorted(animals_of(spine.recording_ids()))
    os.makedirs(config.PATHS.grids_dir, exist_ok=True)
    path = config.PATHS.grid("disposition")
    with open(path, "w") as fh:
        fh.write("\n".join(tags) + "\n")
    log(f"wrote {path}: {len(tags)} animals")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--animal", default=None)
    p.add_argument("--task", type=int, default=None)
    p.add_argument("--n-tasks", type=int, default=30)
    p.add_argument("--split", default="report")
    p.add_argument("--write-grid", action="store_true")
    p.add_argument("--combine", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.write_grid:
        return write_grid(a)
    if a.combine:
        a.out = a.out or config.PATHS.result("disposition.json")
        return combine(a)
    if a.animal:
        tags = [a.animal]
    elif a.task is not None:
        with open(config.PATHS.grid("disposition")) as fh:
            tags = [ln.strip() for ln in fh if ln.strip()][a.task::a.n_tasks]
    else:
        raise SystemExit("pass --animal, --task, --write-grid or --combine")
    out_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "disposition")
    for tag in tags:
        if not a.force and os.path.exists(os.path.join(out_dir, f"{tag}.npz")):
            log(f"[{tag}] shard exists, skipping")
            continue
        shard(a, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
