"""Phase C. Before/after clips: what the cleaning does, where a viewer can see it.

    python3 scripts/compare_clips.py --check     # the gates, first
    python3 scripts/compare_clips.py --render

Five sets, and the second is what keeps the reel honest:

  worst        the highest violation-rate recordings
  random       a seeded stratified sample across the violation distribution --
               a reel of only worst cases overstates how bad the corpus is
  bakeoff      the same segments under several cleaning arms, so the methods are
               compared against each other and not each against raw
  residual     what the best arm STILL gets wrong, three ways
  disposition  Phase D: the frames the corrector moved, before and after, with
               the abstained frames beside them. A correction is 0.50% of frames
               and moves the suspect a median 0.19 body lengths, so this is the
               only place the move is visible rather than tabulated
  anipose      Anipose's Viterbi filter against the incumbent, on the frames it
               actually reassigned. It touches 0.31% of keypoint-frames, so a
               window centred anywhere else shows three identical skeletons

The violation masks are read from `work/bones/<animal>.npz`, bit-packed at Step 1
over that animal's concatenated frames. Recomputing them here would risk a second
implementation disagreeing with the published rate.
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

from recur import anchors, labels as lab                           # noqa: E402
from recur.render import video as vid                              # noqa: E402
from recur.util import log, read_json, write_json                  # noqa: E402
from vieb.clean import arms as clean_arms, viterbi as vit
from vieb.qc import bones                          # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.render import compare                                    # noqa: E402
from vieb.tok import config                                        # noqa: E402

EPS = 0.10
CELL = f"viol|unfiltered|raw|{EPS}|skull"
N_WORST = 6
N_RANDOM = 12
N_BAKEOFF = 4
WINDOWS_PER_RECORDING = 3
SEED = 0


def violation_mask(recording_id: str) -> np.ndarray:
    """The Step 1 skull mask for one recording, unpacked from its animal shard."""
    tag = lab.animal_tag(recording_id)
    with np.load(config.PATHS.bones_shard(tag), allow_pickle=False) as z:
        ids = [str(v) for v in z["recording_ids"]]
        r = ids.index(str(recording_id))
        lo, hi = int(z["bounds"][r]), int(z["bounds"][r + 1])
        total = int(z["bounds"][-1])
        return np.unpackbits(z[CELL])[:total][lo:hi].astype(bool)


def pick(args) -> dict:
    """The three sets, by name -> [(recording_id, rate)]."""
    d = read_json(config.PATHS.result("bones.json"))
    worst = d["worst_recordings"]
    rng = np.random.default_rng(SEED)

    chosen: dict = {"worst": [(w["recording_id"], w["rate"])
                              for w in worst[:N_WORST]]}

    # Stratified across the violation-rate distribution, not uniform: a uniform
    # sample of 3,846 recordings is a sample of the median and shows nothing.
    ids = spine.recording_ids()
    by_rate = {w["recording_id"]: w["rate"] for w in worst}
    pool = [r for r in ids if r not in by_rate]
    rng.shuffle(pool)
    chosen["random"] = [(r, float("nan")) for r in pool[:N_RANDOM]]
    chosen["bakeoff"] = chosen["worst"][:N_BAKEOFF]
    return chosen


def gates(args) -> int:
    ids = spine.recording_ids()
    obj = {"dataset": "luna", "arm": "compare", "split": "all"}
    av = vid.availability_read(ids, scored_object=obj, n_effective=len(ids))
    log(av.line())
    bounds = []
    off = 0
    for rid in ids:
        n = int(spine.clean(rid)["pose"].shape[0])
        bounds.append(off)
        off += n
    bounds.append(off)
    al = vid.alignment_read(ids, np.asarray(bounds), scored_object=obj,
                            n_effective=len(ids), sample=120, seed=SEED)
    log(al.line())
    write_json({"availability": av.to_dict(), "alignment": al.to_dict()},
               config.PATHS.result("compare_gate.json"))
    return 0 if al.verdict in ("PASS", "GRID_LIMITED") else 1


def render(args) -> int:
    gate_path = config.PATHS.result("compare_gate.json")
    if not os.path.exists(gate_path):
        raise SystemExit("run --check first: no clip is cut against an "
                         "unverified pose/video alignment")
    gate = read_json(gate_path)
    if gate["alignment"]["verdict"] not in ("PASS", "GRID_LIMITED"):
        raise SystemExit(f"alignment gate is {gate['alignment']['verdict']}")

    out_dir = os.path.join(config.PATHS.results_dir, "compare", "clips")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    sets = pick(args)
    manifest: list = []

    for kind, picks in sets.items():
        for rid, rate in picks:
            d = spine.clean(rid)
            unfiltered = d["pose_unfiltered"].astype(np.float64)
            wiener = d["pose"].astype(np.float64)
            mask = violation_mask(rid)
            wins = compare.windows(mask, fps=fps)
            if not wins:
                # No violation to centre on: take one window from the middle, so
                # the random set is not silently reduced to its dirty members.
                mid = mask.shape[0] // 2
                wins = [(mid, min(mid + int(4 * fps), mask.shape[0]))]
            wins = wins[:WINDOWS_PER_RECORDING if kind != "random" else 1]

            for w, (a, b) in enumerate(wins):
                if kind == "bakeoff":
                    held = clean_arms.held_array(unfiltered, d["missing"].astype(bool))
                    panes = [("raw", unfiltered), ("wiener", wiener),
                             ("median 0.50s",
                              clean_arms.apply("median_0.50", held, d["conf"], fps))]
                else:
                    panes = [("raw", unfiltered), ("wiener", wiener)]
                name = f"{kind}_{lab.animal_tag(rid)}_{w:02d}.mp4"
                path = os.path.join(out_dir, name)
                ok, why = compare.compare(
                    vid.video_path(rid), a, b, path, fps=fps, panes=panes,
                    crop_from=unfiltered, flags=mask)
                if not ok:
                    log(f"  SKIP {name}: {why}")
                    continue
                manifest.append({
                    "kind": kind, "file": os.path.join("clips", name),
                    "recording_id": rid, "animal": lab.animal_tag(rid),
                    "a": int(a), "b": int(b),
                    "duration_s": round((b - a) / fps, 2),
                    "panes": [p[0] for p in panes],
                    "violation_rate": None if not np.isfinite(rate) else rate,
                    "violations_in_clip": float(mask[a:b].mean()),
                    "bytes": os.path.getsize(path),
                })
                log(f"  {name}  {(b - a) / fps:.1f}s  "
                    f"{os.path.getsize(path) / 1024:.0f} KB")

    doc = {
        **anchors.header(anchors.LUNA, stage="compare",
                         unverified="a curated clip set, not a corpus statistic"),
        "inherited_digest": spine.digest(),
        "eps": EPS, "seed": SEED,
        "n_clips": len(manifest),
        "total_bytes": sum(m["bytes"] for m in manifest),
        "sets": {k: len(v) for k, v in sets.items()},
        "note": ("Left pane is the pose after the gap policy and before the "
                 "filter; right pane is the filtered array every downstream "
                 "stage consumes. Same frames, same crop box, same red outline "
                 "on violating frames -- the only difference between panes is "
                 "the skeleton."),
        "clips": manifest,
    }
    write_json(doc, os.path.join(config.PATHS.results_dir, "compare",
                                 "manifest.json"))
    log(f"{len(manifest)} clips, {doc['total_bytes'] / 1e6:.1f} MB")
    return 0


def residual(args) -> int:
    """What the best arm still gets wrong, and whether composing fixes it.

    The bakeoff ranks arms; it does not show what is LEFT after the best one. So
    this finds frames still violating the skull bound after `median_0.50` and
    renders them three ways -- raw, the smoother alone, and the de-glitcher
    composed ahead of it. If composition helps, this is where it would show.

    The reference length is refitted per arm, the same as the bakeoff, so a
    violation here means the same thing it means there.
    """
    out_dir = os.path.join(config.PATHS.results_dir, "compare", "clips")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    pairs = bones.pair_indices(anchors.LUNA.n_kept_keypoints)
    picks = [w["recording_id"] for w in
             read_json(config.PATHS.result("bones.json"))["worst_recordings"]][:args.n]

    manifest: list = []
    for rid in picks:
        d = spine.clean(rid)
        unfiltered = d["pose_unfiltered"].astype(np.float64)
        held = clean_arms.held_array(unfiltered, d["missing"].astype(bool))
        best = clean_arms.apply("median_0.50", held, d["conf"], fps)
        composed = clean_arms.apply(
            "median_0.50", clean_arms.apply("viterbi", held, d["conf"], fps),
            d["conf"], fps)

        keep = bones.rigid_pairs(bones.log_lengths(best, pairs), pairs)
        lengths = bones.metric_lengths(best, bones.SKULL, "raw",
                                       pairs=pairs, keep=keep)
        l_hat = np.array([bones.reference_length(lengths[:, m], EPS)["l_hat"]
                          for m in range(len(bones.SKULL))])
        mask = bones.frame_mask(bones.violations(lengths, l_hat, EPS))
        wins = compare.windows(mask, fps=fps)[:args.per_recording]
        if not wins:
            log(f"  {rid}: no residual violation at eps={EPS}")
            continue
        for w, (a, b) in enumerate(wins):
            name = f"residual_{lab.animal_tag(rid)}_{w:02d}.mp4"
            path = os.path.join(out_dir, name)
            ok, why = compare.compare(
                vid.video_path(rid), a, b, path, fps=fps,
                panes=[("raw", unfiltered), ("median 0.50s", best),
                       ("viterbi + median", composed)],
                crop_from=unfiltered, flags=mask)
            if not ok:
                log(f"  SKIP {name}: {why}")
                continue
            manifest.append({
                "kind": "residual", "file": os.path.join("clips", name),
                "recording_id": rid, "animal": lab.animal_tag(rid),
                "a": int(a), "b": int(b),
                "duration_s": round((b - a) / fps, 2),
                "panes": ["raw", "median 0.50s", "viterbi + median"],
                "violation_rate": None,
                "violations_in_clip": float(mask[a:b].mean()),
                "bytes": os.path.getsize(path),
            })
            log(f"  {name}  {(b - a) / fps:.1f}s  "
                f"{mask[a:b].mean():.1%} of frames still violating")

    path = os.path.join(config.PATHS.results_dir, "compare", "manifest.json")
    doc = read_json(path)
    doc["clips"] = [c for c in doc["clips"] if c["kind"] != "residual"] + manifest
    doc["sets"]["residual"] = len(manifest)
    doc["n_clips"] = len(doc["clips"])
    doc["total_bytes"] = sum(c["bytes"] for c in doc["clips"])
    write_json(doc, path)
    log(f"{len(manifest)} residual clips; manifest now {doc['n_clips']} total")
    return 0


def disposition(args) -> int:
    """Phase D before/after: what the corrector moved, and what it refused to.

    Windows centre on **corrected** frames rather than on violations, because the
    correction is the subject. The abstained frames are drawn in the same clip
    and are the contrast: a sustained run the corrector deliberately would not
    touch, next to a short one it did.
    """
    out_dir = os.path.join(config.PATHS.results_dir, "compare", "clips")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()
    dis_dir = os.path.join(os.path.dirname(config.PATHS.bones_dir), "disposition")

    # Pick by corrected mass: the recordings where there is something to see.
    rows: list = []
    for f in sorted(glob.glob(os.path.join(dis_dir, "*.npz"))):
        with np.load(f, allow_pickle=False) as z:
            for r in json.loads(str(z["rows_json"])):
                rows.append(r)
    rows.sort(key=lambda r: -r["frac_corrected"])
    picks = [r["recording_id"] for r in rows[:args.n]]

    manifest: list = []
    for rid in picks:
        tag = lab.animal_tag(rid)
        with np.load(os.path.join(dis_dir, f"{tag}.npz"), allow_pickle=False) as z:
            ids = [str(v) for v in z["recording_ids"]]
            k = ids.index(str(rid))
            lo, hi = int(z["bounds"][k]), int(z["bounds"][k + 1])
            after = z["pose"][lo:hi].astype(np.float64)
            corrected = z["corrected"][lo:hi].astype(bool)
            abstain = z["abstain"][lo:hi].astype(bool)
        d = spine.clean(rid)
        before = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                       d["missing"].astype(bool))
        if before.shape[0] != after.shape[0]:
            log(f"  SKIP {rid}: {before.shape[0]} != {after.shape[0]} frames")
            continue
        wins = compare.windows(corrected, fps=fps)[:args.per_recording]
        if not wins:
            log(f"  {rid}: nothing corrected")
            continue
        for w, (a, b) in enumerate(wins):
            name = f"disposition_{tag}_{w:02d}.mp4"
            path = os.path.join(out_dir, name)
            ok, why = compare.compare(
                vid.video_path(rid), a, b, path, fps=fps,
                panes=[("before", before), ("corrected", after)],
                crop_from=before, flags=corrected | abstain)
            if not ok:
                log(f"  SKIP {name}: {why}")
                continue
            moved = float(np.linalg.norm(after[a:b] - before[a:b],
                                         axis=-1).max()) if b > a else 0.0
            manifest.append({
                "kind": "disposition", "file": os.path.join("clips", name),
                "recording_id": rid, "animal": tag,
                "a": int(a), "b": int(b),
                "duration_s": round((b - a) / fps, 2),
                "panes": ["before", "corrected"],
                "violation_rate": None,
                "violations_in_clip": float(corrected[a:b].mean()),
                "abstained_in_clip": float(abstain[a:b].mean()),
                "max_move_px": round(moved, 2),
                "bytes": os.path.getsize(path),
            })
            log(f"  {name}  {(b - a) / fps:.1f}s  "
                f"{corrected[a:b].mean():.1%} corrected, "
                f"{abstain[a:b].mean():.1%} abstained, max move {moved:.1f} px")

    path = os.path.join(config.PATHS.results_dir, "compare", "manifest.json")
    doc = read_json(path)
    doc["clips"] = [c for c in doc["clips"]
                    if c["kind"] != "disposition"] + manifest
    doc["sets"]["disposition"] = len(manifest)
    doc["n_clips"] = len(doc["clips"])
    doc["total_bytes"] = sum(c["bytes"] for c in doc["clips"])
    write_json(doc, path)
    log(f"{len(manifest)} disposition clips; manifest now {doc['n_clips']} total")
    return 0


def anipose(args) -> int:
    """Anipose's Viterbi de-glitcher, seen rather than tabulated.

    `CLEANING.md` calls this the efficiency outlier -- 16.9% of violation
    reduction per pixel of displacement against the incumbent's 4.1%, retaining
    0.586 of the power above f_c against 0.220 -- and no clip on the Atlas has
    ever shown it on its own. `compare/bakeoff` shows raw/wiener/median and
    `compare/residual` shows it only composed with a median.

    **Cut on the frames it reassigned.** It moves 0.31% of keypoint-frames. A
    window chosen any other way is three identical skeletons and shows nothing.
    """
    out_dir = os.path.join(config.PATHS.results_dir, "compare", "clips")
    os.makedirs(out_dir, exist_ok=True)
    fps = spine.fps()

    # Rank by reassigned mass, over a seeded sample -- running Viterbi on all
    # 3,846 recordings to pick six of them is not a reasonable way to pick six.
    rng = np.random.default_rng(SEED)
    pool = list(spine.recording_ids())
    rng.shuffle(pool)
    scored: list = []
    for rid in pool[:args.scan]:
        d = spine.clean(rid)
        held = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                     d["missing"].astype(bool))
        after = clean_arms.apply("viterbi", held, d["conf"], fps)
        moved = np.linalg.norm(after - held, axis=-1) > 1e-9
        scored.append((float(moved.mean()), rid))
    scored.sort(reverse=True)
    log(f"scanned {len(scored)} recordings; top reassigned mass "
        f"{scored[0][0]:.4%}, median {np.median([s for s, _ in scored]):.4%}")

    manifest: list = []
    for rate, rid in scored[:args.n]:
        d = spine.clean(rid)
        held = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                     d["missing"].astype(bool))
        wiener = d["pose"].astype(np.float64)
        after = clean_arms.apply("viterbi", held, d["conf"], fps)
        info = vit.reassignment(held, after)
        moved = np.linalg.norm(after - held, axis=-1) > 1e-9
        frame_moved = moved.any(axis=1)
        wins = compare.windows(frame_moved, fps=fps)[:args.per_recording]
        if not wins:
            log(f"  {rid}: Viterbi reassigned nothing")
            continue
        for w, (a, b) in enumerate(wins):
            name = f"anipose_{lab.animal_tag(rid)}_{w:02d}.mp4"
            path = os.path.join(out_dir, name)
            ok, why = compare.compare(
                vid.video_path(rid), a, b, path, fps=fps,
                panes=[("raw", held), ("anipose viterbi", after),
                       ("wiener", wiener)],
                crop_from=held, flags=frame_moved)
            if not ok:
                log(f"  SKIP {name}: {why}")
                continue
            manifest.append({
                "kind": "anipose", "file": os.path.join("clips", name),
                "recording_id": rid, "animal": lab.animal_tag(rid),
                "a": int(a), "b": int(b),
                "duration_s": round((b - a) / fps, 2),
                "panes": ["raw", "anipose viterbi", "wiener"],
                "violation_rate": None,
                "violations_in_clip": float(frame_moved[a:b].mean()),
                "reassigned_in_recording": float(info["reassigned"]),
                "max_move_px": round(float(info["max_move_px"]), 2),
                "bytes": os.path.getsize(path),
            })
            log(f"  {name}  {(b - a) / fps:.1f}s  "
                f"{frame_moved[a:b].mean():.1%} of frames reassigned, "
                f"max move {info['max_move_px']:.1f} px")

    path = os.path.join(config.PATHS.results_dir, "compare", "manifest.json")
    doc = read_json(path)
    doc["clips"] = [c for c in doc["clips"] if c["kind"] != "anipose"] + manifest
    doc["sets"]["anipose"] = len(manifest)
    doc["n_clips"] = len(doc["clips"])
    doc["total_bytes"] = sum(c["bytes"] for c in doc["clips"])
    write_json(doc, path)
    log(f"{len(manifest)} anipose clips; manifest now {doc['n_clips']} total")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--residual", action="store_true",
                   help="frames the best arm STILL gets wrong, three ways")
    p.add_argument("--disposition", action="store_true",
                   help="Phase D: what the corrector moved, before and after")
    p.add_argument("--anipose", action="store_true",
                   help="Anipose Viterbi, on the frames it actually reassigned")
    p.add_argument("--scan", type=int, default=60,
                   help="recordings to scan when ranking by reassigned mass")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--per-recording", type=int, default=2)
    a = p.parse_args(argv)
    if a.check:
        return gates(a)
    if a.render:
        return render(a)
    if a.residual:
        return residual(a)
    if a.disposition:
        return disposition(a)
    if a.anipose:
        return anipose(a)
    raise SystemExit("pass --check, --render, --residual, --disposition "
                     "or --anipose")


if __name__ == "__main__":
    raise SystemExit(main())
