"""Does the island look like one thing? A blind odd-one-out test.

    python3 scripts/island_look.py --build            # trials, clips, sheets, key
    #   ... read the sheets, write scores.json ...
    python3 scripts/island_look.py --score scores.json
    python3 scripts/island_look.py --manifest   # blind manifest for the site

READ results/ISLAND_LOOK_PREREGISTRATION.md FIRST.

## The build step's stdout is part of the blind

The builder and the scorer are the same process here. Anything printed in trial
order -- which position holds the odd clip, per-arm counts as trials are made --
leaks the answer into the scorer's context as surely as opening the key would.
**This script prints totals only**, and every per-trial log line is deliberately
absent rather than merely quiet.

## Why the control is not drawn from the island member's own recording

It would be the tighter nuisance control -- same box, same light, same animal.
In a TRIAD it puts two clips from one scene into the trial, and a scorer who
notices groups those two and picks the third, which is the wrong answer
systematically. Three distinct animals per trial instead, matched corpus-wide.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, splits                                   # noqa: E402
from recur.render import video as vid                               # noqa: E402
from recur.util import log, write_json                              # noqa: E402
from vieb.clean import arms as clean_arms                           # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.seg import breaks as bk, embed                            # noqa: E402
from vieb.seg import triad as tr                                    # noqa: E402
from vieb.tok import config                                         # noqa: E402

GROUP, K_MAD, CLUMP = "shape", 3.0, 0
SEED = 0
#: Frames shown per clip on a contact sheet, per overlay variant.
SHEET_FRAMES = 4
TILE = 240
#: The lift over chance that would count as a real effect, fixed in the
#: registration so the MDE has something to be compared against.
PLAUSIBLE_LIFT = 0.20
#: Clip length cap. The island runs to 65 s and nobody reviews a 65-second clip;
#: the cap is applied to EVERY arm so it cannot separate them.
MAX_CLIP_S = 6.0


def out_dir() -> str:
    return os.path.join(config.PATHS.results_dir, "island_look")


def _gate() -> None:
    """Refuse to cut against an unverified pose/video alignment."""
    p = config.PATHS.result("compare_gate.json")
    if not os.path.exists(p):
        raise SystemExit("run compare_clips.py --check first: no clip is cut "
                         "against an unverified pose/video alignment")
    with open(p, encoding="utf-8") as fh:
        g = json.load(fh)
    for k in ("availability", "alignment"):
        v = str(g.get(k, {}).get("verdict"))
        if v not in ("PASS", "GRID_LIMITED"):
            raise SystemExit(f"{k} gate is {v}; clips are not cut against it")


def _sr():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg_recur.py")
    spec = importlib.util.spec_from_file_location("sr_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["seg_recur"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def segments() -> tuple[list, list]:
    """Every selectable report segment, with its speed, split by island membership.

    Speed is recomputed because it is persisted nowhere: columns 14 and 15, omega
    excluded, as `quantize.speed` defines it -- a spin is not a displacement and
    adding it would make a turning animal fast.
    """
    sr = _sr()
    fps = spine.fps()
    sd = sr.basis_sd()
    idx = bk.CHANNEL_GROUPS[GROUP]
    split_of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    with np.load(os.path.join(config.PATHS.tok_dir, "seg_vocab",
                              f"{GROUP}__k{K_MAD:g}__corpus.npz"),
                 allow_pickle=False) as z:
        lab = z["labels"]
        an = np.asarray([str(v) for v in z["animal"]])
        rec, frm = z["rec"], z["frame"]
    want = {(an[i], int(rec[i]), int(frm[i])): int(lab[i])
            for i in range(lab.size)}
    island, pool = [], []
    for tag in sorted(set(an.tolist())):
        if split_of.get(tag) != "report":
            continue
        a = sr.load_arm("corpus", tag, sd)
        segs = sr.segments_of(a, idx, fps=fps, k_mad=K_MAD)
        keep = embed.selectable(segs)
        with np.load(os.path.join(config.REPO, "work", "ego",
                                  f"raw__bodylen__{tag}.npz"),
                     allow_pickle=False) as e:
            rids = [str(v) for v in e["recording_ids"]]
        for r, k in zip(segs, keep):
            if not k:
                continue
            key = (tag, int(r["rec"]), int(r["start"]))
            if key not in want:
                continue
            blk = a["X"][int(r["start"]):int(r["stop"])]
            lo = int(a["bounds"][int(r["rec"])])
            row = {"animal": tag, "recording_id": rids[int(r["rec"])],
                   "a": int(r["start"]) - lo, "b": int(r["stop"]) - lo,
                   "n_frames": int(r["n_frames"]),
                   "duration_s": float(r["n_frames"]) / fps,
                   "speed": float(np.sqrt((blk[:, 14:16] ** 2).sum(1)).mean())}
            (island if want[key] == CLUMP else pool).append(row)
        del a
    return island, pool


def build(args) -> int:
    import cv2

    _gate()
    os.makedirs(os.path.join(out_dir(), "clips"), exist_ok=True)
    fps = spine.fps()
    island, pool = segments()
    log(f"  {len(island):,} island segments, {len(pool):,} control candidates")

    rng = np.random.default_rng(SEED)
    trials: list = []
    for arm in tr.ARMS:
        trials += tr.plan_trials(island, pool, arm=arm,
                                 n_trials=tr.N_TRIALS // 2, rng=rng,
                                 n_near=5)
    # Shuffle so the file index carries no information about the arm, and so
    # the two arms are interleaved in the order the scorer meets them.
    order = rng.permutation(len(trials))
    trials = [trials[int(i)] for i in order]
    for n, t in enumerate(trials):
        t["n"] = n

    pose_cache: dict = {}
    made = 0
    for t in trials:
        # Equal frame count within a trial, trimmed to the shortest member and
        # capped: "if the odd one is the short one, a rater picks it by length
        # and the experiment measures a stopwatch."
        w = min(int(c["n_frames"]) for c in t["clips"])
        w = int(min(w, MAX_CLIP_S * fps))
        t["clip_frames"] = w
        tiles_plain, tiles_skel = [], []
        ok = True
        for j, c in enumerate(t["clips"]):
            rid = c["recording_id"]
            if rid not in pose_cache:
                d = spine.clean(rid)
                pose_cache[rid] = clean_arms.held_array(
                    d["pose_unfiltered"].astype(np.float64),
                    d["missing"].astype(bool))
                if len(pose_cache) > 40:
                    pose_cache.pop(next(iter(pose_cache)))
            pose = pose_cache[rid]
            a0, b0 = int(c["a"]), int(c["a"]) + w
            if b0 > pose.shape[0]:
                ok = False
                break
            box = vid.crop_box(pose, a0, b0, size=vid.CROP)
            for kind, ov in (("plain", None), ("skel", pose)):
                dst = os.path.join(out_dir(), "clips",
                                   f"{t['id']}_{j}_{kind}.mp4")
                got, why = vid.cut(vid.video_path(rid), a0, b0, dst, fps=fps,
                                   pose=ov, flags=None, label=None, crop=box)
                if not got:
                    ok = False
                    break
            if not ok:
                break
            tiles_plain.append(_frames(rid, a0, b0, pose, cv2, skel=False))
            tiles_skel.append(_frames(rid, a0, b0, pose, cv2, skel=True))
        if not ok:
            t["rendered"] = False
            continue
        t["rendered"] = True
        _sheet(t, tiles_plain, tiles_skel, cv2)
        made += 1

    kept = [t for t in trials if t.get("rendered")]
    write_json({**anchors.header(anchors.LUNA, stage="island_look",
                                 unverified="a curated trial set"),
                "inherited_digest": spine.digest(),
                "registration": "results/ISLAND_LOOK_PREREGISTRATION.md",
                "group": GROUP, "clump": CLUMP, "k_mad": K_MAD,
                "chance": tr.CHANCE, "seed": SEED,
                "max_clip_s": MAX_CLIP_S,
                "blind": ("filenames are a shuffled index; this key is not read "
                          "until scores are supplied; the builder prints totals "
                          "only, because its stdout is part of the blind"),
                "n_trials": len(kept), "trials": kept},
               os.path.join(out_dir(), "key.json"))
    # Totals only. No per-trial line, no per-arm count in trial order.
    log(f"  rendered {made} trials of {len(trials)} planned")
    log("KEY SEALED. Score the sheets, then: --score scores.json")
    return 0


def _frames(rid: str, a0: int, b0: int, pose, cv2, *, skel: bool) -> list:
    """`SHEET_FRAMES` evenly spaced stills, cropped on the animal."""
    cap = cv2.VideoCapture(vid.video_path(rid))
    out = []
    for t in np.linspace(a0, b0 - 1, SHEET_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        got, frame = cap.read()
        if not got or frame is None:
            out.append(np.zeros((TILE, TILE, 3), np.uint8))
            continue
        pts = pose[int(t)]
        if skel:
            vid.draw_skeleton(frame, pts, cv2, flagged=False)
        c = np.nanmean(pts, axis=0)
        h, w = frame.shape[:2]
        x = int(np.clip(c[0] - vid.CROP // 2, 0, max(0, w - vid.CROP)))
        y = int(np.clip(c[1] - vid.CROP // 2, 0, max(0, h - vid.CROP)))
        out.append(cv2.resize(frame[y:y + vid.CROP, x:x + vid.CROP],
                              (TILE, TILE)))
    cap.release()
    return out


def _sheet(t, plain: list, skel: list, cv2) -> None:
    """One image per trial: three clips, each as plain then skeleton rows.

    Both variants in one sheet because the judgement is one judgement -- the
    site gets a toggle over the same two renderings, so the scorer and the
    reviewer see the same information.
    """
    rows = []
    for j in range(len(plain)):
        for kind, tiles in (("A B C"[2 * j], plain[j]), ("", skel[j])):
            strip = np.hstack(tiles)
            if kind:
                cv2.rectangle(strip, (0, 0), (46, 22), (0, 0, 0), -1)
                cv2.putText(strip, kind, (8, 17), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
            rows.append(strip)
        rows.append(np.full((3, plain[j][0].shape[1] * SHEET_FRAMES, 3),
                            60, np.uint8))
    cv2.imwrite(os.path.join(out_dir(), f"trial_{t['n']:02d}.png"),
                np.vstack(rows[:-1]))


def score(args) -> int:
    with open(os.path.join(out_dir(), "key.json"), encoding="utf-8") as fh:
        key_doc = json.load(fh)
    with open(args.score, encoding="utf-8") as fh:
        given = json.load(fh)
    by_n = {int(t["n"]): t for t in key_doc["trials"]}
    rows: list = []
    for k, v in given.items():
        n = int(k)
        if n not in by_n:
            raise SystemExit(f"a score for trial {n}, which was not rendered")
        if str(v).upper() not in ("A", "B", "C"):
            raise SystemExit(f"{v!r} is not one of A, B, C")
        t = by_n[n]
        pick = "ABC".index(str(v).upper())
        rows.append({"n": n, "id": t["id"], "arm": t["arm"], "type": t["type"],
                     "response": str(v).upper(),
                     "correct": bool(pick == int(t["odd_position"])),
                     "animals": t["animals"], "odd_is": t["odd_is"]})

    obj = {"dataset": "luna", "arm": "island_look", "group": GROUP,
           "clump": CLUMP, "pose_arm": "raw", "split": "report",
           "chance": tr.CHANCE}
    reads: dict = {}
    # The MDE FIRST, before any accuracy is looked at.
    reads["mde"] = tr.mde_read(len(rows), scored_object=obj,
                               n_effective=len(rows),
                               plausible=PLAUSIBLE_LIFT).to_dict()
    log("  " + tr.mde_read(len(rows), scored_object=obj, n_effective=len(rows),
                           plausible=PLAUSIBLE_LIFT).line())
    for arm in tr.ARMS:
        for ttype in (*tr.TYPES, "both"):
            sel = [r for r in rows if r["arm"] == arm
                   and (ttype == "both" or r["type"] == ttype)]
            if not sel:
                continue
            rd = tr.accuracy_read([r["correct"] for r in sel],
                                  [r["animals"][0] for r in sel], arm=arm,
                                  ttype=ttype,
                                  scored_object={**obj, "control": arm,
                                                 "trial_type": ttype},
                                  n_effective=len({a for r in sel
                                                   for a in r["animals"]}),
                                  seed=SEED)
            reads[f"{arm}|{ttype}"] = rd.to_dict()
            log("  " + rd.line())
    write_json({**anchors.header(anchors.LUNA, stage="island_look_score",
                                 unverified="a curated trial set"),
                "inherited_digest": spine.digest(),
                "registration": "results/ISLAND_LOOK_PREREGISTRATION.md",
                "chance": tr.CHANCE, "plausible_lift": PLAUSIBLE_LIFT,
                "n_scored": len(rows), "reads": reads, "responses": rows},
               config.PATHS.result("island_look.json"))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--score", default=None)
    p.add_argument("--manifest", action="store_true",
                   help="write the blind publication manifest for the site")
    a = p.parse_args(argv)
    if a.build:
        return build(a)
    if a.score:
        return score(a)
    if a.manifest:
        return manifest(a)
    raise SystemExit("pass --build, --score or --manifest")




# --------------------------------------------------------------------------
# The publication manifest — the same trials, with the answers removed
# --------------------------------------------------------------------------

#: Every key of a key.json trial that would tell a reader the answer, or let
#: them infer it. `arm` and `type` are not the answer, but they partition the
#: trials into the two questions the design asks and a reader who sees them can
#: treat the arms differently; `speed` is the island's single strongest cue at
#: 0.205x, so a published speed is the answer in a column.
LEAKS = ("odd_position", "odd_is", "arm", "type", "index", "speed",
         "a", "b", "n_frames")


def manifest(args) -> int:
    """Write the blind publication manifest over the clips already rendered.

    The site publishes the instrument, not the result: trials in key order, no
    arm, no trial type, no odd position, nothing that separates an island clip
    from a control. That is what makes the published page scoreable by someone
    who is not this session — which, after this session failed the positive
    control, is the only way the question gets answered.
    """
    with open(os.path.join(out_dir(), "key.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    fps = spine.fps()
    rows: list = []
    for t in doc["trials"]:
        secs = round(int(t["clip_frames"]) / fps, 3)
        for j, c in enumerate(t["clips"]):
            rel = os.path.join("clips", f"{t['id']}_{j}_plain.mp4")
            skel = os.path.join("clips", f"{t['id']}_{j}_skel.mp4")
            if not all(os.path.exists(os.path.join(out_dir(), p))
                       for p in (rel, skel)):
                continue
            rows.append({
                "trial": int(t["n"]), "id": t["id"], "pos": "ABC"[j],
                "file": rel, "skel": skel,
                "animal": c["animal"], "recording_id": c["recording_id"],
                # The CLIP's duration, which is equal across the three members
                # by construction, not the source segment's -- that differs by
                # class and would be the cue `hstack3` exists to remove.
                "duration_s": secs,
                "bytes": os.path.getsize(os.path.join(out_dir(), rel)),
            })
    for r in rows:
        assert not (set(r) & set(LEAKS)), sorted(set(r) & set(LEAKS))
    out = os.path.join(out_dir(), "manifest.json")
    write_json({**anchors.header(anchors.LUNA, stage="island_look_manifest",
                                 unverified="a curated trial set"),
                "inherited_digest": spine.digest(),
                "registration": "results/ISLAND_LOOK_PREREGISTRATION.md",
                "blind": ("trials in key order; no arm, no trial type, no odd "
                          "position, no per-segment speed"),
                "n_trials": len({r["trial"] for r in rows}),
                "clips": rows}, out)
    log(f"  {len(rows)} clips across {len({r['trial'] for r in rows})} trials")
    log(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
