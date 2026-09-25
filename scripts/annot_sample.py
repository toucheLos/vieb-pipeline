"""Step 1a. Draw the ten minutes a human will annotate, and encode it.

    python3 scripts/annot_sample.py [--seconds 10] [--per-decile 6]

READ results/ANNOTATION_PREREGISTRATION.md FIRST. The sample is fixed there:
60 clips x 10 s, one per TUNE animal, 6 per arena-position decile, abstain
fraction under 0.10, no seam crossed, seed 0.

## Three choices that are not free, and why

**Tune animals, not report.** Step 3 selects its model class by human-boundary
recall, which is fitting. Boundaries fitted against report animals would
contaminate Step 5's coverage number on the report split. The three-way split
exists for exactly this.

**Plain video, no skeleton.** The instruction is about the animal, not the
tracker. An overlay invites a rater to mark tracking failures instead of
behaviour. Flag density goes into the manifest and is NOT shown, so boundaries
piling up on flagged frames can be detected afterwards rather than caused
beforehand.

**All-intra.** The finest tolerance band is +/-2 frames -- 67 ms at 30 fps.
Browsers seek accurately only to keyframes, so an ordinary GOP would put the
instrument's own measurement error at the scale of the thing it measures. This
is the one place a clip in this repo is not encoded like every other clip, and
`-g 1 -bf 0` is the reason.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab, splits                    # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.render import video as vid                               # noqa: E402
from recur.util import frames, log, write_json                      # noqa: E402
from vieb import seeds                                              # noqa: E402
from vieb.clean import arms as clean_arms                           # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.qc import concentration as cc                             # noqa: E402
from vieb.tok import config                                         # noqa: E402

SEED = 0
#: Column of the ego/pose array holding the body centre, as
#: `scripts/concentration.py:43` defines it. Edgeness is computed on this and
#: nothing else.
CENTRE = 3
#: The registered sample.
SECONDS = 10.0
PER_DECILE = 6
N_DECILES = 10
#: A window with more abstained frames than this is not offered to a rater. A
#: clip of lost tracking measures the tracker, not the animal.
MAX_ABSTAIN = 0.10
#: Candidate windows per animal, before the decile stratification chooses
#: between them. More than one is what lets the stratification do any work.
PER_ANIMAL = 12
FFMPEG = "/home/tul26194/bin/ffmpeg"


def out_dir() -> str:
    return os.path.join(config.PATHS.results_dir, "annot")


def _windows(rid: str, w: int, rng: np.random.Generator) -> list[dict]:
    """Candidate windows in one recording, with edgeness and flag density.

    Never crosses a recording boundary: this reads one recording's own arrays
    and indexes only inside them.
    """
    d = spine.clean(rid)
    held = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                 d["missing"].astype(bool))
    n = int(held.shape[0])
    if n < w * 2:
        return []
    edge = cc.edgeness(held[:, CENTRE])
    bone = np.asarray(d["bone_flagged"], dtype=bool)
    miss = np.asarray(d["missing"], dtype=bool).any(axis=1)
    interp = np.asarray(d["interpolated"], dtype=bool).any(axis=1)
    bad = bone | miss | interp
    # Abstain is the union the rest of the programme uses; see
    # recur/render/meanskel.py:232.
    starts = np.arange(0, n - w, w)
    rng.shuffle(starts)
    out: list[dict] = []
    for s in starts[:40]:
        s = int(s)
        sl = slice(s, s + w)
        ab = float(bad[sl].mean())
        if ab > MAX_ABSTAIN:
            continue
        e = edge[sl]
        if not np.isfinite(e).any():
            continue
        out.append({"recording_id": rid, "animal": lab.animal_tag(rid),
                    "a": s, "b": s + w, "n_frames": w,
                    "edgeness_median": float(np.nanmedian(e)),
                    "abstain_frac": ab,
                    "flag_frac": float(bone[sl].mean())})
    return out


def _encode(rid: str, a: int, b: int, dst: str, *, fps: float, crop) -> bool:
    """All-intra H.264, plain, no overlay. See the module docstring.

    `vid.cut` is the encoder everywhere else here and is the only one that
    produces playable H.264 on this machine, but it has no all-intra option, so
    this re-encodes its output rather than reimplementing the pipe.
    """
    tmp = dst + ".gop.mp4"
    got, why = vid.cut(vid.video_path(rid), a, b, tmp, fps=fps,
                       pose=None, flags=None, label=None, crop=crop)
    if not got:
        log(f"  SKIP {rid} [{a},{b}): {why}")
        return False
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", tmp,
                    "-c:v", "libx264", "-g", "1", "-bf", "0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                    dst], check=True)
    os.remove(tmp)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=SECONDS)
    p.add_argument("--per-decile", type=int, default=PER_DECILE)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    os.makedirs(os.path.join(out_dir(), "clips"), exist_ok=True)
    fps = spine.fps()
    w = frames(a.seconds, fps)

    doc = splits.load(spine.sf("results/splits.json"))
    split_of = splits.split_of_animal(doc)
    by_animal: dict[str, list[str]] = {}
    for rid in spine.recording_ids():
        tag = lab.animal_tag(rid)
        if split_of.get(tag) == "tune":
            by_animal.setdefault(tag, []).append(rid)
    tags = sorted(by_animal)
    log(f"  {len(tags)} tune animals, window {w} frames ({a.seconds:g} s)")

    # The decile edges are the CORPUS's, read from the published artifact, not
    # quantiles of whatever 60 windows happen to be in hand. That is what makes
    # this sample stratified against the distribution CONCENTRATION.md measured
    # the 3.7x rise across, rather than against itself.
    with open(config.PATHS.result("concentration.json"), encoding="utf-8") as fh:
        prof = json.load(fh)["edge_profile"]
    cuts = np.asarray([b["hi_iqr"] for b in prof[:-1]], dtype=np.float64)
    ceiling = float(prof[-1]["hi_iqr"])
    log(f"  corpus deciles {prof[0]['lo_iqr']:.3f} -> {ceiling:.3f} IQR")

    # Many candidate windows per animal, so the stratification chooses rather
    # than merely labels. One recording per animal, and at most one window per
    # animal survives: "twenty clips of one animal is one observation, not
    # twenty" (recur.render.select.one_per_recording).
    rng = np.random.default_rng(seeds.stable_seed(SEED, "annot"))
    cands: list[dict] = []
    n_far = 0
    for tag in tags:
        rid = sorted(by_animal[tag])[int(rng.integers(len(by_animal[tag])))]
        got = _windows(rid, w, np.random.default_rng(
            seeds.stable_seed(SEED, "annotwin", tag)))
        for c in got[:PER_ANIMAL]:
            # Beyond the corpus's own top decile the centroid has left the
            # animal: this is a tracking failure wearing the costume of a mouse
            # at the wall. Excluded, and the bound is the published profile's
            # own upper edge rather than a number chosen here. Recorded as an
            # amendment in ANNOTATION.md -- added before any rater saw anything.
            if not np.isfinite(c["edgeness_median"]) or \
                    c["edgeness_median"] > ceiling:
                n_far += 1
                continue
            c["decile"] = int(np.searchsorted(cuts, c["edgeness_median"],
                                              side="right"))
            cands.append(c)
    log(f"  {len({c['animal'] for c in cands})} animals, {len(cands)} windows "
        f"({n_far} beyond the corpus's top decile, dropped)")

    chosen: list[dict] = []
    taken: set[str] = set()
    pick = np.random.default_rng(seeds.stable_seed(SEED, "annotpick"))
    by_dec = {d: [c for c in cands if c["decile"] == d]
              for d in range(N_DECILES)}
    # Scarcest decile first. At most one window per animal, so a decile filled
    # late competes for animals the earlier ones already took -- and the
    # scarcest decile is the WALL, which is the regime this stratification
    # exists to protect. Filling in index order starved decile 9 to 3 of 6.
    # Ties broken by decile index so the order is a function of the data alone.
    order = sorted(range(N_DECILES), key=lambda d: (len(by_dec[d]), d))
    for dec in order:
        pool = sorted(by_dec[dec], key=lambda r: (r["recording_id"], r["a"]))
        pick.shuffle(pool)
        got = 0
        for c in pool:
            if c["animal"] in taken:
                continue
            taken.add(c["animal"])
            chosen.append(c)
            got += 1
            if got >= a.per_decile:
                break
        log(f"  decile {dec}: {got} of {len(pool)} windows, "
            f"{a.per_decile} wanted")

    rows: list[dict] = []
    for k, c in enumerate(sorted(chosen, key=lambda r: (r["decile"],
                                                        r["recording_id"]))):
        cid = f"a{k:03d}"
        rel = os.path.join("clips", f"{cid}.mp4")
        if not a.dry_run:
            d = spine.clean(c["recording_id"])
            held = clean_arms.held_array(
                d["pose_unfiltered"].astype(np.float64),
                d["missing"].astype(bool))
            box = vid.crop_box(held, c["a"], c["b"], size=vid.CROP)
            if not _encode(c["recording_id"], c["a"], c["b"],
                           os.path.join(out_dir(), rel), fps=fps, crop=box):
                continue
        rows.append({"clip": cid, "file": rel, "fps": float(fps), **c})

    write_json({**provenance.header(anchors.LUNA, stage="annot_sample",
                                 unverified="a curated sample"),
                "inherited_digest": spine.digest(),
                "registration": "results/ANNOTATION_PREREGISTRATION.md",
                "split": "tune", "seconds": a.seconds, "window_frames": w,
                "per_decile": a.per_decile, "n_deciles": N_DECILES,
                "max_abstain": MAX_ABSTAIN, "seed": SEED,
                "decile_edges": "results/concentration.json#/edge_profile",
                "edgeness_ceiling_iqr": ceiling,
                "encoding": "libx264 -g 1 -bf 0 (all-intra, frame-accurate seek)",
                "overlay": "none; flag density recorded, never shown",
                "n_clips": len(rows),
                # Per decile, so a shortfall is visible rather than implied by
                # a total. Deciles are filled scarcest-first and each animal is
                # used once, so the last decile filled is the one that comes up
                # short -- by design that is the most abundant one, never the
                # wall.
                "per_decile_realised": {
                    str(d): sum(1 for r in rows if r["decile"] == d)
                    for d in range(N_DECILES)},
                "clips": rows},
               os.path.join(out_dir(), "manifest.json"))
    total = sum(r["n_frames"] for r in rows) / fps
    log(f"  {len(rows)} clips, {total / 60:.1f} minutes, wrote "
        f"{os.path.join(out_dir(), 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
