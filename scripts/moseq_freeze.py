"""MOSEQ §7: per-frame pixel freezing on every report recording of days 3-7, A/B.

    python3 scripts/moseq_freeze.py --phase manifest
    sbatch jobs/moseq_freeze.slurm          # --phase shard, array 0-11
    # writes work/moseq/freeze/<rid>.npz

The motion scan is `pixel_pilot.py`'s (`vieb.pixel.motion.scan`, full frame,
`mt_cutoff` = 10 grey levels). Freezing is `freeze.freeze_mask` at the
registered FreezeThresh = 800 changed pixels (ezTrack's 200, area-scaled to
640 x 480) and MinDuration = 15 frames. Motion is a frame DIFFERENCE, so it has
T - 1 entries; frame t >= 1 takes motion[t - 1] and frame 0 is never freezing,
as in ezTrack.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab, splits                     # noqa: E402
from recur.render import video as vid                                # noqa: E402
from recur.util import log, write_json                               # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.pixel import freeze as fz, motion as mo                    # noqa: E402
from vieb.tok import config, ego as tego                             # noqa: E402

DAYS = (3, 4, 5, 6, 7)
CONTEXTS = ("A", "B")
MT_CUTOFF = 10.0
FREEZE_THRESH = 800.0
MIN_FRAMES = 15
OUT = os.path.join(config.REPO, "work", "moseq", "freeze")
MANIFEST = os.path.join(config.REPO, "work", "moseq", "freeze_manifest.json")


def frame_freeze(motion: np.ndarray, n_frames: int) -> np.ndarray:
    """``(n_frames,)`` freezing per frame from the ``(n_frames - 1,)`` motion."""
    out = np.zeros(n_frames, dtype=bool)
    out[1:] = fz.freeze_mask(motion, FREEZE_THRESH, MIN_FRAMES)[:n_frames - 1]
    return out


def manifest(a) -> int:
    split_of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    rows = []
    for rid in spine.recording_ids():
        p = lab.parse(rid)
        if split_of.get(lab.animal_tag(rid)) != "report":
            continue
        if str(p["context_letter"]) in CONTEXTS and int(p["day"]) in DAYS:
            rows.append({"recording_id": rid, "animal": lab.animal_tag(rid),
                         "context": str(p["context_letter"]), "day": int(p["day"])})
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    write_json({**provenance.header(anchors.LUNA, stage="moseq_freeze_manifest",
                                   unverified="a report-split subset (days 3-7, A/B)"),
                "registration": "results/MOSEQ_PREREGISTRATION.md §7",
                "n_recordings": len(rows),
                "n_animals": len({r["animal"] for r in rows}),
                "recordings": rows}, MANIFEST)
    log(f"  {len(rows)} report recordings, days 3-7 x A/B -> {MANIFEST}")
    return 0


def shard(a) -> int:
    with open(MANIFEST, encoding="utf-8") as fh:
        rows = json.load(fh)["recordings"][a.shard::a.of]
    os.makedirs(OUT, exist_ok=True)
    for n, r in enumerate(rows, 1):
        rid = r["recording_id"]
        out = os.path.join(OUT, f"{rid}.npz")
        if os.path.exists(out):
            continue
        pose = np.asarray(spine.clean(rid)["pose"], dtype=np.float64)
        bl = float(np.nanmedian(tego.body_length(pose)))
        s = mo.scan(vid.video_path(rid), pose, dilate_px=bl * mo.DILATE_BODY_LENGTHS)
        motion = mo.motion_from_hist(s["hist_full"], MT_CUTOFF).astype(np.int32)
        frz = frame_freeze(motion, int(s["n_frames"]))
        np.savez_compressed(out, motion=motion, freeze=frz,
                            n_frames=int(s["n_frames"]), n_pixels=s["n_pixels"],
                            freeze_thresh=FREEZE_THRESH, min_frames=MIN_FRAMES,
                            mt_cutoff=MT_CUTOFF)
        log(f"  {n}/{len(rows)} {rid}: freeze {frz.mean():.3f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("manifest", "shard"), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=12)
    a = p.parse_args(argv)
    return {"manifest": manifest, "shard": shard}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
