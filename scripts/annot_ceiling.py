"""Step 1b. The inter-rater ceiling, from the raters' exported files.

    python3 scripts/annot_ceiling.py results/annot/annot_*.json

READ results/ANNOTATION_PREREGISTRATION.md FIRST.

## This runs BEFORE any detector is scored

The ceiling is the benchmark, and it is not 1.0. A detector that matches
rater-rater agreement is at ceiling, not failing. Computing it after seeing a
detector's recall would make it a number chosen to flatter one, so it is
computed, written and committed first.

## What it refuses

* **One rater.** There is no ceiling from a single person, and a detector
  scored against one rater is scored against nothing. `annot.ceiling_read`
  returns NOT_A_RESULT and this script exits non-zero.
* **Agreement that only reaches chance.** If the +/-5 interval includes the
  chance F1 -- what two raters would score scattering the same mark counts at
  random in the same clips -- the question is ill-posed at this tracking
  quality. Report and stop; no detector is scored against it.

## Order was per rater, and that is recorded rather than assumed

Each rater met the clips in an order seeded from their own name, so fatigue and
calibration drift do not land on the same clips for everybody. Each export
carries the order it was rated in; this script checks the sets match and stores
every rater's order in the result.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors                                           # noqa: E402
from recur.util import log, write_json                              # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.seg import annot as an                                    # noqa: E402
from vieb.tok import config                                         # noqa: E402

SEED = 0


def out_dir() -> str:
    return os.path.join(config.PATHS.results_dir, "annot")


def load_exports(paths: list[str]) -> dict:
    """rater -> {clip: [frames]}, plus the metadata each export carries.

    A clip a rater never opened is **missing data, not an empty answer**, and
    only clips carrying an explicit verdict -- marks, or "nothing changes here"
    -- enter the comparison. An unfinished pass must not read as a rater who
    saw nothing happen.
    """
    marks: dict = {}
    meta: dict = {}
    sources: dict = {}
    for p in sorted(paths):
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        rater = str(doc.get("rater") or os.path.basename(p)).strip().lower()
        if not rater:
            raise SystemExit(f"{p}: no rater name")
        if rater in marks:
            raise SystemExit(f"two exports both name rater {rater!r}")
        got: dict = {}
        src: dict = {}
        for cid, row in (doc.get("clips") or {}).items():
            if not (row.get("marks") or row.get("empty")):
                continue
            got[cid] = [int(m) for m in (row.get("marks") or [])]
            # How the mark was placed. The tool permitted "paused"
            # (`currentTime`) and "playing" (`rvfc`) and specified neither,
            # so it is carried rather than assumed uniform.
            src[cid] = str(row.get("frame_source") or "")
        marks[rater] = got
        sources[rater] = src
        meta[rater] = {"file": os.path.basename(p),
                       "exported_at": doc.get("exported_at"),
                       "n_rated": len(got),
                       "frame_api": doc.get("frame_api"),
                       "order": doc.get("order") or []}
    return {"marks": marks, "meta": meta, "sources": sources}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("exports", nargs="*",
                   help="the rater files; default results/annot/annot_*.json")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    paths = a.exports or sorted(glob.glob(os.path.join(out_dir(),
                                                       "annot_*.json")))
    if not paths:
        raise SystemExit("no exports found; pass them, or drop them in "
                         + out_dir())

    with open(os.path.join(out_dir(), "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    clips = {r["clip"]: {"animal": r["animal"], "n_frames": int(r["n_frames"]),
                         "fps": float(r["fps"]), "decile": int(r["decile"])}
             for r in man["clips"]}

    got = load_exports(paths)
    marks, meta, sources = got["marks"], got["meta"], got["sources"]
    for rater, by_clip in marks.items():
        unknown = sorted(set(by_clip) - set(clips))
        if unknown:
            raise SystemExit(f"{rater} rated clips not in the manifest: "
                             f"{unknown[:5]}")
        log(f"  {rater}: {len(by_clip)} clips rated, "
            f"{sum(len(v) for v in by_clip.values())} marks, "
            f"order of {len(meta[rater]['order'])}")

    obj = {"dataset": "luna", "arm": "annot", "split": "tune",
           "pose_arm": "raw", "n_raters": len(marks),
           "n_clips": len(clips)}
    rows = an.pair_rows(marks, clips, seed=SEED)
    n_eff = len({clips[r["clip"]]["animal"] for r in rows}) if rows else 0

    reads: dict = {}
    ok = True
    for tol in an.TOLERANCES:
        rd = an.ceiling_read(rows, tol=tol, scored_object=obj,
                             n_effective=n_eff, seed=SEED)
        reads[f"ceiling|{tol}"] = rd.to_dict()
        log("  " + rd.line())
        if tol == 5 and rd.verdict != "PASS":
            ok = False

    cov = an.coverage_read(marks, clips, scored_object=obj, n_effective=n_eff,
                           seed=SEED)
    reads["coverage"] = cov.to_dict()
    log("  " + cov.line())

    # What the registered ceiling is MADE of, and the shape of the
    # disagreement. All three are NOT_A_RESULT by construction: they describe
    # the number above, they do not restate or replace it. Registration §7
    # forbids re-scoring at a band chosen after the fact, and none of these
    # scores anything -- the decomposition partitions rows already computed,
    # and the offsets are one nearest-neighbour distribution.
    for tol in an.TOLERANCES:
        dec = an.decompose_read(rows, tol=tol, scored_object=obj,
                                n_effective=n_eff, seed=SEED)
        reads[f"decompose|{tol}"] = dec.to_dict()
        log("  " + dec.line())

    offs = an.offsets(marks, clips, sources=sources)
    off = an.offset_read(offs, scored_object=obj, n_effective=n_eff,
                         seed=SEED)
    reads["offsets"] = off.to_dict()
    log("  " + off.line())

    src = an.source_read(offs, scored_object=obj, n_effective=n_eff)
    reads["frame_source"] = src.to_dict()
    log("  " + src.line())

    out = a.out or config.PATHS.result("annot_ceiling.json")
    write_json({**anchors.header(anchors.LUNA, stage="annot_ceiling",
                                 unverified="a curated sample, human-rated"),
                "inherited_digest": spine.digest(),
                "registration": "results/ANNOTATION_PREREGISTRATION.md",
                "n_raters": len(marks), "raters": meta,
                "n_clips": len(clips), "n_pair_rows": len(rows),
                "tolerances": list(an.TOLERANCES),
                "reads": reads, "rows": rows, "offsets": offs}, out)
    log(f"  wrote {out}")
    if not ok:
        log("  THE CEILING DOES NOT STAND at ±5 frames. No detector is scored "
            "against this.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
