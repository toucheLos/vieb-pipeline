"""H3. Score flagged and unflagged frames by eye, blind, and cross-tabulate.

    python3 scripts/adjudicate.py --sample     # writes frames + a sealed key
    python3 scripts/adjudicate.py --score FILE # after scoring, unblinds

The aggregate says 2.134% of frames violate, concentrated 350x and rising 3.7x
toward the arena wall. That is consistent with a footage limit. It is also
consistent with a mask that is measuring the wrong thing, and no aggregate can
tell those apart -- only looking can.

## Why this is blind, and why that is not optional

The scorer is a language model reading contact sheets, and its one recorded
attempt at this in the programme was **wrong**: two frames were read as "rearing
against a wall" and a nine-frame contact sheet showed they were teleports. So the
eye here is the least reliable instrument in the pipeline, and the design has to
assume that rather than hope.

`draw_skeleton` is therefore called with `flagged=False` for **every** frame --
no red border, no marker, nothing that says which set a frame came from. Filenames
are a shuffled index. The mapping from index to (recording, frame, flagged,
stratum) is written to a key file that is **not read until scores are supplied**.

## The four cells

Frames are sampled from a 2x2: recordings in the **worst decile** by violation
rate against recordings that **pass** (below the corpus median), crossed with
frames the mask **flags** against frames it does not. Twenty-five each.

That crossing is what makes the third outcome visible. If flagged frames look
fine and unflagged frames look broken, the mask is measuring the wrong thing --
and a design that only rendered flagged frames could not have seen it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import labels as lab, splits                            # noqa: E402
from recur.render import video as vid                              # noqa: E402
from recur.util import log, write_json                             # noqa: E402
from vieb.clean import arms as clean_arms                          # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.tok import config                                        # noqa: E402

EPS = 0.10
CELL = f"viol|unfiltered|raw|{EPS}|skull"
PER_CELL = 25
SEED = 0
CROP = 320
SCORES = ("occluded", "visible_but_wrong", "fine", "unsure")


def out_dir() -> str:
    return os.path.join(config.PATHS.results_dir, "adjudicate")


def violation_mask(rid: str) -> np.ndarray:
    tag = lab.animal_tag(rid)
    with np.load(config.PATHS.bones_shard(tag), allow_pickle=False) as z:
        ids = [str(v) for v in z["recording_ids"]]
        r = ids.index(str(rid))
        lo, hi = int(z["bounds"][r]), int(z["bounds"][r + 1])
        total = int(z["bounds"][-1])
        return np.unpackbits(z[CELL])[:total][lo:hi].astype(bool)


def recording_rates() -> list:
    """Per-recording violation rates on the report split, from H1's shards."""
    d = os.path.join(os.path.dirname(config.PATHS.bones_dir), "concentration")
    rows: list = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            rows += json.load(fh)["rows"]
    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    return [r for r in rows if of.get(r["animal"]) == "report"]


def render_frame(rid: str, t: int, pose: np.ndarray, path: str) -> bool:
    """One frame, skeleton drawn, cropped on the animal. No flag, ever."""
    import cv2

    src = vid.video_path(rid)
    if not os.path.exists(src):
        return False
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    pts = pose[t]
    # flagged=False ALWAYS. The border is what would leak the answer.
    vid.draw_skeleton(frame, pts, cv2, flagged=False)
    c = np.nanmean(pts, axis=0)
    h, w = frame.shape[:2]
    x = int(np.clip(c[0] - CROP // 2, 0, max(0, w - CROP)))
    y = int(np.clip(c[1] - CROP // 2, 0, max(0, h - CROP)))
    cv2.imwrite(path, frame[y:y + CROP, x:x + CROP])
    return True


def contact_sheets(files: list, labels: list, *, cols: int = 5,
                   rows: int = 5) -> list:
    """5x5 image sheets with each tile's index burned in.

    `vid.tile` composes VIDEOS -- it writes an H.264 stream -- so it cannot be
    used here. The index is drawn on the tile because a score has to be
    attributable to a frame, and position in a grid is easy to miscount.
    """
    import cv2

    out: list = []
    per = cols * rows
    for s in range(0, len(files), per):
        chunk = files[s:s + per]
        tiles = []
        for i, f in enumerate(chunk):
            im = cv2.imread(f)
            if im is None:
                im = np.zeros((CROP, CROP, 3), dtype=np.uint8)
            im = cv2.resize(im, (CROP, CROP))
            cv2.rectangle(im, (0, 0), (52, 20), (0, 0, 0), -1)
            cv2.putText(im, str(labels[s + i]), (4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            tiles.append(im)
        while len(tiles) < per:
            tiles.append(np.zeros((CROP, CROP, 3), dtype=np.uint8))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols])
                          for r in range(rows)])
        path = os.path.join(out_dir(), f"sheet_{s // per}.png")
        cv2.imwrite(path, grid)
        out.append(os.path.basename(path))
    return out


def sample(args) -> int:
    os.makedirs(out_dir(), exist_ok=True)
    rows = recording_rates()
    rates = np.array([r["rate"] for r in rows])
    hi_cut = float(np.percentile(rates, 90))
    lo_cut = float(np.median(rates))
    worst = [r for r in rows if r["rate"] >= hi_cut]
    passing = [r for r in rows if r["rate"] <= lo_cut]
    log(f"{len(rows)} recordings; worst decile >= {hi_cut:.3%} (n={len(worst)}), "
        f"passing <= {lo_cut:.3%} (n={len(passing)})")

    rng = np.random.default_rng(SEED)
    picks: list = []
    for stratum, pool in (("worst_decile", worst), ("passing", passing)):
        for want_flagged in (True, False):
            got = 0
            tries = 0
            order = rng.permutation(len(pool))
            while got < PER_CELL and tries < len(pool) * 6:
                r = pool[int(order[tries % len(pool)])]
                tries += 1
                rid = r["recording_id"]
                try:
                    mask = violation_mask(rid)
                except Exception:
                    continue
                idx = np.flatnonzero(mask if want_flagged else ~mask)
                if idx.size == 0:
                    continue
                t = int(rng.choice(idx))
                picks.append({"recording_id": rid, "animal": r["animal"],
                              "frame": t, "flagged": bool(want_flagged),
                              "stratum": stratum,
                              "recording_rate": float(r["rate"])})
                got += 1
            log(f"  {stratum:13s} flagged={want_flagged}  {got} frames")

    # Shuffle so the filename index carries no information about the cell.
    order = rng.permutation(len(picks))
    key: list = []
    pose_cache: dict = {}
    for n, i in enumerate(order):
        p = picks[int(i)]
        rid = p["recording_id"]
        if rid not in pose_cache:
            d = spine.clean(rid)
            pose_cache[rid] = clean_arms.held_array(
                d["pose_unfiltered"].astype(np.float64),
                d["missing"].astype(bool))
            if len(pose_cache) > 40:
                pose_cache.pop(next(iter(pose_cache)))
        path = os.path.join(out_dir(), f"f{n:03d}.png")
        if not render_frame(rid, p["frame"], pose_cache[rid], path):
            log(f"  SKIP f{n:03d}: {rid} frame {p['frame']}")
            continue
        key.append({"index": n, "file": os.path.basename(path), **p})

    # Contact sheets: 25 per sheet, the format that corrected the one previous
    # attempt at scoring frames by eye in this programme.
    sheets = contact_sheets([os.path.join(out_dir(), k["file"]) for k in key],
                            [k["index"] for k in key])

    write_json({"seed": SEED, "eps": EPS, "cell": CELL,
                "per_cell": PER_CELL, "n_rendered": len(key),
                "worst_decile_cut": hi_cut, "passing_cut": lo_cut,
                "sheets": sheets, "scores_allowed": list(SCORES),
                "blind": ("draw_skeleton called with flagged=False for every "
                          "frame; filenames are a shuffled index; this key is "
                          "not read until scores are supplied"),
                "key": key},
               os.path.join(out_dir(), "key.json"))
    log(f"rendered {len(key)} frames, {len(sheets)} sheets -> {out_dir()}")
    log("KEY SEALED. Score the sheets, then: --score scores.json")
    return 0


def score(args) -> int:
    with open(os.path.join(out_dir(), "key.json"), encoding="utf-8") as fh:
        key_doc = json.load(fh)
    with open(args.score, encoding="utf-8") as fh:
        scores = json.load(fh)
    key = {int(k["index"]): k for k in key_doc["key"]}

    cells: dict = {}
    for idx_s, verdict in scores.items():
        idx = int(idx_s)
        if idx not in key:
            raise SystemExit(f"score for index {idx} which was not rendered")
        if verdict not in SCORES:
            raise SystemExit(f"{verdict!r} is not one of {SCORES}")
        k = key[idx]
        cell = cells.setdefault((k["stratum"], k["flagged"]), {})
        cell[verdict] = cell.get(verdict, 0) + 1

    table = {f"{st}|flagged={fl}": v for (st, fl), v in sorted(cells.items())}
    write_json({"n_scored": len(scores), "by_cell": table,
                "scores": {str(i): s for i, s in scores.items()},
                "key_digest": key_doc["n_rendered"]},
               os.path.join(out_dir(), "adjudication.json"))
    log(f"{'cell':32s} " + "  ".join(f"{s:>18s}" for s in SCORES))
    for name, v in table.items():
        log(f"{name:32s} " + "  ".join(f"{v.get(s, 0):>18d}" for s in SCORES))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true")
    p.add_argument("--score", default=None, help="path to a scores JSON")
    a = p.parse_args(argv)
    if a.sample:
        return sample(a)
    if a.score:
        return score(a)
    raise SystemExit("pass --sample or --score")


if __name__ == "__main__":
    raise SystemExit(main())
