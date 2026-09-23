"""§3 of the grooming registration: the clips get LOOKED AT.

    python3 scripts/grooming_clips.py [--max 30]

The claim under test is "these windows are grooming". That is a claim about
**content**, and no statistic in `grooming_gate.py` establishes it -- the
candidates were detected by a threshold on head-region pixel motion, and calling
them grooming is exactly the step a threshold cannot take.

So 30 candidates and 15 **speed-matched controls** are rendered at the
headline settings, shuffled into ONE panel under opaque ids, and the key is
written beside them and **not published**. The confirmation rate goes in
`work/grooming/confirmation.json` and is published as the detector's own
precision.

**§9.4: below a 50% confirmation rate the spectral gate is not read as an answer
about grooming at all.** It would then be measuring some other
held-still-with-head-motion behaviour, and §9.6 says that is what gets named.

## Why controls are in the panel

A rater shown only candidates has no way to be wrong. Mixing in speed-matched
non-candidates at a ratio the rater is not told means the confirmation rate can
come out at chance, which is the outcome that would falsify the detector.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors                                          # noqa: E402
from recur.render import video as vid                              # noqa: E402
from recur.util import log, write_json                             # noqa: E402
from vieb import seeds                                             # noqa: E402
from vieb.clean import arms as clean_arms                          # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.tok import config                                        # noqa: E402

SALT = "vieb-grooming-panel"
SEED = 0
#: Lead-in and lead-out, in seconds. The same convention `island_clips.py`
#: invented and named, applied identically to both arms so it cannot separate
#: them.
PAD_S = 0.5


def _gg():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "grooming_gate.py")
    spec = importlib.util.spec_from_file_location("gg_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["grooming_gate"]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def _id(arm: str, i: int) -> str:
    return "g" + hashlib.sha256(
        f"{SALT}|{arm}|{i}".encode()).hexdigest()[:10]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=0)
    a = p.parse_args(argv)
    gg = _gg()
    fps = spine.fps()
    rad, win_s, pct = gg.HEADLINE
    got = gg._collect(rad, win_s, pct, fps)
    rows, cand, partners = got["rows"], got["cand"], got["partners"]
    t_idx = np.flatnonzero(cand)
    ok = partners >= 0
    log(f"  {int(cand.sum())} candidates, {int(ok.sum())} matched pairs")
    if not ok.any():
        raise SystemExit("no matched pairs; run the scan first")

    rng = np.random.default_rng(seeds.stable_seed(SEED, "grooming-panel"))
    # §3: at least 15 distinct animals, so the panel is not one animal's quirk.
    order = rng.permutation(int(ok.sum()))
    pick_t, pick_c, seen = [], [], set()
    n_t = a.max or gg.N_CLIPS_CANDIDATE
    for j in order:
        i = int(t_idx[ok][j])
        if len(pick_t) < n_t and (rows[i]["animal"] not in seen
                                  or len(seen) >= gg.MIN_CLIP_ANIMALS):
            pick_t.append(i)
            seen.add(rows[i]["animal"])
        if len(pick_c) < gg.N_CLIPS_CONTROL:
            pick_c.append(int(partners[ok][j]))
        if len(pick_t) >= n_t and len(pick_c) >= gg.N_CLIPS_CONTROL:
            break
    log(f"  {len(pick_t)} candidates over {len(seen)} animals, "
        f"{len(pick_c)} controls")

    out = os.path.join(config.PATHS.results_dir, "grooming")
    os.makedirs(os.path.join(out, "clips"), exist_ok=True)
    pad = int(round(PAD_S * fps))
    cache: dict = {}
    panel, key = [], []
    for arm, picks in (("candidate", pick_t), ("control", pick_c)):
        for n, i in enumerate(picks):
            r = rows[i]
            rid = r["rid"]
            if rid not in cache:
                d = spine.clean(rid)
                cache[rid] = clean_arms.held_array(
                    d["pose_unfiltered"].astype(np.float64),
                    d["missing"].astype(bool))
                if len(cache) > 20:
                    cache.pop(next(iter(cache)))
            held = cache[rid]
            a0 = max(0, int(r["a"]) - pad)
            b0 = min(int(held.shape[0]), int(r["b"]) + pad)
            if b0 - a0 < 2:
                continue
            cid = _id(arm, n)
            rel = os.path.join("clips", f"{cid}.mp4")
            box = vid.crop_box(held, a0, b0, size=vid.CROP)
            ok_cut, why = vid.cut(vid.video_path(rid), a0, b0,
                                  os.path.join(out, rel), fps=fps, pose=held,
                                  flags=None, label=None, crop=box)
            if not ok_cut:
                log(f"  skip {cid}: {why}")
                continue
            panel.append({"clip": cid, "file": rel,
                          "duration_s": round((b0 - a0) / fps, 3)})
            key.append({"clip": cid, "arm": arm, "recording_id": rid,
                        "a": int(r["a"]), "b": int(r["b"]),
                        "animal": r["animal"], "speed": r["speed"],
                        "energy": r["energy"],
                        "peak_excess": r["peak_excess"]})
    rng.shuffle(panel)
    write_json({**anchors.header(anchors.LUNA, stage="grooming_clips",
                                 unverified="a detected candidate set"),
                "inherited_digest": spine.digest(),
                "registration": "results/GROOMING_PREREGISTRATION.md §3",
                "headline": [rad, win_s, pct], "pad_s": PAD_S,
                "blind": ("panel order shuffled; the arm lives ONLY in key.json "
                          "and the mix of candidates to controls is not stated "
                          "on the panel"),
                "question": ("For each clip: is the animal GROOMING -- forepaws "
                             "moving at the face or body while the trunk is "
                             "still? Answer yes / no / unsure."),
                "n_clips": len(panel), "clips": panel},
               os.path.join(out, "manifest.json"))
    write_json({**anchors.header(anchors.LUNA, stage="grooming_clips_key",
                                 unverified="a detected candidate set"),
                "n": len(key), "key": key,
                "not_published": True},
               os.path.join(out, "key.json"))
    log(f"  {len(panel)} clips -> {os.path.join(out, 'manifest.json')} "
        f"(key.json held back)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
