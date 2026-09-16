"""Look at the clump. Contact sheets, because the aggregate cannot settle this.

    python3 scripts/behaviour_clips.py --group shape --clump 0

`speed_read` says clump 0 is 3.7x slower than the same animals' other segments
and that this "is what immobility looks like -- and it is also what a flat-line
tracking artifact looks like". No aggregate separates those two. Only looking
does, and a contact sheet is the format that corrected the one previous attempt
at scoring frames by eye in this programme: two frames read as "rearing against a
wall" turned out to be teleports, and a nine-frame sheet showed it.

Three frames per segment -- first, middle, last -- so a stretch that is genuinely
still looks still across all three, while a frozen *tracker* on a moving animal
shows the skeleton parked while the scene changes.

Segments are drawn across the whole duration range on purpose. The clump spans
0.53 s to 65 s and `one_clump_chaining_read` calls it a chain, so a sample from
one end would describe one end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import labels as lab                                     # noqa: E402
from recur.render import video as vid                               # noqa: E402
from recur.util import log, write_json                              # noqa: E402
from vieb.clean import arms as clean_arms                           # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.tok import config                                         # noqa: E402

CROP = 320
SEED = 0


def out_dir() -> str:
    return os.path.join(config.PATHS.results_dir, "behaviour")


def shard(group: str, k_mad: float) -> str:
    return os.path.join(config.PATHS.tok_dir, "seg_vocab",
                        f"{group}__k{k_mad:g}__corpus.npz")


def render(rid: str, t: int, pose: np.ndarray, path: str) -> bool:
    import cv2

    src = vid.video_path(rid)
    if not os.path.exists(src):
        return False
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None or t >= pose.shape[0]:
        return False
    pts = pose[t]
    vid.draw_skeleton(frame, pts, cv2, flagged=False)
    c = np.nanmean(pts, axis=0)
    h, w = frame.shape[:2]
    x = int(np.clip(c[0] - CROP // 2, 0, max(0, w - CROP)))
    y = int(np.clip(c[1] - CROP // 2, 0, max(0, h - CROP)))
    cv2.imwrite(path, frame[y:y + CROP, x:x + CROP])
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="shape")
    p.add_argument("--clump", type=int, default=0)
    p.add_argument("--k-mad", type=float, default=3.0)
    p.add_argument("--n", type=int, default=12)
    a = p.parse_args(argv)
    import cv2

    os.makedirs(out_dir(), exist_ok=True)
    with np.load(shard(a.group, a.k_mad), allow_pickle=False) as z:
        labels = z["labels"]
        animal = np.asarray([str(v) for v in z["animal"]])
        nfr, rec, frm = z["n_frames"], z["rec"], z["frame"]
    m = np.flatnonzero(labels == a.clump)
    if m.size == 0:
        raise SystemExit(f"clump {a.clump} is empty in {a.group}")
    # Spread across the duration range rather than sampled uniformly: the clump
    # is a chain spanning 0.53 s to 65 s and a uniform draw would over-represent
    # the short end, which holds most of the members.
    order = m[np.argsort(nfr[m])]
    pick = order[np.linspace(0, order.size - 1, min(a.n, order.size)).astype(int)]

    rows: list = []
    tiles: list = []
    fps = spine.fps()
    for n, i in enumerate(pick.tolist()):
        tag = animal[i]
        with np.load(os.path.join(config.REPO, "work", "ego",
                                  f"raw__bodylen__{tag}.npz"),
                     allow_pickle=False) as z:
            rid = [str(v) for v in z["recording_ids"]][int(rec[i])]
            lo = int(z["bounds"][int(rec[i])])
        d = spine.clean(rid)
        pose = clean_arms.held_array(d["pose_unfiltered"].astype(np.float64),
                                     d["missing"].astype(bool))
        s0 = int(frm[i]) - lo          # segment start, within its own recording
        n_f = int(nfr[i])
        got = []
        for name, t in (("first", s0), ("mid", s0 + n_f // 2),
                        ("last", s0 + n_f - 1)):
            path = os.path.join(out_dir(), f"c{a.clump}_{n:02d}_{name}.png")
            if render(rid, t, pose, path):
                got.append(path)
        if len(got) == 3:
            tiles.append(got)
            rows.append({"index": n, "animal": tag, "recording_id": rid,
                         "start_frame": s0, "n_frames": n_f,
                         "duration_s": n_f / fps})
            log(f"  {n:02d} {tag:>6s} {n_f:5d} fr ({n_f / fps:6.2f} s) {rid[-30:]}")

    if tiles:
        grid = []
        for k, trio in enumerate(tiles):
            imgs = []
            for pth in trio:
                im = cv2.imread(pth)
                im = cv2.resize(im if im is not None
                                else np.zeros((CROP, CROP, 3), np.uint8),
                                (CROP, CROP))
                imgs.append(im)
            strip = np.hstack(imgs)
            cv2.rectangle(strip, (0, 0), (150, 22), (0, 0, 0), -1)
            cv2.putText(strip, f"{k}: {rows[k]['duration_s']:.1f}s",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            grid.append(strip)
        sheet = os.path.join(out_dir(), f"sheet_{a.group}_c{a.clump}.png")
        cv2.imwrite(sheet, np.vstack(grid))
        log(f"wrote {sheet}")
    write_json({"group": a.group, "clump": a.clump, "n": len(rows),
                "inherited_digest": spine.digest(),
                "columns": "first / mid / last frame of each segment",
                "rows": rows},
               os.path.join(out_dir(), f"{a.group}_c{a.clump}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
