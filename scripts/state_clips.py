"""Clips per state: MoSeq's syllables (real_s0) and STATES' macrostates.

    python3 scripts/state_clips.py --model moseq     # syllables in use, report split
    python3 scripts/state_clips.py --model states    # PCCA+ macrostates at tau = 1
    # writes results/state_clips/<model>/<state>_<k>.mp4 (+ .jpg) and manifest.json

Neither model licenses its states as behaviour: MOSEQ FAILED on seed stability
(`MOSEQ.md`), and STATES named none (`STATES.md`). These clips are for looking,
which is the only way to say what a label tracks.

STATES at tau = 1 is a descriptive choice, not a registered one: it is the lag
where the corpus beats the white-noise floor (`states_diag.json`), whereas the
registered tau* = 64 sits on the floor and PCCA+ there puts 90% of frames in one
state. M is the number of slow modes before the largest gap ratio at tau = 1.

Each clip is 2 s (60 frames) starting 0.5 s before a bout of >= 0.5 s, cropped
on the animal, with the repaired skeleton drawn; a green border marks the frames
inside the state. Bouts are drawn from distinct report animals, seed 0.
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

from recur import labels as lab                                     # noqa: E402
from recur.render import video as vid                               # noqa: E402
from recur.util import log                                          # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.states import micro, operator as op                        # noqa: E402
from vieb.tok import config                                         # noqa: E402

OUT = os.path.join(config.PATHS.results_dir, "state_clips")
N_CLIPS = 4
MIN_BOUT = 15
PRE, LEN = 15, 60
CROP = 320
IN_USE = 0.005
SEED = 0


def moseq_labels() -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    z = np.load(os.path.join(config.REPO, "work", "moseq", "labels", "real_s0.npz"))
    s = z["syllable"].astype(np.int64)
    frac = np.bincount(s) / s.size
    s = np.where(frac[s] > IN_USE, s, -1)
    return ([str(r) for r in z["recording_ids"]], s, z["bounds"],
            {"what": "MoSeq syllables (real, seed 0), in use > 0.5% of report frames",
             "fraction": {int(k): float(f) for k, f in enumerate(frac) if f > IN_USE}})


def states_labels() -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    sys.path.insert(0, os.path.join(config.REPO, "scripts"))
    import states as S

    sel = S._read("select.json")
    z = np.load(os.path.join(S.WORK, "corpus.npz"))
    fit_b = S.load("fit", sel["use_mask"])["bounds"]
    p, keep = op.transition_matrix(micro.counts(z["lab_fit"], fit_b, 1, micro.N_MICRO))
    ts = op.timescales(p, 1, k=S.N_TS)
    gaps = op.gap_ratios(ts)
    m = int(np.argmax(gaps)) + 2
    chi = op.pcca(op.reversibilise(p), m)
    st = op.assign(z["lab_rep"], keep, chi, threshold=0.8)
    rids = []
    for tag in S._split_tags()["report"]:
        with np.load(os.path.join(S.FEAT, f"{tag}.npz")) as f:
            rids += [str(r) for r in f["recording_ids"]]
    rep_b = S.load("report", sel["use_mask"])["bounds"]
    assert rep_b[-1] == st.size and len(rids) == rep_b.size - 1
    frac = {int(k): float((st == k).mean()) for k in range(m)}
    return rids, st, rep_b, {
        "what": "STATES PCCA+ macrostates at tau = 1 (descriptive; none licensed)",
        "M": m, "gap_ratios_tau1": [float(g) for g in gaps],
        "timescales_tau1": [float(t) for t in ts],
        "fraction": frac, "abstain": float((st < 0).mean())}


def bouts(s: np.ndarray, b: np.ndarray, state: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(b.size - 1):
        seg = s[b[i]:b[i + 1]] == state
        e = np.flatnonzero(np.diff(np.r_[0, seg.astype(np.int8), 0]))
        out += [(i, int(lo), int(hi)) for lo, hi in zip(e[::2], e[1::2])
                if hi - lo >= MIN_BOUT]
    return out


def pose_of(rid: str) -> np.ndarray:
    tag = lab.animal_tag(rid)
    z = np.load(os.path.join(config.REPO, "work", "repaired", f"{tag}.npz"))
    i = [str(r) for r in z["recording_ids"]].index(rid)
    return z["pose"][z["bounds"][i]:z["bounds"][i + 1]].astype(np.float64)


def render(rid: str, t0: int, inside: np.ndarray, title: str, path: str) -> bool:
    import cv2

    pose = pose_of(rid)
    t0 = max(0, t0)
    t1 = min(pose.shape[0], t0 + LEN)
    cen = np.nanmedian(pose[t0:t1, 3], axis=0)          # body centre keypoint
    if not np.isfinite(cen).all():
        return False
    cap = cv2.VideoCapture(vid.video_path(rid))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0 = int(np.clip(cen[0] - CROP // 2, 0, W - CROP))
    y0 = int(np.clip(cen[1] - CROP // 2, 0, H - CROP))
    cap.set(cv2.CAP_PROP_POS_FRAMES, t0)
    proc = subprocess.Popen(
        [vid.FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{CROP}x{CROP}", "-r", f"{spine.fps()}", "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "26", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    poster = None
    for t in range(t0, t1):
        ok, fr = cap.read()
        if not ok:
            break
        pts = pose[t]
        if np.isfinite(pts).all():
            vid.draw_skeleton(fr, pts, cv2)
        c = fr[y0:y0 + CROP, x0:x0 + CROP].copy()
        if inside[t - t0]:
            cv2.rectangle(c, (0, 0), (CROP - 1, CROP - 1), (90, 208, 95), 6)
        cv2.rectangle(c, (6, 6), (14 + 9 * len(title), 28), (0, 0, 0), -1)
        cv2.putText(c, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)
        if poster is None and inside[t - t0]:
            poster = c.copy()
        assert proc.stdin is not None
        proc.stdin.write(c.tobytes())
    assert proc.stdin is not None
    proc.stdin.close()
    proc.wait()
    if poster is not None:
        cv2.imwrite(path[:-4] + ".jpg", poster, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return proc.returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("moseq", "states"), required=True)
    a = ap.parse_args(argv)
    rids, s, b, meta = moseq_labels() if a.model == "moseq" else states_labels()
    out = os.path.join(OUT, a.model)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    man = {**meta, "states": {}}
    for state in sorted(k for k in np.unique(s) if k >= 0):
        bs = bouts(s, b, int(state))
        dw = [hi - lo for _, lo, hi in bs]
        by_animal: dict[str, list] = {}
        for bt in bs:
            by_animal.setdefault(lab.animal_tag(rids[bt[0]]), []).append(bt)
        animals = list(by_animal)
        rng.shuffle(animals)
        clips = []
        for an in animals:
            if len(clips) == N_CLIPS:
                break
            i, lo, hi = by_animal[an][int(rng.integers(len(by_animal[an])))]
            t0 = max(0, lo - PRE)
            inside = np.zeros(LEN, dtype=bool)
            for t in range(LEN):
                g = t0 + t
                inside[t] = 0 <= g < b[i + 1] - b[i] and s[b[i] + g] == state
            name = f"{a.model}_{int(state):03d}_{len(clips)}.mp4"
            title = f"{'syllable' if a.model == 'moseq' else 'state'} {int(state)}"
            if render(rids[i], t0, inside, title, os.path.join(out, name)):
                clips.append({"file": name, "recording_id": rids[i], "start": int(t0),
                              "bout_frames": int(hi - lo)})
        man["states"][int(state)] = {"fraction": meta["fraction"].get(int(state)),
                                     "n_bouts_ge_0.5s": len(bs),
                                     "median_bout_frames": float(np.median(dw)) if dw else None,
                                     "clips": clips}
        log(f"  {a.model} {int(state)}: {len(clips)} clips from {len(bs)} bouts")
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
