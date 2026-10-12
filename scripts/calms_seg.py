"""STEP 1 §3 D and C on CalMS21: the boundary detector and the segment clumps.

    python3 scripts/calms_seg.py --arm m1_s0 --device cuda
    # -> work/calms/states/<arm>/D.npz, C.npz, seg.json

Both are Luna's instruments called, not rebuilt:

* **D** is `scripts/breaks.py`'s primary cell ("both", 0.133 s, degree 3) on the
  `raw` arm, standardised by Luna's frozen `basis.json` `sd_used`.
* **C** is `scripts/seg_vocab.py`'s construction: `seg_recur.segments_of` on the
  shape group, `embed.selectable`, time-normalised to SEG_GRID, full-dimension
  PCA, cross-unit kNN, the bank's own ambient scale, components at the
  **inherited** theta. The "animal" is the video (no identities in CalMS21);
  PCA is fitted on the fit partition, as Luna fitted it on tune.

A frame's C state is the clump of the selectable segment containing it, or -1
(unassigned) -- including frames in no selectable segment. -1 is one state to
the map, like any other.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors                                           # noqa: E402
from recur.recurrence import bank as bk_                            # noqa: E402
from recur.util import frames, log, peak_rss_gb, write_json         # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.calms import io as cio                                    # noqa: E402
from vieb.seg import breaks as bk, embed, vocab as vb               # noqa: E402
from vieb.tok import config, ego                                    # noqa: E402

WORK = os.path.join(config.REPO, "work", "calms")
PARTS = os.path.join(config.REPO, "results", "step1", "partitions.json")
FPS = 30.0
D_CELL = ("both", 0.133, 3)          # scripts/breaks.py PRIMARY
C_GROUP = "shape"                    # scripts/seg_vocab.py PRIMARY_GROUP
K_MAD = 3.0
THETA = 0.18998060778738327          # seg_recur.json shape|k3 segment|microstate
SEED = 0


def _module(name: str):
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_mod", p)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def theta_inherited() -> float:
    with open(config.PATHS.result("seg_recur.json"), encoding="utf-8") as fh:
        g = json.load(fh)
    t = float(g["groups"][f"{C_GROUP}|k{K_MAD:g}"]["excess"]["segment|microstate"]["theta"])
    assert abs(t - THETA) < 1e-12, (t, THETA)
    return t


def load(arm: str, v: str, sd: np.ndarray) -> dict:
    with np.load(os.path.join(WORK, "raw", arm, f"{v}.npz")) as z:
        pose = np.asarray(z["pose"], dtype=np.float64)
        ab = np.asarray(z["abstain"], dtype=bool)
    X, valid = ego.transform(pose, cio.LUNA_ELL_PX, FPS)
    return {"X": X / sd[None, :], "abstain": ab | ~valid,
            "bounds": np.array([0, X.shape[0]], dtype=np.int64)}


def detect(a: dict) -> np.ndarray:
    group, deriv_sec, degree = D_CELL
    h = max(2, frames(deriv_sec, FPS))
    guard = bk.guard_frames(FPS, deriv_sec=deriv_sec)
    d = bk.discontinuity(a["X"][:, list(bk.CHANNEL_GROUPS[group])], h)
    return np.asarray(bk.boundaries(d, bk.mad_threshold(d, K_MAD),
                                    min_gap=bk.min_segment_frames(degree, guard),
                                    blocked=a["abstain"]), dtype=np.int64)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="smoke: videos per partition")
    a = p.parse_args(argv)
    sr, sv = _module("seg_recur"), _module("seg_vocab")
    sr.no_tf32()
    sd = sr.basis_sd()
    theta = theta_inherited()
    parts = cio.load(PARTS)
    fit = parts["fit"][: a.limit or None]
    scored = cio.scored(parts)[: a.limit or None]
    vids = fit + scored
    idx = bk.CHANNEL_GROUPS[C_GROUP]

    D: dict[str, np.ndarray] = {}
    feats, meta = [], []
    n_frames: dict[str, int] = {}
    for v in vids:
        x = load(a.arm, v, sd)
        n_frames[v] = int(x["X"].shape[0])
        if v in scored:
            D[v] = detect(x)
        rows = sr.segments_of(x, idx, fps=FPS, k_mad=K_MAD)
        sel = [r for r, k in zip(rows, embed.selectable(rows)) if k]
        if sel:
            f, _s = embed.stack_segments(x["X"], sel, cols=idx, n=embed.SEG_GRID)
            feats.append(f)
            meta += [(v, int(r["start"]), int(r["stop"])) for r in sel]
    Xs = np.concatenate(feats)
    start = np.arange(len(meta), dtype=np.int64) * embed.SEG_GRID
    animal = np.asarray([m[0] for m in meta])
    cols = np.arange(Xs.shape[1])
    is_fit = np.isin(animal, np.asarray(fit))
    pca = bk_.fit_pca(Xs, start[is_fit], embed.SEG_GRID, cols,
                      n_components=embed.SEG_GRID * len(idx), sample=200_000,
                      seed=SEED, device=a.device)
    B = bk_.build(Xs, start, embed.SEG_GRID, cols, pca=pca)
    (B, an, order_meta), _o = bk_.sort_by_animal(B, animal, np.arange(len(meta)))
    codes = {u: i for i, u in enumerate(sorted(set(an.tolist())))}
    code = np.asarray([codes[u] for u in an], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    i = rng.integers(0, B.shape[0], 2_000_000)
    j = rng.integers(0, B.shape[0], 2_000_000)
    ok = an[i] != an[j]
    scale = float(np.median(np.linalg.norm(B[i[ok]] - B[j[ok]], axis=1)))
    nn_i, nn_d = sv.knn(B, code, k=vb.KNN, device=a.device)
    lab, summary = vb.components(nn_i, nn_d / scale, theta=theta, min_size=vb.MIN_CLUMP)

    seg_label = np.full(len(meta), -1, dtype=np.int64)
    seg_label[np.asarray(order_meta, dtype=np.int64)] = lab
    C = {v: np.full(n_frames[v], -1, dtype=np.int64) for v in scored}
    for (v, s0, s1), L in zip(meta, seg_label.tolist()):
        if v in C and L >= 0:
            C[v][s0:s1] = L

    out = os.path.join(WORK, "states", a.arm if not a.limit else f"smoke_{a.arm}")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "D.npz"), videos=np.array(scored),
                        **{f"b_{v}": D[v] for v in scored})
    np.savez_compressed(os.path.join(out, "C.npz"), videos=np.array(scored),
                        **{f"s_{v}": C[v] for v in scored})
    cov = float(np.mean(np.concatenate([C[v] >= 0 for v in scored])))
    write_json({**provenance.header(anchors.LUNA, stage="calms_seg",
                                    unverified="CalMS21 external; Luna instruments at their commits"),
                "arm": a.arm, "d_cell": list(D_CELL), "c_group": C_GROUP, "k_mad": K_MAD,
                "theta_inherited": theta, "scale": scale, "n_videos": len(vids),
                "n_segments": len(meta), "n_fit_segments": int(is_fit.sum()),
                "pca_retained": pca["retained_variance"],
                "clumps": summary, "frame_coverage_scored": cov,
                "boundaries_per_s": float(sum(D[v].size for v in scored)
                                          / (sum(n_frames[v] for v in scored) / FPS)),
                "peak_rss_gb": peak_rss_gb()},
               os.path.join(out, "seg.json"))
    log(f"{a.arm}: {len(meta):,} segments, {summary['n_clumps']} clumps, "
        f"{summary['unassigned_fraction']:.1%} unassigned; scored frame coverage {cov:.1%}; "
        f"D {sum(D[v].size for v in scored):,} boundaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
