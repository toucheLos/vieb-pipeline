"""Step 0. Is the planted instance findable at all?

    python3 scripts/probe_audit.py [--group shape] [--n-animals 12]

READ results/PROBE_AUDIT_PREREGISTRATION.md FIRST.

## What this is for

Step B closed the segmentation route at 12.8% planted-instance isolation against
a 20% gate, with `merged` 0.0% and `split` 0.0% in every cell -- so the failure
is `no_edge`, and `DETECTOR.md:83-86` already suspects why: the template is the
mean of 40 real windows crossfaded over 3 frames, and may have no acceleration
discontinuity to find.

If that is right, 12.8% is a statement about crossfades, and tuning any
criterion against it builds a crossfade detector. This programme has that failure
in its ledger already.

## Nothing here is tuned, and that is enforced by import

Every setting -- `deriv_sec`, `EDGE_TOL`, the planted duration, the occupancy,
the arm planted into, the channel group, the seed -- is taken from
`detector_sweep.py` **as an attribute**, never retyped. A constant that is
retyped is a constant that can drift, and the whole point of this stage is that
it shares Step B's probe exactly.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors                                           # noqa: E402
from vieb import provenance                                         # noqa: E402
from recur.util import frames, log, write_json                      # noqa: E402
from vieb import seeds                                              # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.seg import breaks as bk, embed                            # noqa: E402
from vieb.seg import planted as pl                                  # noqa: E402
from vieb.seg import probe_audit as pa                              # noqa: E402
from vieb.tok import config                                         # noqa: E402


def _load(name: str):
    """Import a sibling script by path, with argv neutralised."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_mod", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = saved
    return m


def run(a) -> int:
    ds = _load("detector_sweep")
    sr = _load("seg_recur")

    # Every parameter, inherited. See the module docstring.
    deriv = bk.DERIV_SEC
    tol = int(ds.EDGE_TOL)
    duration_s = 0.5
    occ = float(ds.PRIMARY_OCC)
    into = "microstate"
    seed = int(ds.SEED)

    fps = spine.fps()
    sd = sr.basis_sd()
    idx = list(bk.CHANNEL_GROUPS[a.group])
    tags = sorted(sr.animals_of(spine.recording_ids()))[:a.n_animals]
    h = max(2, frames(deriv, fps))
    guard = bk.guard_frames(fps, deriv_sec=deriv)
    min_gap = bk.min_segment_frames(3, guard)
    w_tpl = frames(duration_s, fps)

    # The template, built exactly as the sweep builds it.
    donors = []
    for tag in list(tags)[:3]:
        arm = sr.load_arm(into, tag, sd)
        donors.append((arm["X"], ~arm["abstain"]))
    tpl = pl.ego_template(np.concatenate([d[0] for d in donors]), w_tpl,
                          np.random.default_rng(seeds.stable_seed(seed, "tpl")),
                          valid=np.concatenate([d[1] for d in donors]))
    del donors
    log(f"  template {w_tpl} frames from {len(tags)} animals, h={h}, "
        f"tol=±{tol}")

    rows: list[dict] = []
    n_above = n_edges = 0
    for tag in tags:
        arm = sr.load_arm(into, tag, sd)
        rng = np.random.default_rng(
            seeds.stable_seed(seed, f"det{duration_s:g}{occ:g}", tag))
        arm["X"], mask, _meta = pl.ego_plant(arm["X"], tpl, rng,
                                             bounds=arm["bounds"],
                                             occupancy=occ,
                                             blocked=arm["abstain"])
        sub = arm["X"][:, idx]
        draw = np.random.default_rng(seeds.stable_seed(seed, "pa", tag))
        for r in range(arm["bounds"].shape[0] - 1):
            lo, hi = int(arm["bounds"][r]), int(arm["bounds"][r + 1])
            d = bk.discontinuity(sub[lo:hi], h)
            thr = bk.mad_threshold(d, bk.K_MAD)
            # Slice-local throughout. `boundaries` returns slice-local peaks and
            # the sweep adds `lo` back only to index the corpus table; here D is
            # the slice, so the peaks stay where they are.
            pk = bk.boundaries(d, thr, min_gap=min_gap,
                               blocked=arm["abstain"][lo:hi])
            edges = pa.edge_frames(mask, lo=lo, hi=hi, tol=tol)
            got = pa.populations(d, edges, pk, draw,
                                 selectable=~arm["abstain"][lo:hi])
            if got["n_planted"] == 0:
                continue
            # The quantity the detector actually thresholds: could this edge
            # fire at all?
            n_edges += int(edges.size)
            n_above += int(np.count_nonzero(d[edges] > thr))
            rows.append({"animal": tag, "rec": r, "threshold": float(thr),
                         **got})
        del arm

    obj = {"dataset": "luna", "arm": "probe_audit", "group": a.group,
           "pose_arm": "raw", "deriv_sec": deriv, "duration_s": duration_s,
           "occupancy": occ, "planted_into": into, "edge_tolerance": tol}
    n_eff = len({r["animal"] for r in rows})

    reads: dict = {}
    # The anchor FIRST. If it moved, nothing else here is readable.
    anchor = pa.anchor_read(rows, scored_object=obj, n_effective=n_eff,
                            seed=seed)
    reads["anchor"] = anchor.to_dict()
    log("  " + anchor.line())
    if anchor.verdict == "PASS":
        sep = pa.separation_read(rows, scored_object=obj, n_effective=n_eff,
                                 seed=seed)
        reads["separation"] = sep.to_dict()
        log("  " + sep.line())
    else:
        log("  separation not computed: the anchor moved")

    share = (n_above / n_edges) if n_edges else float("nan")
    log(f"  {n_above}/{n_edges} planted-edge frames exceed their own "
        f"recording's threshold ({share:.2%})")

    out = a.out or config.PATHS.result("probe_audit.json")
    write_json({**provenance.header(anchors.LUNA, stage="probe_audit",
                                 unverified="a subset of animals"),
                "inherited_digest": spine.digest(),
                "registration": "results/PROBE_AUDIT_PREREGISTRATION.md",
                "scored_object": obj, "n_animals": n_eff,
                "n_recordings": len(rows),
                "edge_frames": n_edges, "edge_frames_above_threshold": n_above,
                "edge_share_above_threshold": share,
                "reads": reads, "rows": rows}, out)
    log(f"  wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="shape", choices=tuple(bk.CHANNEL_GROUPS))
    p.add_argument("--n-animals", type=int, default=12)
    p.add_argument("--out", default=None)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
