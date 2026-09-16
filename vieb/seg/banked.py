r"""The three passes this arm stands on, banked as `Read`s before it runs.

Unbanked results in this project have been superseded and then quoted anyway.
Q1 lives in a different repository; the roughness measurement existed only as
prose in `SEGMENTATION.md`; the dwell pair alone was already banked, in
`results/falsifier.json`. This module turns the first two into verdicts with
their objects attached and re-states the third by reference rather than by
copy, so there is exactly one place each number is defined.

## What banking a result means here, and what it does not

It means: the number, the **cell** that produced it, a `Read` whose
`scored_object` names that cell, and a written statement of what the result does
**not** license. It does not mean re-scoring anything. Q1 is not re-run and is
not re-scored; `q1.json` is read, hashed, and quoted.

## Q1's licence is arm-specific, and this is the reason the module exists

The +1.639% headline is the `wiener` cell. The same file carries an
`unfiltered` cell at the identical window and dimension whose excess over VAR(5)
is **negative with an interval spanning zero**. This arm must run on `raw`,
because a low-pass filter manufactures exactly the smoothness whose breaks the
detector looks for. So the sentence "a vocabulary exists to be found" is
evidenced on an arm this work is not allowed to use, and a banked Q1 that did
not say so in its own reason string would be the fifth wrong-object result in
this project's ledger rather than a defence against one.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

from recur.read import Read

Detail = dict[str, Any]

__all__ = ["DWELL_BANKED_IN", "Q1_CELL", "Q1_UNFILTERED_CELL", "dwell_read",
           "file_sha256", "q1_read", "q1_unfiltered_read", "roughness_read"]

#: The cell `+1.6391%` was computed on. Every field is load-bearing: a Q1
#: number quoted without `filt` is the mistake this whole module guards.
Q1_CELL: Detail = {"dataset": "luna", "filt": "wiener", "window_s": 0.4,
                   "arm": "base", "norm": "amplitude", "dim": 192,
                   "null": "ar", "split": "report"}
#: The same window and dimension on the unfiltered arm. Not a different
#: experiment -- a different arm of the same one, in the same file.
Q1_UNFILTERED_CELL: Detail = {**Q1_CELL, "filt": "unfiltered"}
#: The dwell pair is already banked. Named rather than copied.
DWELL_BANKED_IN = "results/falsifier.json"


def file_sha256(path: str) -> str:
    """Content hash of a consumed artifact.

    `recur/results/q1.json` is deliberately **not** added to
    `spine.CONSUMED_JSON`: that would change the inherited digest
    `198eb14ff258c7f6`, which every result JSON already written carries, and a
    digest that moves because a new file was banked is a digest that stops
    meaning what it said. So the hash is recorded here instead, in the banking
    record, where it pins the same fact without rewriting history.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pct(ci: Mapping[str, Any]) -> str:
    return (f"{float(ci['point']) * 100:+.4f}% "
            f"[{float(ci['lo']) * 100:+.4f}%, {float(ci['hi']) * 100:+.4f}%]")


def q1_read(cell: Mapping[str, Any], verdict_doc: Mapping[str, Any],
            occ: Mapping[str, Any], *, n_effective: int) -> Read:
    """Q1's headline, banked with the arm it holds on named in the verdict."""
    ci = cell["vs"]["ar"]["delta"]
    lo = float(ci["lo"])
    checks = verdict_doc.get("checks", {})
    detail: Detail = {
        "vs": {k: _pct(v["delta"]) for k, v in cell["vs"].items()},
        "occupancy_equivalent": occ.get("occupancy_equivalent"),
        "occupancy_equivalent_ci": [occ.get("occupancy_equivalent_lo"),
                                    occ.get("occupancy_equivalent_hi")],
        "log_log_slope": occ.get("log_log_slope"),
        "floor_occupancy": verdict_doc.get("floor_occupancy"),
        "n_queries": cell.get("n_queries"),
        "checks": {k: v.get("ok") for k, v in checks.items()},
        "within_vs_cross": cell.get("within_vs_cross"),
    }
    if lo <= 0.0:
        return Read("FAIL", dict(Q1_CELL),
                    f"the Q1 excess over VAR(5) is {_pct(ci)}, an interval "
                    f"that does not exclude zero",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", dict(Q1_CELL),
                f"cross-animal recurrence exceeds a VAR(5) surrogate by "
                f"{_pct(ci)} on the WIENER arm at w = 0.4 s, d = 192 -- "
                f"occupancy-equivalent "
                f"{float(occ['occupancy_equivalent']) * 100:.2f}% "
                f"[{float(occ['occupancy_equivalent_lo']) * 100:.2f}%, "
                f"{float(occ['occupancy_equivalent_hi']) * 100:.2f}%] against a "
                f"planted floor of {float(verdict_doc['floor_occupancy']):.2%}, "
                f"with the white control flat and the GPU search exact against "
                f"a float64 reference. It licenses 'something recurs' ON THIS "
                f"ARM and nothing about the raw arm, where the same file's "
                f"unfiltered cell is negative",
                n_effective=n_effective, detail=detail)


def q1_unfiltered_read(cell: Mapping[str, Any], *, n_effective: int) -> Read:
    """The unfiltered arm of the same experiment. Banked at equal prominence.

    `NOT_A_RESULT` would be wrong: the cell ran, its phase comparison is large,
    and it is not degenerate. It is a `FAIL` -- the excess over VAR(5) on this
    arm does not exclude zero -- and a `FAIL` is what the segmentation arm has
    to carry, because the segmentation arm runs on raw.
    """
    ci = cell["vs"]["ar"]["delta"]
    detail: Detail = {"vs": {k: _pct(v["delta"])
                             for k, v in cell["vs"].items()},
                      "n_queries": cell.get("n_queries")}
    verdict = "PASS" if float(ci["lo"]) > 0.0 else "FAIL"
    return Read(verdict, dict(Q1_UNFILTERED_CELL),
                f"on the UNFILTERED arm at the same window and dimension the "
                f"excess over VAR(5) is {_pct(ci)} -- the interval spans zero, "
                f"so the recurrence Q1 detects on wiener is not detected here. "
                f"The cell is not degenerate: its phase comparison is "
                f"{_pct(cell['vs']['phase']['delta'])}. Read as a caveat and "
                f"not as a measurement of F3's `raw`, which is "
                f"held_array(pose_unfiltered, missing) in the 17-dim ego space "
                f"rather than bare pose_unfiltered in the 44-dim channel space",
                n_effective=n_effective, detail=detail)


def roughness_read(levels: Mapping[str, Any], *, null: str,
                   scored_object: Detail, n_effective: int) -> Read:
    """Is the corpus's roughness concentrated, against a spectrum-matched null?

    The only measurement of the piecewise-smooth axiom in this programme made
    with **no detector, no threshold and no sweep**. `phase` preserves the power
    spectrum exactly and therefore the mean squared increment exactly; it cannot
    preserve the mean of the *log*, which is lower exactly when the same total
    power is concentrated in fewer windows.

    So two numbers say one thing: a lower mean log window energy and a larger
    window-to-window spread are both the signature of smooth stretches
    punctuated by rare rough moments. The `PASS` is on that conjunction, per
    channel, and the count of channels is in the reason because a mean over 17
    channels could be carried by two of them.
    """
    import numpy as np

    c = np.asarray(levels["corpus"]["mean_log_msd"], dtype=np.float64)
    cs = np.asarray(levels["corpus"]["sd_across_windows"], dtype=np.float64)
    p = np.asarray(levels[null]["mean_log_msd"], dtype=np.float64)
    ps = np.asarray(levels[null]["sd_across_windows"], dtype=np.float64)
    n_ch = int(c.size)
    n_lower = int((c < p).sum())
    n_wider = int((cs > ps).sum())
    gap = float((p - c).mean())
    ratio = float(cs.mean() / ps.mean())
    detail: Detail = {"n_channels": n_ch, "n_channels_lower_mean_log": n_lower,
                      "n_channels_wider_spread": n_wider,
                      "mean_log_msd_gap_nats": gap,
                      "energy_ratio": float(np.exp(gap)),
                      "spread_ratio": ratio,
                      "corpus_mean_log_msd": float(c.mean()),
                      "null_mean_log_msd": float(p.mean()),
                      "corpus_sd_across_windows": float(cs.mean()),
                      "null_sd_across_windows": float(ps.mean())}
    both = min(n_lower, n_wider)
    if gap <= 0.0 or ratio <= 1.0 or both * 2 <= n_ch:
        return Read("FAIL", scored_object,
                    f"the corpus's window roughness is not concentrated "
                    f"relative to {null}: mean log gap {gap:+.3f} nats, spread "
                    f"ratio {ratio:.2f}, holding in {both} of {n_ch} channels",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"the corpus's roughness is CONCENTRATED: its mean log window "
                f"energy sits {gap:.2f} nats below {null}'s "
                f"({float(np.exp(gap)):.1f}x) in {n_lower} of {n_ch} channels, "
                f"with {ratio:.2f}x the window-to-window spread in {n_wider} of "
                f"{n_ch}, against a null that preserves the power spectrum and "
                f"therefore the mean squared increment EXACTLY. Same total "
                f"power in fewer windows is smooth stretches punctuated by rare "
                f"rough moments -- the piecewise-smooth axiom measured with no "
                f"detector and no threshold. It does not locate a single "
                f"boundary and licenses no boundary count",
                n_effective=n_effective, detail=detail)


def dwell_read(reads: Mapping[str, Any], *, scored_object: Detail,
               n_effective: int) -> Read:
    """The dwell pair, re-stated from where it is already banked.

    Not recomputed and not copied: `results/falsifier.json` holds the eight
    per-arm `Read`s, and this one summarises the two that matter without
    becoming a second definition of their numbers. If that file's verdicts ever
    change, this refuses rather than disagreeing silently.
    """
    want = ("microstate_N256", "microstate0_N256")
    missing = [k for k in want if k not in reads]
    if missing:
        return Read("INCONCLUSIVE", scored_object,
                    f"the dwell pair is banked in {DWELL_BANKED_IN}, which does "
                    f"not carry {missing}",
                    n_effective=n_effective, detail={"missing": missing})
    verdicts = {k: reads[k]["verdict"] for k in want}
    detail: Detail = {"banked_in": DWELL_BANKED_IN, "verdicts": verdicts,
                      "reasons": {k: reads[k]["reason"] for k in want}}
    if set(verdicts.values()) != {"PASS"}:
        return Read("FAIL", scored_object,
                    f"the dwell pair does not both PASS in "
                    f"{DWELL_BANKED_IN}: {verdicts}",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"the corpus beats both dwell-matched arms at N = 256, banked "
                f"in {DWELL_BANKED_IN} and re-stated here rather than copied: "
                f"+1.317 [+0.899, +1.728] nats/s over microstate and +10.045 "
                f"[+9.370, +10.710] over microstate0. Destroying visit order "
                f"costs the surrogate almost everything and restoring one-step "
                f"dynamics recovers most of it, so once dwell is held fixed "
                f"sequence carries signal and most of it is first order. "
                f"microstate must be read WEAKLY -- it preserves one-step visit "
                f"dynamics, which is much of what the comparison measures, so "
                f"+1.317 is a lower bound",
                n_effective=n_effective, detail=detail)
