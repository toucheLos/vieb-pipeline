r"""Step 2. Continuous 17-vector to discrete symbol, granularity not assumed.

The tokenizer **is** the dimensionality reduction on this branch. Nothing else
reduces: 17 dims is already small, and PCA is linear and variance-seeking, so on
a corpus whose density is 96-99% single-mode it would spend its components on
immobility. Stage A measured the corpus sitting 24x further from every surrogate
than from itself *in a PCA basis* while matching ambient scale to 2% -- direct
evidence that a fitted basis need not preserve the structure a comparison depends
on. So the reduction happens here, and its granularity is selected by held-out
description length rather than argued for.

## Two arms, because k-means will otherwise buy immobility

Plain k-means finds density modes. The density is one mode, so at N = 2048 most
symbols would subdivide freezing and the moving frames -- which on a
fear-conditioning corpus are the signal -- would share a handful. That is the
frequency-versus-importance problem entering at the tokenizer, and it does not
fix itself downstream.

The second arm allocates symbols per **speed quintile**: N/5 centroids fitted
within each quintile of frames, so the fast tail gets the same alphabet budget as
the slow bulk. Which arm is right is not argued here. Both are scored by MDL and
the objective decides.

## What is fitted where

`fit_partition` is called on **tune frames only** -- 60 animals, 770 recordings.
Choosing a vocabulary on the animals a result is read from is exactly how a sweep
manufactures its own win, and it is the failure this whole branch exists to
avoid.

The standardising SD is **shared between the arms and computed once**. Two arms
standardised differently are not two tokenizations of one representation; they
are two representations, and the MDL comparison between them would be measuring
the wrong difference.

## Why the SD is segment-balanced and pooled

`recur.geom.state.segment_balanced_sd` weights **bouts** equally rather than
frames, because a frame-uniform SD is dominated by whatever the animal spends its
time doing -- freezing -- which would set the variance scale from the least
informative behaviour. And it is pooled over the corpus, never per recording:
per-recording standardisation removes exactly the between-recording differences a
null has to be able to see, and it has disarmed a control on this project once
already.

Bouts are cut **per recording** (`represent.speed_terciles` says why: px/s is
uncalibrated across boxes, so a global cut would make bout structure a property
of the arena) and then offset into corpus coordinates. No bout crosses a seam.

## Abstain is carried, never scrubbed

A flagged frame becomes `ABSTAIN` and stays in the stream. Cleaning harder means
deciding in advance what is real; abstaining carries the uncertainty forward and
lets the objective measure whether it mattered. The code is imported from
`recur.label.nonparametric` rather than redeclared, so there is one definition of
-1 in this project and not three.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, cast

import numpy as np
import numpy.typing as npt
from recur.geom import represent as rep
from recur.label.nonparametric import ABSTAIN
from recur.null import microstate
from recur.read import Read

from vieb.checks import assert_pmf, assert_share

F64 = npt.NDArray[np.float64]
F32 = npt.NDArray[np.float32]
I32 = npt.NDArray[np.int32]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

#: recur predates annotation discipline and `mypy.ini` follows its imports
#: silently, so a fully unannotated function there reads as untyped here. This
#: is the boundary, declared once and named, rather than a blanket `ignore` on
#: the call site. `speed_terciles` needs no wrapper -- it carries an annotation
#: already and mypy accepts it.
_segment_balanced_sd = cast(
    "Callable[..., tuple[Any, int]]", rep.segment_balanced_sd)

__all__ = ["ABSTAIN", "ARMS", "N_STRATA", "STATE_GRID", "allocate", "assign",
           "bouts", "corpus_sd", "fit", "occupancy", "speed", "strata_edges",
           "stratum_of", "alphabet_read"]

#: The two arms. `plain` is k-means over all tune frames; `speed` allocates the
#: alphabet across speed quintiles. Names are stable -- they appear in shard
#: paths and in every result document.
ARMS: tuple[str, ...] = ("plain", "speed")

#: Deliberately over-fine. The stop condition on run length retires whichever
#: end of this grid is fitting quantization noise; it is not chosen in advance.
STATE_GRID: tuple[int, ...] = (256, 512, 1024, 2048)

#: Quintiles. Five because it is the coarsest split that gives the fast tail its
#: own budget while leaving >= 51 symbols per stratum at the smallest N.
N_STRATA = 5

#: The twist translation channels, `ego.CHANNELS` 14 and 15. Speed is their
#: magnitude, in body lengths per second. `omega` is NOT included: it is
#: rotation, it carries a different unit, and a stratum defined by their sum
#: would put a fast pivot in the same bin as a slow walk.
SPEED_COLS: tuple[int, int] = (14, 15)

#: Fraction of frames one symbol may hold before the alphabet is degenerate.
#: Inherited from the brief's own phrasing, not selected against an outcome.
DOMINANT_SHARE = 0.40
#: Fraction of the alphabet that may go unused before N is above what the data
#: supports.
DEAD_SHARE = 0.10


def speed(x: npt.ArrayLike) -> F64:
    """Translational speed in body lengths per second, from the twist columns.

    Read off `X` rather than recomputed from keypoints. A second pass over the
    pose would be a second implementation of a quantity the representation
    already carries exactly, and the two could disagree.
    """
    a = np.asarray(x, dtype=np.float64)
    return np.hypot(a[:, SPEED_COLS[0]], a[:, SPEED_COLS[1]])


def bouts(sp: npt.ArrayLike, fps: float,
          bounds: npt.ArrayLike) -> list[tuple[int, int]]:
    """Speed-tercile bouts for a whole animal, cut per recording.

    `bounds` are the animal's recording seams. Each recording is cut on its own
    speed quantiles and the resulting spans are offset into animal coordinates,
    so no bout spans a seam. Cutting the concatenated array instead would make
    the terciles a property of whichever recordings happened to be adjacent.
    """
    s = np.asarray(sp, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.int64)
    out: list[tuple[int, int]] = []
    for r in range(b.shape[0] - 1):
        lo, hi = int(b[r]), int(b[r + 1])
        for a0, a1 in rep.speed_terciles(s[lo:hi], fps):
            out.append((lo + int(a0), lo + int(a1)))
    return out


def corpus_sd(x: npt.ArrayLike, sp: npt.ArrayLike, fps: float,
              bounds: npt.ArrayLike,
              valid: npt.ArrayLike | None = None) -> tuple[F64, int]:
    """Segment-balanced per-channel SD for one animal, plus its bout count.

    Returned as a `(sum-ready sd, n_bouts)` pair so a caller can pool across
    animals by bout count. Pooling by animal count instead would weight a
    13-recording animal the same as a 3-recording one.
    """
    sd, n = _segment_balanced_sd(
        np.asarray(x, dtype=np.float64), bouts(sp, fps, bounds),
        usable=None if valid is None else np.asarray(valid, dtype=bool))
    if sd is None:
        return np.zeros(np.asarray(x).shape[1], dtype=np.float64), 0
    return np.asarray(sd, dtype=np.float64), int(n)


def strata_edges(sp: npt.ArrayLike, mask: npt.ArrayLike,
                 *, n_strata: int = N_STRATA) -> F64:
    """Interior quantile cuts of speed, from tune frames only.

    `n_strata - 1` edges. Computed on the same frames the partition is fitted
    on, for the same reason: a stratum boundary is a hyperparameter.
    """
    s = np.asarray(sp, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite speeds to cut strata on")
    q = np.linspace(0.0, 1.0, int(n_strata) + 1)[1:-1]
    return np.asarray(np.quantile(s, q), dtype=np.float64)


def stratum_of(sp: npt.ArrayLike, edges: npt.ArrayLike) -> I32:
    """Which speed stratum each frame falls in. Non-finite speeds give -1."""
    s = np.asarray(sp, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    out = np.searchsorted(e, s, side="right").astype(np.int32)
    out[~np.isfinite(s)] = -1
    return out


def allocate(n_states: int, n_strata: int = N_STRATA) -> list[int]:
    """Symbols per stratum, summing to `n_states` exactly.

    Remainder goes to the low strata rather than being dropped, so the grid's N
    is the alphabet size in fact and not approximately -- the MDL codebook term
    charges `|codebook|`, and an arm quietly carrying 2045 symbols where the
    other carries 2048 would be charged differently for no stated reason.
    """
    if n_strata <= 0 or n_states < n_strata:
        raise ValueError(f"cannot split {n_states} states into {n_strata}")
    base, extra = divmod(int(n_states), int(n_strata))
    return [base + (1 if i < extra else 0) for i in range(int(n_strata))]


def fit(x: npt.ArrayLike, sd: npt.ArrayLike, cols: npt.ArrayLike,
        frame_mask: npt.ArrayLike, *, arm: str, n_states: int, seed: int = 0,
        sp: npt.ArrayLike | None = None, sample: int = 2_000_000,
        logger: Any = None) -> Detail:
    """Fit one alphabet on `frame_mask`, which must select the tune split.

    Both arms go through `recur.null.microstate.fit_partition`, which is already
    MiniBatchKMeans over `sd`-standardised frames with a recorded `input_hash`.
    The `speed` arm calls it once per stratum with the mask narrowed, so the two
    arms differ in which frames each centroid is fitted on and in nothing else.
    """
    m = np.asarray(frame_mask, dtype=bool)
    if arm == "plain":
        part = microstate.fit_partition(x, sd, cols, m, n_states=n_states,
                                        seed=seed, sample=sample, logger=logger)
        return {"arm": "plain", "n_states": int(n_states), "seed": int(seed),
                "centroids": np.asarray(part["centroids"], dtype=np.float32),
                "sd": np.asarray(sd, dtype=np.float64),
                "cols": np.asarray(cols, dtype=np.int64),
                "offsets": np.array([0], dtype=np.int64),
                "edges": np.zeros(0, dtype=np.float64),
                "n_fit_frames": int(part["n_fit_frames"]),
                "inertia": float(part["inertia"]),
                "input_hash": str(part["input_hash"])}
    if arm != "speed":
        raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")
    if sp is None:
        raise ValueError("the speed arm needs a speed vector")

    edges = strata_edges(sp, m)
    strat = stratum_of(sp, edges)
    sizes = allocate(int(n_states))
    cents, offsets, fit_frames, inertia, hashes = [], [0], 0, 0.0, []
    for i, k in enumerate(sizes):
        sub = m & (strat == i)
        if int(sub.sum()) < k:
            raise ValueError(
                f"stratum {i} has {int(sub.sum())} tune frames for {k} states")
        part = microstate.fit_partition(x, sd, cols, sub, n_states=k, seed=seed,
                                        sample=max(1, sample // len(sizes)),
                                        logger=logger)
        cents.append(np.asarray(part["centroids"], dtype=np.float32))
        offsets.append(offsets[-1] + k)
        fit_frames += int(part["n_fit_frames"])
        inertia += float(part["inertia"])
        hashes.append(str(part["input_hash"]))
    return {"arm": "speed", "n_states": int(n_states), "seed": int(seed),
            "centroids": np.concatenate(cents),
            "sd": np.asarray(sd, dtype=np.float64),
            "cols": np.asarray(cols, dtype=np.int64),
            "offsets": np.asarray(offsets, dtype=np.int64),
            "edges": edges, "sizes": sizes,
            "n_fit_frames": fit_frames, "inertia": inertia,
            "input_hash": "+".join(hashes)}


def assign(x: npt.ArrayLike, model: Mapping[str, Any], *,
           abstain: npt.ArrayLike | None = None,
           sp: npt.ArrayLike | None = None) -> I32:
    """Symbol per frame, with `ABSTAIN` where the frame is flagged.

    Abstain is applied **after** assignment rather than by dropping rows, so the
    output is frame-aligned with `X` and a run-length encoder can see where the
    gaps are. Dropping them would splice two distant moments into one run.
    """
    a = np.asarray(x)
    n = a.shape[0]
    out = np.full(n, ABSTAIN, dtype=np.int32)
    cent = np.asarray(model["centroids"], dtype=np.float32)
    sd = np.asarray(model["sd"], dtype=np.float64)
    cols = np.asarray(model["cols"], dtype=np.int64)

    if str(model["arm"]) == "plain":
        out = microstate.assign_states(a, cent, sd, cols).astype(np.int32)
    else:
        if sp is None:
            raise ValueError("the speed arm needs a speed vector to assign")
        offsets = np.asarray(model["offsets"], dtype=np.int64)
        strat = stratum_of(sp, np.asarray(model["edges"], dtype=np.float64))
        for i in range(offsets.shape[0] - 1):
            sel = strat == i
            if not sel.any():
                continue
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            local = microstate.assign_states(a[sel], cent[lo:hi], sd, cols)
            out[sel] = (local + lo).astype(np.int32)
        out[strat < 0] = ABSTAIN

    if abstain is not None:
        out[np.asarray(abstain, dtype=bool)] = ABSTAIN
    return out


def occupancy(labels: npt.ArrayLike, n_states: int) -> Detail:
    """Frame share per symbol, and the two numbers that make it readable.

    `top_share` and `dead_frac` are reported because a mean occupancy of 1/N
    holds by construction and says nothing. Abstain is excluded from the
    denominator and reported separately -- folding it in would let a high
    abstain rate flatter the distribution.
    """
    y = np.asarray(labels, dtype=np.int64)
    used = y[y != ABSTAIN]
    counts = np.bincount(used, minlength=int(n_states)).astype(np.float64)
    total = float(counts.sum())
    base: Detail = {
        "n_states": int(n_states),
        "n_frames": int(y.shape[0]),
        "n_labelled": int(used.shape[0]),
        "abstain_frac": float((y == ABSTAIN).mean()) if y.size else float("nan"),
        "counts": counts.astype(np.int64).tolist(),
    }
    if total <= 0:
        # Every share is 0/0. Returning zeros instead would make a degenerate
        # alphabet read as one with no dominant symbol -- the most flattering
        # possible description of having labelled nothing.
        return {**base, "top_share": float("nan"), "top10_share": float("nan"),
                "dead_frac": 1.0, "gini": float("nan")}
    share = assert_pmf(counts / total, name="occupancy frame share")
    order = np.sort(share)[::-1]
    cum = np.cumsum(order)
    return {**base,
            "top_share": float(order[0]),
            "top10_share": float(cum[min(9, cum.size - 1)]),
            "dead_frac": float((counts == 0).mean()),
            "gini": _gini(counts)}


def _gini(counts: npt.ArrayLike) -> float:
    c = np.sort(np.asarray(counts, dtype=np.float64))
    n = c.shape[0]
    if n == 0 or c.sum() <= 0:
        return float("nan")
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * c).sum()) / (n * c.sum()) - (n + 1.0) / n)


def alphabet_read(occ: Mapping[str, Any], *, scored_object: Detail,
                  n_effective: int) -> Read:
    """Is this alphabet a usable tokenization of the representation?

    Two ways it is not, and they are different failures. **One symbol holding
    most of the frames** means the run-length stream will be dominated by it and
    every transition statistic will be about that symbol. **Most of the alphabet
    unused** means N is above what the data supports at this granularity, which
    is a statement about the grid rather than about the corpus -- hence
    `GRID_LIMITED` rather than `FAIL`.
    """
    top = float(occ["top_share"])
    dead = float(occ["dead_frac"])
    n = int(occ["n_states"])
    detail: Detail = {k: occ[k] for k in
                      ("n_states", "n_labelled", "abstain_frac", "top_share",
                       "top10_share", "dead_frac", "gini")}
    tail = (f"top symbol holds {top:.2%} of labelled frames, the top ten hold "
            f"{float(occ['top10_share']):.2%}, {dead:.2%} of the {n} symbols "
            f"are unused, Gini {float(occ['gini']):.3f}, and "
            f"{float(occ['abstain_frac']):.2%} of frames abstain")
    if not np.isfinite(top):
        return Read("NOT_A_RESULT", scored_object,
                    "no labelled frames: every frame in this arm abstained, so "
                    "there is no occupancy to report",
                    n_effective=n_effective, degenerate=True, detail=detail)
    if top >= DOMINANT_SHARE:
        return Read("FAIL", scored_object,
                    f"one symbol dominates the alphabet -- {tail}. At this "
                    f"share the run-length stream is mostly that symbol and "
                    f"every transition statistic would be about it",
                    n_effective=n_effective, detail=detail)
    if dead >= DEAD_SHARE:
        return Read("GRID_LIMITED", scored_object,
                    f"N = {n} is above what the data supports at this "
                    f"granularity -- {tail}. That is a property of the sweep "
                    f"grid, not of the corpus, and a smaller N on the same "
                    f"grid may not have it",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"the alphabet is non-degenerate: {tail}. No symbol reaches the "
                f"{DOMINANT_SHARE:.0%} dominance threshold and under "
                f"{DEAD_SHARE:.0%} of it is unused",
                n_effective=n_effective, detail=detail)
