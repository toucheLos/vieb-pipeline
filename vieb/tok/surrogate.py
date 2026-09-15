r"""Surrogates in the egocentric space, for the falsifier.

`TOK_PREREGISTRATION.md` §7: *if surrogates achieve comparable MDL, the stack is
fitting quantization structure rather than behaviour.* The claim at risk is rung
1 beating rung 0 by +11.3 nats/s on 89 of 89 animals. This module builds the
signals that test it.

## Why in ego space and not in pose

The pipeline under test consumes the 17-dimensional egocentric array. A
surrogate built on raw pose would then pass through the SE(2) transform, which
is a bijection and would faithfully carry whatever the surrogate has -- but it
would also be one more stage between the null and the thing being nulled. Making
the surrogate where the pipeline starts keeps the comparison to one substitution.

## The dead directions, and why they are copied rather than generated

Removing SE(2) equivariance from 14 coordinates costs three degrees of freedom.
Two of them are whole columns -- `s` at the origin keypoint is identically
`(0, 0)` -- and the third is a linear combination, the aligned axis's
y-component. A surrogate carrying variance in a direction the corpus never moves
in would be trivially separable from it, and would hand the quantizer structure
the corpus does not have.

**Phase randomisation preserves all three for free.** One phase per frequency,
shared across channels, multiplies every column by the same `exp(i*phi)`, so any
linear relationship between columns survives exactly and a zero column stays
zero. That is a substantive reason to prefer it as the primary arm, not just a
convenience.

**A VAR does not.** Its residual noise is full-rank in whatever subspace it was
fitted on, so it is fitted on the live channels only and the dead ones are
copied.

## What each destroys

`phase` keeps every channel's power spectrum and the cross-spectrum exactly --
so autocorrelation, so the run-length structure that quantizing a smooth signal
produces -- and destroys everything above second order. `var5` keeps what five
lags can express and destroys the rest. Neither contains a stereotyped sequence,
a repeated motif, or a behaviour.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
from recur.read import Read
from recur.null import microstate as ms
from recur.recurrence import phase as ph
from recur.recurrence import surrogate as sg

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

__all__ = ["KINDS", "N_MICROSTATES", "OU_DIM", "OU_TAU_S", "VAR_ORDER",
           "emit_visits", "falsifier_read", "generate", "live_columns",
           "simulate_marginal"]

#: Every arm. `phase` was primary for the falsifier; `microstate`/`microstate0`
#: are the dwell-matched pair registered in `DWELL_PREREGISTRATION.md`; `white`
#: and `ou` bracket the segmentation nulls.
KINDS: tuple[str, ...] = ("phase", "var5", "white", "ou",
                          "microstate", "microstate0")
#: Kinds that need a fitted partition (and, at order 1, a chain).
VISIT_KINDS: tuple[str, ...] = ("microstate", "microstate0")

#: Lag order for the VAR arm. Five, because VAR(5) is the registered null behind
#: Q1's +1.639% and the two results should sit on one footing.
VAR_ORDER = 5
#: Microstate partition size, matching recur's `microstate_N500k1` arm. This is
#: the NULL's internal granularity and is not the tokenizer's alphabet -- the
#: tokenizer fits its own partition on the surrogate afterwards.
N_MICROSTATES = 500
#: OU latent dimension and time constant, inherited from
#: `recur.recurrence.surrogate.ou_continuum` unchanged.
OU_DIM = 3
OU_TAU_S = 0.5

#: What each arm actually holds fixed, in words, so a verdict cannot describe
#: the wrong null. The first version of `falsifier_read` was written for `phase`
#: and `var5` and said "second-order-matched" for every arm -- which is false of
#: the visit-stitching pair, whose whole point is that they match DWELL and not
#: the spectrum.
WHAT_IS_MATCHED: dict[str, str] = {
    "phase": "a smooth signal with the corpus's exact power spectrum and "
             "cross-spectrum",
    "var5": "a VAR(5) fitted to each recording's own channels",
    "white": "i.i.d. noise at each recording's own per-channel mean and SD",
    "ou": "a smooth, aperiodic flow with no discrete states",
    "microstate": "a signal built from the corpus's own frames, preserving both "
                  "its dwell distribution and its one-step visit dynamics",
    "microstate0": "a signal built from the corpus's own frames, preserving its "
                   "dwell distribution with the visit order destroyed",
}

#: Arms whose construction overlaps what rung 1 measures, so a result against
#: them is weaker in a way the verdict must say out loud.
CIRCULAR_KINDS: tuple[str, ...] = ("microstate",)

#: recur predates annotation discipline; the boundary is declared once.
_randomize_block = cast("Callable[..., F64]", ph.randomize_block)
_fit_var = cast("Callable[..., Any]", sg.fit_var)
_simulate_var = cast("Callable[..., F64]", sg.simulate_var)
_white_surrogate = cast("Callable[..., F64]", sg.white_surrogate)
_assign_states = cast("Callable[..., I64]", ms.assign_states)
_visits = cast("Callable[..., I64]", ms.visits)
_simulate_visits = cast("Callable[..., I64]", ms.simulate_visits)


def live_columns(x: npt.ArrayLike, *, tol: float = 0.0) -> BOOL:
    """Columns that actually vary. The rest are the SE(2) rank deficit."""
    a = np.asarray(x, dtype=np.float64)
    return np.asarray(a.std(axis=0) > tol, dtype=bool)


def _white_block(block: F64, rng: np.random.Generator) -> F64:
    """i.i.d. Gaussian at this recording's own per-channel mean and SD.

    `recur.recurrence.surrogate.white_surrogate` is already shape-generic, so it
    is called rather than re-derived.
    """
    return np.asarray(_white_surrogate(block, rng), dtype=np.float64)


def _ou_block(block: F64, rng: np.random.Generator, *, fps: float,
              dim: int = OU_DIM, tau_s: float = OU_TAU_S) -> F64:
    """A smooth, aperiodic flow with no discrete states, in `(T, C)`.

    `recur.recurrence.surrogate.ou_continuum` reshapes `(T, K, 2)` and cannot be
    called on a channel matrix, so the same process is written here on `(T, C)`.
    The parameters are its defaults, not new choices: mean-reverting and
    non-periodic, because a periodic flow would recur trivially and manufacture
    the very thing a recurrence test looks for.
    """
    t, c = block.shape
    theta = 1.0 / max(tau_s * fps, 1.0)
    z = np.zeros((t, int(dim)), dtype=np.float64)
    for i in range(1, t):
        z[i] = z[i - 1] * (1.0 - theta) + np.sqrt(2.0 * theta) * \
            rng.standard_normal(int(dim))
    w = rng.standard_normal((int(dim), c)) / np.sqrt(float(dim))
    flat = z @ w
    scale = float(block.std())
    flat = flat / max(float(flat.std()), 1e-9) * scale
    return np.asarray(flat + block.mean(axis=0), dtype=np.float64)


def simulate_marginal(visit_states: npt.ArrayLike, n: int,
                      rng: np.random.Generator,
                      *, max_passes: int = 32) -> tuple[I64, int]:
    """`n` visit states drawn from the marginal, **never repeating**.

    The no-repeat condition is registered in `DWELL_PREREGISTRATION.md` §3 and is
    not cosmetic: run-length encoding guarantees `AA` cannot occur in the data,
    and `pooled_chain` zeroes its diagonal for the same reason. An unconditioned
    marginal draw would hand the surrogate a transition the corpus cannot
    contain, which the tokenizer would then have to encode.

    Drawn in a batch and repaired, rather than one rejection-sampled draw per
    visit: a recording holds thousands of visits and the corpus holds 3,846
    recordings. The residual repeat count is returned rather than assumed zero --
    a recording that visits only one microstate cannot avoid repeating, and that
    is a fact about the recording.
    """
    pool = np.asarray(visit_states, dtype=np.int64)
    uniq, cnt = np.unique(pool, return_counts=True)
    p = cnt.astype(np.float64) / float(cnt.sum())
    out = np.asarray(rng.choice(uniq, size=int(n), p=p), dtype=np.int64)
    for _ in range(int(max_passes)):
        bad = np.flatnonzero(out[1:] == out[:-1]) + 1
        if bad.size == 0:
            break
        out[bad] = rng.choice(uniq, size=bad.size, p=p)
    return out, int(np.sum(out[1:] == out[:-1]))


def emit_visits(x: npt.ArrayLike, states: npt.ArrayLike,
                order_seq: npt.ArrayLike, rng: np.random.Generator,
                *, min_len: int = 1) -> tuple[F64, Detail]:
    r"""Stitch this recording's own real visits into the given order.

    **Concatenation, not integration, and that is the whole reason this exists
    rather than calling `microstate.emit`.** recur's emitter carries `cur_mu`
    and `cur_th` across every join and integrates the donor's increments onto a
    running pose, because its channel space holds **absolute** position and
    heading and a naive concatenation would teleport the animal. The egocentric
    representation holds shape in the body frame plus SE(2) *increments*: there
    is no absolute to be discontinuous in, so the joins need no integration and
    the stitch is a copy.

    Every emitted frame is a real observed frame and every emitted visit has a
    real observed length, so the manifold and the dwell distribution are
    preserved by construction. What is destroyed is the order.
    """
    a = np.asarray(x, dtype=np.float64)
    v = np.asarray(_visits(np.asarray(states, dtype=np.int64)),
                   dtype=np.int64)
    if v.shape[0]:
        v = v[(v[:, 2] - v[:, 1]) >= int(min_len)]
    t = a.shape[0]
    if v.shape[0] < 2:
        return a.copy(), {"n_visits": 0, "n_fallback": 0, "degenerate": True,
                          "median_visit_frames": float("nan")}
    pools = {int(sid): np.flatnonzero(v[:, 0] == sid)
             for sid in np.unique(v[:, 0])}
    present = np.array(sorted(pools), dtype=np.int64)
    seq = np.asarray(order_seq, dtype=np.int64)

    out = np.empty_like(a)
    pos = i = n_fall = 0
    while pos < t:
        want = int(seq[i % seq.size])
        i += 1
        if want not in pools:
            n_fall += 1
            want = int(present[np.argmin(np.abs(present - want))])
        row = int(rng.choice(pools[want]))
        lo, hi = int(v[row, 1]), int(v[row, 2])
        n = min(hi - lo, t - pos)
        if n <= 0:
            continue
        out[pos:pos + n] = a[lo:lo + n]
        pos += n
    return out, {"n_visits": int(i), "n_fallback": int(n_fall),
                 "degenerate": False,
                 "n_states_present": int(present.size),
                 "median_visit_frames": float(np.median(v[:, 2] - v[:, 1]))}


def _phase_block(block: F64, rng: np.random.Generator) -> F64:
    return np.asarray(_randomize_block(block, rng), dtype=np.float64)


def _var_block(block: F64, rng: np.random.Generator, *, order: int) -> F64:
    """VAR(`order`) on the live columns; dead columns copied through.

    Falls back to phase randomisation when the fit is degenerate -- too few
    frames for the lag order, or a singular residual covariance -- which is
    `recur.recurrence.surrogate.ar_surrogate`'s own behaviour and is recorded
    per recording rather than hidden.
    """
    live = live_columns(block)
    if block.shape[0] <= order * (int(live.sum()) + 1) or not live.any():
        return _phase_block(block, rng)
    sub = block[:, live]
    mu = sub.mean(axis=0)
    fit = _fit_var(sub - mu, order)
    if fit is None:
        return _phase_block(block, rng)
    sim = _simulate_var(fit, sub.shape[0], rng, seed_state=(sub[:order] - mu))
    out = block.copy()
    out[:, live] = np.asarray(sim, dtype=np.float64) + mu
    return out


def generate(x: npt.ArrayLike, bounds: npt.ArrayLike, kind: str,
             rng: np.random.Generator, *, order: int = VAR_ORDER,
             fps: float = 30.0, fit: Mapping[str, Any] | None = None) -> Detail:
    """A surrogate of `x`, one recording at a time. Never across a seam.

    A surrogate built across a boundary would splice two recordings into one
    spectrum and then be compared against data where crossing a seam is
    forbidden everywhere else in this repo.

    `fit` is required by the visit-stitching arms and carries the microstate
    partition (and, at order 1, the pooled chain). It is fitted on the
    **corpus's** tune split, once, because a per-recording chain at 500 states
    leaves the Laplace prior holding almost every row — `pooled_chain`'s own
    docstring measures 99.5% at N = 1000.
    """
    a_ = np.asarray(x, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.int64)
    if kind not in KINDS:
        raise ValueError(f"unknown surrogate {kind!r}; one of {KINDS}")
    if kind in VISIT_KINDS and fit is None:
        raise ValueError(f"{kind!r} needs a fitted partition")

    out = np.empty_like(a_)
    n_fallback = n_repeat = n_degenerate = 0
    visit_lens: list[float] = []
    for r in range(b.shape[0] - 1):
        lo, hi = int(b[r]), int(b[r + 1])
        blk = a_[lo:hi]
        if blk.shape[0] < 4:
            out[lo:hi] = blk
            continue
        if kind == "phase":
            out[lo:hi] = _phase_block(blk, rng)
        elif kind == "white":
            out[lo:hi] = _white_block(blk, rng)
        elif kind == "ou":
            out[lo:hi] = _ou_block(blk, rng, fps=fps)
        elif kind == "var5":
            out[lo:hi] = _var_block(blk, rng, order=order)
            live = live_columns(blk)
            if blk.shape[0] <= order * (int(live.sum()) + 1) or not live.any():
                n_fallback += 1
        else:
            assert fit is not None          # checked above; narrows for mypy
            emitted, diag = _visit_block(blk, kind, rng, fit=fit)
            out[lo:hi] = emitted
            n_fallback += int(diag.get("n_fallback", 0))
            n_repeat += int(diag.get("n_repeats", 0))
            n_degenerate += int(bool(diag.get("degenerate")))
            if np.isfinite(diag.get("median_visit_frames", np.nan)):
                visit_lens.append(float(diag["median_visit_frames"]))

    # A zero column must stay a zero column, whatever the generator did: the
    # quantizer would otherwise find structure in a direction the corpus has
    # none in. Free under phase randomisation and under visit stitching (real
    # frames), asserted here for every arm.
    dead = ~live_columns(a_)
    out[:, dead] = a_[:, dead]
    det: Detail = {"x": out.astype(np.float32), "kind": kind,
                   "n_recordings": int(b.shape[0] - 1),
                   "n_fallback": int(n_fallback),
                   "n_dead_columns": int(dead.sum())}
    if kind in VISIT_KINDS:
        det["n_residual_repeats"] = int(n_repeat)
        det["n_degenerate_recordings"] = int(n_degenerate)
        det["median_visit_frames"] = (float(np.median(visit_lens))
                                      if visit_lens else float("nan"))
    return det


def _visit_block(block: F64, kind: str, rng: np.random.Generator, *,
                 fit: Mapping[str, Any]) -> tuple[F64, Detail]:
    """One recording, stitched from its own real visits.

    The partition is the corpus's; the **visits, their lengths and their pool
    are this recording's own**, so the emitted signal has this recording's dwell
    distribution exactly. `microstate` orders them by the fitted chain;
    `microstate0` draws from the marginal with no self-repeats.
    """
    states = np.asarray(
        _assign_states(block, fit["centroids"], fit["sd"], fit["cols"]),
        dtype=np.int64)
    v = np.asarray(_visits(states), dtype=np.int64)
    if v.shape[0] < 2:
        return block.copy(), {"n_visits": 0, "n_fallback": 0, "n_repeats": 0,
                              "degenerate": True,
                              "median_visit_frames": float("nan")}
    n_want = max(int(v.shape[0]), 2)
    if kind == "microstate":
        seq = np.asarray(
            _simulate_visits(fit["chain"], n_want, start=v[:, 0], rng=rng,
                             support=np.unique(v[:, 0])), dtype=np.int64)
        n_rep = int(np.sum(seq[1:] == seq[:-1]))
    else:
        seq, n_rep = simulate_marginal(v[:, 0], n_want, rng)
    emitted, diag = emit_visits(block, states, seq, rng)
    diag["n_repeats"] = n_rep
    return emitted, diag


def falsifier_read(delta: Mapping[str, Any], corpus: Mapping[str, Any],
                   surrogate: Mapping[str, Any], *, kind: str,
                   scored_object: Detail, n_effective: int) -> Read:
    """Does the corpus beat its surrogate on `rung 1 - rung 0`?

    The stopping rule is `FALSIFIER_PREREGISTRATION.md` §7 and is read exactly
    as registered: the verdict turns on whether the animal-level interval on
    `d_corpus - d_surrogate` excludes zero, and on **no threshold for the size
    of the gap**. A threshold set now would be arbitrary; one set after seeing
    the number would be selected against the outcome.
    """
    ci = dict(delta.get("animal_interval", {}))
    lo, hi = float(ci.get("lo", np.nan)), float(ci.get("hi", np.nan))
    point = float(ci.get("point", np.nan))
    detail: Detail = {
        "kind": kind,
        "corpus_advantage": corpus.get("mean_delta"),
        "surrogate_advantage": surrogate.get("mean_delta"),
        "gap": point, "animal_interval": ci,
        "frame_interval": dict(delta.get("frame_interval", {})),
        "frame_interval_note": ("printed to show how much narrower the wrong "
                                "method looks; the verdict does not read it"),
        "n_animals": int(n_effective),
    }
    if not np.isfinite(point):
        return Read("NOT_A_RESULT", scored_object,
                    f"the {kind} comparison produced no finite per-animal "
                    f"difference, so there is nothing to test",
                    n_effective=n_effective, degenerate=True, detail=detail)
    tail = (f"corpus {float(corpus.get('mean_delta', np.nan)):+.3f} against "
            f"{kind} {float(surrogate.get('mean_delta', np.nan)):+.3f} nats/s, "
            f"a gap of {point:+.3f} [{lo:+.3f}, {hi:+.3f}] animal-level over "
            f"{n_effective} animals")

    what = WHAT_IS_MATCHED.get(kind, "this surrogate")
    caveat = ""
    if kind in CIRCULAR_KINDS:
        caveat = (" READ THIS ONE WEAKLY: the arm preserves one-step visit "
                  "dynamics and rung 1 IS a one-step model, so the two overlap "
                  "by construction and the gap is a lower bound rather than a "
                  "measurement")
    detail["what_is_matched"] = what
    detail["circular"] = kind in CIRCULAR_KINDS
    if lo > 0:
        return Read("PASS", scored_object,
                    f"the corpus beats its {kind} surrogate: {tail}, interval "
                    f"excluding zero. Rung 1's advantage over rung 0 is not "
                    f"explained by quantizing {what}. This does NOT license "
                    f"'the model learned behaviour' -- one alternative is not "
                    f"all of them.{caveat}",
                    n_effective=n_effective, detail=detail)
    if hi < 0:
        return Read("FAIL", scored_object,
                    f"the {kind} surrogate BEATS the corpus: {tail}, interval "
                    f"excluding zero on the wrong side. Every ladder number is "
                    f"withdrawn",
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                f"the corpus does not beat its {kind} surrogate: {tail}, and "
                f"the interval contains zero. The stack is fitting quantization "
                f"structure rather than behaviour, the coarse sweep does not "
                f"run, and that is the finding",
                n_effective=n_effective, detail=detail)
