r"""The three rungs, the history encoding, and the covariate slot.

Each rung must charge for **both** the next symbol and the duration, or the
comparison between them is not a comparison. A rung that charged only the symbol
would win by hiding information in dwell time -- which is the single most
important detail in this design, and is why a 2-symbol vocabulary gets an
excellent NLL per transition and means nothing.

| rung | symbol | duration |
|---|---|---|
| 0 | marginal `p(u)` over runs | marginal `f(d)`, pooled |
| 1 | first-order `p(u_next | u)`, zero diagonal | marginal `f(d)`, **unchanged from rung 0** |
| 2 | cause-specific hazard | the same hazard, jointly |

Rungs 0 and 1 share a duration model deliberately, so that `rung1 - rung0`
isolates the transition table and `rung2 - rung1` isolates duration structure and
elapsed-time dependence.

## History, and why it is looked up rather than indexed

At `k = 1` the context is `(previous state, previous duration bin, current
state)`. At 2,048 symbols and 12 bins that is 50 million cells, and at `k = 2` it
is 100 billion -- so contexts are **hashed to observed keys**: the vocabulary is
the set of contexts seen on `fit`, and anything on `report` that is not in it
falls into a single unseen bucket and backs off to the Laplace prior.

That backoff is not a convenience. It is what makes a deep history *lose*
honestly: a model whose contexts were never observed pays `log N` per prediction
and is charged for its parameters anyway.

## Elapsed time is not part of the history window

At prediction time the current segment is right-censored and has no completed
duration, so no `(u, d)` history can carry it. It enters the hazard as its own
axis. It is probably the single most informative variable in the model, and the
ladder is built so that it is visible as such rather than folded into `k`.

## The covariate slot is built now and left empty

`recur.labels.parse` already yields day, context and protocol. A zero-width
array goes through the same code path, and a test asserts the zero-width case.
Adding the slot later would reshape every fitted model and invalidate the scoring
harness, which is the kind of change that quietly retires a set of results.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from recur.label.nonparametric import ABSTAIN

from vieb.checks import assert_log_pmf, assert_pmf
from vieb.tok import hazard as hz

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]
Detail = dict[str, Any]

__all__ = ["ABSTAIN_MODES", "CENSORED", "K_GRID", "RUNGS", "context_ids",
           "covariate_block", "duration_pmf", "prepare", "fit_rung",
           "score_rung"]

#: Next-state sentinel for a run whose successor was not observed.
CENSORED = -2

#: The two ways abstain is handled, and both are reported. If MDL improves only
#: when abstain is a symbol, the model is predicting tracking dropout.
ABSTAIN_MODES: tuple[str, ...] = ("symbol", "conditioned")

RUNGS: tuple[int, ...] = (0, 1, 2)
#: Completed `(u, d)` pairs of history, inside rung 2. Depth >= 3 is backlog and
#: explicitly parametric.
K_GRID: tuple[int, ...] = (0, 1, 2)


def prepare(runs: Mapping[str, Any], n_states: int, *,
            abstain: str = "conditioned") -> Detail:
    """Runs into `(code, duration, recording, next_state, censored)`.

    **`symbol`** maps abstain to index `n_states`, widening the alphabet by one,
    so transitions into and out of tracking failure are modelled.

    **`conditioned`** drops abstain runs and **censors the run before each one**.
    A run that ended because tracking failed did not end because the animal did
    something else, and scoring it as a genuine exit would let the model earn
    likelihood for predicting dropout. Dropping the abstain run without censoring
    its predecessor would splice two distant moments into one transition.
    """
    code = np.asarray(runs["code"], dtype=np.int64)
    dur = np.asarray(runs["duration"], dtype=np.int64)
    rec = np.asarray(runs["recording"], dtype=np.int64)
    if abstain not in ABSTAIN_MODES:
        raise ValueError(f"unknown abstain mode {abstain!r}; one of {ABSTAIN_MODES}")

    if abstain == "symbol":
        code = np.where(code == ABSTAIN, int(n_states), code)
        alphabet = int(n_states) + 1
        keep = np.ones(code.shape[0], dtype=bool)
    else:
        alphabet = int(n_states)
        keep = code != ABSTAIN

    # Successor within the same recording, before any dropping.
    nxt = np.full(code.shape[0], CENSORED, dtype=np.int64)
    same = np.zeros(code.shape[0], dtype=bool)
    if code.shape[0] > 1:
        same[:-1] = rec[1:] == rec[:-1]
        nxt[:-1] = np.where(same[:-1], code[1:], CENSORED)
    if abstain == "conditioned":
        # A successor that is abstain is not an observed exit.
        nxt = np.where((nxt == ABSTAIN) | (nxt == int(n_states)), CENSORED, nxt)
    censored = nxt == CENSORED

    return {"code": code[keep], "duration": dur[keep], "recording": rec[keep],
            "next_state": nxt[keep], "censored": censored[keep],
            "alphabet": alphabet, "abstain": abstain,
            "n_runs": int(keep.sum()), "n_dropped": int((~keep).sum())}


def covariate_block(n_runs: int, columns: npt.ArrayLike | None = None) -> F64:
    """The covariate slot. Zero-width unless something is passed.

    Built and wired now so that adding day, context or protocol later is a change
    to what is passed rather than a change to every fitted model's shape.
    """
    if columns is None:
        return np.zeros((int(n_runs), 0), dtype=np.float64)
    a = np.atleast_2d(np.asarray(columns, dtype=np.float64))
    if a.shape[0] != int(n_runs):
        a = a.T
    if a.shape[0] != int(n_runs):
        raise ValueError(f"covariate block has {a.shape[0]} rows for "
                         f"{n_runs} runs")
    return a


def _raw_keys(code: I64, dbin: I64, rec: I64, k: int, alphabet: int,
              n_bins: int, covariates: F64) -> I64:
    """History key per run: the last `k` completed `(u, d)` pairs plus `u`.

    A pair that would come from a different recording is replaced by a reserved
    out-of-range value, so a context never spans a seam. `covariates` is folded
    in only when it is non-empty, which is what keeps the zero-width case on the
    same code path as a populated one.
    """
    key = code.astype(np.int64)
    span = int(alphabet) * int(n_bins) + 1          # +1 for the "no pair" value
    for lag in range(1, int(k) + 1):
        pair = np.full(code.shape[0], span - 1, dtype=np.int64)
        if code.shape[0] > lag:
            ok = rec[lag:] == rec[:-lag]
            prev = code[:-lag] * int(n_bins) + dbin[:-lag]
            pair[lag:] = np.where(ok, prev, span - 1)
        key = key * span + pair
    if covariates.shape[1]:
        for c in range(covariates.shape[1]):
            col = np.asarray(covariates[:, c], dtype=np.int64)
            key = key * (int(col.max()) + 2) + col
    return key


def context_ids(code: npt.ArrayLike, duration: npt.ArrayLike,
                recording: npt.ArrayLike, *, k: int, alphabet: int,
                edges: npt.ArrayLike, vocab: I64 | None = None,
                covariates: F64 | None = None) -> tuple[I64, I64, int]:
    """Dense context ids, the vocabulary, and the context count.

    With `vocab=None` the vocabulary is built here -- call that on **fit**. With a
    vocabulary supplied, unseen contexts map to one reserved bucket at the end,
    which holds no counts and therefore backs off to the Laplace prior.
    """
    c = np.asarray(code, dtype=np.int64)
    d = np.asarray(duration, dtype=np.int64)
    r = np.asarray(recording, dtype=np.int64)
    e = np.asarray(edges, dtype=np.int64)
    dbin = hz.bin_of(np.maximum(d - 1, 0), e)
    cov = covariate_block(c.shape[0]) if covariates is None else covariates
    keys = _raw_keys(c, dbin, r, int(k), int(alphabet), int(e.shape[0] - 1), cov)
    if vocab is None:
        vocab = np.unique(keys)
    idx = np.searchsorted(vocab, keys)
    idx = np.clip(idx, 0, max(vocab.shape[0] - 1, 0))
    hit = (vocab.shape[0] > 0) & (vocab[idx] == keys)
    ids = np.where(hit, idx, vocab.shape[0]).astype(np.int64)
    return ids, vocab, int(vocab.shape[0]) + 1


def duration_pmf(duration: npt.ArrayLike, edges: npt.ArrayLike, *,
                 alpha: float = hz.ALPHA) -> Detail:
    """Pooled `f(d)` at one-frame resolution, uniform within each elapsed bin.

    The registered formula charges `-log f(d) + log delta` for a continuous
    density. With `f` held as a probability mass over bins and spread uniformly
    inside them, `f(d) = p_bin / width` and the two terms collapse to
    `-log p_bin + log width` exactly. So this **is** the registered charge, at
    `delta` = one frame, and not a deviation from it.
    """
    d = np.asarray(duration, dtype=np.int64)
    e = np.asarray(edges, dtype=np.int64)
    n_bins = int(e.shape[0] - 1)
    b = hz.bin_of(np.maximum(d - 1, 0), e)
    counts = np.bincount(b, minlength=n_bins).astype(np.float64)
    p = (counts + alpha) / (counts.sum() + alpha * n_bins)
    # Bin b covers elapsed [e[b], e[b+1]), so it holds exactly e[b+1] - e[b]
    # whole durations d. Computed in float because the open last edge is
    # int64 max and differencing it overflows.
    widths = np.diff(e.astype(np.float64))
    # The last edge is open at infinity; coding is truncated at the longest
    # duration actually seen, or the grid would charge an infinite number of
    # nats for a bin nothing occupies. That bin holds d in [e[-2]+1, top], which
    # is top - e[-2] values -- NOT one more: the +1 is already in the lower
    # endpoint, and including it made the pmf sum to 1.12 rather than 1.
    top = float(max(int(d.max()) if d.size else 1, int(e[-2]) + 1))
    widths[-1] = max(top - float(e[-2]), 1.0)
    log_p, log_w = np.log(p), np.log(widths)
    # The assertion is at FRAME resolution, not bin resolution, because the bin
    # masses sum to 1 by construction and the bug they hid did not: the widths
    # were off by one, so `f(d) = p_bin / width` summed to 1.115 over whole
    # durations while `p` itself looked perfect. Checking `p` would not have
    # caught it. See vieb/checks.py.
    whole = np.arange(1, int(top) + 1, dtype=np.int64)
    assert_pmf(np.exp(log_p[hz.bin_of(whole - 1, e)] - log_w[hz.bin_of(whole - 1, e)]),
               name="duration_pmf f(d) over whole frames", tol=1e-6)
    return {"log_p_bin": log_p, "log_width": log_w,
            "n_bins": n_bins, "edges": e, "counts": counts.tolist()}


def _duration_nats(pmf: Mapping[str, Any], duration: npt.ArrayLike) -> F64:
    d = np.asarray(duration, dtype=np.int64)
    b = hz.bin_of(np.maximum(d - 1, 0), np.asarray(pmf["edges"], dtype=np.int64))
    return np.asarray(-pmf["log_p_bin"][b] + pmf["log_width"][b],
                      dtype=np.float64)


def fit_rung(rung: int, data: Mapping[str, Any], *, edges: npt.ArrayLike,
             k: int = 0, alpha: float = hz.ALPHA) -> Detail:
    """Fit one rung on the `fit` split. Returns everything `score_rung` needs."""
    code = np.asarray(data["code"], dtype=np.int64)
    dur = np.asarray(data["duration"], dtype=np.int64)
    rec = np.asarray(data["recording"], dtype=np.int64)
    nxt = np.asarray(data["next_state"], dtype=np.int64)
    cen = np.asarray(data["censored"], dtype=bool)
    a = int(data["alphabet"])
    e = np.asarray(edges, dtype=np.int64)
    pmf = duration_pmf(dur, e, alpha=alpha)
    done = ~cen

    if rung == 0:
        counts = np.bincount(nxt[done], minlength=a).astype(np.float64) \
            if done.any() else np.zeros(a)
        logp = assert_log_pmf(
            np.log(counts + alpha) - np.log(counts.sum() + alpha * a),
            name="rung 0 marginal next-symbol")
        return {"rung": 0, "k": 0, "alphabet": a, "pmf": pmf, "log_p": logp,
                "n_params": (a - 1) + (pmf["n_bins"] - 1),
                "n_runs_fit": int(code.shape[0])}

    if rung == 1:
        # Zero diagonal: RLE makes AA impossible, so `u` is not a cause and the
        # row is a simplex over the other a-1 states.
        allowed = float(a - 1)
        counts = np.zeros((a, a), dtype=np.float64)
        if done.any():
            np.add.at(counts, (code[done], nxt[done]), 1.0)
        np.fill_diagonal(counts, 0.0)
        row = counts.sum(axis=1, keepdims=True)
        logp = np.log(counts + alpha) - np.log(row + alpha * allowed)
        np.fill_diagonal(logp, -np.inf)
        # Per row, over the a-1 allowed causes. The -inf diagonal is a
        # forbidden outcome rather than a missing one, and exp maps it to the
        # zero it means. `allowed` matching the zeroed diagonal is exactly what
        # this catches if either ever moves without the other.
        assert_log_pmf(logp, name="rung 1 transition rows", axis=1)
        return {"rung": 1, "k": 0, "alphabet": a, "pmf": pmf, "log_p": logp,
                "n_params": int(a * (a - 2)) + (pmf["n_bins"] - 1),
                "n_runs_fit": int(code.shape[0])}

    if rung != 2:
        raise ValueError(f"unknown rung {rung!r}; one of {RUNGS}")

    ids, vocab, n_ctx = context_ids(code, dur, rec, k=k, alphabet=a, edges=e)
    model = hz.fit(ids, dur, nxt, cen, edges=e, n_context=n_ctx, n_states=a,
                   alpha=alpha)
    return {"rung": 2, "k": int(k), "alphabet": a, "hazard": model,
            "vocab": vocab, "pmf": pmf,
            "n_params": int(model["n_cells_observed"]) * (a - 1),
            "n_runs_fit": int(code.shape[0])}


def score_rung(model: Mapping[str, Any], data: Mapping[str, Any], *,
               edges: npt.ArrayLike) -> F64:
    """Per-run code length in **nats**, for symbol and duration together.

    A censored run pays for its duration and, at rung 2, its survival -- never
    for a next symbol that was not observed.
    """
    code = np.asarray(data["code"], dtype=np.int64)
    dur = np.asarray(data["duration"], dtype=np.int64)
    rec = np.asarray(data["recording"], dtype=np.int64)
    nxt = np.asarray(data["next_state"], dtype=np.int64)
    cen = np.asarray(data["censored"], dtype=bool)
    e = np.asarray(edges, dtype=np.int64)

    if int(model["rung"]) == 2:
        ids, _v, _n = context_ids(code, dur, rec, k=int(model["k"]),
                                  alphabet=int(model["alphabet"]), edges=e,
                                  vocab=np.asarray(model["vocab"],
                                                   dtype=np.int64))
        return -hz.log_likelihood(model["hazard"], ids, dur, nxt, cen)

    nats = _duration_nats(model["pmf"], dur)
    done = ~cen
    if done.any():
        lp = model["log_p"]
        if int(model["rung"]) == 0:
            got = lp[nxt[done]]
        else:
            # The diagonal is -inf, so a self-transition would return an
            # infinite code length and quietly make every total infinite.
            # It cannot happen on a run stream that came out of run-length
            # encoding, so if it is here the stream is wrong -- refuse with the
            # diagnosis rather than returning inf or flooring it away.
            bad = code[done] == nxt[done]
            if bad.any():
                u = int(code[done][bad][0])
                raise ValueError(
                    f"{int(bad.sum())} scored transitions are self-repeats "
                    f"(first: state {u} -> {u}). Run-length encoding makes "
                    f"that impossible, so this run stream was concatenated "
                    f"across a seam or had an abstain run spliced out. "
                    f"`rle.self_transitions` is the check for it")
            got = lp[code[done], nxt[done]]
        nats[done] += -np.asarray(got, dtype=np.float64)
    return nats
