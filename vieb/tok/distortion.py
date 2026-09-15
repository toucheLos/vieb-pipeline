r"""What the alphabet throws away: `D(N) = E||x - c_q(x)||^2`.

Pose is continuous. Between any two poses there is another, so `N` is not
approximating a true vocabulary size the way a token count approximates a
language's -- it is **choosing a resolution**. Every symbol necessarily spans a
range of distinct positions, and the question is not whether that is an error
but how much of it there is.

That is the same trade MDL is already adjudicating: a coarser alphabet costs
fewer bits and loses more detail. But **MDL reports the total and never splits
it**, so `LADDER.md`'s selection of N = 256 cannot currently be read as either
"cheap to encode" or "genuinely tighter prediction". Distortion is the missing
column.

## The trap this module exists to avoid

`recur.null.microstate.assign_states` computes

.. code-block:: python

    d = cn[None, :] - 2.0 * (F @ C.T)          # |f|^2 is constant per row

and the comment is correct: the `||f||^2` term is constant **within a row**, so
dropping it leaves the argmin unchanged and saves a pass. It also means `d` is
**not a squared distance** and is negative about half the time. A distortion
built on it would be wrong by a per-frame offset that varies across frames --
invisible in the pooled mean, fatal in the per-symbol breakdown, and it would
not look wrong.

So this module computes the residual directly from the assigned centroid. It
never re-derives the assignment: the labels are on disk, and recomputing an
argmin here would introduce a second implementation that could disagree with
the one the whole ladder was scored on. `verify_assignment` checks a sample
against the true argmin instead, which is the cheap version of the same
guarantee.

## Units, twice

Distortion is computed in the **standardised** space, because that is the space
k-means minimised in and the only one where the 17 channels are commensurable.
It is also reported in the corpus's own units -- body lengths for the shape
block, body lengths per second and radians per second for the twist -- by
multiplying the per-channel squared error by `sd^2`. A number in standardised
units is a number about the algorithm; a number in body lengths is a number
about the mouse.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt
from recur.null import microstate
from recur.read import Read

from vieb.checks import assert_share
from vieb.tok import ego

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

#: The typed boundary. `recur.null.microstate.standardise` carries no
#: annotations, so mypy reads it as untyped; declared once and named here rather
#: than silenced at each call site.
_standardise = cast("Callable[..., F64]", microstate.standardise)

__all__ = ["CHANNEL_GROUPS", "accumulate", "distortion_read", "merge",
           "residuals", "summarise", "verify_assignment"]

#: The two blocks of `ego.CHANNELS`, which carry different units and must not be
#: pooled into one "distortion" without saying so.
CHANNEL_GROUPS: dict[str, tuple[int, ...]] = {
    "shape": tuple(range(ego.N_POSE)),
    "twist": tuple(range(ego.N_POSE, ego.N_DIMS)),
}

#: Frames per chunk. `microstate.CHUNK` is sized for a (chunk, N) distance
#: block; here the working array is (chunk, n_cols), which is 120x smaller at
#: N = 2048, so the chunk exists for cache behaviour rather than for memory.
CHUNK = 1_000_000


def residuals(x: npt.ArrayLike, labels: npt.ArrayLike,
              centroids: npt.ArrayLike, sd: npt.ArrayLike,
              cols: npt.ArrayLike) -> F64:
    """Standardised residual `x/sd - c[label]`, one row per frame.

    Computed against the **assigned** centroid, never against a freshly
    recomputed argmin.
    """
    f = _standardise(x, sd, cols).astype(np.float64)
    c = np.asarray(centroids, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    return np.asarray(f - c[lab], dtype=np.float64)


def accumulate(x: npt.ArrayLike, labels: npt.ArrayLike,
               centroids: npt.ArrayLike, sd: npt.ArrayLike,
               cols: npt.ArrayLike, *, n_states: int,
               keep: npt.ArrayLike | None = None,
               chunk: int = CHUNK) -> Detail:
    """Running sums for one animal, chunked. Additive across animals.

    Returns sums rather than means so that `merge` can pool them across shards
    without weighting a 3-recording animal the same as a 13-recording one.
    `keep` excludes abstained and invalid frames -- an abstained frame has a
    symbol only by convention and its residual is not a quantization error.
    """
    a = np.asarray(x)
    lab = np.asarray(labels, dtype=np.int64)
    m = (np.ones(a.shape[0], dtype=bool) if keep is None
         else np.asarray(keep, dtype=bool))
    n_cols = int(np.asarray(cols).shape[0])
    sse_channel = np.zeros(n_cols, dtype=np.float64)
    sse_symbol = np.zeros(int(n_states), dtype=np.float64)
    n_symbol = np.zeros(int(n_states), dtype=np.int64)
    n_frames = 0
    for i in range(0, a.shape[0], int(chunk)):
        sl = slice(i, i + int(chunk))
        sub = m[sl]
        if not sub.any():
            continue
        r = residuals(a[sl][sub], lab[sl][sub], centroids, sd, cols)
        sq = r * r
        sse_channel += sq.sum(axis=0)
        per_frame = sq.sum(axis=1)
        ll = lab[sl][sub]
        sse_symbol += np.bincount(ll, weights=per_frame, minlength=int(n_states))
        n_symbol += np.bincount(ll, minlength=int(n_states))
        n_frames += int(sub.sum())
    return {"sse_channel": sse_channel, "sse_symbol": sse_symbol,
            "n_symbol": n_symbol, "n_frames": int(n_frames),
            "n_states": int(n_states)}


def frame_sq_error(x: npt.ArrayLike, labels: npt.ArrayLike,
                   centroids: npt.ArrayLike, sd: npt.ArrayLike,
                   cols: npt.ArrayLike) -> F64:
    """Squared distance from each frame to its assigned centroid.

    The per-frame quantity the run-level covariate is built from. Kept separate
    from `accumulate` because that one deliberately never materialises it.
    """
    r = residuals(x, labels, centroids, sd, cols)
    return np.asarray((r * r).sum(axis=1), dtype=np.float64)


def merge(parts: Sequence[Mapping[str, Any]]) -> Detail:
    """Pool per-animal accumulators."""
    if not parts:
        raise ValueError("nothing to merge")
    n_states = int(parts[0]["n_states"])
    out = {"sse_channel": np.zeros_like(np.asarray(parts[0]["sse_channel"])),
           "sse_symbol": np.zeros(n_states, dtype=np.float64),
           "n_symbol": np.zeros(n_states, dtype=np.int64),
           "n_frames": 0, "n_states": n_states}
    for p in parts:
        out["sse_channel"] = out["sse_channel"] + np.asarray(p["sse_channel"])
        out["sse_symbol"] = out["sse_symbol"] + np.asarray(p["sse_symbol"])
        out["n_symbol"] = out["n_symbol"] + np.asarray(p["n_symbol"])
        out["n_frames"] = int(cast(int, out["n_frames"])) + int(p["n_frames"])
    return out


def summarise(acc: Mapping[str, Any], sd: npt.ArrayLike,
              channels: Sequence[str] = ego.CHANNELS) -> Detail:
    """`D(N)` pooled, by channel group, per channel, and per symbol.

    The per-channel figure appears twice: in standardised units, which is the
    space k-means minimised in, and multiplied back by `sd` into body lengths
    (shape) or body lengths per second and radians per second (twist). The
    second is the one a reader can judge.
    """
    n = int(acc["n_frames"])
    sse_c = np.asarray(acc["sse_channel"], dtype=np.float64)
    s = np.asarray(sd, dtype=np.float64)
    if n == 0:
        return {"n_frames": 0, "d_total": float("nan"), "why": "no frames"}

    mse_c = sse_c / n
    groups = {g: {"d": float(mse_c[list(idx)].sum()),
                  "rms_standardised": float(np.sqrt(mse_c[list(idx)].mean())),
                  "n_channels": len(idx)}
              for g, idx in CHANNEL_GROUPS.items()}
    per_channel = [
        {"channel": str(channels[i]),
         "mse_standardised": float(mse_c[i]),
         "rms_native": float(np.sqrt(mse_c[i]) * s[i])}
        for i in range(mse_c.shape[0])]

    n_sym = np.asarray(acc["n_symbol"], dtype=np.float64)
    sse_s = np.asarray(acc["sse_symbol"], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_sym = np.where(n_sym > 0, sse_s / np.maximum(n_sym, 1.0), np.nan)
    live = n_sym > 0
    share = (sse_s / sse_s.sum()) if sse_s.sum() > 0 else np.zeros_like(sse_s)
    order = np.argsort(share)[::-1]
    top = int(max(1, round(0.1 * share.shape[0])))
    return {
        "n_frames": n,
        "d_total": float(mse_c.sum()),
        "rms_standardised": float(np.sqrt(mse_c.sum())),
        "by_group": groups,
        "per_channel": per_channel,
        "per_symbol": {
            "d_median": float(np.nanmedian(d_sym[live])) if live.any() else float("nan"),
            "d_p90": float(np.nanpercentile(d_sym[live], 90)) if live.any() else float("nan"),
            "d_max": float(np.nanmax(d_sym[live])) if live.any() else float("nan"),
            "worst_decile_share_of_error": float(
                assert_share([share[order[:top]].sum()],
                             name="worst-decile share of distortion")[0]),
            "n_live": int(live.sum()),
        },
    }


def verify_assignment(x: npt.ArrayLike, labels: npt.ArrayLike,
                      model: Mapping[str, Any], *, rng: np.random.Generator,
                      sp: npt.ArrayLike | None = None,
                      n_sample: int = 20_000) -> Detail:
    """On a sample, is the stored label the label THIS ARM's rule produces?

    The stored labels came from `microstate.assign_states`, which drops the
    `||f||^2` term. Dropping it cannot change an argmin, so this should pass --
    and that is exactly why it is worth asserting rather than assuming. If the
    labels were ever produced under a different `sd` or a different column
    subset than the one handed here, every residual in this module would be
    measured against the wrong centroid and nothing else would notice.

    **The rule is the arm's, not a global argmin**, and the first version of
    this function got that wrong. The `speed` arm assigns within the frame's
    speed-quintile block: a global nearest-centroid check read 30% agreement on
    it and 99.998% on `plain`, which looks exactly like a corrupted stratified
    arm and is in fact a correct stratified arm measured against the wrong
    question. The distortion itself was never affected -- it is the distance to
    the *assigned* centroid either way -- but the guard was.
    """
    a = np.asarray(x)
    lab = np.asarray(labels, dtype=np.int64)
    cent = np.asarray(model["centroids"], dtype=np.float64)
    sd, cols = model["sd"], model["cols"]
    arm = str(np.asarray(model["arm"]).item()
              if np.asarray(model["arm"]).ndim == 0 else model["arm"])
    take = (np.arange(a.shape[0]) if a.shape[0] <= n_sample else
            rng.choice(a.shape[0], int(n_sample), replace=False))
    f = _standardise(a[take], sd, cols).astype(np.float64)

    if arm == "plain":
        d = ((f[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        want = np.argmin(d, axis=1).astype(np.int64)
    else:
        if sp is None:
            raise ValueError("the speed arm needs a speed vector to verify")
        from vieb.tok import quantize as qz
        offsets = np.asarray(model["offsets"], dtype=np.int64)
        strat = qz.stratum_of(np.asarray(sp)[take],
                              np.asarray(model["edges"], dtype=np.float64))
        want = np.full(take.shape[0], -1, dtype=np.int64)
        for i in range(offsets.shape[0] - 1):
            sel = strat == i
            if not sel.any():
                continue
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            d = ((f[sel][:, None, :] - cent[None, lo:hi, :]) ** 2).sum(axis=2)
            want[sel] = np.argmin(d, axis=1) + lo

    got = lab[take]
    agree = int((want == got).sum())
    return {"n_checked": int(take.shape[0]), "n_agree": agree,
            "arm": arm,
            "frac_agree": float(agree / max(take.shape[0], 1))}


def distortion_read(summary: Mapping[str, Any], check: Mapping[str, Any], *,
                    scored_object: Detail, n_effective: int,
                    min_agreement: float = 0.999) -> Read:
    """Reports `D(N)`; fails only if the assignment it measured is not the one
    on disk.

    There is no threshold on distortion itself and there must not be. Distortion
    falls monotonically with `N` and never reaches zero, so no value of it is
    pass or fail -- it is a coordinate, and the verdict here is about whether
    the coordinate was computed against the right centroids.
    """
    frac = float(check["frac_agree"])
    detail: Detail = {
        "d_total": summary.get("d_total"),
        "rms_standardised": summary.get("rms_standardised"),
        "by_group": summary.get("by_group"),
        "per_symbol": summary.get("per_symbol"),
        "assignment_check": dict(check),
    }
    if frac < min_agreement:
        return Read("FAIL", scored_object,
                    f"the stored labels are not the nearest centroid under the "
                    f"basis handed to this module: {frac:.4%} agreement over "
                    f"{int(check['n_checked']):,} sampled frames. Every "
                    f"residual here would be measured against the wrong "
                    f"centroid, so the distortion is not reportable",
                    n_effective=n_effective, degenerate=True, detail=detail)
    g = summary["by_group"]
    return Read("PASS", scored_object,
                f"D(N) = {float(summary['d_total']):.4f} in standardised units "
                f"over {int(summary['n_frames']):,} frames, split "
                f"{float(g['shape']['d']):.4f} shape and "
                f"{float(g['twist']['d']):.4f} twist; the worst decile of "
                f"symbols carries "
                f"{float(summary['per_symbol']['worst_decile_share_of_error']):.1%} "
                f"of it. Distortion falls monotonically with N and never "
                f"reaches zero, so this is a coordinate and not a verdict -- "
                f"the PASS is that the labels are the nearest centroid "
                f"({frac:.4%} over {int(check['n_checked']):,} sampled frames)",
                n_effective=n_effective, detail=detail)
