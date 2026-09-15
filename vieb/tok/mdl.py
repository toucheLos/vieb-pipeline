r"""Held-out description length, per animal, with the duration charged.

.. math::

    \mathcal{L} = \frac{1}{T_{\text{sec}}}\Big[
        -\sum_i \log p(u_i \mid h_i)
        - \sum_i \log f(d_i \mid u_i, h_i)
        + |\text{codebook}| \cdot c \Big]

**Duration must be charged**, and it is the single most important detail in this
design. Without it a coarse vocabulary wins by hiding information in dwell time:
a two-symbol alphabet gets an excellent NLL per transition and has described
nothing. Bits-above-marginal only partly fixes it, because the marginal moves
with the vocabulary too.

## Per animal, not only pooled

Pooled code length is dominated by the animals with the most transitions, and can
improve while getting worse for most animals. So the per-animal distribution is
the reported object and the pooled scalar sits beside it.

The codebook is a property of the model rather than of any animal, so it is
shared out **in proportion to time**: each animal carries
`codebook / T_total`, an identical offset for every animal, and the animal-mean
then reproduces the pooled figure exactly. Charging every animal the whole
codebook instead would weight it by `1 / T_a` and make a short animal look
disproportionately expensive for a cost it did not incur.

## The interval is animal-level and the frame-level one is printed to be dismissed

`boot.animal_interval` resamples animals; `boot.frame_interval` resamples rows and
comes out roughly nineteen times narrower on this corpus. The second is printed
beside the first **only** so that the gap is visible in the document. It never
licenses a claim, and `ladder_read` gates on the animal-level interval alone.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur import boot
from recur.read import Read

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

__all__ = ["N_BOOT", "bic_c", "code_length", "improvement", "ladder_read",
           "per_animal", "rare_transition_recall"]

#: Animal-level bootstrap replicates, matching every other interval in this repo.
N_BOOT = 2000
#: How many of the busiest edges count as "common" when recall is split out.
COMMON_EDGES = 5


def bic_c(n_runs_fit: int) -> float:
    """The per-parameter charge, `0.5 * log(n)`.

    The standard BIC parameter cost, named in the pre-registration so it is not
    mistaken for a knob. It is a function of the fitting sample size and nothing
    here chooses it.
    """
    return 0.5 * float(np.log(max(int(n_runs_fit), 2)))


def code_length(nats: npt.ArrayLike, duration: npt.ArrayLike, fps: float, *,
                n_params: int, n_runs_fit: int) -> Detail:
    """Total code length, split into the terms a reader needs to see separately.

    The split matters: a rung can lose entirely on the codebook term while
    describing the data better, and a total that hides which term dominates
    cannot be argued with.
    """
    x = np.asarray(nats, dtype=np.float64)
    t_sec = float(np.asarray(duration, dtype=np.float64).sum()) / float(fps)
    c = bic_c(n_runs_fit)
    book = float(n_params) * c
    data = float(x.sum())
    return {
        "data_nats": data, "codebook_nats": book, "total_nats": data + book,
        "t_seconds": t_sec, "n_runs": int(x.shape[0]),
        "n_params": int(n_params), "c_per_param": c,
        "nats_per_second": (data + book) / t_sec if t_sec > 0 else float("nan"),
        "data_nats_per_second": data / t_sec if t_sec > 0 else float("nan"),
        "nats_per_run": data / x.shape[0] if x.shape[0] else float("nan"),
    }


def per_animal(nats: npt.ArrayLike, duration: npt.ArrayLike,
               animals: Sequence[Any], fps: float, *, n_params: int,
               n_runs_fit: int) -> Detail:
    """Per-animal nats per second, with the codebook shared out by time."""
    x = np.asarray(nats, dtype=np.float64)
    d = np.asarray(duration, dtype=np.float64)
    a = np.asarray(animals)
    tags = np.unique(a)
    t_total = float(d.sum()) / float(fps)
    book = float(n_params) * bic_c(n_runs_fit)
    offset = book / t_total if t_total > 0 else 0.0
    vals, secs = [], []
    for t in tags:
        m = a == t
        ts = float(d[m].sum()) / float(fps)
        vals.append(float(x[m].sum()) / ts + offset if ts > 0 else float("nan"))
        secs.append(ts)
    return {"animals": [str(t) for t in tags],
            "nats_per_second": np.asarray(vals, dtype=np.float64),
            "t_seconds": np.asarray(secs, dtype=np.float64),
            "codebook_offset": offset, "n_animals": int(tags.shape[0])}


def improvement(lower: Mapping[str, Any], higher: Mapping[str, Any]) -> Detail:
    """Per-animal `lower - higher`. Positive means the higher rung is cheaper.

    Both must have been scored on the same animals in the same order, and this
    refuses rather than aligning them: a silent re-alignment is how a paired
    comparison stops being paired.
    """
    if list(lower["animals"]) != list(higher["animals"]):
        raise ValueError("the two rungs were scored on different animals")
    d = (np.asarray(lower["nats_per_second"], dtype=np.float64)
         - np.asarray(higher["nats_per_second"], dtype=np.float64))
    return {"animals": list(lower["animals"]), "delta": d,
            "n_animals": int(d.shape[0]),
            "n_better": int(np.sum(d > 0)), "n_worse": int(np.sum(d < 0))}


def ladder_read(imp: Mapping[str, Any], *, name: str, scored_object: Detail,
                n_boot: int = N_BOOT, seed: int = 0) -> Read:
    """Does the higher rung beat the lower one, on an animal-level interval?

    `boot.frame_interval` is computed and carried in the detail purely so the
    document can show how much narrower the wrong method looks. The verdict is
    taken from the animal-level interval and from nothing else.
    """
    d = np.asarray(imp["delta"], dtype=np.float64)
    animals = list(imp["animals"])
    n = int(d.shape[0])
    detail: Detail = {"comparison": name, "n_animals": n,
                      "n_better": int(imp["n_better"]),
                      "n_worse": int(imp["n_worse"]),
                      "mean_delta": float(np.mean(d)) if n else float("nan"),
                      "median_delta": float(np.median(d)) if n else float("nan")}
    if n < 2 or not np.isfinite(d).all():
        return Read("NOT_A_RESULT", scored_object,
                    f"{name}: {n} animals with a finite per-animal code length, "
                    f"which is too few to resample",
                    n_effective=max(n, 0), degenerate=True, detail=detail)

    ci = boot.animal_interval(d, animals, how="mean", n_boot=n_boot, seed=seed)
    detail["animal_interval"] = dict(ci)
    detail["frame_interval"] = dict(boot.frame_interval(d, n_boot=n_boot,
                                                        seed=seed))
    detail["frame_interval_note"] = (
        "printed to show how much narrower the wrong method looks; it licenses "
        "nothing and the verdict does not read it")
    lo, hi = float(ci.get("lo", np.nan)), float(ci.get("hi", np.nan))
    mean = float(np.mean(d))
    tail = (f"{mean:+.4f} nats/s [{lo:+.4f}, {hi:+.4f}], animal-level over "
            f"{n} animals, better on {int(imp['n_better'])} of them")

    if not boot.excludes_zero(ci):
        return Read("FAIL", scored_object,
                    f"{name} does not beat the rung below it: {tail}, and the "
                    f"interval contains zero",
                    n_effective=n, detail=detail)
    if mean < 0:
        return Read("FAIL", scored_object,
                    f"{name} is WORSE than the rung below it: {tail}, with the "
                    f"interval excluding zero on the wrong side",
                    n_effective=n, detail=detail)
    return Read("PASS", scored_object,
                f"{name} beats the rung below it: {tail}, interval excluding "
                f"zero",
                n_effective=n, detail=detail)


def rare_transition_recall(nats: npt.ArrayLike, code: npt.ArrayLike,
                           next_state: npt.ArrayLike, censored: npt.ArrayLike,
                           fit_counts: npt.ArrayLike, *,
                           common: int = COMMON_EDGES) -> Detail:
    """Code length on the busiest edges against everything else.

    Average NLL rewards a model that only nails the top five edges, so the two
    are reported apart. `fit_counts` is the transition count matrix from the
    **fit** split -- which edges are common is not decided on the scored data.
    """
    x = np.asarray(nats, dtype=np.float64)
    c = np.asarray(code, dtype=np.int64)
    j = np.asarray(next_state, dtype=np.int64)
    done = ~np.asarray(censored, dtype=bool)
    counts = np.asarray(fit_counts, dtype=np.float64)
    if not done.any():
        return {"n_common": 0, "n_rare": 0, "common_nats_per_run": float("nan"),
                "rare_nats_per_run": float("nan"), "common_edges": []}
    flat = counts.ravel()
    top = np.argsort(flat)[::-1][:int(common)]
    edges = {(int(t // counts.shape[1]), int(t % counts.shape[1])) for t in top
             if flat[t] > 0}
    key = np.array([(int(u), int(v)) in edges for u, v in zip(c[done], j[done])])
    xs = x[done]
    return {
        "n_common": int(key.sum()), "n_rare": int((~key).sum()),
        "common_nats_per_run": float(xs[key].mean()) if key.any() else float("nan"),
        "rare_nats_per_run": float(xs[~key].mean()) if (~key).any() else float("nan"),
        "common_edges": sorted(edges),
        "note": ("which edges are common is decided on the fit split, never on "
                 "the scored one"),
    }
