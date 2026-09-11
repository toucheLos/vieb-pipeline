r"""Can a probe tell corrected frames from clean ones? It must not be able to.

A minimum-norm projection puts every corrected frame on the constraint boundary —
a codimension-1 surface. Depositing even 1% of frames onto it writes an **atom**
into an otherwise continuous feature distribution. k-means finds atoms. It would
then be read as a behavioural state, and this project has already produced one
manufactured state signature, from a persistence prior rather than a corrector.

So the corrector has to be shown NOT to have left a signature. **The polarity is
inverted from every other probe in this codebase**: here a high score is the
failure.

## Why `recur/audit/leak.py:leak_read` cannot be used unchanged

It is the right shape and the wrong instrument, on three counts, each of which
would have produced a confident wrong answer:

1. **Polarity.** Its ceiling is `log N` and it PASSes when the leak is below
   `max_frac` of it. For two classes the ceiling is `log 2 = 0.693`, and its
   prose — "an inventory that identifies the animal is partly an inventory of
   animals" — is written for a different question. A positive finding here would
   be reported as `FAIL` with a reason about animal identity.
2. **Class balance.** Corrected frames are ~1% of the corpus. `identity_leak`
   has no `class_weight` and reports plain accuracy against `chance = 1/n = 0.5`,
   so a classifier predicting "clean" always scores 99% against a 50% baseline
   and the clamp guard becomes meaningless. Balanced accuracy is the statistic
   that survives the imbalance.
3. **The split is row-wise.** `train_test_split` over rows puts adjacent frames
   of the same event on both sides, and pose at 30 fps is autocorrelated over
   seconds. The probe would separate the classes from temporal proximity alone
   and the result would say nothing about the corrector.

What is kept is the lesson the leak module learned the hard way: report the
classifier's own diagnostics beside the statistic, and refuse rather than report
when the probe did not converge.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read

F64 = npt.NDArray[np.float64]
Detail = dict[str, Any]

#: Pre-registered. Above this the correction has written a detectable signature
#: into the feature distribution.
MAX_BALANCED_ACCURACY = 0.60

#: Below this many minority-class rows the probe cannot say anything and says so.
MIN_MINORITY = 50


def separability(x: npt.ArrayLike, corrected: npt.ArrayLike,
                 groups: Sequence[Any], *, seed: int = 0,
                 max_rows: int = 200_000) -> Detail:
    """Balanced accuracy of a probe separating corrected frames from clean ones.

    Held out **by group** — a group is a contiguous run, so no frame of an event
    appears on both sides of the split. Classes are balanced by subsampling the
    majority, so the reported accuracy is against a real 0.5 baseline rather than
    against the prevalence.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    feats = np.asarray(x, dtype=np.float64)
    y = np.asarray(corrected, dtype=bool)
    g = np.asarray([str(v) for v in groups])
    ok = np.isfinite(feats).all(axis=1)
    feats, y, g = feats[ok], y[ok], g[ok]
    if int(y.sum()) < MIN_MINORITY or int((~y).sum()) < MIN_MINORITY:
        return {"balanced_accuracy": None, "n_corrected": int(y.sum()),
                "why": f"fewer than {MIN_MINORITY} rows in one class"}

    rng = np.random.default_rng(seed)
    # Balance first, so the group split does not have to preserve prevalence.
    pos = np.flatnonzero(y)
    neg = rng.permutation(np.flatnonzero(~y))[: pos.size]
    idx = np.concatenate([pos, neg])
    if idx.size > max_rows:
        idx = rng.permutation(idx)[:max_rows]
    feats, y, g = feats[idx], y[idx], g[idx]

    uniq = np.array(sorted(set(g.tolist())))
    held = set(rng.permutation(uniq)[: max(1, uniq.size // 3)].tolist())
    te = np.array([v in held for v in g])
    if te.all() or not te.any() or len(set(y[te])) < 2 or len(set(y[~te])) < 2:
        return {"balanced_accuracy": None, "n_corrected": int(y.sum()),
                "why": "the group split left a side with one class"}

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(feats[~te], y[~te])
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)

    pred = clf.predict(feats[te])
    prob = clf.predict_proba(feats[te])[:, 1]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
        "auc": float(roc_auc_score(y[te], prob)),
        "n_train": int((~te).sum()), "n_test": int(te.sum()),
        "n_corrected": int(y.sum()), "n_groups": int(uniq.size),
        "converged": bool(converged),
        "method": ("logistic probe, classes balanced by subsampling, held out by "
                   "GROUP (contiguous run) so no event spans the split"),
    }


def separable_read(res: Detail, *, scored_object: Detail, n_effective: int,
                   limit: float = MAX_BALANCED_ACCURACY) -> Read:
    """PASS means the correction left no detectable signature. High is failure."""
    acc = res.get("balanced_accuracy")
    detail: Detail = {**res, "limit": limit}
    if acc is None:
        return Read("INCONCLUSIVE", scored_object,
                    f"the separability probe could not run: {res.get('why')}",
                    n_effective=n_effective, detail=detail)
    if res.get("converged") is False:
        return Read("NOT_A_RESULT", scored_object,
                    "the separability probe did not converge, so neither its "
                    "accuracy nor its refusal is a measurement",
                    n_effective=n_effective, degenerate=True, detail=detail)
    if acc <= limit:
        return Read(
            "PASS", scored_object,
            f"a probe held out by run separates corrected frames from clean ones "
            f"at {acc:.3f} balanced accuracy (AUC {res['auc']:.3f}), at or below "
            f"the {limit} limit. The correction has not written a signature into "
            f"the feature distribution that a downstream quantizer could find and "
            f"report as a state",
            n_effective=n_effective, detail=detail)
    return Read(
        "FAIL", scored_object,
        f"corrected frames are separable from clean ones at {acc:.3f} balanced "
        f"accuracy (AUC {res['auc']:.3f}), above the {limit} limit. The "
        f"correction has left a detectable signature; k-means will find it "
        f"downstream and it will look like a behavioural state. This project has "
        f"manufactured one state signature already, and a corrector depositing "
        f"mass on a constraint surface is the same failure one step earlier",
        n_effective=n_effective, detail=detail)


#: Groups are fixed-length blocks of frames, not the violating runs themselves.
#: Holding out by run would put a corrected frame on one side of the split and
#: its own clean neighbours on the other, and at 30 fps a pose three frames away
#: is nearly the same pose -- the probe would then separate the classes from
#: temporal proximity, which is the exact failure grouping exists to prevent. A
#: three-second block keeps a corrected frame and its whole neighbourhood
#: together, so it is a strictly stronger control than grouping by run.
GROUP_SECONDS = 3.0


def blocks(recording_id: str, n: int, *, fps: float,
           seconds: float = GROUP_SECONDS) -> list[str]:
    """Group labels for one recording: contiguous blocks of `seconds`.

    Prefixed by the recording id, so no group ever spans a seam.
    """
    size = max(1, int(round(fps * seconds)))
    return [f"{recording_id}:{t // size}" for t in range(int(n))]
