"""STEP 1 §4-7: scoring. **The only module that reads CalMS21 labels.**

`tests/test_calms.py` asserts no other module under `vieb/` or `scripts/`
opens the label store or imports this module, except `scripts/calms_score.py`.
Nothing here is consulted before every instrument's states exist.

## Units

Every pooled metric is a sum over evaluation **sequences** of per-sequence
counts (TP, n_true, n_pred), so the bootstrap resamples sequences and re-sums.
`n_effective` in every Read is the number of evaluation sequences.

## The map is refitted under every shift

A many-to-one map fitted on true labels and scored against shifted ones would
compare an informed map with an uninformed target. Chance is "the same
procedure on labels that no longer line up with the pose", so the map is
refitted on the shifted map-side labels each time.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

CALMS = os.environ.get("CALMS", os.path.expanduser("~/data/calms21"))
LABELS = os.path.join(CALMS, "npz", "labels")
POSE = os.path.join(CALMS, "npz", "pose")

FPS = 30.0
IOU = 0.5
TOL_S = 0.5
N_SHIFTS = 1000
N_BOOT = 2000
SEED = 0
MIN_EVENTS = 30
MIN_UNITS = 10
BAR = 0.25

#: label set -> (pose/label sub-directory, vocabulary in id order)
SETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "t1": ("task1/annotator-id_0", ("attack", "investigation", "mount", "other")),
    "a3": ("task2/annotator-id_3", ("attack", "investigation", "mount", "other")),
    "a4": ("task2/annotator-id_4", ("attack", "investigation", "mount", "other")),
    "g": ("task3/groom", ("other", "groom")),
}
T1_LABELS = ("attack", "investigation", "mount")
EASY = T1_LABELS
LUNA_RELEVANT = ("groom",)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def labels(label_set: str) -> dict[str, I64]:
    """``{video sha1: annotations}`` for one label set, as class ids."""
    sub, _vocab = SETS[label_set]
    out: dict[str, I64] = {}
    for lp in sorted(glob.glob(os.path.join(LABELS, sub, "*.npz"))):
        pp = os.path.join(POSE, sub, os.path.basename(lp))
        with np.load(pp, allow_pickle=False) as z:
            sha = str(z["sha1"])
        with np.load(lp, allow_pickle=False) as z:
            vocab = json.loads(str(z["vocab"]))
            ann = np.asarray(z["annotations"], dtype=np.int64)
        want = SETS[label_set][1]
        assert [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])] == list(want), (
            label_set, vocab)
        out.setdefault(sha, ann)
    return out


def class_id(label_set: str, name: str) -> int:
    return SETS[label_set][1].index(name)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

def runs(mask: npt.ArrayLike) -> I64:
    """``(n, 2)`` [lo, hi) of the True runs."""
    c = np.asarray(mask, dtype=np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, c, 0]))
    return np.asarray(e.reshape(-1, 2), dtype=np.int64)


def event_counts(y: npt.ArrayLike, yhat: npt.ArrayLike, cls: int,
                 iou: float = IOU) -> tuple[int, int, int]:
    """``(TP, n_true, n_pred)``: greedy one-to-one bout matching at IoU >= `iou`."""
    t = runs(np.asarray(y) == cls)
    p = runs(np.asarray(yhat) == cls)
    if t.size == 0 or p.size == 0:
        return 0, int(t.shape[0]), int(p.shape[0])
    # Candidate pairs: overlapping intervals only (both lists are sorted).
    pairs: list[tuple[float, int, int]] = []
    j0 = 0
    for i, (a, b) in enumerate(t.tolist()):
        while j0 < p.shape[0] and p[j0, 1] <= a:
            j0 += 1
        j = j0
        while j < p.shape[0] and p[j, 0] < b:
            c, d = int(p[j, 0]), int(p[j, 1])
            inter = min(b, d) - max(a, c)
            union = max(b, d) - min(a, c)
            v = inter / union
            if v >= iou:
                pairs.append((v, i, j))
            j += 1
    pairs.sort(reverse=True)
    ut: set[int] = set()
    up: set[int] = set()
    tp = 0
    for _v, i, j in pairs:
        if i in ut or j in up:
            continue
        ut.add(i)
        up.add(j)
        tp += 1
    return tp, int(t.shape[0]), int(p.shape[0])


def true_boundaries(y: npt.ArrayLike, cls: int) -> I64:
    """Onsets and offsets of the label's bouts, excluding the sequence's edges."""
    a = np.asarray(y)
    r = runs(a == cls)
    b = np.concatenate([r[:, 0], r[:, 1]]) if r.size else np.zeros(0, np.int64)
    return np.asarray(np.sort(b[(b > 0) & (b < a.size)]), dtype=np.int64)


def boundary_counts(true_b: npt.ArrayLike, pred_b: npt.ArrayLike,
                    tol: int) -> tuple[int, int, int]:
    """``(TP, n_true, n_pred)``: greedy one-to-one matching within ±`tol` frames."""
    t = np.sort(np.asarray(true_b, dtype=np.int64))
    p = np.sort(np.asarray(pred_b, dtype=np.int64))
    if t.size == 0 or p.size == 0:
        return 0, int(t.size), int(p.size)
    pairs: list[tuple[int, int, int]] = []
    for i, x in enumerate(t.tolist()):
        lo, hi = np.searchsorted(p, x - tol, "left"), np.searchsorted(p, x + tol, "right")
        for j in range(int(lo), int(hi)):
            pairs.append((abs(int(p[j]) - x), i, j))
    pairs.sort()
    ut: set[int] = set()
    up: set[int] = set()
    tp = 0
    for _d, i, j in pairs:
        if i in ut or j in up:
            continue
        ut.add(i)
        up.add(j)
        tp += 1
    return tp, int(t.size), int(p.size)


def f1(c: npt.ArrayLike) -> float:
    """F1 from summed ``(TP, n_true, n_pred)``; 0 when nothing exists on either side."""
    tp, nt, npred = (float(v) for v in np.asarray(c, dtype=np.float64))
    return 2.0 * tp / (nt + npred) if (nt + npred) > 0 else 0.0


# --------------------------------------------------------------------------
# The many-to-one map
# --------------------------------------------------------------------------

def fit_map(states: Sequence[npt.ArrayLike], ys: Sequence[npt.ArrayLike],
            n_classes: int, other: int) -> dict[int, int]:
    """Each state -> the class with the most frames in it, on the map side."""
    s = np.concatenate([np.asarray(v, dtype=np.int64) for v in states])
    y = np.concatenate([np.asarray(v, dtype=np.int64) for v in ys])
    assert s.shape == y.shape
    offset = int(s.min()) if s.size else 0
    ss = s - offset
    n_states = int(ss.max()) + 1 if ss.size else 1
    tab = np.bincount(ss * n_classes + y, minlength=n_states * n_classes
                      ).reshape(n_states, n_classes)
    out = {}
    for k in np.flatnonzero(tab.sum(1) > 0):
        out[int(k) + offset] = int(np.argmax(tab[k]))
    del other
    return out


def apply_map(states: npt.ArrayLike, m: Mapping[int, int], other: int) -> I64:
    s = np.asarray(states, dtype=np.int64)
    u, inv = np.unique(s, return_inverse=True)
    lut = np.asarray([m.get(int(v), other) for v in u], dtype=np.int64)
    return np.asarray(lut[inv].reshape(s.shape), dtype=np.int64)


# --------------------------------------------------------------------------
# Per-sequence counts for one scoring pass
# --------------------------------------------------------------------------

Counts = dict[str, F64]        # label -> (n_eval, 3) per-sequence counts


def event_pass(states: Mapping[str, npt.ArrayLike], y: Mapping[str, npt.ArrayLike],
               map_ids: Sequence[str], eval_ids: Sequence[str], label_set: str,
               targets: Sequence[str]) -> Counts:
    """Fit the map on `map_ids`, then per-sequence event counts on `eval_ids`."""
    vocab = SETS[label_set][1]
    other = vocab.index("other")
    mp = fit_map([states[v] for v in map_ids], [y[v] for v in map_ids],
                 len(vocab), other)
    out: Counts = {}
    for lab in targets:
        cls = vocab.index(lab)
        rows = []
        for v in eval_ids:
            yhat = apply_map(states[v], mp, other)
            rows.append(event_counts(y[v], yhat, cls))
        out[lab] = np.asarray(rows, dtype=np.float64)
    return out


def boundary_pass(bounds: Mapping[str, npt.ArrayLike], y: Mapping[str, npt.ArrayLike],
                  eval_ids: Sequence[str], label_set: str,
                  targets: Sequence[str], fps: float = FPS) -> Counts:
    vocab = SETS[label_set][1]
    tol = int(round(TOL_S * fps))
    out: Counts = {}
    for lab in targets:
        cls = vocab.index(lab)
        out[lab] = np.asarray([boundary_counts(true_boundaries(y[v], cls), bounds[v], tol)
                               for v in eval_ids], dtype=np.float64)
    return out


def shifted(y: Mapping[str, npt.ArrayLike], ids: Sequence[str],
            rng: np.random.Generator) -> dict[str, I64]:
    """Circular shift of each sequence's labels by a uniform offset in [1, T)."""
    out = {}
    for v in ids:
        a = np.asarray(y[v], dtype=np.int64)
        out[v] = np.roll(a, int(rng.integers(1, a.size))) if a.size > 1 else a
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def support(y: Mapping[str, npt.ArrayLike], eval_ids: Sequence[str], label_set: str,
            lab: str) -> dict[str, Any]:
    cls = class_id(label_set, lab)
    per = [int(runs(np.asarray(y[v]) == cls).shape[0]) for v in eval_ids]
    n_ev, n_units = int(sum(per)), int(sum(1 for c in per if c > 0))
    return {"events": n_ev, "units": n_units,
            "ok": bool(n_ev >= MIN_EVENTS and n_units >= MIN_UNITS)}


def boot_index(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> I64:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.integers(0, n, size=(n_boot, n)), dtype=np.int64)


def seed_mean_f1(per_seed: Sequence[npt.ArrayLike], idx: I64 | None = None) -> float:
    """Mean over seeds of the pooled F1 (optionally on resampled sequences)."""
    vals = []
    for c in per_seed:
        a = np.asarray(c, dtype=np.float64)
        vals.append(f1(a[idx].sum(0) if idx is not None else a.sum(0)))
    return float(np.mean(vals))


def normalised(s: float, chance: float, ceiling: float) -> float:
    assert ceiling > chance, f"ceiling {ceiling} must exceed chance {chance}"
    return (s - chance) / (ceiling - chance)


def normalised_reps(per_seed: Sequence[npt.ArrayLike], null_per_seed: npt.ArrayLike,
                    ceiling: float, boot: npt.ArrayLike) -> tuple[float, F64]:
    """``(point, replicates)`` of the normalised seed-mean score.

    `null_per_seed` is ``(shifts, seeds)``; chance is the mean over shifts of
    the seed mean, i.e. the same statistic as the score, under the null.
    """
    chance = float(np.asarray(null_per_seed, dtype=np.float64).mean(axis=1).mean())
    point = normalised(seed_mean_f1(per_seed), chance, ceiling)
    reps = np.asarray([normalised(seed_mean_f1(per_seed, i), chance, ceiling)
                       for i in np.asarray(boot, dtype=np.int64)], dtype=np.float64)
    return point, reps


def abs_delta_lower(lo: float, hi: float) -> float:
    """Lower CI bound of |Delta|: 0 if the CI contains 0."""
    return 0.0 if lo <= 0.0 <= hi else float(min(abs(lo), abs(hi)))


def cell(per_seed: Sequence[npt.ArrayLike], null: npt.ArrayLike, ceiling: float,
         boot: npt.ArrayLike) -> dict[str, Any]:
    """Score, chance, normalised score with its sequence-bootstrap CI, and the rule."""
    s = seed_mean_f1(per_seed)
    nl = np.asarray(null, dtype=np.float64)
    chance, p95 = float(nl.mean()), float(np.quantile(nl, 0.95))
    reps = np.asarray([normalised(seed_mean_f1(per_seed, i), chance, ceiling)
                       for i in np.asarray(boot, dtype=np.int64)])
    lo, hi = (float(v) for v in np.quantile(reps, [0.025, 0.975]))
    z = normalised(s, chance, ceiling)
    return {"score": s, "chance": chance, "null_p95": p95, "ceiling": ceiling,
            "normalised": z, "ci": [lo, hi],
            "recoverable": bool(lo > BAR and s > p95),
            "straddles": bool(lo <= BAR < hi)}


# --------------------------------------------------------------------------
# Breakdown and the decision rule (§5, §7)
# --------------------------------------------------------------------------

def breakdown(levels: Sequence[float], recoverable: Sequence[bool]) -> dict[str, Any]:
    """Lowest swept level at which a behaviour recoverable just below stops being so."""
    lv = list(levels)
    rc = list(recoverable)
    assert len(lv) == len(rc) and lv == sorted(lv)
    for i in range(1, len(lv)):
        if rc[i - 1] and not rc[i]:
            assert lv[0] <= lv[i] <= lv[-1]
            return {"level": lv[i], "bound": None}
    if all(rc):
        return {"level": None, "bound": f"> {lv[-1]:g}x"}
    if not any(rc):
        return {"level": None, "bound": f"< {lv[0]:g}x (never recoverable)"}
    return {"level": None, "bound": "recovers only above a non-recoverable level"}


def decide(cells: Mapping[tuple[str, str, float], Mapping[str, Any]], *,
           native: float, at: float, controls_fail: bool, r_ge_1: bool,
           groom_grid_limited: bool) -> dict[str, Any]:
    """REGISTRATION §7. `cells[(instrument, label, level)]` -> `cell()` output.

    GRID_LIMITED labels are absent from `cells`.
    """
    if controls_fail:
        return {"outcome": "NOT_A_RESULT", "reason": "a negative control fired the recovery rule"}
    if r_ge_1:
        return {"outcome": "NOT_A_RESULT", "reason": "CalMS21 native jitter >= Luna's"}
    nat = {k: v for k, v in cells.items() if k[2] == native}
    any_native = any(v["recoverable"] for v in nat.values())
    if not any_native:
        if any(v["straddles"] for v in nat.values()):
            return {"outcome": "INCONCLUSIVE",
                    "reason": "nothing recoverable at native, but a native CI straddles the bar"}
        return {"outcome": "O1", "branch": "3A", "reason": "no behaviour recoverable at native"}
    if groom_grid_limited:
        return {"outcome": "NOT_A_RESULT",
                "reason": "groom GRID_LIMITED: O2 vs O3 undefined; O1 is excluded"}
    g = {k: v for k, v in cells.items() if k[1] == "groom" and k[2] >= at}
    if any(v["recoverable"] for v in g.values()):
        return {"outcome": "O3", "branch": "3C",
                "reason": "groom recoverable at >= 1.0x on at least one instrument"}
    g1 = {k: v for k, v in cells.items() if k[1] == "groom" and k[2] == at}
    if any(v["straddles"] for v in g1.values()):
        return {"outcome": "INCONCLUSIVE",
                "reason": "groom not recoverable at 1.0x, but a groom CI there straddles the bar"}
    return {"outcome": "O2", "branch": "3B",
            "reason": "a behaviour is recoverable at native; groom is not at 1.0x"}


def nmi(y: npt.ArrayLike, states: npt.ArrayLike) -> float:
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(np.asarray(y), np.asarray(states)))


def null_distribution(fn: Callable[[np.random.Generator], float], n: int = N_SHIFTS,
                      seed: int = SEED) -> F64:
    rng = np.random.default_rng(seed)
    return np.asarray([fn(rng) for _ in range(n)], dtype=np.float64)
