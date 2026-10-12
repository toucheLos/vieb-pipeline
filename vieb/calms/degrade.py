"""STEP 1 §2: Luna's noise, put onto CalMS21. No labels here.

Every operator takes ONE sequence and its own generator, so nothing is computed
across a sequence boundary (the seam test in `tests/test_calms.py`).

## Jitter

Amplitude is Luna's, per keypoint, in body lengths (REGISTRATION §2.1):
``sigma_L,k = sigma_bar_L * s_k / S``. ``s_k`` is shapeflow's `sigma_hat_px`;
``S`` is the RMS over nose and ears, because that is what the skull estimator
measures. The colour is Luna's measured spectrum, from `vieb.seg.noise.draw`,
rescaled per keypoint so the amplitude is the target and nothing else changes.

## Dropout

Masks are **transplanted** from Luna recordings, not simulated, so rates, run
lengths and cross-keypoint co-occurrence come across exactly. Level m thins
(m < 1) or unions (m > 1) whole runs, never single frames, so a thinned mask
still has Luna's run-length distribution.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

sys.path.insert(0, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from vieb.calms import io as cio                                    # noqa: E402
from vieb.clean import arms                                         # noqa: E402
from vieb.moseq import inputs                                       # noqa: E402
from vieb.seeds import stable_seed                                  # noqa: E402
from vieb.seg import jitter, noise                                  # noqa: E402
from vieb.tok import config                                         # noqa: E402

F64 = npt.NDArray[np.float64]
B1 = npt.NDArray[np.bool_]

FPS = 30.0
LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
LUNA_ORDER: tuple[str, ...] = ("left_ear", "right_ear", "nose", "center",
                               "left_hip", "right_hip", "tail_base")
SKULL_NAMES: tuple[str, ...] = ("nose", "left_ear", "right_ear")
#: s1 is measured on one draw of this length at seed 0 (REGISTRATION §2.3).
S1_FRAMES = 1_000_000


def rng_for(seed: int, m: float, sequence: str) -> np.random.Generator:
    """The one generator a sequence's draw at (seed, level) may use (§2.5)."""
    return np.random.default_rng(stable_seed("calms", int(seed), f"{float(m):g}", sequence))


def sigma_bar_luna() -> float:
    with open(config.PATHS.result("plant.json"), encoding="utf-8") as fh:
        return float(json.load(fh)["sigma_bar_bl"])


def luna_profile(sigma_bar: float | None = None) -> F64:
    """``(7,)`` Luna's per-keypoint jitter in body lengths, Luna order."""
    with open(os.path.join(config.REPO, "results", "inherited", "sf_calibration.json"),
              encoding="utf-8") as fh:
        s_px = json.load(fh)["sigma_hat_px"]
    s = np.asarray([float(s_px[k]) for k in LUNA_ORDER], dtype=np.float64)
    S = float(np.sqrt(np.mean([float(s_px[k]) ** 2 for k in SKULL_NAMES])))
    sb = sigma_bar_luna() if sigma_bar is None else float(sigma_bar)
    return np.asarray(sb * s / S, dtype=np.float64)


def count_weighted(table: Mapping[str, Any]) -> float:
    """`scripts/plant.py sigma_bar`'s statistic, on any calibration table."""
    sg = np.asarray(table["sigma_bl"], dtype=np.float64)
    ct = np.asarray(table["counts"], dtype=np.float64)
    ok = np.isfinite(sg)
    assert ok.any(), "no confidence bin was estimable"
    return float(np.sum(sg[ok] * ct[ok]) / np.sum(ct[ok]))


def native_sigma_bar(poses: Sequence[npt.ArrayLike],
                     scores: Sequence[npt.ArrayLike]) -> tuple[float, dict[str, Any]]:
    """CalMS21's skull-estimator jitter, with Luna's estimator, unchanged."""
    rs, cs = [], []
    for p, s in zip(poses, scores):
        p = np.asarray(p, dtype=np.float64)
        none = np.zeros(p.shape[:2], dtype=bool)
        r, c = jitter.residuals(p, s, missing=none, interpolated=none,
                                ell=cio.LUNA_ELL_PX)
        rs.append(r)
        cs.append(c)
    table = jitter.calibrate(np.concatenate(rs), np.concatenate(cs))
    return count_weighted(table), table


def s1(seed: int = 0) -> float:
    """Per-axis SD of `noise.draw` at scale 1, measured once."""
    x = noise.draw(np.random.default_rng(seed), S1_FRAMES, len(LUNA_ORDER), fps=FPS)
    return float(np.std(x))


def added_bl(m: float, r: float, profile: npt.ArrayLike, gain: float = 1.0) -> F64:
    """``(7,)`` sigma_add in body lengths: Luna's total at m, minus native, in
    quadrature, times the estimator calibration `gain` (Amendment 1)."""
    p = np.asarray(profile, dtype=np.float64)
    return np.asarray(float(gain) * p * np.sqrt(max(0.0, float(m) ** 2 - float(r) ** 2)),
                      dtype=np.float64)


#: Amendment 1: the calibration pose and the length of the calibrating draw.
CAL_FRAMES = 300_000
CAL_CONF = 0.97


def calibration_pose(skull_bl: Sequence[float], n_frames: int = CAL_FRAMES,
                     seed: int = 0) -> F64:
    """A rigid, slowly turning mouse whose skull has the given bone lengths.

    `skull_bl` follows `bones.SKULL` order: (left_ear-right_ear,
    left_ear-nose, right_ear-nose), in body lengths. Only the skull matters to
    the estimator; the trunk is placed on the nose-tail axis at 1 BL.
    """
    lr, ln, rn = (float(v) for v in skull_bl)
    x = (ln ** 2 - rn ** 2 + lr ** 2) / (2.0 * lr)
    y = float(np.sqrt(max(ln ** 2 - x ** 2, 0.0)))
    le, re, nose = np.array([0.0, 0.0]), np.array([lr, 0.0]), np.array([x, y])
    mid = (le + re) / 2.0
    ax = (nose - mid) / np.linalg.norm(nose - mid)
    tail = nose - ax                                  # nose-tail = 1 BL
    perp = np.array([-ax[1], ax[0]])
    center = nose - 0.35 * ax
    lh, rh = nose - 0.6 * ax + 0.1 * perp, nose - 0.6 * ax - 0.1 * perp
    body = np.stack([le, re, nose, center, lh, rh, tail]) * cio.LUNA_ELL_PX
    rng = np.random.default_rng(seed)
    th = np.cumsum(rng.normal(0.0, 0.01, int(n_frames)))
    c, s_ = np.cos(th), np.sin(th)
    R = np.stack([np.stack([c, -s_], -1), np.stack([s_, c], -1)], -2)
    return np.asarray(np.einsum("tij,kj->tki", R, body), dtype=np.float64)


def estimator_reading(pose: npt.ArrayLike, m: float, profile: npt.ArrayLike,
                      s1_px: float, gain: float, seed: int = 0) -> float:
    """What the skull estimator reads after the operator adds level m (r = 0)."""
    p = np.asarray(pose, dtype=np.float64)
    x = jitter_noise(np.random.default_rng(seed), p.shape[0],
                     added_bl(m, 0.0, profile, gain), s1_px)
    got, _t = native_sigma_bar([p + x], [np.full(p.shape[:2], CAL_CONF)])
    return got


def gain(skull_bl: Sequence[float], profile: npt.ArrayLike, s1_px: float,
         sigma_bar: float) -> float:
    """g = sigma_bar_L / the estimator's reading of the operator at 1.0x (Amendment 1)."""
    return float(sigma_bar) / estimator_reading(calibration_pose(skull_bl), 1.0,
                                                profile, s1_px, 1.0)


def reachable(m: float, r: float) -> bool:
    """A level below native cannot be reached by adding noise (§2.3)."""
    return float(m) == 0.0 or float(m) >= float(r)


def jitter_noise(rng: np.random.Generator, n_frames: int, sigma_add_bl: npt.ArrayLike,
                 s1_px: float) -> F64:
    """``(T, 7, 2)`` pixels: Luna's colour, at each keypoint's target amplitude."""
    sig_px = np.asarray(sigma_add_bl, dtype=np.float64) * cio.LUNA_ELL_PX
    if not np.any(sig_px > 0):
        return np.zeros((int(n_frames), sig_px.size, 2))
    x = noise.draw(rng, int(n_frames), sig_px.size, fps=FPS)
    return np.asarray(x * (sig_px / float(s1_px))[None, :, None], dtype=np.float64)


# --------------------------------------------------------------------------
# Dropout
# --------------------------------------------------------------------------

def runs(col: npt.ArrayLike) -> list[tuple[int, int]]:
    """``[(lo, hi), ...]`` of the True runs in a 1-D mask."""
    c = np.asarray(col, dtype=np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, c, 0]))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2])]


def thin(mask: npt.ArrayLike, p: float, rng: np.random.Generator) -> B1:
    """Keep each run of each column with probability `p` (whole runs only)."""
    m = np.asarray(mask, dtype=bool)
    flat = m.reshape(m.shape[0], -1)
    out = np.zeros_like(flat)
    for k in range(flat.shape[1]):
        for lo, hi in runs(flat[:, k]):
            if rng.random() < p:
                out[lo:hi, k] = True
    return np.asarray(out.reshape(m.shape), dtype=bool)


def transplant(n_frames: int, donors: Mapping[str, Any],
               rng: np.random.Generator) -> tuple[B1, B1]:
    """One full-strength ``(missing (T, 7), abstain (T,))`` from Luna donors.

    `donors`: ``missing (N, 7)``, ``abstain (N,)``, ``bounds (R + 1,)`` over R
    Luna recordings. Donors are drawn uniformly and read from a random offset
    to their end, concatenated until `n_frames` are covered.
    """
    b = np.asarray(donors["bounds"], dtype=np.int64)
    mi = np.asarray(donors["missing"], dtype=bool)
    ab = np.asarray(donors["abstain"], dtype=bool)
    n = int(n_frames)
    out_m = np.zeros((n, mi.shape[1]), dtype=bool)
    out_a = np.zeros(n, dtype=bool)
    pos = 0
    while pos < n:
        r = int(rng.integers(b.size - 1))
        lo, hi = int(b[r]), int(b[r + 1])
        start = lo + int(rng.integers(hi - lo))
        take = min(hi - start, n - pos)
        out_m[pos:pos + take] = mi[start:start + take]
        out_a[pos:pos + take] = ab[start:start + take]
        pos += take
    return out_m, out_a


def dropout(n_frames: int, m: float, donors: Mapping[str, Any],
            rng: np.random.Generator) -> tuple[B1, B1]:
    """Dropout at level m (§2.4): thinned below 1, unions of masks above."""
    k = int(np.floor(float(m)))
    frac = float(m) - k
    miss = np.zeros((int(n_frames), donors["missing"].shape[1]), dtype=bool)
    abst = np.zeros(int(n_frames), dtype=bool)
    for _ in range(k):
        a, b = transplant(n_frames, donors, rng)
        miss |= a
        abst |= b
    if frac > 0:
        a, b = transplant(n_frames, donors, rng)
        miss |= thin(a, frac, rng)
        abst |= thin(b, frac, rng)
    return miss, abst


# --------------------------------------------------------------------------
# Luna's fill, per instrument (§2.4)
# --------------------------------------------------------------------------

def moseq_input(pose: npt.ArrayLike, scores: npt.ArrayLike, missing: npt.ArrayLike,
                abstain: npt.ArrayLike) -> tuple[F64, F64]:
    """MoSeq's Luna input: `inputs.real` with no veto (CalMS21 has no mask)."""
    p = np.asarray(pose, dtype=np.float64)
    veto = np.zeros(p.shape[:2], dtype=bool)
    return inputs.real(p, missing, veto, abstain, scores)


def raw_input(pose: npt.ArrayLike, missing: npt.ArrayLike,
              abstain: npt.ArrayLike) -> tuple[F64, B1]:
    """The detector's Luna input: the `raw` arm (every gap held), and abstain."""
    ab = np.asarray(abstain, dtype=bool)
    miss = np.asarray(missing, dtype=bool) | ab[:, None]
    return arms.held_array(pose, miss), ab


def degraded(pose: npt.ArrayLike, m: float, *, r: float, profile: npt.ArrayLike,
             s1_px: float, donors: Mapping[str, Any], rng: np.random.Generator,
             gain: float = 1.0) -> tuple[F64, B1, B1]:
    """One sequence at level m: ``(pose + jitter, missing, abstain)``, unfilled.

    The jitter is drawn first and the dropout second from the same generator,
    so a sequence's draw depends only on (seed, m, sequence).
    """
    p = np.asarray(pose, dtype=np.float64)
    T = p.shape[0]
    noisy = p + jitter_noise(rng, T, added_bl(m, r, profile, gain), s1_px)
    if float(m) == 0.0:
        return noisy, np.zeros((T, p.shape[1]), dtype=bool), np.zeros(T, dtype=bool)
    miss, ab = dropout(T, m, donors, rng)
    return noisy, miss, ab
