"""Per-window dynamics descriptors: AR poles, and band power that is a shape.

READ results/DYNAMICS_PREREGISTRATION.md FIRST.

## Why poles are primary

Measured on this corpus, the high-frequency noise plateau is **182x lower in
still windows than in moving ones** (1.67e-7 against 3.05e-5 bl^2/Hz), while the
SNR profile barely moves -- SNR >= 4 holds to 4.69 Hz still and 3.98 Hz moving.

So an absolute-amplitude descriptor -- band power, motion energy -- is dominated
by the motion regime and means something different in a freeze than in a run.
**Pole frequency and damping are scale-free**: multiply the signal by any
constant and the fitted AR coefficients, hence the poles, are unchanged. They
are the only descriptor in the proposed set that survives a 182x swing in the
thing underneath them.

Band power is kept, but **normalised per window** so it is a spectral *shape*
and not an amplitude.

## What a pole is, in behavioural terms

An AR(p) fit has `p` poles. A complex-conjugate pair `r*exp(+-i*w)` is a damped
oscillator:

    frequency = w * fps / (2*pi)   Hz        -- how fast it oscillates
    radius    = r                            -- how long it persists
    damping   = -log(r)                      -- decay rate per sample

A real pole near +1 is drift; a real pole near 0 is white; a complex pair with
`r` near 1 is a sustained rhythm. That is the "(configuration, dynamical
regime)" decomposition written as three numbers.

## The trap this module is built around

**Coloured tracking noise has poles too.** The residual's measured lag-1
autocorrelation is +0.5817 over 267 ms, which is a real pole at ~0.58 and an
apparent timescale of ~3.7 Hz -- inside the only usable signal band. Every
descriptor here therefore has to be read against its own distribution under the
measured noise colour, which is what `vieb/seg/noise.py` exists to generate.
Nothing in this module decides anything on its own.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from scipy.signal.windows import dpss

__all__ = ["AR_ORDER", "BANDS", "NW", "N_TAPERS", "ar_fit", "ar_fit_batch", "poles_of", "poles_batch",
           "dominant_pole", "pole_features", "multitaper_psd", "band_power",
           "motion_energy", "describe"]

F64 = npt.NDArray[np.float64]

#: Order of the local fit. 4 gives two complex pairs -- enough for one rhythm
#: plus one drift or decay term -- without spending degrees of freedom a
#: sub-second window does not have.
AR_ORDER = 4
#: Registered bands, from the frozen `f_c = 4.833 Hz`. `fast` is swept rather
#: than excluded so its null is demonstrated instead of assumed.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("slow", 0.5, 2.0),
    ("mid", 2.0, 4.0),
    ("marginal", 4.0, 6.0),
    ("fast", 6.0, 15.0),
)
#: Time-bandwidth product and taper count for the multitaper estimate. NW = 3
#: with 5 tapers is the standard conservative choice: it buys variance
#: reduction at a resolution cost of 2*NW/T, which at a 1 s window is 6 Hz --
#: wide, and the reason band power is a coarse descriptor here and poles are
#: the fine one.
NW = 3.0
N_TAPERS = 5


def ar_fit(x: npt.ArrayLike, p: int = AR_ORDER, *,
           ridge: float = 1e-8) -> F64:
    """Univariate AR(`p`) coefficients by least squares, mean removed.

    Univariate and per channel, not a VAR: the ego array is rank 11 + 3, so a
    17-channel joint fit has a singular Gram and a singular residual
    covariance. Per channel there is no such problem, and a shared boundary is
    recovered later by agreement across channels rather than by a joint fit.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a - a.mean()
    n = a.size
    if n <= p + 2:
        return np.zeros(p)
    rows = np.lib.stride_tricks.sliding_window_view(a, p)[:-1][:, ::-1]
    y = a[p:]
    g = rows.T @ rows
    g.flat[:: p + 1] += ridge * max(float(np.trace(g)), 1e-300) / max(p, 1)
    try:
        return np.asarray(np.linalg.solve(g, rows.T @ y), dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.zeros(p)


def poles_of(coef: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Roots of the AR characteristic polynomial -- companion eigenvalues."""
    c = np.asarray(coef, dtype=np.float64).ravel()
    if c.size == 0 or not np.isfinite(c).all():
        return np.zeros(0, dtype=np.complex128)
    return np.asarray(np.roots(np.concatenate([[1.0], -c])),
                      dtype=np.complex128)


def dominant_pole(poles: npt.ArrayLike, *, fps: float) -> dict[str, float]:
    """The longest-lived oscillatory pole: frequency, radius, damping.

    Oscillatory means a non-negligible imaginary part. A purely real pole is
    drift or decay and carries no frequency, so it is reported separately
    rather than being handed a frequency of zero that would average in.
    """
    z = np.asarray(poles, dtype=np.complex128)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {"frequency_hz": float("nan"), "radius": float("nan"),
                "damping": float("nan"), "real_radius": float("nan"),
                "n_oscillatory": 0.0}
    r = np.abs(z)
    osc = np.abs(z.imag) > 1e-6
    out = {"n_oscillatory": float(int(osc.sum())),
           "real_radius": float(np.max(r[~osc])) if (~osc).any()
           else float("nan")}
    if not osc.any():
        out.update({"frequency_hz": float("nan"), "radius": float("nan"),
                    "damping": float("nan")})
        return out
    k = int(np.argmax(np.where(osc, r, -np.inf)))
    rad = float(min(r[k], 1.0 - 1e-12))
    out.update({"frequency_hz": float(abs(np.angle(z[k])) * fps
                                      / (2.0 * np.pi)),
                "radius": rad, "damping": float(-np.log(max(rad, 1e-12)))})
    return out


def ar_fit_batch(block: npt.ArrayLike, p: int = AR_ORDER, *,
                 ridge: float = 1e-8) -> F64:
    """`(C, p)` AR coefficients, every channel solved in one batched call.

    Identical in result to `ar_fit` per column and roughly ten times faster,
    which is what makes a sliding two-window contrast affordable at all: the
    per-channel Python loop was the entire cost of the dynamics channel.
    """
    a = np.asarray(block, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    n, c = a.shape
    if n <= p + 2:
        return np.zeros((c, p))
    rows = np.lib.stride_tricks.sliding_window_view(
        a, p, axis=0)[:-1][:, :, ::-1]                    # (n-p, C, p)
    rows = np.ascontiguousarray(np.moveaxis(rows, 1, 0))  # (C, n-p, p)
    y = a[p:].T[:, :, None]                               # (C, n-p, 1)
    g = rows.transpose(0, 2, 1) @ rows                    # (C, p, p)
    tr = np.trace(g, axis1=1, axis2=2)[:, None, None]
    g = g + ridge * np.maximum(tr, 1e-300) / max(p, 1) * np.eye(p)[None]
    rhs = rows.transpose(0, 2, 1) @ y                     # (C, p, 1)
    try:
        return np.asarray(np.linalg.solve(g, rhs)[:, :, 0], dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.zeros((c, p))


def poles_batch(coef: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """`(C, p)` roots, via batched eigenvalues of the companion matrices."""
    a = np.asarray(coef, dtype=np.float64)
    c, p = a.shape
    if p == 0:
        return np.zeros((c, 0), dtype=np.complex128)
    comp = np.zeros((c, p, p))
    comp[:, 0, :] = a
    if p > 1:
        idx = np.arange(p - 1)
        comp[:, idx + 1, idx] = 1.0
    bad = ~np.isfinite(comp).all(axis=(1, 2))
    comp[bad] = 0.0
    return np.asarray(np.linalg.eigvals(comp), dtype=np.complex128)


def pole_features(block: npt.ArrayLike, *, fps: float,
                  order: int = AR_ORDER) -> dict[str, F64]:
    """Per-channel `(frequency_hz, radius, damping, real_radius)`."""
    b = np.asarray(block, dtype=np.float64)
    n_c = b.shape[1]
    out = {k: np.full(n_c, np.nan) for k in
           ("frequency_hz", "radius", "damping", "real_radius")}
    z = poles_batch(ar_fit_batch(b, order))
    for c in range(n_c):
        d = dominant_pole(z[c], fps=fps)
        for k in out:
            out[k][c] = d[k]
    return out


def multitaper_psd(x: npt.ArrayLike, *, fps: float, nw: float = NW,
                   k: int = N_TAPERS) -> tuple[F64, F64]:
    """`(freq, psd)` by Thomson's method. Averaged over channels.

    Multitaper rather than a single window because a sub-second window has too
    few degrees of freedom for a periodogram: the variance of a one-taper
    estimate does not fall with window length, and every band comparison here
    is between two short windows.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a.reshape(a.shape[0], -1)
    n = a.shape[0]
    if n < 8:
        return np.zeros(0), np.zeros(0)
    kk = max(1, min(int(k), int(2 * nw) - 1))
    tapers = dpss(n, nw, kk)
    a = a - a.mean(axis=0, keepdims=True)
    freq = np.asarray(np.fft.rfftfreq(n, d=1.0 / float(fps)),
                      dtype=np.float64)
    acc = np.zeros((freq.size, a.shape[1]))
    for t in range(kk):
        spec = np.fft.rfft(a * tapers[t][:, None], axis=0)
        acc += (np.abs(spec) ** 2)
    psd = 2.0 * acc.mean(axis=1) / (kk * float(fps))
    return freq, np.asarray(psd, dtype=np.float64)


def band_power(freq: npt.ArrayLike, psd: npt.ArrayLike, *,
               bands: Sequence[tuple[str, float, float]] = BANDS,
               normalise: bool = True) -> dict[str, float]:
    """Power in each registered band, normalised to a SHAPE by default.

    Normalised because the floor moves 182x between still and moving windows:
    an unnormalised band power is mostly a readout of how fast the animal was
    going. The normalised version asks *where* the power is, which is the
    question the dynamics channel is actually posing.
    """
    f = np.asarray(freq, dtype=np.float64)
    p = np.asarray(psd, dtype=np.float64)
    out: dict[str, float] = {}
    tot = 0.0
    for name, lo, hi in bands:
        sel = (f >= lo) & (f < hi)
        v = float(np.trapezoid(p[sel], f[sel])) if int(sel.sum()) > 1 else 0.0
        out[name] = v
        tot += v
    if normalise:
        d = tot if tot > 0 else 1.0
        out = {k: v / d for k, v in out.items()}
    out["total_power"] = tot
    return out


def motion_energy(block: npt.ArrayLike) -> float:
    """Mean squared first difference. A covariate, never a boundary source."""
    b = np.asarray(block, dtype=np.float64)
    if b.shape[0] < 2:
        return float("nan")
    return float((np.diff(b, axis=0) ** 2).sum(axis=1).mean())


def describe(block: npt.ArrayLike, *, fps: float,
             order: int = AR_ORDER) -> dict[str, Any]:
    """Every descriptor for one window, as a flat record."""
    b = np.asarray(block, dtype=np.float64)
    pf = pole_features(b, fps=fps, order=order)
    freq, psd = multitaper_psd(b, fps=fps)
    bp = band_power(freq, psd)
    with np.errstate(invalid="ignore"):
        rec: dict[str, Any] = {
            "pole_frequency_hz": float(np.nanmedian(pf["frequency_hz"])),
            "pole_radius": float(np.nanmedian(pf["radius"])),
            "pole_damping": float(np.nanmedian(pf["damping"])),
            "pole_real_radius": float(np.nanmedian(pf["real_radius"])),
            "n_oscillatory_channels": float(
                np.isfinite(pf["frequency_hz"]).sum()),
            "motion_energy": motion_energy(b)}
    rec.update({f"band_{k}": v for k, v in bp.items()})
    return rec
