"""Tracking noise with the measured COLOUR, not just the measured amplitude.

READ results/DYNAMICS_PREREGISTRATION.md FIRST.

## The error this module exists to correct

`vieb/seg/jitter.py:draw` samples independent noise per frame. It is **white**.
`shapeflow`'s frozen calibration -- 3,846 recordings, 17,160 quiet segments,
inside digest `198eb14ff258c7f6` -- says the residual is not:

    whiteness: FAIL
    lag-1 autocorrelation  +0.5817,  decaying over 267 ms
    cross-keypoint correlation +0.1180
    common mode 0.2257 of residual power against 0.1429 if independent

So `NOISEFLOOR.md`'s floor and `PLANT.md`'s recovery curve were both measured
against noise of the wrong colour. For an amplitude threshold that is probably
survivable -- `PLANT.md` measured near-invariance to amplitude. For a
**dynamics** channel it is fatal if unfixed: a 267 ms correlation time is
~3.7 Hz, and the usable signal band is 0.476-4.83 Hz, so coloured tracking noise
lives exactly where a slow behavioural rhythm would and looks exactly like one.

## Why this is synthesised and not re-estimated

`calibration.json` already publishes `psd_noise_pooled` on a 32-point grid and
the residual `acf` at 11 lags, and it is a **frozen consumed artifact**. Fitting
a new noise model to the corpus would be a second estimate of a quantity the
programme has already measured more carefully than this stage could. So the
generator is **spectral synthesis against the published PSD**, which reproduces
the whole autocovariance by Wiener-Khinchin rather than approximating it.

**An AR(1) would not do.** The measured ACF falls 1.0 -> 0.582 -> 0.346 ->
0.207 and then **flattens near 0.15**. One pole gives a pure geometric decay and
cannot produce that floor; the floor is the common-mode component, which is
modelled separately and explicitly.

## Units

**Pixels**, because the calibration is in pixels. `jitter.draw` returns body
lengths and its callers multiply by `ell`; this returns pixels and is added to a
pose directly. The contract differs deliberately and the registration says so,
to stop the difference being read later as a bug.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

sys.path.insert(0, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur.read import Read                                         # noqa: E402
from vieb.io import spine                                           # noqa: E402

__all__ = ["ACF_TOL", "TIME_TOL_FRAMES", "PSD_TOL", "calibration",
           "noise_psd", "measured_acf", "common_share", "segment_frames",
           "draw", "detrended_acf", "acf_of", "correlation_time", "validate",
           "model_read"]

F64 = npt.NDArray[np.float64]

#: Registered acceptance for the model itself (§2). Mechanical: if the
#: synthesised noise does not reproduce the measured autocorrelation, no
#: descriptor floor computed under it means anything.
ACF_TOL = 0.05
TIME_TOL_FRAMES = 1.0
PSD_TOL = 1.25


def calibration() -> dict[str, Any]:
    """The frozen shapeflow calibration. Never re-estimated here."""
    return dict(spine.json_("sf_calibration"))


def noise_psd(cal: Mapping[str, Any] | None = None) -> tuple[F64, F64]:
    """`(frequency_hz, psd_noise_pooled)` -- the measured residual spectrum."""
    c = dict(cal) if cal is not None else calibration()
    return (np.asarray(c["frequency_hz"], dtype=np.float64),
            np.asarray(c["psd_noise_pooled"], dtype=np.float64))


def measured_acf(cal: Mapping[str, Any] | None = None) -> tuple[F64, F64]:
    """`(lags_s, acf)` of the residual, from the frozen whiteness read."""
    c = dict(cal) if cal is not None else calibration()
    d = c["whiteness"]["detail"]
    return (np.asarray(d["acf_lags_s"], dtype=np.float64),
            np.asarray(d["acf"], dtype=np.float64))


def common_share(cal: Mapping[str, Any] | None = None) -> float:
    """The fraction of residual power shared across keypoints.

    Taken from the measured **cross-keypoint correlation**, not from
    `common_mode_power_share`. For `K` keypoints with a shared fraction `c`,
    the common-mode power share is `c + (1-c)/K`, so the two are different
    statistics of the same model: at the published `c = 0.1180` and `K = 7`
    that predicts `0.244` against the measured `0.2257`. Using the correlation
    directly makes the generator's parameter the thing it is set from.
    """
    c = dict(cal) if cal is not None else calibration()
    return float(c["whiteness"]["detail"]["cross_keypoint_correlation"])


def segment_frames(cal: Mapping[str, Any] | None = None, *,
                   fps: float = 30.0) -> int:
    """The calibration's own segment length, in frames.

    From its published rule, `2 * 10^(min_decades + 0.5) / fps`, which at
    `min_decades = 1.0` and 30 fps is the 2.11 s `CALIBRATION.md` quotes. It
    matters because the published ACF is measured on segments of exactly this
    length, **after linear detrending**, and a synthetic sample measured any
    other way is not comparable to it.
    """
    c = dict(cal) if cal is not None else calibration()
    dec = float(c["params"]["min_decades"])
    return max(8, int(round(2.0 * 10.0 ** (dec + 0.5))))


def _amplitudes(freq: npt.ArrayLike, psd: npt.ArrayLike, n: int,
                fps: float) -> F64:
    """Per-bin Fourier magnitudes reproducing a one-sided PSD at length `n`.

    `|X_k|^2 = S(f_k) * fps * n / 2` is the normalisation that makes
    `irfft` return a series whose variance integrates the target PSD.
    Interpolated in **log power**, because the spectrum spans three decades and
    a linear interpolation between coarse bins would sag.
    """
    f = np.asarray(freq, dtype=np.float64)
    s = np.asarray(psd, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(s) & (s > 0)
    grid = np.fft.rfftfreq(n, d=1.0 / float(fps))
    if int(ok.sum()) < 2:
        return np.zeros(grid.size)
    lg = np.interp(grid, f[ok], np.log(s[ok]),
                   left=float(np.log(s[ok][0])), right=float(np.log(s[ok][-1])))
    amp = np.sqrt(np.exp(lg) * float(fps) * n / 2.0)
    amp[0] = 0.0                      # no DC offset: a constant is not noise
    return np.asarray(amp, dtype=np.float64)


def _synth(rng: np.random.Generator, amp: F64, n: int,
           shape: tuple[int, ...]) -> F64:
    """Random-phase spectral synthesis, one independent series per element."""
    phi = rng.uniform(0.0, 2.0 * np.pi, size=shape + (amp.size,))
    spec = amp[None, :] * np.exp(1j * phi).reshape(-1, amp.size)
    out = np.asarray(np.fft.irfft(spec, n=n, axis=-1), dtype=np.float64)
    return np.asarray(np.moveaxis(out.reshape(shape + (n,)), -1, 0),
                      dtype=np.float64)


def draw(rng: np.random.Generator, n_frames: int, n_keypoints: int, *,
         fps: float, cal: Mapping[str, Any] | None = None,
         scale: float = 1.0) -> F64:
    """`(T, K, 2)` tracking noise in **pixels**, with the measured colour.

    A shared component carrying `common_share` of the power, plus an
    independent component carrying the rest, so both the spectrum and the
    cross-keypoint correlation come out right. `scale` multiplies the
    amplitude and is how a dose-response arm is built.
    """
    c = dict(cal) if cal is not None else calibration()
    f, s = noise_psd(c)
    share = common_share(c)
    n = int(n_frames)
    if n < 8:
        return np.zeros((max(n, 0), int(n_keypoints), 2))
    amp = _amplitudes(f, s, n, fps) * float(scale)
    shared = _synth(rng, amp * np.sqrt(share), n, (2,))          # (T, 2)
    indep = _synth(rng, amp * np.sqrt(1.0 - share), n,
                   (int(n_keypoints), 2))                        # (T, K, 2)
    return indep + shared[:, None, :]


def detrended_acf(x: npt.ArrayLike, n_lags: int, *, seg: int) -> F64:
    """ACF over linearly-detrended segments, averaged -- the published recipe.

    `shapeflow/calibrate.py:591 residual_structure` detrends each fixed-length
    segment before correlating, so the published lag-1 of **+0.5817** is a
    *detrended* statistic. Measuring a synthetic sample without detrending
    compares two different quantities and overstates the correlation -- on the
    first run here, 0.724 against 0.582, which reads as a broken generator when
    what is broken is the comparison.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a.reshape(a.shape[0], -1)
    n_seg = a.shape[0] // int(seg)
    if n_seg < 1:
        return acf_of(a, n_lags)
    t = np.arange(int(seg), dtype=np.float64)
    des = np.column_stack([np.ones(int(seg)), (t - t.mean()) / max(t.std(), 1.0)])
    out = []
    for i in range(n_seg):
        blk = a[i * int(seg):(i + 1) * int(seg)]
        coef, *_ = np.linalg.lstsq(des, blk, rcond=None)
        out.append(acf_of(blk - des @ coef, n_lags))
    return np.asarray(np.mean(out, axis=0), dtype=np.float64)


def acf_of(x: npt.ArrayLike, n_lags: int) -> F64:
    """Autocorrelation to `n_lags`, averaged over every column."""
    a = np.asarray(x, dtype=np.float64)
    a = a.reshape(a.shape[0], -1)
    a = a - a.mean(axis=0, keepdims=True)
    var = (a ** 2).mean(axis=0)
    out = np.zeros(n_lags + 1)
    for k in range(n_lags + 1):
        num = (a[k:] * a[:a.shape[0] - k]).mean(axis=0) if k else var
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = float(np.nanmean(np.where(var > 0, num / var, np.nan)))
    return out


def correlation_time(acf: npt.ArrayLike, lags_s: npt.ArrayLike,
                     *, floor: float = 0.05) -> float:
    """First lag at which the ACF drops below `floor`. The published rule."""
    a = np.asarray(acf, dtype=np.float64)
    t = np.asarray(lags_s, dtype=np.float64)
    below = np.flatnonzero(a < floor)
    return float(t[below[0]]) if below.size else float(t[-1])


def validate(sample: npt.ArrayLike, *, fps: float,
             cal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Does the synthesised noise reproduce the measured colour?"""
    c = dict(cal) if cal is not None else calibration()
    lags_s, want = measured_acf(c)
    n_lags = int(want.size) - 1
    seg = segment_frames(c, fps=fps)
    got = detrended_acf(sample, n_lags, seg=seg)
    f, s = noise_psd(c)
    a = np.asarray(sample, dtype=np.float64)
    flat = a.reshape(a.shape[0], -1)
    n = flat.shape[0]
    spec = np.fft.rfft(flat - flat.mean(axis=0, keepdims=True), axis=0)
    grid = np.fft.rfftfreq(n, d=1.0 / float(fps))
    got_psd = (2.0 * (np.abs(spec) ** 2).mean(axis=1)) / (float(fps) * n)
    band = (grid >= float(f[f > 0].min())) & (grid <= float(f.max()))
    tgt = np.interp(grid[band], f, s)
    ratio = got_psd[band] / np.maximum(tgt, 1e-300)
    good = np.isfinite(ratio) & (ratio > 0)
    med = float(np.median(ratio[good])) if good.any() else float("nan")
    return {"acf_measured": want.tolist(), "acf_synth": got.tolist(),
            "acf_lags_s": lags_s.tolist(),
            "acf_lag1_measured": float(want[1]), "acf_lag1_synth": float(got[1]),
            "correlation_time_measured": correlation_time(want, lags_s),
            "correlation_time_synth": correlation_time(got, lags_s),
            "psd_ratio_median": med,
            "psd_ratio_p05": float(np.quantile(ratio[good], 0.05))
            if good.any() else float("nan"),
            "psd_ratio_p95": float(np.quantile(ratio[good], 0.95))
            if good.any() else float("nan"),
            "common_share": common_share(c), "fps": float(fps),
            "segment_frames": int(seg), "n_frames": int(n)}


def model_read(v: Mapping[str, Any], *, scored_object: dict[str, Any],
               n_effective: int) -> Read:
    """§2's acceptance. Mechanical, and it gates everything downstream.

    If the generator does not reproduce the measured autocorrelation there is
    no point quoting a descriptor floor computed under it, so this returns
    `NOT_A_RESULT` and the stage stops.
    """
    d_acf = abs(float(v["acf_lag1_synth"]) - float(v["acf_lag1_measured"]))
    d_t = abs(float(v["correlation_time_synth"])
              - float(v["correlation_time_measured"])) * float(v["fps"])
    med = float(v["psd_ratio_median"])
    ok_psd = np.isfinite(med) and (1.0 / PSD_TOL) <= med <= PSD_TOL
    detail = {**dict(v), "d_acf_lag1": d_acf, "d_correlation_time_frames": d_t,
              "acf_tol": ACF_TOL, "time_tol_frames": TIME_TOL_FRAMES,
              "psd_tol": PSD_TOL}
    if d_acf <= ACF_TOL and d_t <= TIME_TOL_FRAMES and ok_psd:
        return Read(
            "PASS", scored_object,
            (f"the synthesised noise carries the MEASURED colour: lag-1 ACF "
             f"{v['acf_lag1_synth']:.4f} against the calibration's "
             f"{v['acf_lag1_measured']:.4f} (|d| {d_acf:.4f} <= {ACF_TOL}), "
             f"correlation time {v['correlation_time_synth']:.4f}s against "
             f"{v['correlation_time_measured']:.4f}s ({d_t:.2f} frames), PSD "
             f"ratio median {med:.3f} within {PSD_TOL:g}x. White noise at the "
             f"same variance would give lag-1 0.0"),
            n_effective=n_effective, detail=detail)
    return Read(
        "NOT_A_RESULT", scored_object,
        (f"the noise model does NOT reproduce the measured colour: lag-1 ACF "
         f"{v['acf_lag1_synth']:.4f} against {v['acf_lag1_measured']:.4f} "
         f"(|d| {d_acf:.4f}), correlation time off by {d_t:.2f} frames, PSD "
         f"ratio median {med:.3f}. No descriptor floor computed under this is "
         f"a measurement, and the model is the result"),
        n_effective=n_effective, detail=detail)
