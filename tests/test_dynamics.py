"""The dynamics channel: noise colour, descriptors, and the plant."""
from __future__ import annotations

import numpy as np

from vieb.seg import descriptors as ds, dynplant as dp, noise as nz

OBJ = {"dataset": "luna", "arm": "dynamics", "split": "tune"}
FPS = 30.0


# --- the noise colour ---------------------------------------------------

def test_the_synthesised_noise_is_not_white():
    """The whole point. White noise has lag-1 zero; the measured residual
    has +0.5817, and a floor computed under the wrong colour is not a
    measurement of anything."""
    rng = np.random.default_rng(0)
    x = nz.draw(rng, 20_000, 7, fps=FPS)
    acf = nz.acf_of(x, 4)
    assert acf[1] > 0.3


def test_the_detrended_acf_is_the_published_recipe():
    """`residual_structure` detrends each segment before correlating, so the
    published +0.5817 is a DETRENDED statistic. Measuring a synthetic sample
    raw overstates it -- 0.724 against 0.582 on the first run here, which
    reads as a broken generator when the comparison is what is broken."""
    rng = np.random.default_rng(0)
    x = nz.draw(rng, 20_000, 7, fps=FPS)
    raw = nz.acf_of(x, 4)[1]
    det = nz.detrended_acf(x, 4, seg=nz.segment_frames(fps=FPS))[1]
    assert det < raw


def test_the_common_mode_makes_keypoints_correlated():
    rng = np.random.default_rng(0)
    x = nz.draw(rng, 8000, 7, fps=FPS)[:, :, 0]
    c = np.corrcoef(x.T)
    off = c[~np.eye(7, dtype=bool)]
    assert 0.02 < float(off.mean()) < 0.35


def test_scaling_the_noise_scales_its_amplitude():
    rng = np.random.default_rng(0)
    one = float(np.std(nz.draw(rng, 4000, 7, fps=FPS)))
    two = float(np.std(nz.draw(rng, 4000, 7, fps=FPS, scale=2.0)))
    assert abs(two / one - 2.0) < 0.15


def test_a_model_that_misses_the_colour_is_refused():
    v = {"acf_lag1_synth": 0.0, "acf_lag1_measured": 0.58,
         "correlation_time_synth": 0.0, "correlation_time_measured": 0.27,
         "psd_ratio_median": 1.0, "fps": FPS}
    r = nz.model_read(v, scored_object=OBJ, n_effective=1)
    assert r.verdict == "NOT_A_RESULT" and "does NOT reproduce" in r.reason


# --- the descriptors ----------------------------------------------------

def test_poles_recover_a_planted_frequency():
    n = 128
    t = np.arange(n) / FPS
    x = np.cos(2 * np.pi * 3.0 * t)[:, None] * np.ones((1, 8))
    got = ds.describe(x + 1e-6, fps=FPS)
    assert abs(got["pole_frequency_hz"] - 3.0) < 0.2


def test_poles_are_scale_free():
    """The reason poles are primary: the noise floor moves 182x between still
    and moving windows, and an amplitude descriptor moves with it."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(200, 6)), axis=0)
    a = ds.describe(x, fps=FPS)
    b = ds.describe(x * 1000.0, fps=FPS)
    assert abs(a["pole_frequency_hz"] - b["pole_frequency_hz"]) < 1e-6
    assert abs(a["pole_radius"] - b["pole_radius"]) < 1e-9


def test_band_power_is_normalised_to_a_shape():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(256, 6)), axis=0)
    a = ds.describe(x, fps=FPS)
    b = ds.describe(x * 50.0, fps=FPS)
    for k in ("band_slow", "band_mid", "band_fast"):
        assert abs(a[k] - b[k]) < 1e-9
    assert b["band_total_power"] > a["band_total_power"]


def test_the_batched_fit_matches_the_per_channel_loop():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(300, 9)), axis=0)
    loop = np.array([ds.ar_fit(x[:, c], 4) for c in range(9)])
    assert np.abs(loop - ds.ar_fit_batch(x, 4)).max() < 1e-9


def test_a_pure_drift_has_no_oscillatory_pole_frequency():
    x = np.arange(200, dtype=float)[:, None] * np.ones((1, 4))
    d = ds.dominant_pole(ds.poles_of(ds.ar_fit(x[:, 0], 4)), fps=FPS)
    assert not np.isfinite(d["frequency_hz"]) or d["frequency_hz"] < 1.0


# --- the plant ----------------------------------------------------------

def test_the_plant_changes_only_the_named_parameter():
    """Both sides carry an oscillation, so a frequency plant is not an
    amplitude step wearing a frequency step's name."""
    rng = np.random.default_rng(0)
    n = 240
    base = np.zeros((n, 14))
    y = dp.plant_into(base, [n // 2], kind="frequency", level=2.0, fps=FPS,
                      win=30, amp_bl=0.3, idx=range(14),
                      rng=np.random.default_rng(1))
    pre = float(np.std(y[:n // 2]))
    post = float(np.std(y[n // 2:]))
    assert abs(pre - post) / max(pre, 1e-12) < 0.35      # amplitude held


def test_the_plant_spans_the_whole_slice_with_no_envelope():
    """D12's lesson, arriving twice. A windowed or tapered plant puts the
    rhythm's ARRIVAL next to the parameter change, and an arrival is always
    the larger event -- measured, the detector fired 28 then 108-118 frames
    away before this was fixed."""
    rng = np.random.default_rng(0)
    n = 240
    y = dp.plant_into(np.zeros((n, 14)), [n // 2], kind="damping", level=0.5,
                      fps=FPS, win=30, amp_bl=0.3, idx=range(14), rng=rng)
    edge = float(np.abs(y[:5]).max())
    assert edge > 0.0            # the oscillation is present at the edges


def test_the_oscillator_has_the_requested_frequency():
    rng = np.random.default_rng(0)
    y = dp.oscillation(2048, freq_hz=4.0, radius=0.99, fps=FPS, rng=rng)
    f = np.fft.rfftfreq(y.size, d=1.0 / FPS)
    peak = float(f[int(np.argmax(np.abs(np.fft.rfft(y - y.mean())) ** 2))])
    assert abs(peak - 4.0) < 0.5


def test_a_thin_cell_is_refused_not_caveated():
    r = dp.recovery_read([], kind="frequency", level=1.0, n_instances=10,
                         scored_object=OBJ, n_effective=60)
    assert r.verdict == "NOT_A_RESULT" and "Refused" in r.reason


def test_recall_at_chance_fails():
    rows = [{"animal": f"a{i}", "recall": 0.03, "chance": 0.03}
            for i in range(20)]
    r = dp.recovery_read(rows, kind="frequency", level=1.0, n_instances=5000,
                         scored_object=OBJ, n_effective=20)
    assert r.verdict == "FAIL"


def test_the_floor_read_names_persistence_as_the_discriminator():
    null = {"pole_radius": [0.4] * 50, "pole_frequency_hz": [8.0] * 50}
    r = dp.floor_read(null, scored_object=OBJ, n_effective=60)
    assert r.verdict == "NOT_A_RESULT" and "PERSISTENCE" in r.reason


def test_the_registered_constants_are_what_the_registration_says():
    assert dp.WIN_S == 1.0 and dp.ALPHA == 3.0
    assert ds.BANDS[0][0] == "slow" and ds.BANDS[-1] == ("fast", 6.0, 15.0)
    assert nz.ACF_TOL == 0.05 and nz.PSD_TOL == 1.25
