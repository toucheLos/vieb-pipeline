"""The banking module's job is to refuse to state a number without its arm."""
from __future__ import annotations

import json
import os

import pytest

from vieb.seg import banked as bkd


def _ci(point, lo, hi, n=89):
    return {"point": point, "lo": lo, "hi": hi, "n_animals": n}


def _cell(point=0.016391, lo=0.012049, hi=0.020796):
    return {"n_queries": 3022508,
            "vs": {"ar": {"delta": _ci(point, lo, hi)},
                   "phase": {"delta": _ci(0.150757, 0.131211, 0.168849)}}}


def _occ():
    return {"occupancy_equivalent": 0.10874, "occupancy_equivalent_lo": 0.07687,
            "occupancy_equivalent_hi": 0.14220, "log_log_slope": 0.8873}


def _verdict():
    return {"floor_occupancy": 0.0025, "checks": {"1_above_both_nulls": {"ok": 1}}}


def test_q1_read_names_the_wiener_arm_in_its_own_reason():
    """The banked headline must say which arm it holds on, in words.

    A `+1.639%` quoted without `filt=wiener` is the wrong-object failure this
    module exists to prevent, and a scored_object nobody reads is not enough --
    the reason string is what gets pasted into a later document.
    """
    rd = bkd.q1_read(_cell(), _verdict(), _occ(), n_effective=89)
    assert rd.verdict == "PASS"
    assert rd.scored_object["filt"] == "wiener"
    assert "WIENER" in rd.reason
    assert "raw" in rd.reason


def test_q1_read_fails_when_the_interval_spans_zero():
    rd = bkd.q1_read(_cell(0.001, -0.004, 0.006), _verdict(), _occ(),
                     n_effective=89)
    assert rd.verdict == "FAIL"


def test_unfiltered_read_fails_and_refuses_to_stand_for_the_raw_arm():
    """It is a caveat about `unfiltered`, not a measurement of F3's `raw`.

    Those are different arrays in different spaces -- bare `pose_unfiltered` in
    44 channel dims against `held_array(pose_unfiltered, missing)` in 17 ego
    dims -- and conflating them would make this banking worse than none.
    """
    rd = bkd.q1_unfiltered_read(_cell(-0.001991, -0.004636, 0.000418),
                                n_effective=89)
    assert rd.verdict == "FAIL"
    assert rd.scored_object["filt"] == "unfiltered"
    assert "held_array" in rd.reason


def test_roughness_read_needs_both_halves_of_the_signature():
    """Lower mean-log AND wider spread. Either alone is not concentration.

    A null that is simply smoother everywhere gives the first without the
    second; a null that is noisier in a few windows gives the second without
    the first. Only both together say the same total power sits in fewer
    windows.
    """
    n = 17
    good = {"corpus": {"mean_log_msd": [-7.0] * n,
                       "sd_across_windows": [2.8] * n},
            "phase": {"mean_log_msd": [-5.0] * n,
                      "sd_across_windows": [1.3] * n}}
    obj = {"dataset": "t", "null": "phase"}
    assert bkd.roughness_read(good, null="phase", scored_object=obj,
                              n_effective=9).verdict == "PASS"
    # Lower mean log, but no wider spread: not concentration.
    half = {"corpus": {"mean_log_msd": [-7.0] * n,
                       "sd_across_windows": [1.0] * n},
            "phase": {"mean_log_msd": [-5.0] * n,
                      "sd_across_windows": [1.3] * n}}
    assert bkd.roughness_read(half, null="phase", scored_object=obj,
                              n_effective=9).verdict == "FAIL"


def test_roughness_read_refuses_a_mean_carried_by_a_minority_of_channels():
    """17 channels, and a gap in 2 of them is not a gap.

    The mean over channels can be dragged by a single outlier, so the verdict
    is on the per-channel count as well as the mean.
    """
    mean_log = [-7.0, -7.0] + [-4.0] * 15
    lvl = {"corpus": {"mean_log_msd": mean_log,
                      "sd_across_windows": [2.8, 2.8] + [1.0] * 15},
           "phase": {"mean_log_msd": [-5.0] * 17,
                     "sd_across_windows": [1.3] * 17}}
    rd = bkd.roughness_read(lvl, null="phase", scored_object={"a": 1},
                            n_effective=9)
    assert rd.verdict == "FAIL"


def test_dwell_read_refuses_rather_than_disagreeing_with_its_source():
    """It re-states falsifier.json; it must not become a second definition."""
    rd = bkd.dwell_read({}, scored_object={"a": 1}, n_effective=89)
    assert rd.verdict == "INCONCLUSIVE"
    flipped = {"microstate_N256": {"verdict": "FAIL", "reason": "x"},
               "microstate0_N256": {"verdict": "PASS", "reason": "y"}}
    assert bkd.dwell_read(flipped, scored_object={"a": 1},
                          n_effective=89).verdict == "FAIL"


def test_file_sha256_matches_hashlib_on_a_known_string(tmp_path):
    import hashlib
    p = tmp_path / "x.json"
    p.write_bytes(b'{"a": 1}')
    assert bkd.file_sha256(str(p)) == hashlib.sha256(b'{"a": 1}').hexdigest()


def test_the_banked_q1_record_on_disk_carries_a_hash_and_both_arms():
    """The committed artifact, not the function that wrote it."""
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "q1_banked.json")
    if not os.path.exists(path):
        pytest.skip("q1_banked.json not written yet")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert set(doc["reads"]) == {"q1_wiener", "q1_unfiltered"}
    assert doc["reads"]["q1_wiener"]["verdict"] == "PASS"
    assert doc["reads"]["q1_unfiltered"]["verdict"] == "FAIL"
    for v in doc["banked_from"].values():
        assert len(v["sha256"]) == 64
