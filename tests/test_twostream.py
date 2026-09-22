"""The two streams, their different bands, and the split that is the output."""
from __future__ import annotations

import numpy as np

from vieb.seg import twostream as ts

OBJ = {"dataset": "luna", "arm": "twostream", "split": "tune"}


def test_the_two_streams_have_different_registered_bands():
    """Stage 0 measured them to localise differently -- +/-2 for the frozen
    detector, +/-15 derived from the dynamics probe's own offset spread.
    Averaging them to one band would flatter the worse instrument."""
    assert ts.CONFIG_TOL == 2 and ts.DYN_TOL == 15


def test_labelling_is_one_to_one():
    """One configuration boundary cannot absorb three dynamics boundaries and
    make the streams look more agreed than they are."""
    got = ts.label_boundaries([100], [98, 100, 102], tol=15)
    assert got["n_both"] == 1 and got["n_dyn_only"] == 2


def test_disjoint_streams_are_all_only():
    got = ts.label_boundaries([100, 200], [500, 600], tol=15)
    assert got["n_both"] == 0
    assert got["n_config_only"] == 2 and got["n_dyn_only"] == 2


def test_the_labels_partition_every_boundary():
    rng = np.random.default_rng(0)
    c = np.sort(rng.choice(5000, 200, replace=False))
    d = np.sort(rng.choice(5000, 150, replace=False))
    g = ts.label_boundaries(c, d)
    assert g["n_config_only"] + g["n_both"] == g["n_config"]
    assert g["n_dyn_only"] + g["n_both"] == g["n_dyn"]


def test_a_dynamics_stream_that_adds_nothing_fails():
    """Registered prediction 1: if it fires only where configuration already
    did, the two-stream idea is refuted on this corpus."""
    rows = [{"animal": f"a{i}", "n_config": 100, "n_dyn": 50, "n_both": 50,
             "n_dyn_only": 0} for i in range(20)]
    r = ts.split_read(rows, scored_object=OBJ, n_effective=20)
    assert r.verdict == "FAIL" and "adds nothing" in r.reason


def test_a_dynamics_stream_with_its_own_boundaries_passes():
    rows = [{"animal": f"a{i}", "n_config": 100, "n_dyn": 60, "n_both": 20,
             "n_dyn_only": 40} for i in range(20)]
    r = ts.split_read(rows, scored_object=OBJ, n_effective=20)
    assert r.verdict == "PASS" and "OF ITS OWN" in r.reason


def test_clearing_a_floor_is_not_enough_it_must_beat_the_incumbent():
    """M12. A gate without a reference point admits bad results as readily as
    it refuses good ones -- which is exactly what happened here."""
    corpus = {"point": 0.24, "lo": 0.235, "hi": 0.245}
    floor = {"point": 0.23, "lo": 0.228, "hi": 0.232}
    r = ts.separation_read(corpus, floor, incumbent=0.5475,
                           scored_object=OBJ, n_effective=60)
    assert r.verdict == "FAIL" and "does NOT beat the incumbent" in r.reason


def test_a_stream_well_clear_of_its_floor_passes():
    corpus = {"point": 1.00, "lo": 0.95, "hi": 1.05}
    floor = {"point": 0.20, "lo": 0.19, "hi": 0.21}
    r = ts.separation_read(corpus, floor, incumbent=0.5475,
                           scored_object=OBJ, n_effective=60)
    assert r.verdict == "PASS"


def test_overlapping_intervals_fail_before_the_incumbent_is_consulted():
    corpus = {"point": 0.24, "lo": 0.20, "hi": 0.28}
    floor = {"point": 0.23, "lo": 0.21, "hi": 0.26}
    r = ts.separation_read(corpus, floor, incumbent=0.99,
                           scored_object=OBJ, n_effective=60)
    assert r.verdict == "FAIL" and "NOT distinguishable" in r.reason


def test_location_read_carries_no_verdict():
    r = ts.location_read({"config15": {"t0": 0.3, "t1": 0.4, "t2": 0.6}},
                         scored_object=OBJ, n_effective=60)
    assert r.verdict == "NOT_A_RESULT"
    assert abs(r.detail["config15"]["wall_over_centre"] - 2.0) < 1e-9
