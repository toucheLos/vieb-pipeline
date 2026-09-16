"""Odd-one-out triads: the cues the design exists to remove."""
from __future__ import annotations

import numpy as np
import pytest

from vieb.seg import triad as tr


def _seg(animal, n, speed):
    return {"animal": animal, "n_frames": n, "speed": speed,
            "recording_id": f"r{animal}", "a": 0, "b": n}


def _island(k=40):
    # Slow and long, as the real island is: 0.205x the corpus speed.
    return [_seg(f"i{i % 13}", 40 + i, 0.08 + 0.001 * i) for i in range(k)]


def _pool(k=400):
    return [_seg(f"p{i % 47}", 10 + (i % 200), 0.05 + 0.01 * (i % 40))
            for i in range(k)]


def test_every_trial_uses_three_distinct_animals():
    """Two clips of one mouse in one box resemble each other because it is one
    mouse in one box. `select.one_per_recording` states the same rule."""
    rng = np.random.default_rng(0)
    trials = tr.plan_trials(_island(), _pool(), arm="matched", n_trials=20,
                            rng=rng)
    assert trials
    for t in trials:
        assert len(set(t["animals"])) == 3


def test_the_odd_position_is_balanced_by_construction():
    """An i.i.d. draw leaves the marginal uneven at this n, and a scorer who
    notices is answering a different question."""
    rng = np.random.default_rng(1)
    trials = tr.plan_trials(_island(60), _pool(), arm="unmatched",
                            n_trials=36, rng=rng)
    counts = np.bincount([t["odd_position"] for t in trials], minlength=3)
    assert counts.min() >= counts.max() - 1


def test_both_trial_types_are_built_and_the_odd_one_matches_the_type():
    rng = np.random.default_rng(2)
    trials = tr.plan_trials(_island(60), _pool(), arm="matched", n_trials=20,
                            rng=rng)
    kinds = {t["type"] for t in trials}
    assert kinds == {"island_pair", "control_pair"}
    for t in trials:
        want = "control" if t["type"] == "island_pair" else "island"
        assert t["odd_is"] == want


def test_the_odd_clip_sits_at_its_declared_position():
    rng = np.random.default_rng(3)
    for t in tr.plan_trials(_island(40), _pool(), arm="matched", n_trials=12,
                            rng=rng):
        odd = t["clips"][t["odd_position"]]
        others = [c for i, c in enumerate(t["clips"])
                  if i != t["odd_position"]]
        # The two non-odd clips are the pair, so they share a class; the
        # cheapest observable proxy is that they are not the odd one.
        assert odd not in others


def test_matching_picks_a_control_close_in_duration_and_speed():
    """And searches the WHOLE pool: narrowing first means a 20-frame target can
    be matched against a label whose clips are all 180, and the match fails
    while appearing to have been made."""
    target = {"n_frames": 40, "speed": 0.08}
    pool = [_seg("a", 400, 2.0), _seg("b", 41, 0.081), _seg("c", 12, 0.9)]
    j = tr.match_control(target, pool, used=set(), forbid_animals=set(),
                         rng=np.random.default_rng(0), n_near=1)
    assert pool[j]["animal"] == "b"


def test_matching_respects_the_forbidden_animals():
    target = {"n_frames": 40, "speed": 0.08}
    pool = [_seg("a", 41, 0.081), _seg("b", 400, 2.0)]
    j = tr.match_control(target, pool, used=set(), forbid_animals={"a"},
                         rng=np.random.default_rng(0), n_near=1)
    assert pool[j]["animal"] == "b"


def test_matching_refuses_when_the_pool_is_exhausted():
    j = tr.match_control({"n_frames": 40, "speed": 0.08}, [_seg("a", 40, 0.08)],
                         used={0}, forbid_animals=set(),
                         rng=np.random.default_rng(0))
    assert j == -1


def test_the_matched_arm_is_closer_in_speed_than_the_unmatched_arm():
    """The whole point of the primary control: without it the scorer sorts on
    'is it moving', and the island runs at 0.205x the corpus speed."""
    isl, pool = _island(60), _pool()
    m = tr.plan_trials(isl, pool, arm="matched", n_trials=30,
                       rng=np.random.default_rng(4))
    u = tr.plan_trials(isl, pool, arm="unmatched", n_trials=30,
                       rng=np.random.default_rng(4))

    def gap(trials):
        out = []
        for t in trials:
            sp = [c["speed"] for c in t["clips"]]
            out.append(max(sp) / max(min(sp), 1e-9))
        return float(np.median(out))

    assert gap(m) < gap(u)


def test_trial_ids_do_not_encode_the_answer():
    """A function of arm, type and index only. Hashing the odd position in would
    make the key recoverable by re-running the builder."""
    a = tr.trial_id("matched", "island_pair", 7)
    b = tr.trial_id("matched", "island_pair", 7)
    c = tr.trial_id("unmatched", "island_pair", 7)
    assert a == b and a != c and a.startswith("t") and len(a) == 11


def test_mde_falls_with_n_and_is_reported_against_a_named_lift():
    assert tr.mde(72) > tr.mde(400)
    rd = tr.mde_read(72, scored_object={"a": 1}, n_effective=72, plausible=0.20)
    assert rd.verdict in ("PASS", "FAIL")
    assert "MDE" in rd.reason and "0.20" in rd.reason


def test_accuracy_read_uses_the_animal_clustered_interval_to_decide():
    """percall.py: batch 1 returned 43.1% at binomial p = 0.025 with a clustered
    CI of [0.320, 0.546] that INCLUDES chance."""
    # Perfect scores from two animals only: the clustered interval is wide.
    rd = tr.accuracy_read([True] * 20, ["a"] * 10 + ["b"] * 10, arm="matched",
                          ttype="both", scored_object={"a": 1}, n_effective=2)
    assert rd.verdict == "PASS"
    # 50% over only 30 trials does NOT clear chance once the interval is
    # animal-clustered -- which is why N_TRIALS was amended upward before any
    # trial was built. The naive binomial would have called this significant.
    mixed = [True, False] * 15
    rd2 = tr.accuracy_read(mixed, [f"a{i % 12}" for i in range(30)],
                           arm="matched", ttype="both", scored_object={"a": 1},
                           n_effective=12)
    assert rd2.verdict == "FAIL"
    chance_like = [True] * 10 + [False] * 20
    rd3 = tr.accuracy_read(chance_like, [f"a{i % 12}" for i in range(30)],
                           arm="matched", ttype="both", scored_object={"a": 1},
                           n_effective=12)
    assert rd3.verdict == "FAIL" and "includes chance" in rd3.reason


def test_accuracy_read_refuses_on_too_few_trials():
    rd = tr.accuracy_read([True, False], ["a", "b"], arm="matched",
                          ttype="both", scored_object={"a": 1}, n_effective=2)
    assert rd.verdict == "INCONCLUSIVE"


def test_the_registered_constants_match_the_registration():
    assert tr.N_TRIALS == 180
    assert tr.ARMS == ("matched", "unmatched")
    assert tr.TYPES == ("island_pair", "control_pair")
    assert tr.CHANCE == pytest.approx(1 / 3)


def test_published_manifest_carries_no_answer():
    """The site publishes the instrument, not the result.

    A published `arm` or `type` partitions the trials into the two questions the
    design asks; a published `speed` is the island's 0.205x cue in a column; a
    published `odd_position` is simply the key. Any of them makes the page
    unscoreable by the next reader, which is the only reason it is published.
    """
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "island_look", "manifest.json")
    if not os.path.exists(path):
        return                       # nothing rendered in this checkout
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    banned = {"odd_position", "odd_is", "arm", "type", "index", "speed",
              "a", "b", "n_frames"}
    for row in doc["clips"]:
        assert not (set(row) & banned), sorted(set(row) & banned)
    # And the clip duration must be constant within a trial, or the reader picks
    # the odd one with a stopwatch.
    by_trial: dict = {}
    for row in doc["clips"]:
        by_trial.setdefault(row["trial"], set()).add(row["duration_s"])
    assert all(len(v) == 1 for v in by_trial.values())
