"""STATES end to end on synthetic features, before any real feature exists.

Fake feature files for real tune/fit/report animals carry a hidden two-state
switching process (dwell ~3 s) in several pose channels. Every phase of
`scripts/states.py` runs through, at reduced size (N_MICRO, N_BOOT, FIT_ROWS
overridden), into temporary directories. The point is that the registered
pipeline runs and finds planted slow structure -- not any number about mice.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))


def _fake(path: str, rng: np.random.Generator, n_rec=2, t=3000) -> None:
    xs, vm, xm = [], [], []
    for _ in range(n_rec):
        s = np.zeros(t, dtype=bool)
        i, on = 0, False
        while i < t:
            n = int(rng.gamma(4.0, 90 / 4.0)) + 1
            s[i:i + n] = on
            i += n
            on = not on
        e = rng.normal(0, 1, (t, 17))
        x = np.zeros((t, 17))
        for j in range(1, t):                     # AR(1): short, stationary memory
            x[j] = 0.8 * x[j - 1] + 0.6 * e[j]
        x = x.astype(np.float32)
        x[:, [4, 5, 14]] += 2.5 * s[:, None]
        xs.append(x)
        xm.append(rng.normal(0, 1, (t, 4)).astype(np.float32))
        vm.append(np.ones(t, dtype=bool))
    X = np.concatenate(xs)
    np.savez_compressed(
        path, X_pose=X, valid_pose=np.ones(X.shape[0], bool),
        X_mask=np.concatenate(xm), valid_mask=np.concatenate(vm),
        head_covered=np.ones(X.shape[0], bool), head_defined=np.ones(X.shape[0], bool),
        raw_speed=np.where(rng.random(X.shape[0]) < 0.2, 0.0, 1.0).astype(np.float32),
        bounds=np.arange(0, (n_rec + 1) * t, t, dtype=np.int64),
        recording_ids=np.array([f"fake_{k}" for k in range(n_rec)]), ell_a=100.0,
        context=np.array(["A", "B"][:n_rec]), day=np.array([3, 4][:n_rec]),
        pose_arm=np.array("repaired"))



def test_states_pipeline_runs_end_to_end_and_finds_the_planted_switch(tmp_path):
    from recur import splits
    sys.path.insert(0, REPO)
    from vieb.io import spine

    of = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    feat = tmp_path / "features"
    feat.mkdir()
    rng = np.random.default_rng(0)
    for split in ("tune", "fit", "report"):
        for tag in sorted(t for t, s in of.items() if s == split)[:4]:
            _fake(str(feat / f"{tag}.npz"), rng)
    env = {**os.environ, "VIEB_STATES_FEAT": str(feat),
           "VIEB_STATES_WORK": str(tmp_path / "work"),
           "VIEB_STATES_NMICRO": "40", "VIEB_STATES_NBOOT": "20",
           "VIEB_STATES_FITROWS": "6000"}
    script = os.path.join(REPO, "scripts", "states.py")

    def run(*args):
        r = subprocess.run([sys.executable, script, *args], env=env,
                           capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stderr[-2000:]
        return r.stdout + r.stderr

    run("--phase", "leak")
    run("--phase", "select")
    run("--phase", "fit")
    for kind in ("phase", "var5", "microstate", "microstate0", "white"):
        run("--phase", "surrogate", "--kind", kind)
    out = tmp_path / "states.json"
    run("--phase", "combine", "--out", str(out))
    res = json.load(open(out))["results"]
    assert res["M"] >= 2, res["counting"]
    assert res["gate3_plant"]["planted_recordings"] >= 1
    assert res["verdict"] in ("PASS", "FAIL")
