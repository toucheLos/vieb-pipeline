"""Paper 1 optional analysis (registered, approved): Keypoint-MoSeq's own synthetic
null (Weinreb et al. 2024, Extended Data Fig. 2g,h) scored with Finding 1's dwell
statistic. Registration: `paper1/registrations/KPMS_SYNTHETIC_DWELL.md` (82b7489).

    python3 scripts/kpms_synth.py build        # LDS fit on tune, one synthetic recording per tune recording
    # GPU: jobs/kpms_synth.slurm (scripts/moseq_kpms.py with MOSEQ_WORK=work/kpms_synth, arm 'lds')
    python3 scripts/kpms_synth.py score        # -> paper1/registrations/kpms_synthetic_dwell.json

The LDS follows their Methods ("fit a linear dynamical system to egocentrically
aligned keypoint trajectories and then sampled randomly generated outputs from
the fitted model"): egocentric keypoints (`ego.egocentric`), PCA to the
labeller's latent_dim (10), one VAR of the labeller's nlags (3) with a bias,
Gaussian innovations, and a Gaussian observation residual. Each synthetic
recording has its source recording's length and starts from its first latent
frames. Every keypoint has confidence 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab, splits                   # noqa: E402
from recur.util import log, write_json                             # noqa: E402
from vieb import provenance                                        # noqa: E402
from vieb.clean import arms                                        # noqa: E402
from vieb.io import spine                                          # noqa: E402
from vieb.seeds import stable_seed                                 # noqa: E402
from vieb.tok import config, ego                                   # noqa: E402

WORK = os.path.join(config.REPO, "work", "kpms_synth")
OUT = os.path.join(config.REPO, "paper1", "registrations", "kpms_synthetic_dwell.json")
LATENT, NLAGS, SEED = 10, 3, 0
LUNA_SETS = os.path.join(config.REPO, "work", "moseq", "sets.json")


def tune_ids() -> list[str]:
    sp = splits.split_of_animal(splits.load(spine.sf("results/splits.json")))
    return sorted(r for r in spine.recording_ids() if sp.get(lab.animal_tag(r)) == "tune")


def ego_pose(rid: str, ell: float) -> np.ndarray:
    d = spine.clean(rid)
    miss = np.asarray(d["missing"], dtype=bool)
    p = arms.held_array(np.asarray(d["pose_unfiltered"], dtype=np.float64), miss)
    return ego.egocentric(p, ell)                     # (T, 14), body lengths


def build(_a) -> int:
    ids = tune_ids()
    blocks: dict = {}
    for r in ids:
        d = spine.clean(r)
        miss = np.asarray(d["missing"], bool)
        p = arms.held_array(np.asarray(d["pose_unfiltered"], np.float64), miss)
        blocks.setdefault(lab.animal_tag(r), ([], []))
        blocks[lab.animal_tag(r)][0].append(p)
        blocks[lab.animal_tag(r)][1].append(~miss.any(1))
    ells = {a: float(ego.ell_a(b[0], b[1])) for a, b in blocks.items()}
    Y = {r: ego_pose(r, ells[lab.animal_tag(r)]) for r in ids}
    allY = np.concatenate(list(Y.values()))
    mu = allY.mean(0)
    _u, s, vt = np.linalg.svd(allY - mu, full_matrices=False)
    C = vt[:LATENT].T                                   # (14, 10)
    X = {r: (y - mu) @ C for r, y in Y.items()}
    # VAR(NLAGS) + bias, least squares over per-recording design matrices (no seams)
    phis, tgts = [], []
    for x in X.values():
        T = x.shape[0]
        phis.append(np.concatenate([x[NLAGS - j:T - j] for j in range(1, NLAGS + 1)]
                                   + [np.ones((T - NLAGS, 1))], axis=1))
        tgts.append(x[NLAGS:])
    phi, tgt = np.concatenate(phis), np.concatenate(tgts)
    A, *_ = np.linalg.lstsq(phi, tgt, rcond=None)       # (L*NLAGS+1, L)
    Q = np.cov((tgt - phi @ A).T)
    R = np.var(allY - mu - (allY - mu) @ C @ C.T, axis=0)   # observation residual, per coordinate
    comp = np.zeros((LATENT * NLAGS, LATENT * NLAGS))
    comp[:LATENT, :] = A[:-1].T
    comp[LATENT:, :-LATENT] = np.eye(LATENT * (NLAGS - 1))
    rho = float(np.max(np.abs(np.linalg.eigvals(comp))))
    assert rho < 1.0, f"fitted LDS is not stationary (spectral radius {rho})"
    os.makedirs(os.path.join(WORK, "input", "lds"), exist_ok=True)
    Lq = np.linalg.cholesky(Q + 1e-12 * np.eye(LATENT))
    for r in ids:
        rng = np.random.default_rng(stable_seed("kpms-synth", SEED, r))
        T = X[r].shape[0]
        x = np.zeros((T, LATENT))
        x[:NLAGS] = X[r][:NLAGS]
        for t in range(NLAGS, T):
            ph = np.concatenate([x[t - j] for j in range(1, NLAGS + 1)] + [np.ones(1)])
            x[t] = ph @ A + Lq @ rng.standard_normal(LATENT)
        y = x @ C.T + mu + rng.standard_normal((T, 14)) * np.sqrt(R)
        ell = ells[lab.animal_tag(r)]
        coords = (y.reshape(T, 7, 2) * ell + 400.0).astype(np.float32)   # px, fixed placement
        np.savez_compressed(os.path.join(WORK, "input", "lds", f"{r}.npz"),
                            coords=coords, conf=np.ones((T, 7), np.float32))
    with open(LUNA_SETS, encoding="utf-8") as fh:
        fit = json.load(fh)["fit"]
    assert set(fit) <= set(ids), "Luna's MoSeq fit set must be tune recordings"
    with open(os.path.join(WORK, "sets.json"), "w", encoding="utf-8") as fh:
        json.dump({"fit": fit, "report": ids}, fh, indent=1)
    write_json({**provenance.header(anchors.LUNA, stage="kpms_synth_lds", unverified="tune split"),
                "n_recordings": len(ids), "latent_dim": LATENT, "nlags": NLAGS,
                "pca_explained": float((s[:LATENT] ** 2).sum() / (s ** 2).sum()),
                "spectral_radius": rho, "n_fit": len(fit)},
               os.path.join(WORK, "lds.json"))
    log(f"LDS built: {len(ids)} synthetic recordings, PCA {LATENT} explains "
        f"{(s[:LATENT] ** 2).sum() / (s ** 2).sum():.3f}, spectral radius {rho:.4f}")
    return 0


def score(_a) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dwell_moseq", os.path.join(os.environ.get("VIEB_RECUR", "/home/tul26194/recur"),
                                    "scripts", "dwell_moseq.py"))
    assert spec and spec.loader
    dm = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["dwell_moseq"]
    try:
        spec.loader.exec_module(dm)
    finally:
        sys.argv = saved
    rows = {}
    for s in (0, 1, 2):
        z = np.load(os.path.join(WORK, "labels", f"lds_s{s}.npz"))
        lab_ = np.asarray(z["syllable"], np.int64)
        b = np.asarray(z["bounds"], np.int64)
        rids = [str(r) for r in z["recording_ids"]]
        animal = np.asarray([lab.animal_tag(r) for r in rids])
        m = dm.measure(lab_, b, animal, seed=SEED)
        frac = np.bincount(lab_) / lab_.size
        top2 = float(np.sort(frac)[-2:].sum())
        rows[f"s{s}"] = {**m, "n_in_use_0.5pct": int((frac > 0.005).sum()), "top_two_share": top2}
    with open(os.path.join(WORK, "lds.json"), encoding="utf-8") as fh:
        lds = json.load(fh)
    write_json({**provenance.header(anchors.LUNA, stage="kpms_synthetic_dwell",
                                    unverified="approved optional analysis; tune split"),
                "registration": "paper1/registrations/KPMS_SYNTHETIC_DWELL.md", "lds": lds,
                "seeds": rows}, OUT)
    log(f"synthetic dwell: {json.dumps(rows)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("build", "score"))
    a = p.parse_args(argv)
    return {"build": build, "score": score}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
