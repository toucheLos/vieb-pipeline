"""MOSEQ §2-3 on the kpms side. Runs in ~/moseq/venv-moseq (numpy 1.26), so it
imports nothing from `vieb` or `recur`: only npz and json cross the boundary.

    python scripts/moseq_kpms.py setup  --arm real            # PCA, once per arm
    python scripts/moseq_kpms.py fit    --arm real --seed 0   # AR-HMM 50 @1e4, full to 550 @1e6
    python scripts/moseq_kpms.py apply  --arm real --seed 0   # report syllables
    python scripts/moseq_kpms.py export --arm real --seed 0   # params + latent x

The protocol is `~/moseq/train_moseq.py`'s, unchanged (latent_dim 10, nlags 3,
kappa 1e4 then 1e6, x64). Each arm (real / white / phase) is its own kpms
project with its own PCA, fitted on the same 100 tune recordings.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

import keypoint_moseq as kpms  # noqa: E402
from jax_moseq.utils import unbatch  # noqa: E402
from jax_moseq.utils.debugging import convert_data_precision  # noqa: E402

REPO = "/home/tul26194/vieb-pipeline"
#: STEP 1 runs the same protocol on CalMS21 by pointing MOSEQ_WORK at
#: work/calms/moseq, whose sets.json and input/<arm>/ have the same layout.
WORK = os.environ.get("MOSEQ_WORK", os.path.join(REPO, "work", "moseq"))
LUNA_ARMS = ("real", "white", "phase")
TEMPLATE = "/home/tul26194/moseq/luna_demo/config.yml"
KEEP = ("left_ear", "right_ear", "nose", "center", "left_hip", "right_hip", "tail_base")
LATENT_DIM = 10
KAPPA_ARHMM = 1e4
#: STEP 1's kappa ablation sets MOSEQ_KAPPA_FULL=1e5; unset, the protocol's 1e6.
KAPPA_FULL = float(os.environ.get("MOSEQ_KAPPA_FULL", "1e6"))
ITERS_ARHMM, ITERS_FULL = 50, 500
APPLY_CHUNK, APPLY_ITERS = 200, 20


SMOKE = False    # set by --limit: a separate project, labels and export dir


def project(arm: str) -> str:
    return os.path.join(WORK, "kpms_smoke" if SMOKE else "kpms", arm)


def outdir(kind: str) -> str:
    return os.path.join(WORK, f"{kind}_smoke" if SMOKE else kind)


def sets() -> dict:
    with open(os.path.join(WORK, "sets.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load(arm: str, rids) -> tuple[dict, dict]:
    coords, confs = {}, {}
    for rid in rids:
        with np.load(os.path.join(WORK, "input", arm, f"{rid}.npz")) as z:
            coords[rid] = np.asarray(z["coords"], dtype=np.float64)
            confs[rid] = np.asarray(z["conf"], dtype=np.float64)
    return coords, confs


def config(arm: str) -> dict:
    cfg = kpms.load_config(project(arm))
    assert tuple(cfg["use_bodyparts"]) == KEEP, cfg["use_bodyparts"]
    return {**cfg, "bodyparts": list(KEEP)}


def formatted(arm: str, rids):
    coords, confs = load(arm, rids)
    data, metadata = kpms.format_data(coords, confs, **config(arm))
    return convert_data_precision(data), metadata


def setup(a) -> int:
    d = project(a.arm)
    os.makedirs(d, exist_ok=True)
    shutil.copy(TEMPLATE, os.path.join(d, "config.yml"))
    kpms.update_config(d, bodyparts=list(KEEP), use_bodyparts=list(KEEP),
                       latent_dim=LATENT_DIM, video_dir=d)
    fit = sets()["fit"][: a.limit or None]
    data, _ = formatted(a.arm, fit)
    pca = kpms.fit_pca(**data, **config(a.arm))
    kpms.save_pca(pca, d)
    ev = np.cumsum(pca.explained_variance_ratio_)[LATENT_DIM - 1]
    print(f"{a.arm}: PCA on {len(fit)} recordings; {LATENT_DIM} PCs explain {ev:.3f}")
    return 0


def name(a) -> str:
    return f"{a.arm}_s{a.seed}"


def fit(a) -> int:
    d = project(a.arm)
    fit_rids = sets()["fit"][: a.limit or None]
    data, metadata = formatted(a.arm, fit_rids)
    pca = kpms.load_pca(d)
    model = kpms.init_model(data, pca=pca, seed=jr.PRNGKey(a.seed), **config(a.arm))
    model = kpms.update_hypparams(model, kappa=KAPPA_ARHMM)
    n_ar, n_full = a.iters_arhmm, a.iters_full
    model, mname = kpms.fit_model(model, data, metadata, d, model_name=name(a),
                                  ar_only=True, num_iters=n_ar,
                                  generate_progress_plots=False)
    model = kpms.update_hypparams(model, kappa=KAPPA_FULL)
    kpms.fit_model(model, data, metadata, d, mname, ar_only=False,
                   start_iter=n_ar, num_iters=n_ar + n_full,
                   parallel_message_passing=False, generate_progress_plots=False)
    kpms.reindex_syllables_in_checkpoint(d, mname)
    with open(os.path.join(d, mname, "fit_manifest.json"), "w") as fh:
        json.dump({"arm": a.arm, "seed": a.seed, "fit": fit_rids,
                   "kappa": [KAPPA_ARHMM, KAPPA_FULL],
                   "iters": [n_ar, n_ar + n_full], "latent_dim": LATENT_DIM}, fh, indent=1)
    print(f"fit {mname}: {len(fit_rids)} recordings, iterations {n_ar}+{n_full}")
    return 0


def apply(a) -> int:
    d = project(a.arm)
    rep = sets()["report"][: a.limit or None]
    model = kpms.load_checkpoint(d, name(a))[0]
    labels, latent = {}, {}
    for i in range(0, len(rep), APPLY_CHUNK):
        batch = rep[i:i + APPLY_CHUNK]
        data, metadata = formatted(a.arm, batch)
        cfg = {k: v for k, v in config(a.arm).items() if k != "num_iters"}
        res = kpms.apply_model(model, data, metadata, d, name(a), save_results=False,
                               overwrite=True, parallel_message_passing=False,
                               num_iters=APPLY_ITERS, **cfg)
        for rid in batch:
            labels[rid] = np.asarray(res[rid]["syllable"], dtype=np.int16)
            latent[rid] = np.asarray(res[rid]["latent_state"], dtype=np.float32)
        print(f"  applied {i + len(batch)}/{len(rep)}", flush=True)
    b = np.cumsum([0] + [labels[r].size for r in rep])
    os.makedirs(outdir("labels"), exist_ok=True)
    np.savez_compressed(os.path.join(outdir("labels"), f"{name(a)}.npz"),
                        syllable=np.concatenate([labels[r] for r in rep]),
                        x=np.concatenate([latent[r] for r in rep]),
                        bounds=b, recording_ids=np.array(rep))
    return 0


def broken(arm: str, rids) -> tuple[np.ndarray, np.ndarray]:
    """Frames with any unobserved (conf 0) keypoint, and the recording bounds."""
    _, confs = load(arm, rids)
    br = [(confs[r] == 0).any(axis=1) for r in rids]
    return np.concatenate(br), np.cumsum([0] + [c.shape[0] for c in br])


def export(a) -> int:
    """Parameters, and MoSeq's OWN inferred latents for the fit recordings.

    Amendment 2: both models are scored on kpms's inferred x -- the final
    Gibbs sample on the fit set (here), `apply_model`'s latent_state on report
    (`apply`) -- because the fitted AR noise Q describes the denoised x, not
    the deterministic projection of the raw keypoints.
    """
    d = project(a.arm)
    model = kpms.load_checkpoint(d, name(a))[0]
    p = {k: np.asarray(v) for k, v in model["params"].items()}
    s = sets()
    fit_rids = s["fit"][: a.limit or None]
    _, metadata = formatted(a.arm, fit_rids)
    xs = unbatch(np.asarray(model["states"]["x"]), *metadata)
    out = {"Ab": p["Ab"], "Q": p["Q"], "pi": p["pi"], "betas": p["betas"], "Cd": p["Cd"]}
    out["x_fit"] = np.concatenate([np.asarray(xs[r], np.float32) for r in fit_rids])
    out["broken_fit"], out["bounds_fit"] = broken(a.arm, fit_rids)
    assert out["bounds_fit"][-1] == out["x_fit"].shape[0]
    rep = s["report"][: a.limit or None]
    out["broken_report"], out["bounds_report"] = broken(a.arm, rep)
    out["rids_fit"], out["rids_report"] = np.array(fit_rids), np.array(rep)
    os.makedirs(outdir("export"), exist_ok=True)
    np.savez_compressed(os.path.join(outdir("export"), f"{name(a)}.npz"), **out)
    print(f"export {name(a)}: Ab {p['Ab'].shape}, x_fit {out['x_fit'].shape}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("setup", "fit", "apply", "export"))
    ap.add_argument("--arm", required=True,
                    help="real/white/phase on Luna; any input/<arm> under MOSEQ_WORK")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    ap.add_argument("--iters-arhmm", type=int, default=ITERS_ARHMM)
    ap.add_argument("--iters-full", type=int, default=ITERS_FULL)
    a = ap.parse_args(argv)
    if "MOSEQ_WORK" not in os.environ and a.arm not in LUNA_ARMS:
        ap.error(f"--arm must be one of {LUNA_ARMS} unless MOSEQ_WORK is set")
    global SMOKE
    SMOKE = bool(a.limit)
    return {"setup": setup, "fit": fit, "apply": apply, "export": export}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
