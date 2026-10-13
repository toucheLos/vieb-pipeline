"""STEP 1 §1-2: partitions, Luna's dropout and skull geometry, the jitter gain,
CalMS21's native level, then the degraded inputs of every arm. No labels.

    python3 scripts/calms_prep.py prep            # -> results/step1/noise.json, partitions.json
    python3 scripts/calms_prep.py build --arm m1_s0
    python3 scripts/calms_prep.py arms            # the registered arm list

Arms: ``m<level>_s<seed>`` for the primary sweep, ``white_s<seed>`` and
``phase_s<seed>`` for the negative controls (level 0, no dropout).

Outputs per arm and video (sha1):
  work/calms/moseq/input/<arm>/<sha>.npz   coords, conf     (MoSeq's Luna input)
  work/calms/raw/<arm>/<sha>.npz           pose, abstain    (the detector's raw arm)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, labels as lab                            # noqa: E402
from recur.util import log, write_json                              # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.calms import arms as A, degrade as dg, io as cio          # noqa: E402
from vieb.io import spine                                           # noqa: E402
from vieb.moseq import inputs                                       # noqa: E402
from vieb.qc import bones as bn                                     # noqa: E402
from vieb.seeds import stable_seed                                  # noqa: E402
from vieb.tok import config, ego                                    # noqa: E402

WORK = os.path.join(config.REPO, "work", "calms")
OUT = os.path.join(config.REPO, "results", "step1")
PARTS = os.path.join(OUT, "partitions.json")
NOISE = os.path.join(OUT, "noise.json")
DONORS = os.path.join(WORK, "luna_masks.npz")
LUNA_SETS = os.path.join(config.REPO, "work", "moseq", "sets.json")


def arms() -> list[str]:
    """The registered primary arms (REGISTRATION §2.5, §6), in their original order."""
    out = [f"m{m:g}_s{s}" for m in dg.LEVELS for s in dg.SEEDS]
    out += [f"{k}_s{s}" for k in ("white", "phase") for s in dg.SEEDS]
    return out


LISTS = {"primary": arms, "build": A.build_list, "seg": A.seg_list, "kpms": A.kpms_list}


# --------------------------------------------------------------------------
# prep
# --------------------------------------------------------------------------

def luna_masks() -> tuple[dict, dict, list]:
    """The 100 Luna fit recordings' (missing, abstain), and skull geometry in BL.

    Built as `scripts/moseq_input.py build` builds the MOSEQ input:
    missing = DLC missing | vetoed, abstain = disposition abstain.
    """
    with open(LUNA_SETS, encoding="utf-8") as fh:
        fit = json.load(fh)["fit"]
    by_tag: dict = {}
    for rid in fit:
        by_tag.setdefault(lab.animal_tag(rid), []).append(rid)
    miss, abst, bounds, skull = [], [], [0], []
    for tag in sorted(by_tag):
        disp = np.load(os.path.join(config.REPO, "work", "disposition", f"{tag}.npz"))
        rep = np.load(os.path.join(config.REPO, "work", "repaired", f"{tag}.npz"))
        rids = [str(r) for r in disp["recording_ids"]]
        assert rids == [str(r) for r in rep["recording_ids"]], tag
        b = disp["bounds"]
        for rid in sorted(by_tag[tag]):
            i = rids.index(rid)
            lo, hi = int(b[i]), int(b[i + 1])
            c = spine.clean(rid)
            m = np.asarray(c["missing"], bool) | np.asarray(rep["vetoed"][lo:hi], bool)
            a = np.asarray(disp["abstain"][lo:hi], bool)
            assert m.shape[0] == a.shape[0] == hi - lo
            miss.append(m)
            abst.append(a)
            bounds.append(bounds[-1] + m.shape[0])
            p = np.asarray(c["pose_unfiltered"], dtype=np.float64)
            ok = ~np.asarray(c["missing"], bool).any(axis=1)
            ell = float(np.median(ego.body_length(p[ok])))
            skull.append(np.median(bn.bone_lengths(p[ok], bn.SKULL), axis=0) / ell)
    donors = {"missing": np.concatenate(miss), "abstain": np.concatenate(abst),
              "bounds": np.asarray(bounds, dtype=np.int64)}
    return donors, {"n_recordings": len(miss)}, np.median(np.asarray(skull), axis=0).tolist()


def dropout_stats(d: dict) -> dict:
    """Per-keypoint rate and the run-length distribution (normalised, asserted)."""
    out: dict = {"rate_per_keypoint": dict(zip(dg.LUNA_ORDER,
                                               d["missing"].mean(axis=0).tolist())),
                 "abstain_rate": float(d["abstain"].mean())}
    lens: list[int] = []
    b = d["bounds"]
    for r in range(b.size - 1):
        blk = d["missing"][b[r]:b[r + 1]]
        for k in range(blk.shape[1]):
            lens += [hi - lo for lo, hi in dg.runs(blk[:, k])]
    lens_a = np.asarray(lens, dtype=np.int64)
    edges = np.array([1, 2, 3, 4, 6, 11, 31, 91, 301, 10 ** 9])
    h = np.histogram(lens_a, bins=edges)[0].astype(np.float64)
    pmf = h / h.sum()
    assert abs(pmf.sum() - 1.0) < 1e-12
    out["run_length"] = {"n_runs": int(lens_a.size), "median": float(np.median(lens_a)),
                         "p90": float(np.quantile(lens_a, 0.9)), "max": int(lens_a.max()),
                         "bins_lo": edges[:-1].tolist(), "pmf": pmf.tolist(),
                         "share_longer_than_hold_3": float((lens_a > 3).mean())}
    return out


def prep(_a) -> int:
    os.makedirs(WORK, exist_ok=True)
    parts = cio.partitions()
    cio.save(parts, PARTS)
    log(f"partitions: " + ", ".join(f"{k} {len(parts[k])}" for k in cio.PARTITIONS)
        + f"; fit frames {parts['fit_frames']:,}; dropped {parts['dropped_for_collision']}")

    donors, dinfo, skull = luna_masks()
    np.savez_compressed(DONORS, **donors)
    dstats = dropout_stats(donors)
    log(f"luna masks: {dinfo['n_recordings']} recordings, skull BL {np.round(skull, 4)}")

    sb = dg.sigma_bar_luna()
    prof = dg.luna_profile()
    s1 = dg.s1()
    g = dg.gain(skull, prof, s1, sb)
    cal_pose = dg.calibration_pose(skull)
    reading = {f"{m:g}": dg.estimator_reading(cal_pose, m, prof, s1, g)
               for m in dg.LEVELS if m > 0}
    log(f"sigma_bar_L {sb:.6f}, s1 {s1:.4f} px, g {g:.4f}; readings {reading}")

    vids = parts["fit"] + cio.scored(parts)
    poses, scores = [], []
    for v in vids:
        p, s, _ell = cio.resident(parts["where"][v])
        poses.append(p)
        scores.append(s)
    sn, table = dg.native_sigma_bar(poses, scores)
    r = sn / sb
    log(f"CalMS21 native sigma_bar {sn:.6f} BL; r = {r:.4f}")
    write_json({**provenance.header(anchors.LUNA, stage="calms_noise",
                                    unverified="CalMS21 is external data; Luna inputs cited"),
                "registration": "results/step1/REGISTRATION.md (+ Amendment 1)",
                "sigma_bar_luna_bl": sb, "profile_luna_bl": dict(zip(dg.LUNA_ORDER, prof.tolist())),
                "s1_px": s1, "gain": g, "luna_skull_bl": dict(zip(
                    ["left_ear-right_ear", "left_ear-nose", "right_ear-nose"], skull)),
                "estimator_reading_by_level": reading,
                "native_sigma_bar_bl": sn, "r": r, "r_ge_1": bool(r >= 1.0),
                "native_table": table, "n_native_sequences": len(vids),
                "reachable": {f"{m:g}": dg.reachable(m, r) for m in dg.LEVELS},
                "luna_dropout": {**dinfo, **dstats}},
               NOISE)
    return 0


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(a) -> int:
    """One arm's inputs, from its registered spec (`vieb.calms.arms`).

    Variants re-use the primary arm's draws: the jitter and the dropout come
    from the same per-sequence generator as `m<level>_s<seed>`, and are then
    dropped (j, d) or post-processed (keypoint substitution, decimation,
    smoothing). So every Delta in the ablation is paired draw for draw.
    """
    sp = A.parse(a.arm)
    if sp.input_from is not None:
        raise SystemExit(f"{a.arm} reuses {sp.input_arm}'s inputs; nothing to build")
    parts = cio.load(PARTS)
    with open(NOISE, encoding="utf-8") as fh:
        nz = json.load(fh)
    prof = np.asarray([nz["profile_luna_bl"][k] for k in dg.LUNA_ORDER])
    donors = dict(np.load(DONORS))
    dm = os.path.join(WORK, "moseq", "input", a.arm)
    dr = os.path.join(WORK, "raw", a.arm)
    os.makedirs(dm, exist_ok=True)
    os.makedirs(dr, exist_ok=True)
    vids = parts["fit"] + cio.scored(parts)
    sets = os.path.join(WORK, "moseq", "sets.json")
    if not os.path.exists(sets):
        with open(sets, "w", encoding="utf-8") as fh:
            json.dump({"fit": parts["fit"], "report": cio.scored(parts)}, fh, indent=1)
    step = int(round(dg.FPS / sp.fps))
    assert step * sp.fps == dg.FPS, sp.fps
    for v in vids:
        pose, scores, _ell = cio.resident(parts["where"][v])
        if sp.kind == "real":
            noisy, miss, ab = dg.degraded(pose, sp.level, r=nz["r"], profile=prof,
                                          s1_px=nz["s1_px"], donors=donors,
                                          rng=dg.rng_for(sp.seed, sp.level, v), gain=nz["gain"])
            if not sp.jitter:
                noisy = pose
            if not sp.dropout:
                miss, ab = np.zeros_like(miss), np.zeros_like(ab)
        else:
            rng = np.random.default_rng(stable_seed("calms", sp.kind, sp.seed, v))
            noisy, _cf = inputs.surrogate(pose, scores, sp.kind, rng)
            miss = np.zeros(noisy.shape[:2], dtype=bool)
            ab = np.zeros(noisy.shape[0], dtype=bool)
        noisy, miss, sc_ = dg.substitute(noisy, miss, scores, sp.keypoints)
        noisy, miss, ab, sc_ = dg.decimate(step, noisy, miss, ab, sc_)
        if sp.smooth == "median":
            coords, conf = dg.moseq_input_smoothed(noisy, sc_, miss, ab, sp.fps)
            held, abst = dg.raw_input_smoothed(noisy, miss, ab, sp.fps)
        elif sp.kind == "real":
            coords, conf = dg.moseq_input(noisy, sc_, miss, ab)
            held, abst = dg.raw_input(noisy, miss, ab)
        else:
            # The primary controls' construction, kept exactly (REGISTRATION §6).
            coords, conf = inputs.surrogate(pose, scores, sp.kind,
                                            np.random.default_rng(stable_seed("calms", sp.kind,
                                                                              sp.seed, v)))
            held, abst = coords, np.zeros(coords.shape[0], dtype=bool)
        np.savez_compressed(os.path.join(dm, f"{v}.npz"), coords=coords.astype(np.float32),
                            conf=conf.astype(np.float32))
        np.savez_compressed(os.path.join(dr, f"{v}.npz"), pose=held.astype(np.float32),
                            abstain=abst)
    log(f"{a.arm}: {len(vids)} videos")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prep", "build", "arms"))
    p.add_argument("--arm", default=None)
    p.add_argument("--list", choices=tuple(LISTS), default="primary",
                   help="which arm list `arms` prints and `--index` indexes")
    p.add_argument("--new", action="store_true", help="only arms not in the primary list")
    p.add_argument("--index", type=int, default=None, help="array index into the list")
    a = p.parse_args(argv)
    names = LISTS[a.list]()
    if a.new:
        names = [n for n in names if n not in set(arms())]
    if a.phase == "arms":
        print("\n".join(names))
        return 0
    if a.phase == "build" and a.arm is None:
        a.arm = names[a.index]
    return {"prep": prep, "build": build}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
