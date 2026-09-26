"""REPAIR: build the repaired pose for every animal, then read its gates.

    sbatch jobs/repair.slurm                     # CPU; 12 shards, after masks
    python3 scripts/repair.py --phase combine

READ results/REPAIR_PREREGISTRATION.md FIRST.

Per animal: disposition's pose (`work/disposition/<animal>.npz`), DLC's missing
flags (`spine.clean`), SAM's masks (`work/masks/<rid>.npz`, required -- a
recording without its mask file stops the shard rather than silently skipping
the veto). Writes `work/repaired/<animal>.npz` and the registered fallback
`work/repaired_noveto/<animal>.npz`, plus per-recording metrics for four arms:
raw (held), disposition, repaired, repaired without the veto.

Metrics are the bakeoff's (`scripts/cleaning.py`): skull violation rate with the
reference refitted per arm, mean displacement from the raw held input (over
measured keypoint-frames only), high-frequency retention; plus the D21 skull
jitter. Violation and spectrum read each arm's contiguous held series.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot, labels as lab, splits               # noqa: E402
from recur.util import log, write_json                               # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.clean import arms as clean_arms, repair as rp, score       # noqa: E402
from vieb.io import spine                                            # noqa: E402
from vieb.qc import bones                                             # noqa: E402
from vieb.tok import config                             # noqa: E402

EPS = 0.10                        # the bakeoff's (scripts/cleaning.py) epsilon
WIN_S = 2.0
ARMS = ("raw", "disposition", "repaired", "noveto")
MASKS = os.path.join(config.REPO, "work", "masks")
OUT = {"repaired": os.path.join(config.REPO, "work", "repaired"),
       "noveto": os.path.join(config.REPO, "work", "repaired_noveto")}
METRICS = os.path.join(config.REPO, "work", "repaired_metrics")
SEED = 0
N_BOOT = 2000
#: REPAIR §2's bars.
JITTER_BAR = 0.0
VIOL_TOL = 0.001
HF_FLOOR = 0.220
DIST_CEIL = 1.555


def skull_jitter(pose: np.ndarray, bl: float, fps: float) -> float:
    """D21: median over 2 s windows of the within-window SD of skull bones / bl."""
    L = bones.bone_lengths(pose, bones.SKULL) / bl
    w = int(round(WIN_S * fps))
    k = L.shape[0] // w
    if k < 1:
        return float("nan")
    sd = np.nanstd(L[:k * w].reshape(k, w, -1), axis=1)
    return float(np.nanmedian(np.nanmean(sd, axis=1)))


def shard(a) -> int:
    fps = spine.fps()
    f_c = float(spine.json_("sf_calibration").get("f_c_hz", score.DEFAULT_F_C_HZ))
    pairs = bones.pair_indices(anchors.LUNA.n_kept_keypoints)
    by_animal: dict = {}
    for rid in spine.recording_ids():
        by_animal.setdefault(lab.animal_tag(rid), []).append(rid)
    tags = sorted(by_animal)[a.shard::a.of]
    if a.limit:
        tags = tags[:a.limit]
    for d in (*OUT.values(), METRICS):
        os.makedirs(d, exist_ok=True)
    for tag in tags:
        if os.path.exists(os.path.join(METRICS, f"{tag}.json")):
            continue
        disp = dict(np.load(os.path.join(config.REPO, "work", "disposition",
                                         f"{tag}.npz")))
        b = disp["bounds"]
        rids = [str(r) for r in disp["recording_ids"]]
        arrays: dict[str, list] = {n: [] for n in ARMS}
        held_for_metric: dict[str, list] = {n: [] for n in ARMS}
        stored: dict[str, dict[str, list]] = {n: {"pose": [], "missing_kp": [],
                                                  "vetoed": [], "has_mask": []}
                                              for n in OUT}
        measured: list = []
        bl_rows: list = []
        for i, rid in enumerate(rids):
            c = spine.clean(rid)
            miss = c["missing"].astype(bool)
            raw_held = clean_arms.held_array(c["pose_unfiltered"].astype(np.float64), miss)
            dp = disp["pose"][b[i]:b[i + 1]].astype(np.float64)
            mpath = os.path.join(MASKS, f"{rid}.npz")
            if not os.path.exists(mpath):
                raise SystemExit(f"no mask file for {rid}; run masks_corpus first")
            mz = dict(np.load(mpath))
            bl = float(mz["body_length"])
            dist = rp.MaskDistance(mz, pad_px=rp.VETO_BL * bl + 4)
            rep = rp.repair(dp, miss, dist, bl, fps, use_veto=True)
            nov = rp.repair(dp, miss, None, bl, fps, use_veto=False)
            arrays["raw"].append(raw_held)
            arrays["disposition"].append(dp)
            arrays["repaired"].append(rep["pose"])
            arrays["noveto"].append(nov["pose"])
            held_for_metric["raw"].append(raw_held)
            held_for_metric["disposition"].append(dp)
            held_for_metric["repaired"].append(rep["pose_held"])
            held_for_metric["noveto"].append(nov["pose_held"])
            for name, r in (("repaired", rep), ("noveto", nov)):
                for key in stored[name]:
                    stored[name][key].append(r[key])
            measured.append(~(miss | rep["vetoed"]))
            bl_rows.append(bl)
        for name in OUT:
            np.savez_compressed(
                os.path.join(OUT[name], f"{tag}.npz"),
                pose=np.concatenate(stored[name]["pose"]).astype(np.float32),
                missing_kp=np.concatenate(stored[name]["missing_kp"]),
                vetoed=np.concatenate(stored[name]["vetoed"]),
                has_mask=np.concatenate(stored[name]["has_mask"]),
                abstain=disp["abstain"], bounds=b, recording_ids=disp["recording_ids"])
        rows = []
        for name in ARMS:
            stacked = np.concatenate(held_for_metric[name])
            keep = bones.rigid_pairs(bones.log_lengths(stacked, pairs), pairs)
            pooled = bones.metric_lengths(stacked, bones.SKULL, "raw",
                                          pairs=pairs, keep=keep)
            l_hat = np.array([bones.reference_length(pooled[:, m], EPS)["l_hat"]
                              for m in range(len(bones.SKULL))])
            for i, rid in enumerate(rids):
                h = held_for_metric[name][i]
                lengths = bones.metric_lengths(h, bones.SKULL, "raw", pairs=pairs,
                                               keep=keep)
                viol = float(bones.frame_mask(bones.violations(lengths, l_hat, EPS)).mean())
                d = np.linalg.norm(arrays[name][i] - arrays["raw"][i], axis=-1)
                dmean = float(np.nanmean(d[measured[i]]))
                hf = score.high_frequency_retention(arrays["raw"][i], h, fps,
                                                    f_c=f_c)["retained"]
                rows.append({"animal": tag, "recording_id": rid, "arm": name,
                             "n_frames": int(h.shape[0]), "violation_rate": viol,
                             "distortion_mean_px": dmean, "hf_retained": float(hf),
                             "skull_jitter": skull_jitter(arrays[name][i],
                                                          bl_rows[i], fps)})
        vet = np.concatenate(stored["repaired"]["vetoed"])
        has = np.concatenate(stored["repaired"]["has_mask"])
        summary = {"animal": tag, "rows": rows,
                   "veto": {"vetoed_kp_frames": int(vet.sum()),
                            "masked_kp_frames": int(has.sum()) * vet.shape[1],
                            "per_keypoint": vet.sum(axis=0).tolist(),
                            "recordings": [{"recording_id": r,
                                            "vetoed": int(v.sum()),
                                            "masked_frames": int(hm.sum())}
                                           for r, v, hm in zip(
                                               rids, stored["repaired"]["vetoed"],
                                               stored["repaired"]["has_mask"])]}}
        with open(os.path.join(METRICS, f"{tag}.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh)
        log(f"  {tag}: {len(rids)} recordings, vetoed {vet.sum()} kp-frames of "
            f"{int(has.sum()) * vet.shape[1]} masked")
    return 0


def combine(a) -> int:
    split_of = splits.split_of_animal(splits.load())
    rows, veto_ctx = [], {"A": [0, 0], "B": [0, 0]}
    per_kp = np.zeros(anchors.LUNA.n_kept_keypoints)
    for f in sorted(os.listdir(METRICS)):
        d = json.load(open(os.path.join(METRICS, f), encoding="utf-8"))
        if split_of.get(d["animal"]) != "report":
            continue
        rows += d["rows"]
        per_kp += np.asarray(d["veto"]["per_keypoint"])
        for r in d["veto"]["recordings"]:
            ctx = "A" if "Context_A" in r["recording_id"] else "B"
            veto_ctx[ctx][0] += r["vetoed"]
            veto_ctx[ctx][1] += r["masked_frames"] * anchors.LUNA.n_kept_keypoints

    def per_animal(arm: str, key: str) -> dict[str, float]:
        acc: dict[str, list] = {}
        for r in rows:
            if r["arm"] == arm and np.isfinite(r[key]):
                acc.setdefault(r["animal"], []).append((r[key], r["n_frames"]))
        return {an: float(np.average([v for v, _ in xs], weights=[w for _, w in xs]))
                for an, xs in acc.items()}

    def interval(vals: dict[str, float]) -> dict:
        return boot.animal_interval(list(vals.values()), list(vals), how="mean",
                                    n_boot=N_BOOT, seed=SEED)

    def paired(key: str, a_: str, b_: str) -> dict:
        x, y = per_animal(a_, key), per_animal(b_, key)
        return interval({an: x[an] - y[an] for an in x if an in y})

    res: dict = {"split": "report"}
    for arm in ARMS:
        res[arm] = {k: interval(per_animal(arm, k))
                    for k in ("skull_jitter", "violation_rate",
                              "distortion_mean_px", "hf_retained")}
    g1a = paired("skull_jitter", "repaired", "raw")
    g1b = paired("skull_jitter", "repaired", "disposition")
    g2 = paired("violation_rate", "repaired", "disposition")
    g3 = res["repaired"]["hf_retained"]
    g4 = res["repaired"]["distortion_mean_px"]
    gates = {
        "1_skull_jitter": {"vs_raw": g1a, "vs_disposition": g1b,
                           "pass": g1a["hi"] < JITTER_BAR and g1b["hi"] < JITTER_BAR},
        "2_violations": {"vs_disposition": g2, "pass": g2["hi"] <= VIOL_TOL},
        "3_hf_retention": {"interval": g3, "pass": g3["lo"] >= HF_FLOOR},
        "4_distortion": {"interval": g4, "pass": g4["hi"] <= DIST_CEIL},
    }
    verdict = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"
    veto = {c: {"vetoed": v[0], "masked_kp_frames": v[1],
                "rate": v[0] / v[1] if v[1] else float("nan")}
            for c, v in veto_ctx.items()}
    ratio = (veto["B"]["rate"] / veto["A"]["rate"]
             if veto["A"]["rate"] and np.isfinite(veto["A"]["rate"]) else float("nan"))
    res.update({"gates": gates, "verdict": verdict, "veto_by_context": veto,
                "veto_ratio_B_over_A": ratio,
                "veto_per_keypoint": per_kp.tolist(),
                "adopted_arm": "repaired" if verdict == "PASS" else "repaired_noveto"})
    for name, g in gates.items():
        log(f"  gate {name}: {'PASS' if g['pass'] else 'FAIL'}")
    log(f"  veto by context: {veto}  B/A {ratio:.2f}")
    log(f"  verdict {verdict}; adopted arm {res['adopted_arm']}")
    out = a.out or config.PATHS.result("repair.json")
    write_json({**provenance.header(anchors.LUNA, stage="repair",
                                    unverified="a pose-cleaning stage"),
                "inherited_digest": spine.digest(),
                "registration": "results/REPAIR_PREREGISTRATION.md",
                "results": res}, out)
    log(f"  wrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("combine",), default=None)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--of", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.phase == "combine":
        return combine(a)
    if a.shard is None:
        raise SystemExit("pass --shard I --of N, or --phase combine")
    return shard(a)


if __name__ == "__main__":
    raise SystemExit(main())
