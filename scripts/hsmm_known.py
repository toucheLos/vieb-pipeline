"""STEP 4: Amendment 4's planted bout recall, and the known-answer check.

Registered in `results/step4/REGISTRATION.md` Amendment 4 (fe42d1a, D36) and
`results/step4/KNOWN_ANSWER.md` (29e84cc). Phases:

    python3 scripts/hsmm_known.py planted_bouts   # -> results/step4/hsmm_planted_bouts.json
    python3 scripts/hsmm_known.py known           # -> results/step4/hsmm_known_answer.json

Both use the run-2 final model (`work/hsmm/final_fit.json`). `known` needs the
tune pixel scan (`scripts/moseq_freeze.py --split tune`) and the saved smoothed
posteriors (`work/hsmm/tune/<rid>_post.npz`). No shock mapping is used or stated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.environ.get("VIEB_RECUR", "/home/tul26194/recur"))

from recur import anchors, boot                                     # noqa: E402
from recur.read import Read                                         # noqa: E402
from recur.util import log, write_json                              # noqa: E402
from vieb import provenance                                         # noqa: E402
from vieb.hsmm import core, mouse as M                              # noqa: E402
from vieb.tok import config                                         # noqa: E402

WORK = os.path.join(config.REPO, "work", "hsmm")
OUT = os.path.join(config.REPO, "results", "step4")
PIXEL = os.path.join(config.REPO, "work", "moseq", "freeze")
MANIFEST = os.path.join(config.REPO, "work", "moseq", "freeze_manifest_tune.json")
FPS = 30.0
SEED = 0
N_BOOT = 2000
IOU = 0.5
RECALL_BAR = 0.80
PLANT_BINS_S = (1.0, 2.0)                                # planted bouts >= 1 s, >= 2 s
BINS_S = ((0.0, 1.0), (1.0, 2.0), (2.0, np.inf))         # [0,1), [1,2), [2,inf)
BIN_NAMES = ("lt1s", "1to2s", "ge2s")
KAPPA_MIN_S = (0.0, 1.0, 2.0)                            # all, bouts >= 1 s, >= 2 s
KAPPA_TIE = 0.4


def model() -> core.HSMM:
    with open(os.path.join(WORK, "final_fit.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    return M.build(np.asarray(d["theta"]), d["family"], FPS)


def bouts(mask: np.ndarray) -> np.ndarray:
    """``(n, 2)`` [start, stop) of the True runs of a boolean mask."""
    return np.asarray([(b, e) for k, b, e in core.runs(mask.astype(np.int64)) if k == 1],
                      dtype=np.int64).reshape(-1, 2)


def match(t: np.ndarray, p: np.ndarray, iou: float = IOU) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching at IoU >= `iou` (Step 1's rule, `vieb/calms/score.py`).

    Returns, per true bout, the index of its matched predicted bout (-1 if none), and
    per predicted bout whether it is matched.
    """
    pairs: list[tuple[float, int, int]] = []
    j0 = 0
    for i, (a, b) in enumerate(t.tolist()):
        while j0 < p.shape[0] and p[j0, 1] <= a:
            j0 += 1
        j = j0
        while j < p.shape[0] and p[j, 0] < b:
            c, d = int(p[j, 0]), int(p[j, 1])
            v = (min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
            if v >= iou:
                pairs.append((v, i, j))
            j += 1
    pairs.sort(reverse=True)
    mt = np.full(t.shape[0], -1, np.int64)
    mp = np.zeros(p.shape[0], bool)
    for _v, i, j in pairs:
        if mt[i] >= 0 or mp[j]:
            continue
        mt[i], mp[j] = j, True
    return mt, mp


# --------------------------------------------------------------------------
# Amendment 4: planted bout recall
# --------------------------------------------------------------------------

def planted_bouts(_a) -> int:
    m = model()
    z = np.load(os.path.join(WORK, "planted.npz"))
    n = len([k for k in z.files if k.startswith("le")])
    per: list[dict] = []
    acc = []
    for i in range(n):
        tr = np.asarray(z[f"tr{i}"])
        _l, post = core.smooth(m, [np.asarray(z[f"le{i}"])])
        pf = post[0][:, M.FREEZE]
        acc.append(float(((pf > 0.5) == (tr == 0)).mean()))
        t = bouts(tr == 0)
        p = bouts(post[0].argmax(1) == M.FREEZE)
        mt, mp = match(t, p)
        td = (t[:, 1] - t[:, 0]) / FPS
        pd = (p[:, 1] - p[:, 0]) / FPS
        frag = np.asarray([int(((p[:, 0] < b) & (p[:, 1] > a)).sum()) for a, b in t.tolist()])
        row: dict = {}
        for thr in PLANT_BINS_S:
            ti, pi = td >= thr, pd >= thr
            row[f"ge{thr:g}s"] = {
                "n_true": int(ti.sum()), "tp": int((mt[ti] >= 0).sum()),
                "n_pred": int(pi.sum()), "pred_matched": int(mp[pi].sum()),
                "dur_ratio": (pd[mt[ti & (mt >= 0)]] / td[ti & (mt >= 0)]).tolist(),
                "fragments": frag[ti].tolist()}
        per.append(row)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (N_BOOT, n))
    res: dict = {}
    for thr in PLANT_BINS_S:
        k = f"ge{thr:g}s"
        tp = np.asarray([r[k]["tp"] for r in per], float)
        nt = np.asarray([r[k]["n_true"] for r in per], float)
        pm = np.asarray([r[k]["pred_matched"] for r in per], float)
        npd = np.asarray([r[k]["n_pred"] for r in per], float)
        reps = tp[idx].sum(1) / nt[idx].sum(1)
        recall = float(tp.sum() / nt.sum())
        ratio = np.concatenate([r[k]["dur_ratio"] for r in per]) if per else np.zeros(0)
        frags = np.concatenate([r[k]["fragments"] for r in per])
        res[k] = {"recall": recall, "lo": float(np.quantile(reps, 0.025)),
                  "hi": float(np.quantile(reps, 0.975)), "n_true": int(nt.sum()),
                  "pass": bool(recall >= RECALL_BAR),
                  "precision": float(pm.sum() / npd.sum()) if npd.sum() else None,
                  "n_pred": int(npd.sum()),
                  "median_duration_ratio_matched": float(np.median(ratio)) if ratio.size else None,
                  "mean_fragments_per_true_bout": float(frags.mean()) if frags.size else None,
                  "frac_true_bouts_fragmented": float((frags > 1).mean()) if frags.size else None}
    acc_pass = bool(np.mean(acc) >= 0.90)
    ok = acc_pass and all(v["pass"] for v in res.values())
    read = Read("PASS" if ok else "FAIL", {"arm": "planted_bouts_amended", "split": "tune"},
                f"frame accuracy {np.mean(acc):.4f} (bar 0.90); freeze-bout recall at IoU >= 0.5: "
                + "; ".join(f"{k} {v['recall']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] (bar {RECALL_BAR})"
                            for k, v in res.items()), n_effective=n).to_dict()
    write_json({**provenance.header(anchors.LUNA, stage="hsmm_planted_bouts",
                                    unverified="tune split; post-result amendment (D36)"),
                "registration": "results/step4/REGISTRATION.md Amendment 4",
                "frame_accuracy": {"mean": float(np.mean(acc)), "min": float(np.min(acc)),
                                   "pass": acc_pass},
                "bout_recall": res, "n_sessions": n, "pass": ok, "reads": [read]},
               os.path.join(OUT, "hsmm_planted_bouts.json"))
    log(f"planted bouts: acc {np.mean(acc):.4f}; {json.dumps(res)}")
    return 0


# --------------------------------------------------------------------------
# Known-answer check
# --------------------------------------------------------------------------

def bin_fractions(mask: np.ndarray) -> np.ndarray:
    """Fraction of frames in True runs of each `BINS_S` duration bin."""
    b = bouts(mask)
    d = (b[:, 1] - b[:, 0]) / FPS
    n = (b[:, 1] - b[:, 0]).astype(float)
    return np.asarray([n[(d >= lo) & (d < hi)].sum() for lo, hi in BINS_S]) / mask.size


def long_only(mask: np.ndarray, min_s: float) -> np.ndarray:
    out = np.zeros_like(mask)
    for a, b in bouts(mask).tolist():
        if (b - a) / FPS >= min_s:
            out[a:b] = True
    return out


def kappa(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    po = float((x == y).mean())
    px, py = float(x.mean()), float(y.mean())
    pe = px * py + (1 - px) * (1 - py)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def known(_a) -> int:
    with open(MANIFEST, encoding="utf-8") as fh:
        rows = json.load(fh)["recordings"]
    per: dict = {}
    mismatched, missing = [], []
    for r in rows:
        rid = r["recording_id"]
        pz = os.path.join(PIXEL, f"{rid}.npz")
        if not os.path.exists(pz):
            missing.append(rid)
            continue
        post = np.load(os.path.join(WORK, "tune", f"{rid}_post.npz"))["post"]
        z = np.load(os.path.join(WORK, "tune", f"{rid}.npz"))
        pix = np.asarray(np.load(pz)["freeze"], bool)
        T = min(post.shape[0], pix.size)
        if post.shape[0] != pix.size:
            mismatched.append({"recording_id": rid, "pose": int(post.shape[0]), "video": int(pix.size)})
        hs = post[:T].argmax(1) == M.FREEZE
        pix = pix[:T]
        fl = np.asarray(z["flag"], bool)[:T]
        v = np.asarray(z["v"], np.float64)[:T]
        rec = {"hsmm": bin_fractions(hs), "pixel": bin_fractions(pix),
               "speed": float(np.nanmedian(v[~fl])) if (~fl).any() else np.nan}
        for ms in KAPPA_MIN_S:
            h = long_only(hs, ms) if ms > 0 else hs
            rec[f"kappa_ge{ms:g}s"] = kappa(h, pix)
            rec[f"kappa_ge{ms:g}s_unflagged"] = kappa(h[~fl], pix[~fl])
        per.setdefault(r["animal"], {}).setdefault(r["context"], []).append(rec)
    if missing:
        raise SystemExit(f"{len(missing)} recordings have no pixel scan yet, e.g. {missing[:3]}")
    animals = sorted(a for a, d in per.items() if {"A", "B"} <= set(d))

    def ctx_mean(a: str, c: str, key: str) -> np.ndarray:
        return np.nanmean(np.asarray([x[key] for x in per[a][c]], dtype=np.float64), axis=0)

    def ci(vals: list[float], how: str = "mean") -> dict:
        return boot.animal_interval(vals, animals, how=how, n_boot=N_BOOT, seed=SEED)

    def effect(c: dict) -> str:
        return "effect" if c["lo"] > 0 else ("reverse" if c["hi"] < 0 else "no_effect")

    out: dict = {}
    for sig in ("hsmm", "pixel"):
        A = np.asarray([ctx_mean(a, "A", sig) for a in animals])
        B = np.asarray([ctx_mean(a, "B", sig) for a in animals])
        s: dict = {}
        for j, nm in enumerate(BIN_NAMES):
            d = ci((A[:, j] - B[:, j]).tolist())
            s[nm] = {"A": ci(A[:, j].tolist()), "B": ci(B[:, j].tolist()), "A_minus_B": d,
                     "reading": effect(d)}
        d = ci((A.sum(1) - B.sum(1)).tolist())
        s["total"] = {"A": ci(A.sum(1).tolist()), "B": ci(B.sum(1).tolist()), "A_minus_B": d,
                      "reading": effect(d)}
        out[sig] = s
    kap: dict = {}
    for ms in KAPPA_MIN_S:
        for suf in ("", "_unflagged"):
            k = f"kappa_ge{ms:g}s{suf}"
            kA = [float(ctx_mean(a, "A", k)) for a in animals]
            kB = [float(ctx_mean(a, "B", k)) for a in animals]
            kap[k] = {"A": ci(kA), "B": ci(kB),
                      "A_minus_B": ci((np.asarray(kA) - np.asarray(kB)).tolist())}
    sA = np.asarray([float(ctx_mean(a, "A", "speed")) for a in animals])
    sB = np.asarray([float(ctx_mean(a, "B", "speed")) for a in animals])
    speed = {"ratio_A_over_B_per_animal": ci((sA / sB).tolist(), how="median"),
             "A_median_bl_s": float(np.median(sA)), "B_median_bl_s": float(np.median(sB))}

    # Registered readings (KNOWN_ANSWER.md)
    anchor = out["pixel"]["total"]["reading"] == "effect"
    lo_, mid, hi_ = (out["hsmm"][b]["reading"] for b in BIN_NAMES)
    if not anchor:
        row, verdict = "pixel anchor absent on tune", "NOT_A_RESULT"
    elif hi_ == "effect" and lo_ in ("no_effect", "reverse"):
        row, verdict = "1: the freeze state mixes micro-pauses with real freezing", "FAIL"
    elif hi_ == "effect" and lo_ == "effect":
        row, verdict = "2: the context difference shows at every bout length", "PASS"
    elif hi_ == "reverse":
        row, verdict = "4: decoded long freeze runs against both signals", "FAIL"
    elif hi_ == "no_effect" and mid == "no_effect":
        row, verdict = ("3: the keypoint signal misses freezing, or the pixel effect was "
                        "partly the visual confound"), "FAIL"
    else:
        row, verdict = (f"unlisted combination (<1 s {lo_}, 1-2 s {mid}, >=2 s {hi_})",
                        "INCONCLUSIVE")
    tie = None
    if row.startswith("3"):
        k1 = kap["kappa_ge1s"]
        tie = ("leans visual confound" if min(k1["A"]["point"], k1["B"]["point"]) >= KAPPA_TIE
               else "leans keypoint signal missing freezing")
    reads = [Read("PASS" if anchor else "NOT_A_RESULT", {"arm": "pixel_anchor", "split": "tune"},
                  f"pixel freeze A - B total {out['pixel']['total']['A_minus_B']['point']:.4f} "
                  f"[{out['pixel']['total']['A_minus_B']['lo']:.4f}, "
                  f"{out['pixel']['total']['A_minus_B']['hi']:.4f}]",
                  n_effective=len(animals)).to_dict()]
    for nm in BIN_NAMES + ("total",):
        d = out["hsmm"][nm]["A_minus_B"]
        reads.append(Read("PASS" if d["lo"] > 0 or d["hi"] < 0 else "INCONCLUSIVE",
                          {"arm": f"hsmm_freeze_{nm}", "split": "tune"},
                          f"decoded freeze fraction A - B {d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}] "
                          f"({out['hsmm'][nm]['reading']}; PASS = interval excludes 0)",
                          n_effective=len(animals)).to_dict())
    reads.append(Read(verdict, {"arm": "known_answer", "split": "tune"},
                      f"row {row}" + (f"; kappa tiebreak: {tie}" if tie else ""),
                      n_effective=len(animals)).to_dict())
    write_json({**provenance.header(anchors.LUNA, stage="hsmm_known_answer",
                                    unverified="tune split; diagnostic gate"),
                "registration": "results/step4/KNOWN_ANSWER.md",
                "n_animals": len(animals), "n_recordings": sum(len(per[a]["A"]) + len(per[a]["B"])
                                                              for a in animals),
                "bins_s": [[lo, None if not np.isfinite(hi) else hi] for lo, hi in BINS_S],
                "freeze_fraction_by_bin": out, "kappa_within_context": kap, "speed": speed,
                "frame_count_mismatches": mismatched, "reading": row, "kappa_tiebreak": tie,
                "verdict": verdict, "reads": reads},
               os.path.join(OUT, "hsmm_known_answer.json"))
    log(f"known answer: {row} ({verdict}); tie {tie}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("planted_bouts", "known"))
    a = p.parse_args(argv)
    return {"planted_bouts": planted_bouts, "known": known}[a.phase](a)


if __name__ == "__main__":
    raise SystemExit(main())
