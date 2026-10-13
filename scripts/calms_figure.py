"""STEP 1 recovery curves: normalised score vs noise multiple, per instrument x label.

    python3 scripts/calms_figure.py   # results/step1/recovery.json -> results/step1/recovery.svg
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "results", "step1", "recovery.json")
OUT = os.path.join(REPO, "results", "step1", "recovery.svg")
COL = {"D": "#1f77b4", "C": "#7f7f7f", "M": "#d62728"}
NAME = {"D": "D boundary detector (boundary F1)", "C": "C segment clumps (event F1)",
        "M": "M Keypoint-MoSeq (event F1)"}


def main() -> int:
    with open(SRC, encoding="utf-8") as fh:
        d = json.load(fh)
    labels = ["investigation", "mount", "groom"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, lab in zip(axes, labels):
        for inst in ("D", "C", "M"):
            c = d["curves"].get(f"{inst}|{lab}")
            if c is None:
                continue
            x = c["levels"]
            lo = [v[0] for v in c["ci"]]
            hi = [v[1] for v in c["ci"]]
            ax.fill_between(x, lo, hi, color=COL[inst], alpha=0.18, lw=0)
            ax.plot(x, c["normalised"], "-o", color=COL[inst], ms=3, label=NAME[inst])
            for xi, yi, r in zip(x, c["normalised"], c["reachable"]):
                if not r:
                    ax.plot([xi], [yi], "o", mfc="white", mec=COL[inst], ms=5)
        ax.axhline(0.25, color="k", ls="--", lw=0.8)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.axvline(1.0, color="#2ca02c", lw=1.0)
        ax.axvline(d["r"], color="#999", lw=0.8, ls=":")
        ax.set_title(lab + (" (Luna-relevant; ceiling = 1)" if lab == "groom" else ""), fontsize=10)
        ax.set_xlabel("noise multiple of Luna (green = Luna, dotted = CalMS21 native)")
    axes[0].set_ylabel("normalised score, 95% CI over sequences")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"STEP 1 CalMS21 degrade-to-match: {d['decision']['outcome']} "
                 f"(bar 0.25; hollow = jitter below native, dropout only); attack GRID_LIMITED",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
