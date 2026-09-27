"""STATES figure for the site: the gap ratios against the surrogates, and t2 by lag.

    python3 scripts/states_figure.py      # -> results/states_spectrum.svg

Reads `results/states.json` (registered) and `results/states_diag.json`
(post-hoc, labelled as such on the figure). Colours are the site's theme.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BG, FG, MUTED, LINE = "#0f1218", "#e8eaed", "#98a2b3", "#2a3140"
ACCENT, WARN, BAD = "#6aa9ff", "#f5b544", "#f2777a"
SURR = {"phase": "#c98bdb", "var5": "#5fd08a", "microstate": "#f5b544",
        "microstate0": "#98a2b3", "white": "#f2777a"}


def _res(name: str) -> dict:
    with open(os.path.join(REPO, "results", name), encoding="utf-8") as fh:
        d = json.load(fh)
    return d.get("results", d)


def main() -> int:
    st, dg = _res("states.json"), _res("states_diag.json")
    plt.rcParams.update({"font.size": 10, "text.color": FG, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.edgecolor": LINE, "font.family": "DejaVu Sans"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=BG)
    for ax in (a, b):
        ax.set_facecolor(BG)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ts = st["corpus_timescales"]
    i = list(range(2, 2 + len(ts["gap_point"])))
    for k, s in st["surrogate_timescales"].items():
        s = s.get("results", s)
        a.plot(i, s["gap_hi"], color=SURR[k], lw=1.2, alpha=0.9,
               label=f"{k} (upper 95%)")
    # The bootstrap interval need not contain the point: a ratio of sorted
    # timescales is >= 1 by construction, so resampling biases it upward.
    a.vlines(i, ts["gap_lo"], ts["gap_hi"], color=ACCENT, lw=1.4, zorder=5)
    a.plot(i, ts["gap_point"], "o", color=ACCENT, ms=5, zorder=6,
           label="mice (point, 95% bootstrap)")
    a.axhline(1, color=LINE, lw=0.8)
    a.set_xticks(i, [f"t{j}/t{j + 1}" for j in i], rotation=45, fontsize=8)
    a.set_ylabel("gap ratio")
    a.set_title("Registered: no gap beats the surrogates (τ = 64)",
                color=FG, fontsize=10.5, loc="left")
    a.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, ncol=2)

    taus = sorted(int(t) for t in dg["corpus"])
    b.plot(taus, [dg["corpus"][str(t)][0] for t in taus], "o-", color=ACCENT,
           lw=1.6, label="mice")
    b.plot(taus, [dg["white"][str(t)][0] for t in taus], "s--", color=BAD,
           lw=1.4, label="white noise")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xticks(taus, [str(t) for t in taus])
    b.set_xlabel("lag τ (frames, 30 fps)")
    b.set_ylabel("slowest timescale t2 (frames)")
    b.set_title("Post-hoc: at τ = 64 the mice sit on the noise floor",
                color=FG, fontsize=10.5, loc="left")
    b.axvline(64, color=WARN, lw=0.8, ls=":")
    b.text(60, b.get_ylim()[0] * 1.15, "τ* chosen", color=WARN, fontsize=8,
           ha="right")
    b.legend(frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.tight_layout()
    out = os.path.join(REPO, "results", "states_spectrum.svg")
    fig.savefig(out, facecolor=BG)
    if os.environ.get("STATES_FIG_PNG"):
        fig.savefig(os.environ["STATES_FIG_PNG"], facecolor=BG, dpi=90)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
