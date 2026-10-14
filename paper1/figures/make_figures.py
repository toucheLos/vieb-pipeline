"""Paper 1 figures, regenerated from committed result files.

    python3 paper1/figures/make_figures.py     # -> paper1/figures/*.svg, *.pdf

Every array comes from a committed JSON through `paper1/sources.committed`
(which refuses dirty files). Every number printed ON a figure is an S1 value,
formatted by `render.fmt` from `paper1/values.json`; `ANNOTATED` records which
keys each figure prints, and `paper1/check_figures.py` verifies them.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.dirname(HERE)
sys.path.insert(0, P1)
import render  # noqa: E402
import sources  # noqa: E402

V = render.values()
ANNOTATED: dict[str, list[str]] = {}
INK, MUTED, REAL, SURR = "#1a1a1a", "#6b7280", "#2563eb", "#9ca3af"
ARM_COL = {"corpus": REAL, "ou": "#a3a3a3", "ar": "#737373", "white": "#d4d4d4"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "svg.fonttype": "none"})


def t(fig: str, key: str, fmt: str | None = None) -> str:
    """An S1 value as text, recorded against the figure that prints it."""
    ANNOTATED.setdefault(fig, []).append([key, fmt or V[key]["fmt"]])
    e = V[key]
    return render.fmt(e["value"], fmt or e["fmt"])


def save(fig, name: str) -> None:
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def J(repo: str, path: str):
    return sources.committed(repo, path)[0]


# --------------------------------------------------------------------------

def fig1() -> None:
    """Where each control attaches. No numbers."""
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.axis("off")
    stages = ["video", "keypoints", "preprocessing", "representation", "labeller",
              "statistic"]
    xs = np.linspace(0.06, 0.94, len(stages))
    for x, s in zip(xs, stages):
        ax.add_patch(plt.Rectangle((x - 0.065, 0.55), 0.13, 0.2, fc="white", ec=INK, lw=0.8,
                                   transform=ax.transAxes))
        ax.text(x, 0.65, s, ha="center", va="center", transform=ax.transAxes)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.annotate("", (b - 0.067, 0.65), (a + 0.067, 0.65), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8})
    controls = [
        (xs[1], "surrogate pose\n(white, phase, VAR, OU)\nthrough every later stage", "A"),
        (xs[2], "the same smoothing\ninside every null", "B"),
        (xs[3], "direct identity probe,\nheld out within recording", "C"),
        (xs[4], "fitted labeller on surrogates;\ntriage against tracking flags", "D"),
        (xs[5], "the estimator's value\non white noise", "E"),
    ]
    for i, (x, txt, tag) in enumerate(controls):
        y = 0.36 if i % 2 == 0 else 0.12
        ax.annotate("", (x, 0.54), (x, y + 0.02), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": REAL, "lw": 1.0})
        ax.text(x, y, f"{tag}  {txt}", ha="center", va="top", color=REAL, fontsize=6.5,
                transform=ax.transAxes)
    ax.text(0.0, 0.95, "Each control re-runs a stage of the identical pipeline on input "
            "that has no behavioural structure, or probes what should not be there.",
            transform=ax.transAxes, color=MUTED, fontsize=7)
    save(fig, "fig1_pipeline")


def fig2() -> None:
    arms = [("corpus", "mouse"), ("ou", "OU"), ("ar", "VAR"), ("white", "white noise")]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.4))
    x = np.arange(len(arms))
    vals = [V[f"dwell.{a}.var_over_mean_sq"]["value"] for a, _ in arms]
    axs[0].bar(x, vals, color=[ARM_COL[a] for a, _ in arms], ec=INK, lw=0.5)
    for xi, (a, _) in zip(x, arms):
        axs[0].text(xi, V[f"dwell.{a}.var_over_mean_sq"]["value"], t("fig2", f"dwell.{a}.var_over_mean_sq"),
                    ha="center", va="bottom")
    axs[0].set_xticks(x, [n for _, n in arms])
    axs[0].set_ylabel("dwell var / mean²\n(median over animals)")
    axs[0].set_title("non-geometric dwell, same fitted model", fontsize=8)
    fr = [V[f"dwell.{a}.frac_reject"]["value"] for a, _ in arms]
    axs[1].bar(x, fr, color=[ARM_COL[a] for a, _ in arms], ec=INK, lw=0.5)
    for xi, (a, _) in zip(x, arms):
        axs[1].text(xi, V[f"dwell.{a}.frac_reject"]["value"],
                    f"{t('fig2', f'dwell.{a}.n_reject')}/{t('fig2', f'dwell.{a}.n_animals')}",
                    ha="center", va="bottom", fontsize=7)
    axs[1].set_xticks(x, [n for _, n in arms])
    axs[1].set_ylim(0, 1.15)
    axs[1].set_ylabel("animals rejecting geometric dwell")
    axs[1].set_title("rejection is universal, real or not", fontsize=8)
    save(fig, "fig2_dwell")


def fig3() -> None:
    arms = [("corpus", "results/grammar_moseq_table.json", "mouse"),
            ("ou", "results/grammar_moseq_surrogate_ou_table.json", "OU"),
            ("ar", "results/grammar_moseq_surrogate_ar_table.json", "VAR"),
            ("white", "results/grammar_moseq_surrogate_white_table.json", "white noise")]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    x = np.arange(len(arms))
    for xi, (a, _p, _n) in zip(x, arms):
        nt, ns = V[f"grammar.{a}.n_tested"]["value"], V[f"grammar.{a}.n_significant"]["value"]
        axs[0].bar(xi, nt, color="white", ec=INK, lw=0.5)
        axs[0].bar(xi, ns, color=ARM_COL[a], ec=INK, lw=0.5)
        axs[0].text(xi, nt, f"{t('fig3', f'grammar.{a}.n_significant')} / "
                    f"{t('fig3', f'grammar.{a}.n_tested')}", ha="center", va="bottom", fontsize=6.5)
    axs[0].set_xticks(x, [n for *_r, n in arms])
    axs[0].set_ylabel("n-grams: surviving FDR (filled)\nof those tested (outline)")
    axs[0].set_title("every arm 'has a grammar' (provisional reads)", fontsize=8)
    data = []
    for a, p, _n in arms:
        rows = J("recur", p)["rows"]
        data.append([r["median_log_ratio"] for r in rows if r.get("significant")])
    parts = axs[1].violinplot(data, positions=x, showmedians=True, widths=0.8)
    for b, (a, *_r) in zip(parts["bodies"], arms):
        b.set_facecolor(ARM_COL[a])
        b.set_edgecolor(INK)
        b.set_alpha(0.9)
    axs[1].set_xticks(x, [n for *_r, n in arms])
    axs[1].set_ylabel("effect of surviving n-grams\n(median log observed / Markov expected)")
    axs[1].set_title("white noise's survivors are not weak ones", fontsize=8)
    save(fig, "fig3_grammar")


def fig4() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.6))
    for ax, null, title in ((axs[0], "ar", "against a VAR null"),
                            (axs[1], "phase", "against a phase-randomised null")):
        for i, f in enumerate(("wiener", "unfiltered")):
            e = V[f"q1.{f}.{null}"]
            col = REAL if f == "wiener" else INK
            ax.errorbar([i], [100 * e["value"]], yerr=[[100 * (e["value"] - e["lo"])],
                                                       [100 * (e["hi"] - e["value"])]],
                        fmt="o", color=col, capsize=3, ms=4)
            ax.text(i + 0.1, 100 * e["value"], t("fig4", f"q1.{f}.{null}"), va="center",
                    fontsize=6.5)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlim(-0.5, 1.9)
        ax.set_xticks([0, 1], ["Wiener-filtered\npose", "unfiltered\npose"])
        ax.set_title(title, fontsize=8)
    axs[0].set_ylabel(f"cross-animal recurrence excess\nover the null (%), "
                      f"{t('fig4', 'param.ci_level', 'conf')} CI")
    save(fig, "fig4_recurrence")


def fig5() -> None:
    st = J("reversibility-tests", "results/surrogate_test.json")
    taus = ["0.2", "0.3", "0.5", "1.0"]
    tx = [float(x) for x in taus]
    g = st["guard_detail"]["ring"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    series = [("white noise", g["white_noise_sigma"], SURR, "s"),
              ("driven ring, estimated", st["ring_sigma_obs"], REAL, "o"),
              ("driven ring, phase-randomised surrogate", st["ring_sigma_surrogate"], "#93c5fd", "o"),
              ("driven ring, its true value", g["ring_true_state_sigma"], INK, "^")]
    for lab, d, c, mk in series:
        axs[0].plot(tx, [d[k] for k in taus], marker=mk, color=c, label=lab, lw=1)
    axs[0].annotate(t("fig5", "irr.white_02"), (0.2, g["white_noise_sigma"]["0.2"]),
                    textcoords="offset points", xytext=(-22, -3), fontsize=6.5, color=MUTED)
    axs[0].annotate(t("fig5", "irr.ring_true_02"), (0.2, g["ring_true_state_sigma"]["0.2"]),
                    textcoords="offset points", xytext=(-22, -3), fontsize=6.5)
    axs[0].set_xticks(tx, [t("fig5", f"irr.tau{i}") for i in range(len(taus))])
    axs[0].set_xlabel("lag τ (s)")
    axs[0].set_ylabel("entropy production σ (nats)")
    axs[0].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2)
    axs[0].set_title(f"planted-irreversibility guard: {t('fig5', 'irr.guard')}", fontsize=8)
    sw = st["embedding_artifact_sweep"]["sweep"]
    ks = [r["embed_k"] for r in sw]
    for i, k_t in enumerate(taus):
        axs[1].plot(ks, [r["sigma"][k_t] for r in sw], marker="o", lw=1,
                    label=f"τ = {t('fig5', f'irr.tau{i}')} s")
    for K in (1, 24):
        axs[1].annotate(t("fig5", f"irr.embed_K{K}"), (K, V[f"irr.embed_K{K}"]["value"]),
                        textcoords="offset points", xytext=(3, 3), fontsize=6.5)
    axs[1].set_xscale("log", base=2)
    axs[1].set_xticks(ks, [t("fig5", f"irr.K{k}") for k in ks])
    axs[1].minorticks_off()
    axs[1].set_xlabel("delay-embedding order K (frames)")
    axs[1].set_ylabel("σ on white-noise pose (nats)")
    axs[1].set_title("overlapping windows manufacture σ", fontsize=8)
    axs[1].legend(fontsize=6, frameon=False)
    save(fig, "fig5_irreversibility")


def fig6() -> None:
    d = J("recur", "results/triage_moseq.json")["read"]["detail"]
    pl = d["per_label"]
    order = []
    for cat in ("artifact", "occlusion_linked"):
        labs = [k for k, v in pl.items() if v["category"] == cat]
        order += sorted(labs, key=lambda k: -pl[k]["enrichment_overall"])
    n_s = max(len(v["enrichment_by_stratum"]) for v in pl.values())
    M = np.full((len(order), n_s), np.nan)
    for i, k in enumerate(order):
        for s, v in pl[k]["enrichment_by_stratum"].items():
            M[i, int(s)] = v
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    im = ax.imshow(np.log2(M), aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
    ax.set_yticks(range(len(order)), [f"{k} ({pl[k]['category'].replace('_', '-')})" for k in order],
                  fontsize=6)
    ax.set_xlabel("posture stratum (speed quintile × extension tercile)")
    nart = sum(1 for k in order if pl[k]["category"] == "artifact")
    ax.axhline(nart - 0.5, color=INK, lw=1)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log₂ enrichment of tracking flags")
    ax.set_title(f"artifact: enriched in every stratum ({t('fig6', 'tri.n.artifact')} syllables, "
                 f"{t('fig6', 'tri.mass.artifact')} of frames); occlusion-linked: only in some",
                 fontsize=7)
    save(fig, "fig6_triage")


# --------------------------------------------------------------------------
# Supplementary
# --------------------------------------------------------------------------

def figS1() -> None:
    probes = [("id.ego.session", "pose windows, session", "id.ego.session.ceiling_nats"),
              ("id.ego.animal", "pose windows, animal", "id.ego.animal.ceiling_nats"),
              ("id.egoraw.session", "unfiltered pose, session", "id.egoraw.session.ceiling_nats"),
              ("id.egoraw.animal", "unfiltered pose, animal", "id.egoraw.animal.ceiling_nats"),
              ("id.g3.recording", "encoder state, recording", "id.g3.recording.ceiling_nats"),
              ("id.g3.animal", "encoder state, animal", "id.g3.animal.ceiling_nats"),
              ("id.moseq.animal", "MoSeq labels, animal", "id.moseq.animal.ceiling_nats")]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for i, (k, lab, ck) in enumerate(probes):
        v = V[f"{k}.leak_nats"]["value"]
        ax.barh(i, v, color=REAL, ec=INK, lw=0.4)
        ax.plot([V[ck]["value"]], [i], "|", color=INK, ms=10)
        ax.text(v, i, f" {t('figS1', f'{k}.leak_nats')} of {t('figS1', ck)}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(probes)), [p[1] for p in probes])
    ax.set_xlabel("identity recoverable by a direct probe (nats); tick = log N ceiling")
    save(fig, "figS1_identity")


def figS2() -> None:
    d = J("recur", "results/montage_moseq.json")["selection"]["detail"]
    labs = sorted(d["frame_mass"], key=lambda k: -d["frame_mass"][k])
    fm = np.array([d["frame_mass"][k] for k in labs])
    em = np.array([d["event_mass"][k] for k in labs])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.loglog(fm, em, "o", ms=3, color=REAL)
    lim = [min(fm.min(), em.min()) * 0.8, 1]
    ax.plot(lim, lim, color=MUTED, lw=0.6)
    for lab in ("1", "0"):
        name = {"1": "most frames", "0": "most runs"}[lab]
        ax.annotate(f"syllable with {name}: {t('figS2', f'mass.{lab}.frame_mass')} of frames, "
                    f"{t('figS2', f'mass.{lab}.event_mass')} of runs",
                    (d["frame_mass"][lab], d["event_mass"][lab]), textcoords="offset points",
                    xytext=(-150, 8 if lab == "1" else -14), fontsize=6)
    ax.set_xlabel("share of frames")
    ax.set_ylabel("share of runs (events)")
    save(fig, "figS2_mass")


def figS3() -> None:
    arms = ["white", "phase", "var5", "ou"]
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    w = 0.38
    for i, a in enumerate(arms):
        for off, pre, col, lab in ((-w / 2, "sep", SURR, "raw window frames"),
                                   (w / 2, "seprough", REAL, "+ roughness feature")):
            v = V[f"{pre}.{a}"]["value"]
            ax.bar(i + off, v, w, color=col, ec=INK, lw=0.5, label=lab if i == 0 else None)
            ax.text(i + off, v, t("figS3", f"{pre}.{a}"), ha="center", va="bottom", fontsize=5.5)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls="--")
    ax.set_xticks(range(len(arms)), ["white", "phase", "VAR", "OU"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("AUC, mouse vs surrogate\n(linear probe)")
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "figS3_separability")


def figS4() -> None:
    amps = [1, 2, 4, 8, 16, 32]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for o, c in ((1, INK), (2, REAL)):
        pts = [V[f"plant.o{o}.a{a}"] for a in amps]
        y = [p["value"] for p in pts]
        ax.errorbar(amps, y, yerr=[[p["value"] - p["lo"] for p in pts], [p["hi"] - p["value"] for p in pts]],
                    marker="o", ms=3, color=c, capsize=2, lw=1,
                    label={1: "first-order break", 2: "second-order break"}[o])
    ax.axhline(V["plant.chance_o2_a16"]["value"], color=MUTED, lw=0.6, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(amps, [str(a) for a in amps])
    ax.minorticks_off()
    ax.set_xlabel("planted break amplitude (multiples of σ̄)")
    ax.set_ylabel("recall")
    ax.annotate(t("figS4", "plant.o2.a16"), (16, V["plant.o2.a16"]["value"]), textcoords="offset points",
                xytext=(4, -10), fontsize=6.5, color=REAL)
    ax.legend(fontsize=6, frameon=False)
    save(fig, "figS4_detection_floor")


def figS5() -> None:
    ds = ["pole_radius", "pole_frequency_hz", "band_slow", "band_mid", "band_marginal", "band_fast"]
    names = ["pole radius", "pole frequency", "slow-band power", "mid-band power",
             "marginal-band power", "fast-band power"]
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    for i, d_ in enumerate(ds):
        ax.barh(i, V[f"dyn.{d_}"]["value"], color=REAL if i < 1 else SURR, ec=INK, lw=0.4)
        ax.text(V[f"dyn.{d_}"]["value"], i, " " + t("figS5", f"dyn.{d_}"), va="center", fontsize=6.5)
    ax.axvline(0.05, color=MUTED, lw=0.6, ls="--")
    ax.set_yticks(range(len(ds)), names)
    ax.set_xlabel(f"corpus windows above the tracking-noise null's "
                  f"{t('figS5', 'param.null_q', 'ord')} percentile")
    save(fig, "figS5_descriptor_floor")


def main() -> int:
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, figS1, figS2, figS3, figS4, figS5):
        f()
    with open(os.path.join(HERE, "annotated.json"), "w", encoding="utf-8") as fh:
        json.dump(ANNOTATED, fh, indent=1)
    print(f"figures written; {sum(len(v) for v in ANNOTATED.values())} annotations from S1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
