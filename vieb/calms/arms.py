"""Every STEP 1 arm, by name: what its input is and which instruments see it.

Registered in `results/step1/REGISTRATION.md` (primary, controls) and
`results/step1/ABLATION_REGISTRATION.md` (secondary arms, ablation). An arm is
``<family><level>_s<seed>`` or, for a control, ``<family>_s<seed>``. No labels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

LEVELS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
SEEDS_ALL: tuple[int, ...] = (0, 1, 2, 3, 4)
SEEDS_M_SECONDARY: tuple[int, ...] = (0, 1, 2)
ABLATION_LEVELS: tuple[float, ...] = (0.0, 1.0)
KAPPA_FULL = 1e6


@dataclass(frozen=True)
class Spec:
    family: str
    level: float
    seed: int
    kind: str = "real"              # real | white | phase
    jitter: bool = True
    dropout: bool = True
    keypoints: str = "seven"        # seven | six | nonose
    smooth: str = "none"            # none | median
    fps: float = 30.0
    input_from: str | None = None   # reuse another family's inputs
    kappa_full: float = KAPPA_FULL
    deriv_sec: float = 0.133
    d_group: str = "both"
    instruments: tuple[str, ...] = ("D", "C", "M")
    primary: bool = False

    @property
    def cond(self) -> str:
        return self.family if self.kind != "real" else f"{self.family}{self.level:g}"

    @property
    def name(self) -> str:
        return f"{self.cond}_s{self.seed}"

    @property
    def input_arm(self) -> str:
        return f"{self.input_from}{self.level:g}_s{self.seed}" if self.input_from else self.name


#: family -> (template, levels, instruments). Controls have levels = (0,).
FAMILIES: dict[str, tuple[Spec, tuple[float, ...], tuple[str, ...]]] = {
    "m": (Spec("m", 0, 0, primary=True), LEVELS, ("D", "C", "M")),
    "white": (Spec("white", 0, 0, kind="white", dropout=False, jitter=False,
                   primary=True), (0.0,), ("D", "C", "M")),
    "phase": (Spec("phase", 0, 0, kind="phase", dropout=False, jitter=False,
                   primary=True), (0.0,), ("D", "C", "M")),
    # ---- secondary (ABLATION_REGISTRATION A)
    "j": (Spec("j", 0, 0, dropout=False), (1.0, 1.5, 2.0, 3.0), ("D", "C", "M")),
    "d": (Spec("d", 0, 0, jitter=False), (0.25, 0.5, 1.0, 1.5, 2.0, 3.0), ("D", "C", "M")),
    "k": (Spec("k", 0, 0, keypoints="six"), LEVELS, ("D", "C", "M")),
    # ---- ablation (ABLATION_REGISTRATION B), at m in {0, 1}
    "a-med-m": (Spec("a-med-m", 0, 0, smooth="median"), ABLATION_LEVELS, ("D", "C", "M")),
    "a-med-white": (Spec("a-med-white", 0, 0, kind="white", jitter=False, dropout=False,
                         smooth="median"), (0.0,), ("D", "C", "M")),
    "a-med-phase": (Spec("a-med-phase", 0, 0, kind="phase", jitter=False, dropout=False,
                         smooth="median"), (0.0,), ("D", "C", "M")),
    "a-nonose-m": (Spec("a-nonose-m", 0, 0, keypoints="nonose"), ABLATION_LEVELS, ("D", "C", "M")),
    "a-fps15-m": (Spec("a-fps15-m", 0, 0, fps=15.0), ABLATION_LEVELS, ("D", "C", "M")),
    "a-fps10-m": (Spec("a-fps10-m", 0, 0, fps=10.0), ABLATION_LEVELS, ("D", "C", "M")),
    "a-k5-m": (Spec("a-k5-m", 0, 0, input_from="m", kappa_full=1e5), ABLATION_LEVELS, ("M",)),
    "a-w067-m": (Spec("a-w067-m", 0, 0, input_from="m", deriv_sec=0.067), ABLATION_LEVELS,
                 ("D", "C")),
    "a-w267-m": (Spec("a-w267-m", 0, 0, input_from="m", deriv_sec=0.267), ABLATION_LEVELS,
                 ("D", "C")),
    "a-shape-m": (Spec("a-shape-m", 0, 0, input_from="m", d_group="shape"), ABLATION_LEVELS,
                  ("D",)),
}

#: Which reference each ablation family's Delta is taken against.
ABLATION_REF: dict[str, str] = {
    "a-med-m": "m", "a-nonose-m": "m", "k": "m", "a-fps15-m": "m", "a-fps10-m": "m",
    "a-k5-m": "m", "a-w067-m": "m", "a-w267-m": "m", "a-shape-m": "m"}
ABLATION_DECISION: dict[str, str] = {
    "a-med-m": "smoothing: median 0.25 s", "k": "keypoints: 6 (no neck)",
    "a-nonose-m": "keypoints: no nose", "a-fps15-m": "frame rate: 15 fps",
    "a-fps10-m": "frame rate: 10 fps", "a-k5-m": "MoSeq kappa: 1e5",
    "a-w067-m": "window: DERIV_SEC 0.067 s", "a-w267-m": "window: DERIV_SEC 0.267 s",
    "a-shape-m": "representation: pose only"}


def seeds(family: str, instrument: str) -> tuple[int, ...]:
    tpl = FAMILIES[family][0]
    return SEEDS_ALL if (tpl.primary or instrument != "M") else SEEDS_M_SECONDARY


def all_specs() -> list[Spec]:
    out = []
    for fam, (tpl, levels, inst) in FAMILIES.items():
        for m in levels:
            n_seeds = max(len(seeds(fam, i)) for i in inst)
            for s in SEEDS_ALL[:n_seeds]:
                out.append(replace(tpl, level=m, seed=s, instruments=inst))
    return out


def parse(name: str) -> Spec:
    for sp in all_specs():
        if sp.name == name:
            return sp
    raise KeyError(name)


def build_list() -> list[str]:
    """Arms that need their own inputs (not `input_from` another family)."""
    return [s.name for s in all_specs() if s.input_from is None]


def seg_list() -> list[str]:
    """Arms D/C run on: every arm with D or C, at D/C's seeds."""
    return [s.name for s in all_specs() if {"D", "C"} & set(s.instruments)]


def kpms_list() -> list[str]:
    """Arms MoSeq is fitted on, at M's seeds."""
    return [s.name for s in all_specs()
            if "M" in s.instruments and s.seed in seeds(s.family, "M")]


def conds() -> list[tuple[str, str, str]]:
    """``(instrument, family, cond)`` for every scoring cell."""
    out = []
    for fam, (tpl, levels, inst) in FAMILIES.items():
        for i in inst:
            for m in levels:
                out.append((i, fam, replace(tpl, level=m).cond))
    return out
