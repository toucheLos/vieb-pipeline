"""Paths and sweep grids for the tokenizer stages. Nothing here is a frame count.

Two rules this module exists to hold, both inherited rather than invented.

**Every root is overridable by environment variable**, the way
`recur.io.spine.SHAPEFLOW` is, so a test can point the whole stage at a fixture
tree without monkeypatching module globals. A hardcoded absolute path is a path
that cannot be tested against anything but the real corpus.

**Temporal parameters stay in seconds.** `recur.util.frames` converts them once,
against the dataset's own fps. The Spence rat corpus runs at 250 fps and a window
written as `63` would silently become a quarter of its intended duration there
while looking correct -- so no constant in this file is expressed in frames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: The repo root. Three levels up, because this module sits at
#: `recur/tok/config.py` and not at `recur/config.py` -- two `dirname`s land on
#: `<repo>/recur` and every path below it would quietly gain a second `recur/`.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Paths:
    """Every root and directory the tokenizer stages read or write."""

    #: ExBias's per-recording segmentation. Read only, and joined ONLY on
    #: recording id and frame range -- never on keypoint index, because ExBias
    #: orders bodyparts alphabetically where shapeflow uses file order.
    exbias_root: str = field(
        default_factory=lambda: _env("RECUR_EXBIAS", "/home/tul26194/exbias"))
    #: Which ExBias run's segments. `segments_v2` is the corrected one; v1's
    #: mean R^2 of 0.877 came from over-segmentation at a 200 ms median.
    exbias_segments: str = "segments_v2"

    bones_dir: str = os.path.join(REPO, "work", "bones")
    #: Overridable so the WHOLE tokenizer stack can be pointed at a different
    #: tree without editing a line of it. That is what the surrogate falsifier
    #: needs: `scripts/quantize.py` and `scripts/ladder.py` run unmodified on
    #: `VIEB_EGO_DIR` / `VIEB_TOK_DIR`, so "the identical pipeline" is true by
    #: construction rather than by inspection. A second implementation of a
    #: stage is a second thing that can differ from the one it is a control for.
    ego_dir: str = field(
        default_factory=lambda: _env("VIEB_EGO_DIR",
                                     os.path.join(REPO, "work", "ego")))
    tok_dir: str = field(
        default_factory=lambda: _env("VIEB_TOK_DIR",
                                     os.path.join(REPO, "work", "tok")))
    grids_dir: str = os.path.join(REPO, "work", "grids")
    results_dir: str = os.path.join(REPO, "results")

    def segments(self, recording_id: str) -> str:
        return os.path.join(self.exbias_root, self.exbias_segments,
                            f"{recording_id}.npz")

    def bones_shard(self, animal_tag: str) -> str:
        return os.path.join(self.bones_dir, f"{animal_tag}.npz")

    def grid(self, name: str) -> str:
        return os.path.join(self.grids_dir, f"{name}.txt")

    def result(self, name: str) -> str:
        return os.path.join(self.results_dir, name)


PATHS = Paths()

#: Which pose array the diagnostic runs on. `unfiltered` is PRIMARY: it is the
#: measurement, and it is the array shapeflow's own `bone_flagged` was computed
#: on, which is what makes the 2x2 between them a like-for-like comparison.
#: `wiener` is what every downstream feature actually consumes, so the gap
#: between the two rates says how much tracking failure the filter absorbs into
#: the feature space rather than removes from it.
POSE_ARMS: tuple[str, ...] = ("unfiltered", "wiener")
POSE_KEY = {"unfiltered": "pose_unfiltered", "wiener": "pose"}

#: Recordings per animal that get a shuffled-keypoint ceiling. The ceiling is a
#: property of the arena and the body, not of the frame, so a seeded handful per
#: animal estimates it and the full 3,846 would be spent for no extra precision.
CEILING_PER_ANIMAL = 4
CEILING_SEED = 0

#: Segments an animal needs before its Spearman rho against ExBias R^2 counts.
MIN_SEGMENTS_PER_ANIMAL = 30
