r"""Before/after clips: the same seconds of the same video, two pose arrays.

Nothing in recur or shapeflow has ever *shown* what the cleaning does. The
numbers exist now -- Phase A puts the Wiener filter's median displacement at
1.77 px with 86% of keypoint-frames moved -- but a displacement distribution is
not a thing anyone can check against their own eyes, and this corpus has already
had one round of statistics read off 64 clips that would not play.

## How it composes

`recur.render.video.cut` already takes a **pose array** and draws the skeleton
over the real frames, and it already burns a caption with `cv2.putText`. So a
comparison is: cut twice from the same `(video, a, b, crop)` with two different
pose arrays, then `hstack`. Nothing new touches pixels.

**The crop box is computed once and shared.** `vid.crop_box` puts a fixed box on
the clip's mean centroid; letting each pane compute its own would move the two
frames relative to each other and the difference a viewer sees would be partly
the crop.

`hstack` is generalised from `recur/scripts/instruments.py:hstack3`, which
hard-codes three inputs. Its `trim=end_frame=N` per input is the load-bearing
part and is kept: **equal frame counts**. If one pane is a frame shorter the
stack runs to the shorter one and the panes desynchronise silently.

## Inherited constraints, every one of which has already cost something

* **cv2 cannot write H.264 here and fails silently** -- it falls back to mp4v
  without raising, which no browser plays. That shipped 64 unplayable clips and
  two rounds of flag statistics read off them. Encoding goes through the pinned
  static ffmpeg, via `vid.cut`.
* **Even dimensions.** `yuv420p` needs them; an odd source is a silent encoder
  failure. `vid.crop_box` already forces the side even, and `hstack` of two
  even-width panes stays even.
* **`avg_frame_rate`, never `r_frame_rate`** -- across a 60-file sample the
  latter reports `60/1` on 9 of them, which renders at half speed. `vid.probe`
  handles this.
* **No `drawtext`**: this ffmpeg build has no such filter. Captions are drawn
  into the frames by `vid.cut`'s `label=`, before encoding.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from recur.render import video as vid

Detail = dict[str, Any]

#: Encoder settings for the stacked output. `crf` a little tighter than the
#: single-pane 24, because a two-pane clip is twice the pixels and the Atlas
#: commits its media to git.
CRF = 26
TIMEOUT_S = 900


def hstack(paths: Sequence[str], out_path: str, n_frames: int, *,
           crf: int = CRF) -> tuple[bool, str | None]:
    """N clips side by side, every one trimmed to the same frame count.

    Generalised from `instruments.hstack3`. Equal duration is not cosmetic: a
    pane one frame shorter desynchronises the whole stack from that point on,
    and the difference a viewer attributes to the cleaning is then partly a
    time offset.
    """
    n = len(paths)
    if n < 2:
        raise ValueError(f"hstack needs at least two inputs, got {n}")
    cmd = [vid.FFMPEG, "-y", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", p]
    fl = "".join(f"[{i}:v]trim=end_frame={int(n_frames)},setpts=PTS-STARTPTS[v{i}];"
                 for i in range(n))
    fl += "".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[out]"
    cmd += ["-filter_complex", fl, "-map", "[out]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    return r.returncode == 0, (r.stderr[:300] if r.returncode else None)


def compare(video: str, a: int, b: int, out_path: str, *, fps: float,
            panes: Sequence[tuple[str, npt.ArrayLike]],
            crop_from: npt.ArrayLike | None = None,
            flags: npt.ArrayLike | None = None,
            crop_size: int = vid.CROP, crf: int = CRF) -> tuple[bool, str | None]:
    """One clip per pane, same frames, same crop, then stacked.

    `panes` is `[(label, pose), ...]` in left-to-right order. `crop_from` is the
    pose the shared crop box is computed from -- one box for every pane, so the
    only thing that differs between them is the skeleton.

    `flags` marks frames to outline in red, and is drawn on **every** pane: the
    question a viewer is being asked is what the cleaning did to *this* frame,
    and highlighting it on only one side answers a different one.
    """
    if len(panes) < 2:
        raise ValueError("a comparison needs at least two panes")
    ref = np.asarray(crop_from if crop_from is not None else panes[0][1],
                     dtype=np.float64)
    box = vid.crop_box(ref, a, b, size=crop_size)
    tmp: list[str] = []
    try:
        for i, (label, pose) in enumerate(panes):
            fd, path = tempfile.mkstemp(prefix=f"cmp{i}_", suffix=".mp4")
            os.close(fd)
            tmp.append(path)
            ok, why = vid.cut(video, a, b, path, fps=fps,
                              pose=np.asarray(pose, dtype=np.float64),
                              flags=flags, label=label, crop=box, crf=crf)
            if not ok:
                return False, f"pane {i} ({label}): {why}"
        return hstack(tmp, out_path, b - a, crf=crf)
    finally:
        for p in tmp:
            if os.path.exists(p):
                os.unlink(p)


def windows(mask: npt.ArrayLike, *, fps: float, pad_s: float = 0.5,
            max_s: float = 6.0, min_gap_s: float = 1.0) -> list[tuple[int, int]]:
    """Frame ranges around runs of `mask`, merged when they nearly touch.

    A clip that starts on the violating frame shows the viewer no context to
    judge it against, so each run is padded. Runs closer than `min_gap_s` are
    merged rather than rendered as two clips of the same event.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    pad = max(int(round(pad_s * fps)), 1)
    gap = max(int(round(min_gap_s * fps)), 1)
    longest = max(int(round(max_s * fps)), 2)
    edges = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    out: list[tuple[int, int]] = []
    for a, b in runs:
        a, b = max(int(a) - pad, 0), min(int(b) + pad, m.shape[0])
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    trimmed = []
    for a, b in out:
        if b - a > longest:
            mid = (a + b) // 2
            a, b = mid - longest // 2, mid + longest // 2
        if b - a >= 2:
            trimmed.append((int(a), int(b)))
    return trimmed
