"""A synthetic top-down mouse, for tests that need pose with known properties.

Lifted from `~/recur/tests/test_swap.py`, where it was written, and carried here
rather than imported. A test fixture is not shared infrastructure: importing it
across repos would make this repo's tests fail when recur reorganises its own,
and both `tests/` directories are packages, so `tests.test_swap` would resolve to
whichever `sys.path` entry came first -- a collision that would silently pick the
wrong module rather than raise.

The y-sign comment below is inherited verbatim because it is the whole point of
the fixture and it cost a debugging session once already.
"""
import numpy as np

FPS = 30.0
T = 900


def mouse(t=T, *, turn_rad=0.0, turn_frames=1, ears_only=False):
    """A synthetic top-down mouse, 7 keypoints, in the anchor's order.

    Built in a body frame and then rotated, so left/right is correct by
    construction and any sign flip in a test is one the test planted. Every bone
    length is constant, which is what makes it usable as a clean baseline for the
    bone diagnostic.
    """
    # NOTE the y signs. DLC writes IMAGE coordinates, y increasing downward, so
    # the animal's left appears at negative y when x runs nose-forward. Measured
    # on the corpus, `signed_sine` is positive in all 30 tune recordings
    # sampled, and this layout is what reproduces that. Writing it in maths
    # convention (y up) inverts every sign and makes a clean mouse read as a
    # wholly swapped one -- which is exactly what the first draft of this
    # fixture did.
    body = np.array([
        [0.9, -0.5],   # left_ear
        [0.9, +0.5],   # right_ear
        [1.6, 0.0],    # nose
        [0.0, 0.0],    # center
        [-0.9, -0.6],  # left_hip
        [-0.9, +0.6],  # right_hip
        [-1.7, 0.0],   # tail_base
    ]) * 20.0
    head = np.zeros(t)
    if turn_rad:
        ramp = np.linspace(0.0, turn_rad, turn_frames)
        head[t // 2: t // 2 + turn_frames] = ramp
        head[t // 2 + turn_frames:] = turn_rad
    out = np.empty((t, 7, 2))
    for f in range(t):
        # Whole-body heading drifts; the head turn is applied to nose+ears only.
        psi = 0.4 * np.sin(2 * np.pi * f / 300.0)
        R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
        pts = body.copy()
        a = head[f]
        Rh = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        turned = [0, 1] if ears_only else [0, 1, 2]
        pts[turned] = pts[turned] @ Rh.T
        out[f] = pts @ R.T + np.array([200.0, 200.0])
    return out
