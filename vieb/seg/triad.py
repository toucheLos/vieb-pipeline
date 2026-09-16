r"""Odd-one-out triads for a blind look at the island.

Registered in `results/ISLAND_LOOK_PREREGISTRATION.md`.

## What the triad structure buys

A binary "island or control?" label tests whether the island **differs** from a
control. It would pass on an island whose members share nothing with each other
but happen to differ from the pool. The question is whether island segments look
like **each other**, and the triad asks it directly: with two island members and
one control, the odd clip can only be found if the two islands cohere.

The mirrored trial type -- two controls and one island -- is built and scored
**separately**, because it tests the weaker property and averaging the two into
one accuracy would let the weaker one carry the claim.

## Three cues that would otherwise decide it

**Speed.** The island runs at 0.205x the mean segment speed in its own animals.
**Duration.** Its median is 1.30 s against the corpus's 0.93 s, and
`partition.py` records a rater scoring +0.183 purely by calling the six longest
clips "same". **Animal and scene.** One animal supplies 56 of the island's 361
segments.

The first two are handled by `match_control`. The third is handled here, by
requiring **three distinct animals per trial** -- which is also why the control
is not drawn from the island member's own recording. In a triad that would place
two clips from one scene into the trial and point the scorer at the wrong
answer.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from recur.read import Read

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
Detail = dict[str, Any]

__all__ = ["ARMS", "CHANCE", "N_TRIALS", "SALT", "TYPES", "accuracy_read",
           "match_control", "mde", "mde_read", "plan_trials", "trial_id"]

#: Chance for a three-alternative forced choice.
CHANCE = 1.0 / 3.0
#: Registered, then AMENDED upward before any trial was built. 72 was chosen
#: against the naive binomial MDE, which is not the statistic the verdict uses:
#: measured, the ANIMAL-CLUSTERED interval needs 65% accuracy to clear chance at
#: 36 trials per arm, 55% at 60 and 50% at 90. 90 per arm is the first size that
#: detects a +17-point lift, which is what the registration said it wanted.
N_TRIALS = 180
#: `matched` answers the question; `unmatched` says whether the scorer can see.
ARMS: tuple[str, str] = ("matched", "unmatched")
#: Which class supplies the pair, and therefore what the trial tests.
TYPES: tuple[str, str] = ("island_pair", "control_pair")
SALT = "vieb-island-3afc"


def trial_id(arm: str, ttype: str, index: int, *, salt: str = SALT) -> str:
    """An opaque, stable trial name that does not encode the answer.

    A function of the arm, the trial type and the index only. Hashing the odd
    position or the member ids in would make the key recoverable by anyone who
    could re-run the builder, which is the same failure as a filename that sorts
    the real arm before the control.
    """
    h = hashlib.sha256(f"{salt}|{arm}|{ttype}|{index}".encode()).hexdigest()
    return f"t{h[:10]}"


def match_control(target: Mapping[str, float], pool: Sequence[Mapping[str, Any]],
                  *, used: set[int], forbid_animals: set[str],
                  rng: np.random.Generator, n_near: int = 5) -> int:
    """Nearest non-island segment on joint (log duration, log speed).

    **The whole pool is searched at once**, never narrowed by animal first.
    `partition.take_other` records the reason: narrowing first means "a target of
    20 frames can be matched against a label whose clips are all 180, and the
    match fails while appearing to have been made."

    Randomised among the `n_near` closest so repeated targets do not all collapse
    onto one control, and so the control arm is not a deterministic function of
    the island arm.
    """
    lt = float(np.log(max(float(target["n_frames"]), 1.0)))
    ls = float(np.log(max(float(target["speed"]), 1e-9)))
    best: list[tuple[float, int]] = []
    for j, c in enumerate(pool):
        if j in used or str(c["animal"]) in forbid_animals:
            continue
        d = ((np.log(max(float(c["n_frames"]), 1.0)) - lt) ** 2
             + (np.log(max(float(c["speed"]), 1e-9)) - ls) ** 2)
        best.append((float(d), j))
    if not best:
        return -1
    best.sort()
    take = best[:max(1, int(n_near))]
    return int(take[int(rng.integers(0, len(take)))][1])


def plan_trials(island: Sequence[Mapping[str, Any]],
                pool: Sequence[Mapping[str, Any]], *, arm: str,
                n_trials: int = N_TRIALS // 2, rng: np.random.Generator,
                n_near: int = 5) -> list[Detail]:
    """Trials for one arm: half `island_pair`, half `control_pair`.

    **Odd-one position is balanced by construction**, not drawn i.i.d. -- a
    permutation of a tiled `[0, 1, 2]`, copying `forced_choice.plan_trials`. An
    i.i.d. draw leaves the position marginal uneven at this n, and a scorer who
    notices is answering a different question.

    **Three distinct animals per trial**, enforced here rather than checked
    later, because the draw has to be retried when it fails.
    """
    per_type = int(n_trials) // 2
    positions = rng.permutation(np.tile(np.arange(3), per_type * 2 // 3 + 1))
    out: list[Detail] = []
    used_pool: set[int] = set()
    used_isl: set[int] = set()
    k = 0
    for ttype in TYPES:
        for i in range(per_type):
            got = _one(island, pool, ttype=ttype, arm=arm, index=len(out),
                       used_isl=used_isl, used_pool=used_pool, rng=rng,
                       n_near=n_near)
            if got is None:
                continue
            got["odd_position"] = int(positions[k % positions.size])
            k += 1
            # Place the odd member at its balanced position; the other two keep
            # their order. Built as a list so the rendered file order IS the
            # presentation order and nothing re-sorts downstream.
            odd = got.pop("_odd")
            pair = got.pop("_pair")
            clips = list(pair)
            clips.insert(got["odd_position"], odd)
            got["clips"] = clips
            got["animals"] = [c["animal"] for c in clips]
            assert len(set(got["animals"])) == 3, (
                "three distinct animals per trial is the design, not a "
                "preference: two clips of one mouse in one box resemble each "
                "other because it is one mouse in one box")
            out.append(got)
    return out


def _one(island: Sequence[Mapping[str, Any]],
         pool: Sequence[Mapping[str, Any]], *, ttype: str, arm: str,
         index: int, used_isl: set[int], used_pool: set[int],
         rng: np.random.Generator, n_near: int) -> Detail | None:
    """One trial, or None if three distinct animals could not be drawn."""
    free_i = [i for i in range(len(island)) if i not in used_isl]
    if len(free_i) < 2:
        return None
    if ttype == "island_pair":
        a = int(rng.choice(free_i))
        mates = [i for i in free_i
                 if i != a and island[i]["animal"] != island[a]["animal"]]
        if not mates:
            return None
        b = int(rng.choice(mates))
        forbid = {island[a]["animal"], island[b]["animal"]}
        c = (match_control(island[a], pool, used=used_pool,
                           forbid_animals=forbid, rng=rng, n_near=n_near)
             if arm == "matched"
             else _uniform(pool, used_pool, forbid, rng))
        if c < 0:
            return None
        used_isl.update({a, b})
        used_pool.add(c)
        return {"id": trial_id(arm, ttype, index), "arm": arm, "type": ttype,
                "index": index, "_pair": [island[a], island[b]],
                "_odd": pool[c], "odd_is": "control"}
    a = int(rng.choice(free_i))
    forbid = {island[a]["animal"]}
    c1 = (match_control(island[a], pool, used=used_pool, forbid_animals=forbid,
                        rng=rng, n_near=n_near)
          if arm == "matched" else _uniform(pool, used_pool, forbid, rng))
    if c1 < 0:
        return None
    forbid2 = forbid | {str(pool[c1]["animal"])}
    c2 = (match_control(island[a], pool, used=used_pool | {c1},
                        forbid_animals=forbid2, rng=rng, n_near=n_near)
          if arm == "matched" else _uniform(pool, used_pool | {c1}, forbid2, rng))
    if c2 < 0:
        return None
    used_isl.add(a)
    used_pool.update({c1, c2})
    return {"id": trial_id(arm, ttype, index), "arm": arm, "type": ttype,
            "index": index, "_pair": [pool[c1], pool[c2]],
            "_odd": island[a], "odd_is": "island"}


def _uniform(pool: Sequence[Mapping[str, Any]], used: set[int],
             forbid: set[str], rng: np.random.Generator) -> int:
    free = [j for j in range(len(pool))
            if j not in used and str(pool[j]["animal"]) not in forbid]
    return int(rng.choice(free)) if free else -1


def mde(n: int, *, chance: float = CHANCE, power: float = 0.8,
        alpha: float = 0.05) -> float:
    """Smallest lift over chance this many trials can detect.

    The standard one-sample proportion MDE, computed at the realised `n` and
    reported **before the accuracy is read** -- the pattern `journeys.py` and
    `learning_curve.py` both use. A null from a design that could not have seen
    the effect is not evidence of absence.
    """
    from scipy import stats

    z_a = float(stats.norm.isf(alpha / 2.0))
    z_b = float(stats.norm.isf(1.0 - power))
    sd = float(np.sqrt(chance * (1.0 - chance)))
    return float((z_a + z_b) * sd / np.sqrt(max(int(n), 1)))


def mde_read(n: int, *, scored_object: Detail, n_effective: int,
             plausible: float, chance: float = CHANCE) -> Read:
    """Could this design have seen an effect worth calling one?

    Reported whether it passes or not. Unlike the gates elsewhere in this
    programme it does **not** stop the contrast -- the accuracy is cheap and is
    wanted either way -- but a null must be quoted with this number beside it.
    """
    m = mde(n, chance=chance)
    detail: Detail = {"mde": m, "n_trials": int(n), "chance": chance,
                      "plausible_effect": plausible, "power": 0.8,
                      "alpha": 0.05}
    if m > plausible:
        return Read("FAIL", scored_object,
                    f"MDE = {m:.4f} over {n} trials, above the {plausible:.4f} "
                    f"lift that would count as a real effect: a null here would "
                    f"mean 'not enormous', not 'nothing', and must be quoted "
                    f"that way",
                    n_effective=n_effective, detail=detail)
    return Read("PASS", scored_object,
                f"MDE = {m:.4f} over {n} trials at 80% power against chance "
                f"{chance:.4f}, below the {plausible:.4f} that would count as a "
                f"real effect, so a null here is informative",
                n_effective=n_effective, detail=detail)


def accuracy_read(correct: Sequence[bool], animals: Sequence[str], *, arm: str,
                  ttype: str, scored_object: Detail, n_effective: int,
                  seed: int = 0, chance: float = CHANCE) -> Read:
    """Accuracy against chance, with an ANIMAL-clustered interval.

    `percall.py` records why the pooled binomial alone was abandoned: batch 1
    returned 43.1% at binomial p = 0.025 with a label-clustered CI of
    [0.320, 0.546] that **includes chance**. Trials sharing animals are not
    independent, and the clustered interval is the one that decides.
    """
    from recur import boot

    c = np.asarray(list(correct), dtype=np.float64)
    if c.size < 5:
        return Read("INCONCLUSIVE", scored_object,
                    f"only {c.size} scored trials on {arm}/{ttype}",
                    n_effective=n_effective, detail={"n": int(c.size)})
    acc = float(c.mean())
    ci = boot.animal_interval(c, list(animals), how="mean", seed=seed)
    detail: Detail = {"n": int(c.size), "n_correct": int(c.sum()),
                      "accuracy": acc, "chance": chance,
                      "animal_interval": ci}
    tail = (f"{int(c.sum())}/{int(c.size)} = {acc:.1%} against chance "
            f"{chance:.1%}, animal-clustered interval "
            f"[{float(ci['lo']):.3f}, {float(ci['hi']):.3f}] over "
            f"{int(ci['n_animals'])} animals")
    if float(ci["lo"]) > chance:
        return Read("PASS", scored_object,
                    f"{arm}/{ttype}: the scorer beats chance — {tail}",
                    n_effective=n_effective, detail=detail)
    return Read("FAIL", scored_object,
                f"{arm}/{ttype}: the scorer does not beat chance — {tail}, an "
                f"interval that includes chance",
                n_effective=n_effective, detail=detail)
