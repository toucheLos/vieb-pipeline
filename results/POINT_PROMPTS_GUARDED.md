# Guarded point prompts: no inflation, most of the gain in context A, and a head-coverage gap between arenas

`scripts/point_prompts.py --mode guarded`, **designed after** the plain run
(`POINT_PROMPTS.md` §2) and committed (`bef921e`) before its own run.
Descriptive. The same every-30th frames, the same metrics.
`results/point_prompts_guarded.json`.

**The combination.** SAM's box-only mask comes first. A keypoint is used as a
point prompt only within 0.2 body lengths of that mask, and the point mask is
kept only if it passes STABILISE 7's rules and is at most 1.3× the box mask's
area. Otherwise the box mask stands. The point mask was used on **89.7%** of
frames.

## 1. Results

| measure | box only | guarded points | plain points (for reference) |
|---|---|---|---|
| ears outside the mask (not prompted) | 14.54% [11.25, 18.25] | **10.26% [7.93, 12.96]** | 7.01% |
| paired change in ears outside | — | **−4.28 pts [−5.47, −3.11]** | −7.26 pts |
| hips outside | 2.84% | 2.69% | 1.69% |
| mask area / bl², 95th percentile | 0.622 | **0.645 (+4%)** | 0.746 (+20%) |
| masks over 1.5× the box mask | — | 0%, **by construction** (1.3× cap), not evidence | 13.2% in B |

Refusals are 0% for both columns here, **by design**: the guarded path only
runs on frames whose box mask STABILISE 7 accepted.

## 2. By context: the gain is mostly in A, and the arenas differ at baseline

| ears outside the mask | context A | context B |
|---|---|---|
| box only | 10.0% | **20.1%** |
| guarded points | **5.9%** | **15.8%** |
| plain points | 5.4% | 9.0% (with wall-ballooning) |

* **In A the guard keeps almost all of the plain prompts' gain**, without their
  over-inclusion.
* **In B it keeps little.** The guard rejects most of the point masks that
  would have restored the head there. The same bright top wall that drew the
  plain prompts onto itself sits next to the head, so an honest point mask and
  a runaway one are hard to tell apart by area.
* **At baseline, SAM's box-only mask clips the head twice as often in B as in
  A.** Head coverage is itself arena-dependent, and the guarded masks widen the
  ratio (5.9% against 15.8%).

## 3. Consequences

* **For the mask track: adopt the guarded prompts.** They cut head clipping by
  about a third overall and by about 40% in context A, with no measurable
  over-inclusion.
* **For any context comparison of a head-region feature, with any of these
  masks:** head coverage differs by arena, so the comparison must **condition
  on per-frame head coverage**. For example, restrict to frames where both ears
  are inside the mask, and report how many frames each context loses to that
  rule. Without it, an A-versus-B difference in a head feature can be an
  arena-dependent difference in how much head the mask contains.
* **Body-scale behaviours are much less exposed.** The body centre falls outside
  the mask on about 0.5% of frames and the hips on about 3%, in both arenas.
