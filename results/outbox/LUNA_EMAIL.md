# One email to Luna — both asks in it

Not sent. Drafted here so the asks are on the record with their evidence.

---

**Subject:** Two questions about the CFC recordings — what differs between
Context A and B, and the next cohort's camera

Hi Luna,

Two things — one I need an answer to, one for planning.

## 1. What is physically different between Context A and Context B, besides the odour?

The two contexts are measurably **different images**, not just different odours,
and I can't tell from the filenames what changed. Four independent measurements,
none of which assumes anything about behaviour:

- **Context A frames cost ~32% more bits to encode** than Context B frames,
  paired within animal and day (1,476 cells, 298 animals).
- The animals **move ~24% less** in Context A. Those two point in **opposite
  directions**, and that is what makes it conclusive — video bitrate goes *up*
  with motion, so motion cannot be what makes A more expensive.
- The **static background** of the same animal, same day, same box differs by
  **58–62 grey levels** between contexts, against **5–10** for same-context
  comparisons across days. Mean brightness is nearly identical (117 vs 115), so
  it is not lighting — it is **spatial pattern**.
- The camera's own **frame-to-frame noise floor** differs too: Context A is
  noisier by **1.30 [0.51, 2.02] grey levels**, paired within animal and day
  over 144 animals and 720 cells. That is consistent with A containing more fine
  spatial detail.

**So: is there a floor insert, a different wall pattern, different bedding, or a
different camera setting between A and B?** Even a one-line answer changes how I
have to analyse this — any pixel-based measure compared across contexts is
otherwise confounded on exactly the axis the main comparison uses.

Two smaller things I could not find documented anywhere:

- **Context A sessions run about 17% longer** than Context B (≈6,300 vs ≈5,400
  frames). Deliberate?
- Tracking dropout is about **6.8× more common** in one context than the other.
  A different-looking arena would explain that, which is part of why the first
  question matters.

## 2. For the next cohort: a side camera, a higher frame rate, and paw-visible views

Nothing needs to change for the existing data — this is only about what we
record next, and it costs nothing to decide now.

The limitation I keep hitting is that a **top-down view at 30 fps cannot see the
paws**. That rules out grooming, which is one of the behaviours most likely to
carry a real effect. In order of how much they would help:

1. **A side or angled camera**, even a cheap second one, roughly synchronised.
   This is the single biggest gain.
2. **A higher frame rate** — 60 fps or more. Grooming strokes run at roughly
   3–8 Hz, so at 30 fps we get under four samples per stroke, which is right at
   the edge of what can be measured.
3. **Any view in which the forepaws are visible** during rearing and grooming.

Happy to show you the plots behind any of this.

Thanks,
Carlos

---

## Provenance for every number above

| claim | source |
|---|---|
| 32% more bits, 24% less speed, 1,476 cells / 298 animals | the compression and speed measurements recorded in `PROGRESS.md` and this plan's context section |
| background 58–62 vs 5–10 grey levels, brightness 117 vs 115 | median-background comparison, same animal / day / box, 3 boxes |
| noise floor 1.30 [0.51, 2.02] grey levels | `PIXEL_NOISEFLOOR.md` §3, census of 1,440 recordings, paired within (animal, day), animal-level bootstrap |
| A sessions ~17% longer (6,302 vs 5,392 frames) | recording lengths; documented nowhere in any of the three repos |
| 6.8× dropout asymmetry | `CONTEXT.md` |
| 3–8 Hz grooming band, <4 samples per stroke at 30 fps | `GROOMING_PREREGISTRATION.md` §0 |
