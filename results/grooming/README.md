# The grooming panel — §3 of `GROOMING_PREREGISTRATION.md`

**45 clips: 30 detector candidates and 15 speed-matched controls, shuffled.**
Which arm a clip came from lives only in `key.json`, which is **not published**
and should not be opened before scoring.

## How to score them

    cd results/grooming && python3 -m http.server 8000
    # then open http://localhost:8000/score.html

(A plain `file://` open will not load the manifest — browsers block `fetch` from
the filesystem. The one-line server is the whole workaround.)

Click **Grooming / Not grooming / Unsure** for each clip, press **Copy JSON**,
and save the result as `work/grooming/confirmation.json`. `grooming_gate.py
--phase combine` picks it up from there and publishes the confirmation rate as
the detector's own precision.

## Why controls are mixed in, unlabelled and at an unstated ratio

A rater shown only candidates has no way to be wrong, and a confirmation rate
computed that way measures nothing. With controls in the panel the rate can come
out at chance, which is the outcome that would falsify the detector.

## What this can and cannot settle

**The spectral gate already refused** — no arm of the registered grid reached
its 500-candidate minimum — so these clips are **not** evidence for or against a
3–8 Hz rhythm. What they settle is the prior question, which no statistic in
this repo can answer: **what are these windows?** The detector selects for a
still body with head-region pixel motion, and whether that is grooming, sniffing,
chewing, twitching or a tracking artefact is a claim about content.

Per §9.6: if the confirmed content is some other held-still-with-head-motion
behaviour, **that** is what gets named, not grooming.
