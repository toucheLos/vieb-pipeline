# Registration (optional; NOT approved to run): Keypoint-MoSeq's own synthetic null, scored with our dwell statistic

**Status:** written for the authors' decision (Paper 1 revision prompt, A1).
**Nothing here has been run, and nothing will run without explicit approval.**

## Question

Weinreb et al. (2024, Extended Data Fig. 2g,h) fit a linear dynamical system to
egocentrically aligned keypoint trajectories, sampled synthetic keypoints from it
(no block structure, no changepoints), and reported that Keypoint-MoSeq "failed to
identify distinct syllables". Paper 1's Finding 1 asks a different question: does
non-geometric **dwell** emerge? This analysis puts their synthetic condition through
our dwell statistic, so the two controls can be compared on the same input.

## Design (fixed before any run)

- **Synthetic input.** Their construction, as stated in their Methods: an LDS of the
  same form as keypoint-MoSeq's latent model, fitted to egocentrically aligned
  trajectories, then sampled. Fitted on the **tune** split; one synthetic recording per
  tune recording, at the same length. No recording boundary is crossed.
- **Labeller.** The labeller of record unchanged (κ = 1e6, the `scripts/moseq_kpms.py`
  protocol), fitted on the synthetic recordings and applied to them, **seeds 0–2**.
- **Statistic.** Exactly Finding 1's: recur `scripts/dwell_moseq.py measure()`. That is a
  per-animal Anderson–Darling-type geometric test at α = 0.05, the count of rejecting
  animals, and the median var/mean² of dwell.
- **Also reported, as their measure:** the number of syllables in use (more than 0.5% of
  frames) and the top-two share. This shows whether we reproduce "failed to identify
  distinct syllables" before we read dwell.

## Reading (fixed now)

- **Their control replicates and dwell still rejects in most animals.** The two controls
  measure different things. The sentence in Finding 1 stands, with the number added.
- **Their control replicates and dwell does not reject.** The LDS null differs from our
  surrogates in a way that matters. This is reported, and Finding 1 adds the contrast.
- **Their control does not replicate** (distinct syllables appear). `NOT_A_RESULT` for
  the comparison, reported as a replication failure of their panel on this corpus.

## Cost

3 kpms fits of about 55 minutes each on a GPU, plus an LDS fit on the CPU.
