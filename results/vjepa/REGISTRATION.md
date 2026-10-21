# Registration — V-JEPA 2 features for the mouse corpus (a new arm, extraction only)

Committed **alone**, before any code for this arm exists.

**Why now.** Cluster access ends around 2026-11-04 (`CLUSTER_STATUS.md`), and the
videos cannot be re-encoded after that. Step 3A found that the keypoint
representations carry no grooming information beyond speed
(`step3a/REPORT.md`, Amendment 1). Pixels are the next candidate.

**Scope.** This registration produces features only. **Nothing is analysed on the
cluster.** Every analysis on these features needs its own registration later, and
must include the controls fixed in §6.

## 1. Encoder

- **Model:** `facebook/vjepa2-vitg-fpc64-256` (ViT-g/16, 1,408-dim, tubelet 2).
  - Revision `875c192b7b704b87d1e1d99345769632dd5f739a`.
  - `model.safetensors` SHA-256
    `f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b`, checked at
    load; a mismatch stops the job.
- **Why 256 and not 384:** the mouse crop is about 255 px at native resolution
  (§3), so a 384 input would only upsample.
- **Inference:** `transformers.VJEPA2Model`, bf16, `skip_predictor=True`,
  last-layer patch tokens, in the environment `~/bjj-envs/video`.
- **Normalisation:** ImageNet mean and SD.
- **Batching:** batches never mix recordings.

## 2. Windows

- **Frame grid:** one global 8 fps grid per recording, `t_m = m / 8` s, at frame
  index `round(t_m × fps)` (fps from `spine.fps()`).
- **Window:** **4 s, 32 grid frames**, giving 16 × 16 × 16 tokens at 256 px.
- **Start times:** window k starts at `k` s. The **pooled stride is 1 s**. The **tune
  full-grid stride is 2 s** (even k).
- **Recording boundaries:** a window is kept only if all 32 frames lie inside one
  recording's valid range, `T = min(pose frames, video frames)`. The partial tail
  window is dropped. **No window, crop or smoothing step uses another recording.**

## 3. Crop: mouse-centred, rotated to the body axis, background masked

The pose is `spine.clean(rid)["pose"]` (T, 7, 2), using CENTER = 3 and
TAIL_BASE = 6.

- **Heading:**
  - the angle of CENTER → TAIL_BASE per frame, at the full frame rate;
  - unwrapped, median-filtered over 0.5 s (15 frames);
  - non-finite values linearly interpolated within the recording.
- **Crop:** a square of side **S = 2.5 × the recording's median body length**
  (`tego.body_length`, about 255 px), centred on CENTER.
- **Warp:** one affine warp maps the square to 256 × 256, rotated so that
  center → tail_base points **down** (+y). Bilinear interpolation for the frame,
  nearest-neighbour for the mask. Outside the image is filled with grey.
- **Mask:** the SAM keyframe masks (`work/masks/<rid>.npz`, every 6th frame).
  - **Frame t is covered** iff its bracketing keyframes are accepted, the same rule
    as `vieb/feat/streams.py` `mask_stream` (t on a keyframe needs only that
    keyframe).
  - The mask used is the nearer accepted keyframe's crop, translated by the
    linearly interpolated mask-centroid shift and dilated by **0.1 BL**.
- **Background:** pixels outside the dilated mask are set to the ImageNet mean
  grey (124, 116, 104 RGB). **An uncovered frame is left unmasked**, and that
  shows in its window's coverage.
- **Coverage:** per window, the fraction of its 32 frames that are covered,
  stored with every window. **Low coverage is < 0.9.**
  - Coverage differs by context (repaired-mask coverage is 98.9% in A and 91.5% in
    B, `INVENTORY.md`).
  - **Every later analysis reports results with and without low-coverage
    windows.**

## 4. Outputs

Under `work/vjepa/<tier>/`, per recording:

- **`<rid>.pooled.npy`** (n, 16, 1408) fp16: the spatial mean of the 16 × 16 patch
  grid, per temporal token. Every 1 s window.
- **`<rid>.grid.npy`** (n2, 16, 16, 16, 1408) fp16 memmap. **Tune only**, every 2 s
  window.
- **`<rid>.ids.json`**: the window start (s), coverage, frame indices, and the
  model and revision.
- **`<rid>.done`**: the resume marker, written last.

## 5. Coverage tiers (priority order; the cut order is in `CLUSTER_STATUS.md`)

1. **The tune split** (by `animal_tag`, `results/splits.json`): full grid plus
   pooled.
2. **The tune-sample controls** (§6): pooled, 4 s stride, on every tune recording.
3. **CFD Days 3–7, all animals** (A and B are paired on the same day): pooled.
4. **The rest of the corpus:** pooled.

## 6. Controls (extracted now, analysed later)

Each control is a feature arm with the identical encoder and window. Each is a
negative control for a later probe.

- **mask_only:** the warped dilated mask as a white silhouette on grey, with no
  pixels. *What the silhouette alone carries.*
- **background_only:** the mask inverted, so the animal's pixels are set to grey
  and the surround is kept. *What the context and arena image carries* (the
  A/B image confound, `step4/HSMM.md`).
- **frame_shuffled:** the window's 32 crops in a seeded random order (seed from
  `stable_seed("vjepa-shuffle", rid, k)`). *Appearance without temporal order.*

Registered now for later probes:
- a probe on these features must beat **mask_only** and **frame_shuffled** to claim
  pixel or motion content;
- it must report **background_only** beside any context result;
- it must report results with and without low-coverage windows.

## 7. Seam and correctness tests (before any run)

`tests/test_vjepa_crop.py`:
- **Seam:** a recording's windows, crops and coverage computed alone equal those
  computed in a run with another recording before it. No window indexes frames
  beyond its recording's T.
- **Rotation:** a synthetic pose with a known heading lands center → tail_base on
  +y.
- **Mask:** a known keyframe crop is reproduced; uncovered frames are left unmasked
  and counted as such.

**Smoke run:** 2 recordings, checking that the features are finite and the shapes
and coverage are as specified, before any array.

## 8. Benchmark (Day 0–1, recorded in `CLUSTER_STATUS.md`)

On 100 windows, measure:
- encoder windows per GPU-hour;
- CPU decode-and-crop seconds per window;
- bytes per window (pooled against grid).

The budget is set from these numbers. If it does not fit, tiers are cut from the
bottom.
