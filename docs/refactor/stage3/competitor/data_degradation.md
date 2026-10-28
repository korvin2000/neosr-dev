# Stage 3 competitor harvest: lane `data_degradation`

T = neosr (`.worktrees/refactor`, `refactor/stabilize` at e0019ee). C = traiNNer-redux fork
(`/home/korvin/apps/traiNNer-redux-dev`, master 3a2d2549). CU = the upstream classic OTF chain that this fork
merged but did not keep (`git show 3a2d2549^2:traiNNer/models/realesrgan_model.py`, the-database master).

Evidence key: **paper** (publication; cited from memory, not re-read in this session) · **code** (read in C/CU/T) ·
**probe** (p1/p2 below, CPU only, capped at 4G, one at a time) · **config** (C's own option files) · **reasoning**.
No training run was made. Prior work this builds on, not repeated here: `analysis/competitor/H2.md` (data
pipeline, 15 port items, written 2026-10-20) and `stage2/analysis/data.md` (D-01..D-20, mostly fixed in stage 2).

**What changed since H2.** C merged upstream on 2026-10-21, after H2. In the fork, `RealESRGANModel.feed_data`
now runs only the "ParagonOTF" chain by Philip Hofmann. The Real-ESRGAN two-stage chain that H2 read is gone
from the fork. It survives only upstream (CU). New since H2: `SingleGtDataset` + `lq_resize_mode` (GT-only LQ
synthesis), a fixed `p_clean`, >8 bpc decode, nested dataroots and ECO (a teacher curriculum; out of this lane).

## 0. Verdict

- **Most of C's degradation additions are marketing, and some are harmful.** The fork's "physically accurate" chain
  has these defects:
  - Lens distortion **transposes the LQ against the GT, even at strength 0** (probe).
  - Rolling shutter and motion blur shift or warp the LQ only, so LQ and GT no longer line up.
  - Exposure and colour-temperature shifts on a 64 px patch ask the network to undo a change it cannot identify.
  - Demosaic, sensor noise and sharpening are applied at GT scale, so the final downscale largely averages them away.
  - The codec stages run per sample in the training process (GPU→PIL→GPU). They truncate instead of rounding, use
    one quality per batch, and silently return the input on any error.
  - The ffmpeg video path, the "sequences" controller, and the block/banding/ringing ops are all dead code. The
    controller is built but never applied.
  - The fork's own showcase config sets about 20 classic keys (`resize_prob`, `noise_range`, `jpeg_prob`,
    `blur_prob2`, ...). The fork's `feed_data` silently ignores them. Nothing in C's tree records a result for any
    of this.
- **The real gaps in T are few and cheap. All of them are in how the LQ distribution is sampled, not in exotic
  effects:**
  1. T's Degrader has **no antialiased or lanczos resampler**. Probe: the owner's own LQ matches antialiased
     Keys-cubic or lanczos2 at 52.7 dB. T's non-AA bicubic/bilinear modes give 31-34 dB and `area` gives 43.7 dB.
  2. T **always applies noise and JPEG twice**. No sample can be clean or JPEG-free, so a model trained this way
     tends to over-smooth clean inputs.
  3. T has **no sharpening halos at LQ scale**.
  4. T has **no dataset-level mixing** (several roots, weights, paired + OTF).
- **Curation is where most of the remaining quality is.** C's tools are a good idea with wrong metrics:
  - The blockiness metric wraps around in uint8 (probe).
  - The "aliasing" score is plain high-frequency energy, so it throws away exactly the detailed tiles SR needs.
  - NIMA (an aesthetics score) is used as a gate.
  - The filtering plan records tile counts but no training result.

  Rewrite the idea as one small scorer that writes `meta_info` lists, then let the owner's A/B harness decide.
- **Relevance to the owner today.** The owner's A/B data is paired and bicubic-like (no OTF), so no degradation item
  changes current results. They matter for real-world OTF training. The curation items apply to both.
- **The owner's held-out split is clean.** Exact 32x32 block hashes found no shared pixels between the 67 validation
  tiles and the 10540 training tiles (probe p2).

## 1. What each pipeline actually does

| step | T `Degrader` (`neosr/data/otf_degradation.py`) | CU (upstream classic) | C fork (`realesrgan_model.py:456-650`) |
|---|---|---|---|
| sharpen input | none | `lq_usm` (USM on the GT-res input, weight 0.5, thr 10) | `oversharpen` (5x5 box, GT res) |
| blur 1 | always, kernel1 (iso/aniso/gen/plateau/**skewed**, sinc) | `blur_prob` | `blur_prob`, kernel1 |
| resize 1 | up/down/keep, `area/bilinear/bicubic` uniform, **no AA** | `resize_mode_list/prob` (AA bilinear/bicubic, `nearest-exact`, "lanczos" = lanczos prefilter + bicubic) | removed |
| noise 1 | always (Gaussian or Poisson) | always (Gaussian or Poisson) | "sensor noise" = Gaussian, GT res |
| JPEG 1 | always (DiffJPEG) | `jpeg_prob` | removed (DiffJPEG constructed, never called) |
| stage 2 | blur `second_blur_prob`, resize, noise always | `blur_prob2`, mode lists, noise always | removed (`kernel2` loaded, unused) |
| final | [resize + sinc] and JPEG in random order | same, `jpeg_prob2`, `resize_mode_list3` | resize (`resize_mode_list3`) + sinc, then PIL JPEG/WebP/AVIF/HEIF + optional recompress |
| extras | color='y' degrades RGB first | `thicklines` (anime) | lens, CA, motion, demosaic, rolling shutter, exposure, colour temp, aliasing (nearest down-up), "editing" |
| clean path | none | none | `p_clean` (AA bicubic GT→LQ, now correct) |
| pair pool | indexed swap, uint8 storage, exact LUT decode | full-queue shuffle + clone, fp32, `.cuda()` | same as CU |
| key handling | validated once (`ConfigError`) | struct defaults | `hasattr` probing, so missing keys silently disable a stage |

Dead or broken code in the fork, with locations:

- `enable_sequences` builds a `SequenceController` (`realesrgan_model.py:64-72`) that `feed_data` never calls.
- `apply_video_compression` (`paragon_otf_degradations.py:580`) is reachable only through those sequences.
- The editing branch computes `sharpen_range` and applies nothing (`:609-616`).
- Chromatic aberration scales R/B by 1.001/0.999: about 0.25 px at the border of a 512 px canvas, so it is a
  no-op (`:485`, H2).
- The motion-blur kernel size comes from `random.randint` and can be even. With `padding=k//2` the output grows by
  a pixel, and that half-pixel shift survives the final resize. The padding is zero, so borders darken (`:251-294`).
- The block-artifact op is scalar quantisation inside a Python 8x8 loop, not a DCT (`:683`).
- The "ringing" op is Sobel × a sharpen kernel, not Gibbs ringing (`:776`).

## 2. Probes

Scripts are in the scratchpad under `s3data/`: `p1_probe.py` and `p2_leak.py`.

**p1.a: separable matmul resampler.** It follows the PIL algorithm: the kernel is stretched by the downscale
factor, truncated at the border and renormalised. Parity on random data:

| torch mode | matmul kernel | max abs diff |
|---|---|---|
| `bicubic, antialias=True` | Keys cubic a=-0.5 | 4.4e-16 (identical in float64) |
| `bicubic, antialias=True` | a=-0.75 | 1.9e-2 |
| `bilinear, antialias=True` | triangle | 2.2e-16 (identical in float64) |

So torch's AA bicubic is the PIL/Matlab kernel. One 40-line function covers every kernel below.

**p1.b: which downscaler made the owner's LQ** (48 pairs, 16-bit, mean PSNR in dB, worst pair in brackets):

| downscaler | mean PSNR (worst) |
|---|---|
| lanczos2 AA | 52.71 (37.8) |
| torch bicubic AA = cubic -0.5 AA | 52.64 (37.8) |
| cubic -0.75 AA | 52.07 |
| lanczos3 AA | 48.42 |
| Mitchell AA | 44.22 |
| `area` = box AA | 43.71 |
| triangle AA | 42.39 |
| torch bilinear (no AA) | 34.07 |
| **torch bicubic (no AA), T's Degrader mode** | **31.16** |

**p1.c: C lens distortion at strength 0** (the lines from `:313-342` copied as they are, on a 5x5 ramp):

- max|out − in| = 16 (the input values run 0 to 24);
- max|out − inᵀ| = 1.5.

The grid feeds a row-varying coordinate as `x`, so the output is the transposed image, slightly resampled. The
showcase config sets `lens_distort_prob: 0.25` (`options/train/ParagonSR2/fidelity/2xS_real_photography_showcase_copy.yml:118`).

**p1.d: C `compute_blockiness`** (`stage1_prefilter.py:90`, copied as is):

| input | score |
|---|---|
| smooth increasing ramp | 0.22 |
| the same ramp flipped | 27.8 |

`uint8 - uint8` wraps around, so the metric depends on gradient sign, not on blocking.

**p1.e: real-codec cost per image**, single thread:

| codec | 64 px | 128 px |
|---|---|---|
| cv2 JPEG | 0.04 ms | 0.08 ms |
| cv2 WebP | 0.32 ms | 1.04 ms |
| PIL AVIF | 3.3 ms | 9.5 ms |

A batch of 16 at 128 px therefore costs about 150 ms for AVIF and 17 ms for WebP. That is affordable in loader
workers or offline. It is not affordable synchronously in the training step (T's compact OTF step is 73.8 ms).

**p2: does the A/B validation set share pixels with the training set?**

- Exact hashes of textured 32x32 blocks on a 16 px grid, all 67 val GT against all 10540 train GT: 0 shared blocks.
- Inside the first 1000 train tiles: 0 shared blocks, so the pool has no overlapping tiles.
- dHash ≤ 4 gave 10 weak hits. Three were inspected: different images with a similar coarse layout, not copies.
- Conclusion: the A/B harness has no tile-overlap leakage.

## 3. Ranked candidates (by expected quality gain for real-world SISR per unit of effort)

### P1

**C1. One resampler with a kernel table, chosen per stage (AA cubic, AA triangle, box/area, lanczos2/3, Mitchell,
nearest, plus the legacy non-AA torch modes).**

- *Value:* resampling-kernel mismatch is the classic failure of non-blind SR (blind-SR literature, **paper**).
  T's OTF cannot draw the most common real-world downscaler class:
  - PIL, Matlab, ffmpeg and browsers all antialias.
  - The owner's LQ sits at 52.7 dB to AA cubic/lanczos2, against 31 dB for T's bicubic mode (**probe**).

  Honest caveat: T's first-stage Gaussian blur (σ 0.2-3 at GT scale) partly acts as an AA prefilter, so the
  effective gap is smaller than the 31 dB figure.
- *C quality:* CU's `resize_pt` (`degradations.py:1004`) forces AA on for bilinear/bicubic, so it loses the non-AA
  modes. Its "lanczos" is a lanczos prefilter followed by non-AA bicubic (H2), and `_ramp` is a Python loop.
- *Port:* rewrite from the PIL reference.
  - `resample(x, size, kernel)` builds one (out × in) weight matrix per axis, vectorised in torch, and applies two
    matmuls. That is about 1 GFLOP for b16 at 512→128, so negligible.
  - The kernel table is an explicit dict.
  - `resize_mode{,2,3}` / `resize_mode_prob{,2,3}` default to today's `area/bilinear/bicubic` non-AA list, so
    current behaviour is unchanged.
  - Parity tests against torch AA bicubic/bilinear, at 1e-6 in fp32.
  - Note: under `fast_matmul` (TF32) it is about 1e-3 relative, which is harmless before 8-bit quantisation.
- *Fit:* a lookup table and validated keys. *Cost:* S.

**C2. Per-operation probabilities and a clean path in the Degrader.**

- *Keys:* `blur_prob` (stage 1), `noise_prob{,2}`, `jpeg_prob{,2}` (`second_blur_prob` already exists), and
  `clean_prob`. A clean sample is LQ = quantised AA resample of the GT, with no blur, noise or JPEG.
- *Value:* T applies noise twice and JPEG twice to every sample (`_noise` always picks Gaussian or Poisson; `_jpeg`
  always runs when `jpeg_compress`). A clean PNG, game capture or AA-downscaled photo is therefore out of
  distribution, and the model learns to treat fine texture as noise or blocking. This is the "waxy texture"
  failure mode of Real-ESRGAN-style training. Evidence is **reasoning** plus **paper**: BSRGAN applies each
  degradation with a probability. CU has `blur_prob`/`jpeg_prob` and the fork has `p_clean`; C's own configs use
  `jpeg_prob` 0.65-0.9.
- *C quality:* the fork's `p_clean` is now correct (AA bicubic GT→LQ, quantise, crop, pool), but it is one more
  `hasattr` branch. CU's probabilities are fine.
- *Port:* port the idea and fix it into T's `_STAGE_KEYS` table with defaults of 1.0/1.0/1.0/0.0, so behaviour is
  unchanged. Each coin is per batch, as resize already is, and the pair pool mixes batches.
- *Fit:* clean. *Cost:* S.

**C3. A curation scorer that writes `meta_info` selections (no training-code change).**

- *What it computes per image or tile, into a CSV:*
  - **SR headroom**: PSNR or 1−SSIM between the tile and its own AA down→up round trip at the training scale. This is
    the learning signal per patch, no model needed, and the same idea as "informative patch" sampling (SamplingAug,
    BMVC 2021, **paper**).
  - Laplacian variance (blur).
  - A **correct** blockiness ratio: mean |∇| on 8-px boundaries over mean |∇| inside the blocks, in int32.
  - Immerkær noise σ.
  - Clipped and flat fractions.
  - Optional IC9600 complexity and pyiqa NR-IQA columns when installed.
- *Selection:* by quantile or threshold, written as `meta_info` text files that T's datasets already read.
- *Value:* the GT quality caps the output quality, and GAN training reproduces GT artifacts. Curated data beats more
  data: LSDIR, and Real-ESRGAN's multi-scale DF2K (**paper**). APISR (CVPR 2024) selects training frames with
  IC9600 (**paper**). The gain per patch is real, but its size on the owner's homogeneous 10.5k-tile set is unknown.
  Decide it on the A/B harness: full set vs. drop the bottom 20-30 % by SR headroom (same updates) vs. score-weighted
  sampling (needs C5).
- *C quality:* three standalone scripts (`datasets/preparation/filtering/stage{1,2,3}_*.py`, `complexity/*`).
  - The blockiness metric wraps (**probe**).
  - The "aliasing" score is the share of FFT magnitude outside 0.75 of Nyquist. It rejects sharp textures, which
    are exactly the useful tiles.
  - The Laplacian-variance upper bound (3000) rejects rich texture.
  - NIMA is an aesthetics score, irrelevant to SR targets.
  - All three read through `convert("RGB")`, so 16-bit sources drop to 8 bit, and save tiles by truncation.
  - The 81 MB `complexity.pth` is vendored with no license file next to it. The IC9600 weights' license must be
    checked before redistribution: fetch them at run time, do not vendor them.
  - The plan doc lists only tile counts and claims "high complexity accelerates convergence" without numbers.
- *Port:* rewrite from the references. *Fit:* `scripts/`, outside the training code. *Cost:* M.
- *Risk:* dropping flat patches can leave flat areas unlearned, so GAN noise appears in skies and gradients. Prefer
  down-weighting to deletion, and keep a floor share of low-score tiles.

### P2

**C4. Multi-scale GT tiling in `scripts/extract_subimages.py`** (`--scales 1,0.75,0.5`, AA downscale before tiling).

- *Value:* downscaling camera images removes noise, demosaic and JPEG residue and raises the sharpness per pixel.
  Real-ESRGAN built DF2K-OST this way (**paper/practice**), and C's stage-1 script does it (0.75/0.5/0.25 plus an
  adaptive "min512" pass).
- *C quality:* the resize itself is fine (cv2 `INTER_AREA` in float), but the output is 8-bit truncated.
- *Port:* idea only; a ~20-line addition to T's tiler, with tiles named per scale. *Cost:* S.

**C5. Multi-source training data: list dataroots, a weighted batch-homogeneous source sampler, paired + OTF mixing.**

- *Value:* this lets the owner:
  - mix domains with explicit weights;
  - up-weight a small, very clean set;
  - mix real or real-codec paired LQ (C7) with OTF batches.

  Evidence is **reasoning** and common practice; there is no controlled number.
- *C quality:*
  - `dataroot_*` lists are plain concatenation, with no weights.
  - `RealESRGANPairedDataset` (`data/realesrgan_paired_dataset.py`) loads **both** a paired and an OTF sample for
    every index, so half the I/O is wasted. Its length is the OTF length, so a shorter paired set raises
    IndexError. `RealESRGANPairedModel` flips one coin per batch, and paired batches bypass the pair pool.
- *Port:* idea only.
  - `datasets.train.sources = [{dataroot..., weight, (lq or otf)}]`.
  - A small `BatchSampler` draws the source for each batch from a seeded generator (`seed + epoch`), takes the
    indices from that source's own permutation, and is rank-aware for DDP.
  - The `otf` model skips the Degrader when a batch carries `lq`. That branch is in `feed_data`, not in the loss
    loop, so the loss Context rule holds.
  - The same sampler later serves score-weighted sampling (C3).
- *Cost:* M.

**C6. Sharpening halos at LQ scale.**

- *Value:* web and broadcast images are usually sharpened **after** the final resize. A model that never saw halos
  treats them as edges and upscales them into double edges (**reasoning**/practice).
- *C quality:* both C variants sharpen at GT scale before the 1/scale downscale, which averages the halos out:
  - CU `lq_usm` uses a fixed weight of 0.5;
  - the fork uses a 5x5 box, which its own comment calls a placeholder.
- *Port:* idea only. USM (Gaussian σ 0.5-2 LQ px, amount 0.3-1.5, small threshold) right after the final resize +
  sinc, with `sharpen_prob` defaulting to 0. *Cost:* S.

**C7. `scripts/degrade.py`: offline pairs and fixed OTF validation sets from the same dataset kernels + Degrader,
with an optional real-codec tail.**

- *Codecs:* cv2 JPEG with an explicit chroma sampling factor, cv2 WebP, PIL AVIF, and ffmpeg x264/x265/AV1 intra
  with `yuv420p`. Each codec is checked at startup and fails loudly.
- *Value:*
  - a reproducible validation set for OTF runs, which today need a hand-made paired val;
  - degradation previews, which catch config mistakes such as C's transposed LQ;
  - real WebP/AVIF/H.264 artifacts (APISR and RealBasicVSR found real codecs matter for anime and video, **paper**),
    fed in as paired sources through C5.
- *C quality:*
  - `scripts/create_validation_otf.py` re-implements the training chain in 478 lines, so it drifts by construction.
  - `helpful-scripts/Hofmann_scripts/create_video_lr.py` works, but it encodes one frame. Its docstring promises
    P/B-frame and motion artifacts that an intra-only encode cannot contain.
  - The in-loop PIL codecs are too slow and silent on failure (p1.e and §0).
- *Port:* idea only. Reuse `otf._load` and `Degrader` with a seed instead of duplicating them. *Cost:* M.

**C8. Near-duplicate and leakage check** (a subcommand of C3): exact block hashes plus dHash, between splits and
inside a set.

- *Value:* it protects every A/B decision, and it removes the duplicate weight from scraped sets.
- C has nothing equivalent.
- p2 shows the current split is clean, so this is for future datasets. *Cost:* S.

### P3

- **C9. Chroma realism.**
  - DiffJPEG upsamples chroma by pixel replication (`utils/diffjpeg.py:391-417`), whereas libjpeg decoders default to
    "fancy" triangle upsampling, so T's JPEG chroma edges are blockier than real ones.
  - Add a bilinear option, plus a standalone `chroma_prob` op (YCbCr 601/709, 4:2:0 or 4:2:2 via avg-pool,
    nearest/bilinear up) for high-quality video-like sources.
  - C only has an offline script (`scripts/data_preparation/create_lrs_bicubic_chroma_subsample_downscale.py`). It
    needs `chainner_ext` and casts Catmull-Rom output to uint8 without a clip (unverified for out-of-range values).
  - Opt-in. S.
- **C10. Configurable kernel ranges** (`kernel_range{,2}`, `final_kernel_range`; H2 #10). T hard-codes 7..21. CU makes
  the ranges configurable but still pads to 21, so a range above 21 crashes `np.pad`. Port and pad to the shared
  maximum. S.
- **C11. Motion-blur kernels in the kernel table** (odd size, centred, anti-aliased line built in the worker; reuses
  `filter2D`, no GPU cost). C's version is buggy (§1). Rewrite from the reference. S.
- **C12. Bit-depth reduction and dither** (uniform or ordered Bayer) at LQ scale, for debanding models. C's banding
  op is correct but trivial and only reachable from dead code. It has no dither. S.
- **C13. GT-only resize mode** (C `lq_resize_mode` + `SingleGtDataset`). With C1 this is a resize-only degrader of
  about 10 lines. T's canvas margin and `_quantize` already avoid C's two defects: C resizes after cropping, so the
  borders see the crop edge, and it never quantises the LQ to 8 bit. The probe says the owner's LQ ≈ AA cubic
  (52.6 dB), so this is a storage and scale convenience, not a quality item. S.
- **C14. >8 bpc decode** (C f809c259 passes the vips dtype through). For SR fidelity this is negligible: the 8-bit
  quantisation floor is about 59 dB, against models at about 30 dB. It matters only for debanding or 16-bit outputs.
  It costs about 1.65x decode time on the owner's 16-bit PNGs (D-14). S.
- **C15. In-loop OTF debug dump** (C `high_order_degradations_debug`: writes into a cwd-relative dir and rescans
  filenames every batch). C7's `degrade.py --count 32` gives the same visibility without touching the loop.

### Skip

- **Fork ParagonOTF chain:**
  - Lens distortion (transposes the image, p1.c), rolling shutter and motion blur: LQ-only geometry, so LQ and GT
    are misaligned and pixel losses blur.
  - Exposure and colour temperature: a global change on the LQ only, not identifiable from a patch, which teaches
    colour shifts.
  - Demosaic, sensor noise, oversharpen and aliasing: applied at GT scale and attenuated by the final downscale.
    Demosaic is also per-sample CPU work.
  - Chromatic aberration: a no-op.
  - Editing: dead code.

  Take only the ideas that fit the right scale: C6, C11, C12.
- **In-loop PIL codecs and the ffmpeg video path; sequences; block/banding/ringing ops.**
  - The PIL codecs have a sync per batch, cost about 150 ms per AVIF batch, truncate instead of rounding, use one
    quality per batch, and silently return the input on error.
  - The ffmpeg path and the sequences are dead code.
  - Real ringing already comes from T's sinc kernels. Real codecs belong in C7.
- **`dynamic_dataloader_wrapper.py`** (VRAM-driven batch and patch changes) and **`utils/dataset_analyzer.py`** (it
  only feeds the DynamicLossScheduler): the owner has rejected that automation.
- **Video datasets** (`paired_video_dataset.py`, `single_video_dataset.py`): T has no temporal archs.
- **ThickLines** (CU, anime): it darkens lines on the LQ only. Revisit only for an anime model.
- **C's filtering and complexity scripts as code**, and the vendored `complexity.pth`: rewrite as C3.
- **`RealESRGANPairedDataset`/Model as code**: rewrite as C5.
- **`create_validation_otf.py` as code**: rewrite as C7.
- **pyvips crops** (H2 deferred): T's pre-tiling covers the large-image case.
- **ECO**: a teacher-target curriculum; it belongs to the training-recipe lanes.

## 4. Degradation gaps that neither repo covers (ideas only, P3)

Neither repo covers these:

- BSRGAN-style random order of degradations (**paper**: generalisation gain in BSRGAN's ablation).
- A camera-ISP noise model, with signal-dependent read/shot noise in linear raw and demosaic at capture resolution.
- Screentones and halftoning.
- Palette (GIF) quantisation.
- JPEG with a misaligned grid after a crop. T's JPEG → resize → JPEG covers this partly.

Each one is domain-specific. Add one only when the target data shows the artifact.

## 5. What T already does better

- **Pair pool.** It does an indexed swap of b slots. It stores uint8 (4x smaller and exact through the producer's
  LUT) and is device-agnostic. C shuffles and copies the whole fp32 queue every batch, clones several times and
  hard-codes `.cuda()`.
- **Degradation keys** are validated once with a `ConfigError`. C probes with `hasattr` and silently ignores keys:
  its own showcase config sets about 20 classic keys that the fork never reads.
- **Kernels.** T has `skewed`/`skewed_aniso` kernels, which C lacks, and kernel keys are checked before the workers
  start.
- **color='y' OTF** degrades RGB first, so the chroma JPEG artifacts are real, then converts to luma.
- **Padding.** The canvas uses reflect-101 padding, whereas C's `embed` zero-pads small images, which puts black
  borders into the GT. `tight_canvas` (opt-in) matches C's gt+32 canvas and also keeps it a multiple of the scale.
- **Quantisation.** The LQ is rounded (`_quantize`). The fork's PIL, demosaic and codec paths truncate, a bias of
  -0.5 to -1 level per pass.
- **Host syncs.** Noise is sync-free (D-05), cuDNN autotune storms are suppressed in the Degrader, and nothing in
  the degradation path makes a per-sample GPU↔CPU round trip.
- **Loader.**
  - Worker RNG is seeded per worker and per epoch from `get_worker_info().seed`; C replays the same stream every
    epoch.
  - `train_retry` skips bad files with the path logged.
  - `PathList` keeps worker memory flat.
  - Sorted pairing, affinity-aware `auto` workers, and persistent workers with pinned memory by default.
- **Batch augmentations** run at native resolution with a correct mixup; C's ResizeMix swaps H and W.
- **DiffJPEG** is device-agnostic, with a vectorised quality→factor mapping.

## 6. Suggested order

1. C2 + C1 (S + S): one Degrader change behind defaults that reproduce today's output exactly. Add parity tests and
   a test that the defaults are bit-identical.
2. C3 + C8 (M + S): the scorer, then one A/B on the owner's harness (full set / bottom 25 % by SR headroom removed /
   score-weighted). The weighted arm needs C5's sampler.
3. C4 (S): multi-scale tiling when the next GT set is built.
4. C6 (S): sharpening, opt-in.
5. C7 (M): `degrade.py`, which also gives OTF runs a fixed validation set and previews.
6. C5 (M): multi-source sampler.
7. P3 items on demand.

Per the owner's rule 2, every distribution change ships opt-in. Before any template default flips, the A/B for the
OTF items needs a **real-world** validation set, not the bicubic val67: for example C7 output with real codecs, or
real LQ with no GT, judged by NR metrics or visually.

## 7. Not verified

- No GPU timing for the matmul resampler. The FLOP estimate is by hand.
- The paper effects are cited from memory.
- The IC9600 weight license has not been checked.
- The out-of-range behaviour of C's `chainner_ext` cast was not run.
- The effect of C1/C2/C6 on quality is reasoning until A/B-tested.

## 8. Incidental (outside the lane)

The `origin` remote URL in `/home/korvin/apps/traiNNer-redux-dev/.git/config` embeds a GitHub personal access token
in plain text. Rotate it and switch to a credential helper. The value is not reproduced here.
