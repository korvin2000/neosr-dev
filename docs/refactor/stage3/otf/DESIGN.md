# OTF realism (stage 3 Phase B): design

Status: **draft 2, 2026-11-10** (Fable 5.2 on the Windows machine; revised after the Codex gpt-6.1-sol critique, section 10).
No code written. Measured here: only the resampler formula of D1 (CPU probe).
Inputs: `../competitor/data_degradation.md` (lane report with probes; ids C1-C15 below), `../CONSENSUS.md` Phase 2,
`../REVIEW.md` Phase B, `../../stage2/analysis/data.md` (D-02, D-16), the code at 92081c6
(`neosr/data/otf_degradation.py`, `otf_dataset.py`, `models/otf.py`, `utils/diffjpeg.py`).
Owner goal (DECISIONS 2026-10-28): real-world photographs are the main target.

## 0. Scope
| in | out (and why) |
|---|---|
| how the LQ distribution is sampled: resampling kernels, per-operation probabilities, a clean path, sharpening at LQ scale | LQ-only geometry (lens, rolling shutter, motion warp): misaligns LQ and GT |
| an offline `degrade.py` that reuses the training code: fixed validation sets, previews | exposure / colour-temperature shifts: not identifiable from a patch |
| the A/B plan for the Linux machine | real codecs inside the training step (sync, ~150 ms per AVIF batch); they belong in `degrade.py` |
| | multi-source and paired+OTF sampling (C5), curation scorer (C3): separate Phase B items, sketched in 8 |

Rules that bind the design (owner): every distribution change ships **opt-in** with defaults that reproduce today's
output; a template changes only after an A/B on the owner's data; no automatic damping or adaptive schedules.

## 1. What the Degrader does today and where it misses real inputs
Chain per batch: blur1 (always) -> resize1 (up/down/keep, `area|bilinear|bicubic`, **no antialiasing**) -> noise1
(always, Gaussian or Poisson) -> JPEG1 (always) -> blur2 (`second_blur_prob`) -> resize2 -> noise2 (always) ->
{[resize to LQ + sinc], JPEG2} in either order -> 8-bit quantisation.

| # | gap | evidence | consequence |
|---|---|---|---|
| G1 | no antialiased / lanczos resampler | lane probe p1.b: the owner's own LQ matches AA cubic or lanczos2 at 52.7 dB; the Degrader's bicubic at 31.2 dB, `area` at 43.7 dB | the most common real downscaler class (PIL, ffmpeg, browsers, Photoshop) cannot be drawn |
| G2 | noise twice and JPEG twice on every sample; blur1 always | code: `_noise` and `_jpeg` have no skip | no clean or JPEG-free sample exists: fine texture is learnt as noise or blocking ("waxy" output on good inputs) |
| G3 | no sharpening at LQ scale | code | halos of web and camera sharpening are upscaled into double edges |
| G4 | the canvas is >= 512 px | D-02: on the owner's 256 px tiles 75 % of the degraded GT pixels are mirror padding, 96.9 % of patches hold a seam; `tight_canvas` exists, opt-in, never A/B-tested | wasted work and symmetric seams in training targets |
| G5 | OTF runs have no fixed synthetic validation set and no real-LQ set | templates validate on a hand-made paired set | no OTF claim can be tested today (CONSENSUS risk list) |
| G6 | DiffJPEG pads with zeros and upsamples chroma by replication | D-16, C9 | border and chroma-edge artifacts differ from libjpeg; small |

G1-G3 are distribution gaps, G4 is an existing switch waiting for its A/B, G5 is tooling, G6 is fidelity of one op.

## 2. Decisions
### D1. One separable resampler with a kernel table (G1)
`resample(x, size, kernel) -> Tensor` in a new `neosr/data/resample.py`: the PIL algorithm, vectorised.
For one axis with `n` input and `m` output samples, `s = n / m`, `t = max(s, 1)` (the kernel is stretched only when
downscaling), centre `c_i = (i + 0.5) s - 0.5`:

    W[i, j] = k((j - c_i) / t),   rows normalised to sum 1 (this is also the border rule: truncate, renormalise)
    out = W_h @ x @ W_w^T          (two matmuls on (b, c, h, w); weights built in float64, cast to x.dtype)

| name | kernel `k`, support radius | models |
|---|---|---|
| `aa_box` | box on the half-open support [-0.5, 0.5) | area averaging (equals torch `area` only for integer factors) |
| `aa_bilinear` | triangle, 1 | torch `bilinear, antialias=True` (parity oracle) |
| `aa_bicubic` | Keys cubic a = -0.5, 2 | PIL / torch `bicubic, antialias=True` (parity oracle; torch uses a = -0.75 only without antialiasing); the owner's LQ |
| `aa_mitchell` | Mitchell-Netravali B = C = 1/3, 2 (not interpolating: it smooths slightly at any size) | ImageMagick's enlargement default |
| `aa_lanczos2`, `aa_lanczos3` | windowed sinc, 2 and 3 | the lanczos family of ffmpeg / ImageMagick / editors; the owner's LQ (lanczos2) |
| `area`, `bilinear`, `bicubic` | unchanged: `F.interpolate` without antialiasing | today's modes, kept bit-identical |

The names stand for kernel families, not for exact application pipelines (Matlab, for one, pads symmetrically
instead of renormalising at the border).

Why not `F.interpolate(antialias=True)` alone: it has two kernels; the table costs ~40 lines and gives the
diversity a blind model needs. Verified on this machine (torch 2.9.1 CPU, fp64, down / up / mixed sizes): the
formula equals torch `antialias=True` to 2.3e-15 for cubic -0.5 and 3.3e-16 for the triangle (Codex's own probe:
7.8e-16 / 2.2e-16, fp32 3.0e-7). Cost: two dense matrices of at most ~768 x 512 and two matmuls per resize;
expected well under 1 ms per batch (O6 measures it).

Contracts:
- **Size, not factor.** The legacy modes keep today's `F.interpolate(scale_factor=...)` call unchanged (torch maps
  coordinates with the factor). An AA mode computes the same output size `floor(n * factor)` and maps coordinates
  with `n / m`; the two differ (Codex probe: 0.47 on a 7 x 9 input at factor 1.3), which is intended and only
  reachable through the new names.
- **An unchanged axis is not resampled** (`m == n`: identity), so "keep" stays a no-op for every kernel including
  Mitchell.
- **TF32.** Under `fast_matmul` the matmuls run in TF32 and some pixels round to a neighbouring 8-bit level, so
  LQ differs between CPU and CUDA runs. Tests run on the CPU; O6 measures the flip rate on CUDA and the owner
  decides whether the resampler pins fp32 matmul precision.

Keys (per stage, list of names): `resize_mode`, `resize_mode2`, `final_resize_mode`, each with an optional
`*_prob` weight list of the same length. Absent: today's tuple and today's `random.choice` (D6).

### D2. Per-operation probabilities (G2)
| key | default | where the coin is drawn | granularity |
|---|---|---|---|
| `blur_prob` | 1 | dataset worker: a pulse kernel replaces `kernel1` (the mechanism `final_sinc_prob` already uses) | per sample |
| `second_blur_prob` | (existing) | Degrader | per batch, unchanged |
| `noise_prob`, `noise_prob2` | 1 | Degrader, torch device RNG, one mask per batch | per sample |
| `jpeg_prob`, `jpeg_prob2` | 1 | Degrader, torch device RNG | per sample |

Per sample, because noise and JPEG already draw their strength per sample and a mask is one `torch.where`; the op
still runs on the whole batch (cheap), which keeps shapes static and needs no host sync.
Boundary rules: a skipped noise row is clamped to [0, 1] exactly as the noise functions clamp an applied row (an
intermediate image is always in range); JPEG keeps its input clamp and contiguous output; with
`jpeg_compress = false` the JPEG gates return before any mask or quality draw. `gaussian_noise_prob` keeps its meaning
(Gaussian vs Poisson **given** noise); the readme row must say so, the names are close.
`jpeg_compress = false` stays the global off switch.

### D3. A clean path (G2)
`clean_prob` (default 0, per sample) with `clean_mode` (default `["aa_bicubic"]`, optional `clean_mode_prob`):
`lq = quantize(resample(gt, (h / scale, w / scale), mode))`, no blur, noise or JPEG. Mixed into the batch with
`torch.where` **on the RGB result before the existing quantisation**, so `color = 'y'` (luma conversion and
re-quantisation in `degrade`) and the pool's uint8 contract are untouched. Shortcuts follow the **configured**
probability only (`clean_prob == 0`: no resample, no draw; `== 1`: no chain); for 0 < p < 1 both paths are computed.
Looking at the mask (`any()` / `all()`) would be a host sync per batch.
Independent probabilities alone do not give this sample: the chain still resamples three times with random factors.

### D4. Sharpening at LQ scale (G3)
Unsharp mask right after the final resize + sinc, in both orders (so it sits before the last JPEG in one order and
after it in the other, as in real pipelines): `out + a (out - G_sigma * out)`, clamped. Keys: `sharpen_prob`
(default 0, per sample), `sharpen_sigma = [0.5, 2.0]` (LQ px), `sharpen_amount = [0.3, 1.5]`; parameters per sample
from the torch RNG; one `filter2D` with a per-sample 13 x 13 Gaussian (`filter2D` already takes per-sample kernels).
With `sharpen_prob = 0` the step returns before any draw or filtering. No threshold parameter: sharpening noise
and JPEG blocks is what re-sharpened downloads look like, but the `degrade.py` previews of O4 must confirm it
looks like real inputs before any run uses it.

### D5. Canvas (G4): no new code
`tight_canvas = true` is required for the owner's 256 px tiles and is part of every arm of the A/B except the
legacy baseline (7). Its own effect is arm B1 vs B0. D-16 (replicate padding in DiffJPEG) joins as an opt-in key
`jpeg_pad = "replicate"` only if B1 previews show border artifacts; otherwise it stays in the backlog.

### D6. Compatibility contract (the test that guards it exists already)
With none of the new keys set the Degrader is **bitwise** today's, including RNG consumption:
- **existing** draws, branches and their order are not touched: the second-blur coin, the Gaussian/Poisson choice
  and the worker's sinc choices keep drawing at probability 0 or 1, as today;
- a **new** gate with `p >= 1` or `p <= 0` draws nothing (`_coin`), so python `random`, `neosr.rng` (numpy) and
  torch's CPU and device generators advance exactly as now;
- an absent `resize_mode*` uses the same tuple and the same `random.choice`; a present list uses
  `random.choices(names, weights)`;
- `tests/test_otf_degradation.py::test_equals_reference_on_three_batches` (the frozen pre-refactor reference)
  stays unchanged and must stay green. It compares outputs only, so a new test also compares the **RNG states**
  (python, numpy, torch CPU, torch CUDA when present) after a default-config batch against the frozen reference,
  for the Degrader, the dataset worker and `jpeg_compress = false`.
New draws come from RNGs the trainer already saves on rank 0, so resume semantics do not change (stochastic
continuation, as documented).

### D7. Validation of the keys
`Degrader.__init__` checks names against the table, list lengths, probabilities in [0, 1], ordered positive ranges
and raises `ConfigError` naming the key (as today); `config._DEGRADATIONS` gains the keys with their types, so an
unknown or misspelt key still warns. `scripts/studio_surface.py` then regenerates the Studio surface and the Studio
contract test lists the fields to add to `studio/shared/catalog/degradations.ts`.

### D8. Rejected or deferred
| idea | reason |
|---|---|
| BSRGAN-style random order of all operations | deferred, not refuted: an opt-in `shuffle` is possible later; it multiplies the arms of the first A/B and some orders give the "twisted lines" the current code avoids |
| per-batch coins for noise/JPEG | simpler, but the pool (180 pairs, batch 8) then holds runs of identical recipes; per-sample masks cost one `where` |
| a `trace` of the drawn parameters per sample | not in the training path; stratified evaluation uses profile-specific sets (4). `degrade.py` may record its own draws later if a set needs per-sample strata |
| camera noise in linear space with denoising traces, centred motion-blur PSFs (C11: aligned, unlike a warp) | **evidence-gated, likely needed for photos of people**: today's Poisson term scales with the count of distinct 8-bit levels (`degradations.py:698`), it is not a camera model. Decided after Q1 and the residual analysis of 4.1 |
| halftone, palette, exposure / white-balance shifts | deferred until the target data shows them |

## 3. The chain after the change (new parts marked +)
    k1 = pulse with prob 1 - blur_prob                                   + (worker)
    out = filter2D(gt, k1)
    out = resize(out, factor1, mode ~ resize_mode)                       + AA kernels selectable
    out = where(mask(noise_prob), noise(out), clamp(out))                +
    out = where(mask(jpeg_prob), jpeg(out), out)                         +
    out = filter2D(out, k2) with prob second_blur_prob
    out = resize(out, size2, mode ~ resize_mode2)                        +
    out = where(mask(noise_prob2), noise(out), clamp(out))               +
    A: out = usm(sinc(resize(out, lq_size, mode ~ final_resize_mode)));  out = where(mask(jpeg_prob2), jpeg(out), out)
    B: out = where(mask(jpeg_prob2), jpeg(out), out);  out = usm(sinc(resize(out, lq_size, ...)))
    lq = quantize(where(mask(clean_prob), resample(gt, lq_size, clean_mode), out))   +

## 4. `scripts/degrade.py` (G5)
One script, no second implementation of the chain: it builds the `otf` dataset and the `Degrader` from a training
TOML (or a `[degradations]` profile file), seeds the three RNGs, runs single-process and writes `gt/`, `lq/` (PNG)
and `manifest.json` (seed, commit, the effective degradation keys, file list).

    python scripts/degrade.py -opt options/train_x_otf.toml --out data/val_v2 --count 300 --seed 2026 \
        [--meta split/dev.txt] [--profile options/degrade/web.toml] [--codec cv2] [--device cpu]

- `--codec cv2`: the last JPEG is a real libjpeg encode (cv2, 4:2:0, the drawn quality) instead of DiffJPEG, so a
  validation set is not produced by the simulator the model trained on. Checked at start-up, fails loudly.
- Output depends on the device and the torch version (CUDA vs CPU kernels): the **files** are the validation set;
  they are generated once on the Linux machine and never regenerated for a comparison.
- Doubles as the preview tool: `--count 32` on a new profile before any run (catches a wrong key in seconds).

Profiles (`options/degrade/*.toml`, each a `[degradations]` table; values are starting points, see Q1):
| set | content | question it answers |
|---|---|---|
| V0 | the owner's paired dev split as is (AA-cubic LQ) | does the model stay good on clean inputs? (non-inferiority) |
| V1 clean-mixed | `clean_prob = 1`, `clean_mode` = all AA kernels | kernel robustness without artifacts |
| V2 web | AA resize, `sharpen_prob 0.5`, one cv2 JPEG q 70-95 | typical downloaded photo |
| V3 recompressed | JPEG q 50-85 -> AA resize -> cv2 JPEG q 60-90 | social-media / messenger copies |
| V4 noisy | Poisson + Gaussian noise, mild blur, cv2 JPEG q 85-98 | phone photo in low light |
| V5 legacy | today's template `[degradations]` | the baseline's home distribution |
| R | real LQ without GT (owner's samples; the four `val8` real-LQ crops to start) | visual sign-off only |

| X independent | the same GT through **Pillow only** (its own resize kernels, `UnsharpMask`, JPEG encoder; ~40 lines, no neosr code) | does a win survive a pipeline the model's simulator did not produce? |

V1-V5 and X are cut from the R5 **dev** tiles that passed the leak check (219 kept; `split/exclude.txt` applies to
dev and confirmation alike); the confirmation tiles get the same sets, untouched until the final check. All variants
of one source group are resampled together in the bootstrap.

### 4.1 What these sets can and cannot show
V1-V5 come from the simulator under test (only the JPEG tail is independent): they are **controlled diagnostics**
(which artifact class does an arm handle), not evidence of real-photo quality. Evidence for the goal needs:
- **set R as a real holdout**: 50+ real LQ photographs from several sources and devices (Q1), frozen before the
  first run, compared blind (arm labels hidden) by the owner; the four `val8` crops come from one photograph and
  are a smoke test only;
- **set X** as the cheap independent check;
- a **residual analysis** before the A/B: put real LQ samples next to `degrade.py` previews of each profile and
  list the artifacts the simulator cannot produce (noise grain and its correlation, denoising smear, motion blur);
  that list decides the evidence-gated items of D8.

## 5. Tests (CPU, tiny tensors, seconds; runnable on the Windows machine)
| test | asserts |
|---|---|
| resampler parity | `aa_bicubic` / `aa_bilinear` equal torch `antialias=True` (fp64 1e-12, fp32 1e-6), down and up, odd sizes, non-square |
| resampler properties | rows sum to 1; a constant image stays constant for every kernel; an unchanged axis is returned as is; lanczos/mitchell/box against a 10-line scalar reference; box 5 -> 3 on an impulse (the half-open tie rule); AA size = `floor(n * factor)` |
| defaults bit-identical | the existing frozen-reference test unchanged; outputs **and RNG states** (python, numpy, torch CPU/CUDA) equal the reference for the Degrader, the worker and `jpeg_compress = false` |
| coins | a new gate at `p` in {0, 1} draws nothing; mask rate at `p = 0.5` within binomial bounds over 4k samples; `blur_prob = 0` gives the pulse kernel; skipped noise rows are clamped; both final orders with skipped JPEG |
| clean path | `clean_prob = 1` equals `quantize(resample(gt))` bitwise and consumes no noise/JPEG draw; mixed batch: clean rows equal the pure-clean result; `color = 'y'`: lq is luma of the quantised RGB, pool stores it as uint8 exactly |
| sharpen | zero amount is the identity; a step edge gains overshoot of the expected sign; output clamped |
| validation | every new key: wrong name, length, range -> `ConfigError` naming the key; shipped templates still validate |
| no host sync | the existing `test_degrader_no_host_sync` with all new keys on |
| `degrade.py` | two runs with one seed give identical files; the manifest lists the effective keys; `--codec cv2` round trip |

## 6. Steps
| step | content | size | machine |
|---|---|---|---|
| O1 | `data/resample.py`, `resize_mode*` keys, tests | S | Windows (code + tests) |
| O2 | `blur_prob`, `noise_prob*`, `jpeg_prob*`, `clean_prob`/`clean_mode`, tests | S | Windows |
| O3 | `sharpen_*`, tests | S | Windows |
| O4 | `scripts/degrade.py` (+ the Pillow-only generator of set X), `options/degrade/*.toml`, tests | M | Windows; the sets are generated on Linux |
| O4b | residual analysis: real LQ samples (Q1) next to the previews; decides the evidence-gated items of D8 | S | Windows (needs the owner's samples) |
| O5 | config schema, readme OTF section, commented template rows, Studio surface + catalog | S | Windows |
| O6 | cost check with `scripts/bench.py` (compact_otf and realplksr_otf, old vs new keys on), full suite | S | **Linux** |
| O7 | A/B (7) | GPU-days | **Linux** |
Each step is one commit with a Codex review of the diff, as in Phase A. O1-O5 change no default.

## 7. A/B plan (Linux machine; rules of `r3gan/RESULTS.md` section 1 reused)
Predeclared before any run, as in R5. All arms: x4 realplksr from the shared L1 pretrain P0, same seed, same
updates, EMA weights.

**Stage S1, selection (no discriminator: template L1 + MS-SSIM + consistency + LDL + FDL):**
| arm | change against the previous arm |
|---|---|
| B0 | shipped `train_realplksr_otf.toml` (canvas 512) |
| B1 | `tight_canvas = true` (the reference of every later comparison) |
| B2 | + AA kernels in all three mode lists (legacy modes kept in the list at reduced weight) |
| B3 | + `blur_prob 0.8`, `noise_prob 0.5`, `noise_prob2 0.3`, `jpeg_prob 0.6`, `jpeg_prob2 0.8`, `clean_prob 0.15` |
| B4 | + `sharpen_prob 0.25` |
The B3/B4 numbers are assumptions of this design (in the range of BSRGAN / Real-ESRGAN practice, cited from
memory), not measurements.

**Stage S2, attribution (only if B3 or B4 wins S1):** B3 without `clean_prob`; B3 with `clean_prob` only (all
other probabilities 1). A bundle win alone does not say which part helped; a template gets only parts that show
an effect.

**Stage S3, with the template GAN:** the S1 winner **and** B1, matched (same D, same seed), so the GAN run has its
own control.

- **Primary endpoint (one, predeclared):** TOPIQ-FR PIPAL, mean paired difference against B1 pooled over V1-V4,
  group bootstrap over sources (all variants of a source together), CI excluding 0 and beyond the R5 margin
  (+0.005). LPIPS, DISTS, PSNR per set are secondary and reported in full.
- **Regression limits (every set, including V0 and X):** PSNR CI lower bound >= -0.3 dB on V0; no set in V1-V4 or X
  worse on TOPIQ beyond the margin with a CI excluding 0. Losing on V5 is expected and reported, not gating.
- **Evidence for the goal:** the blind comparison on set R (4.1) and set X moving in the same direction. A win on
  V1-V4 with a loss or a tie on R is "inconclusive", not a template change.
- **Acceptance as in R5:** both seeds agree on dev; the confirmation sets once, same sign, CI excluding 0; a long
  run (80k+, past `wavelet_init`) before any template default changes.
- **Budget:** set from O6's ms/update; 20k updates per arm as in R5 unless the owner prefers longer runs.

## 8. Later (not designed here)
- Multi-source batches and paired + OTF mixing (C5): `datasets.train.sources = [{...weight...}]` with a
  batch-homogeneous seeded sampler; `otf.feed_data` skips the Degrader when a batch carries `lq`. Needed to feed
  `degrade.py --codec` pairs (real WebP/AVIF/H.264) into training.
- Curation scorer writing `meta_info` lists (C3), multi-scale tiling (C4), chroma realism (C9), kernel ranges (C10).

## 9. Open questions for the owner
- **Q1.** What are the real inputs: which cameras / phones / platforms (phone JPEGs, web or messenger copies, scans,
  video frames)? 50+ real LQ samples from several sources define set R (the only direct evidence, 4.1) and the
  weights of V2-V4; without them the profiles are generic and the result cannot be called real-photo quality.
- **Q1b.** Which matters more: staying faithful on already clean inputs, or repairing visibly damaged ones? It sets
  `clean_prob` and the regression limits.
- **Q2.** Scales: x4 only, or also x2 and 1x restoration (NAFNet)? The Degrader works at scale 1; the profiles differ.
- **Q3.** May O1-O5 (opt-in code, no default change) start before the R3GAN A/B finishes? The recorded order is
  "Phase B after R3GAN"; this design assumes yes for the code and leaves the A/B order to GPU availability.
- **Q4.** Arms without a discriminator first (7): acceptable?

## 10. Critique round (Codex gpt-6.1-sol, read-only, 2026-11-10) and what changed
| # | finding | verdict | change |
|---|---|---|---|
| 1 | a no-draw rule applied to existing coins would shift every later draw; the frozen test checks outputs only | accepted (the draft meant new gates but did not say so) | D6: new gates only; RNG-state test (5) |
| 2 | skipping a path by looking at the mask is a host sync | accepted | D3: shortcuts on the configured probability only |
| 3 | V0-V5 share the simulator: no evidence of real-photo generalisation; four crops of one photo are weak | accepted | 4.1: sets are diagnostics; real holdout R, independent Pillow set X, residual analysis; grouped bootstrap; R5 exclusions explicit |
| 4 | bundled arm, no matched GAN control, "TOPIQ or LPIPS on any two sets" is endpoint picking | accepted, attribution staged to save GPU time (S2 runs only for a winning bundle) | 7: stages S1-S3, one primary endpoint, regression limits, both seeds, 80k+ |
| 5 | resampler formula correct (independent probe); the size-vs-factor contract was missing | accepted | D1 contracts |
| 6 | Mitchell is not an identity at equal size; box is not torch `area`; Matlab pads differently; "TF32 is irrelevant" is wrong | accepted | D1: unchanged axis bypassed, tie rule, narrower attributions, TF32 measured in O6 |
| 7 | skipped noise rows escape the clamp; clean mix must precede quantisation and luma | accepted; **decision differs from the reviewer's default**: skipped rows are clamped (an intermediate image is in range) instead of left with excursions | D2 boundary rules, D3, tests |
| 8 | camera noise, denoising traces and centred motion PSFs dismissed too broadly; random order rejected inconsistently | partly accepted: moved to evidence-gated (Q1, 4.1) instead of added now, because no target sample has been inspected yet | D8 |
Not verified by anyone: CUDA behaviour, cost, and any effect on training.
