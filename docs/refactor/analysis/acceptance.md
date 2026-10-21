# Stability acceptance (lane C7, T2 §9 Wave E)

Ran on 2026-10-21 against `refactor/stabilize` HEAD `b5c951a`, plus the uncommitted edits other lanes
made to the worktree at the same time (dead-code and legacy-options removal: `neosr/utils/{options,registry}.py`,
`optimizers/{fsam,soap_sf}.py`, small edits in `arch_util.py`, `esrgan_arch.py`, `img_util.py`,
`lmdb_util.py`). None of those touch the trainer. Machine: RTX 4090 (shared), torch 2.9.1+cu128.

## 1. Verification lanes (pytest)

| Lane | File | Result |
|---|---|---|
| E1 continuation, CPU, `use_deterministic_algorithms(True)` | `tests/test_resume.py` | 4/4 pass |
| E2 continuation, CUDA AMP | `tests/test_resume_cuda.py` | 2/2 pass (fp16, bf16) |
| E3 2-rank gloo DDP | `tests/test_ddp_gloo.py` | 8/8 pass (CPU and CUDA tensors) |

**Method (E1/E2).** Every build seeds all streams `rng_state()` captures (python, numpy, the neosr
numpy stream, torch + CUDA, and a loader generator standing in for train.py's `loader.generator`),
and every micro-batch draws once from the data-side streams, as loaders and degradations do. Run A
does 6 updates and its `training_state()` is taken right then. Run B does 3 updates and `save()`. A
fresh model is built with seed 5 (every stream perturbed) and `resume_state`, calls `restore_rng()`,
does 3 more updates, and its `training_state()` is taken. Both snapshots are taken right after their
own run, because the RNG streams are process-wide. They are compared recursively with `torch.equal`
(and dtype) for tensors and `==` for everything else. The `run` record is left out because `resume()`
already diffs it. The comparison covers weights and buffers (BN running stats, SN u/v), EMA,
optimizer state, scheduler state, GradScaler state, counters, best metrics and the python, numpy,
neosr, torch, CUDA (E2) and loader RNG streams.

- (a) compact G, L1 + FDL (vgg, `num_proj=4`) + MS-SWD, `ema=0.999`, MultiStepLR: equal. To show
  the check can fail, the same run without `restore_rng()` must differ in each of the five CPU
  streams and in the weights, and it does (MS-SWD draws its projections from the torch stream). A
  scratch mutation run that skips the python, numpy, neosr or loader restore in turn makes (c) fail
  on exactly that stream.
- (b) `adan_sf` with SF on, `ema=0`. Validation (psnr) + `save()` (state + exports through the SF x
  swap) run between updates 2 and 3 in both runs: equal, best metrics included. Each run's
  `net_g_2.pth` equals an oracle x computed from the optimizer state without touching the model
  (`y.lerp(z, 1 - 1/beta1)`), which differs from the live y; a mutation that exports y fails.
- (c) sebica_mini G (BatchNorm) + small metagan D (spectral norm, DropPath, dropout, attention), GAN,
  `accumulate=2`, MultiStepLR on both phases: equal.
- (d) The (c) state resumed with no `network_d`/`gan_opt`/`optim_d` raises
  `ConfigError: nets ['net_d', 'net_g'] saved, ['net_g'] configured`.
- E2: (c) on cuda:0 with deterministic cuDNN/cuBLAS and channels_last. A hook makes the G gradient
  of update 3 NaN in both runs. fp16: G skipped once, D not skipped, scaler 32 -> 16 in both runs,
  everything equal. bf16 (scaler off, plain skip): equal.
- E3: two ranks, compact G with `adan_sf` SF + unet D (SN) with adamw, GAN, `accumulate=2`,
  rank-specific batches. Results:
  - Weights and buffers are equal across ranks after 2 and 4 updates.
  - The rank-0 validation + `save()` between barriers leaves rank 0 bit-identical to its
    pre-validation snapshot, and the ranks stay equal. `net_g_2.pth` equals the SF x oracle and
    `net_d_2.pth` the live D.
  - Both ranks resume rank 0's `2.state` and run 2 more updates. Weights, buffers and optimizer
    state equal the uninterrupted run on rank 0, and rank 1's weights equal it too.
  - A real `SIGINT` to rank 1 only, inside update 3's window, stops both ranks at update 3 / micro 6.
    Only rank 1's flag is set.

  Workers use a 60 s process-group timeout and the parent kills them after 240 s, so a hung
  collective fails the test instead of stalling the suite.

  The models have no model-side randomness on purpose. Other ranks' RNG is not part of the state
  (T2 §11 "continuation scope"). A G or D with dropout under DDP would make the resumed run on
  ranks != 0 differ from an uninterrupted one. That is documented, not tested.

## 2. GPU acceptance runs

Configs are temporary copies in the lane scratch dir. Data: `/home/korvin/apps/data` GT/LQ 256/64,
x4, 10607 pairs. Val: 7 images, 128 -> 512. psnr/ssim metrics were added. Not deterministic (no
`manual_seed`). The runs ran one after another, but the GPU was shared with other lanes' pytest
jobs, so the it/s figures are indicative only.

| Run | Updates | AMP | Exit | Wall | Steady it/s (>= u300) | Skipped G/D | NaN / abort | Val PSNR / SSIM @1000 | @end |
|---|---|---|---|---|---|---|---|---|---|
| `w5_esrgan2_bf16`: train_esrgan2 (esrgan + metagan, adan_sf SF both, ema 0.999, mssim+consistency+ldl+fdl(dinov2)+gan bce, wavelet_guided, **wavelet_init 500**) | 2000 | bf16 | 0 | 10:04 | 3.64 | 0 / 0 | none | 6.55 / 0.145 | 19.15 / 0.665 |
| `w5_esrgan2_fp16`: same, `bfloat16=false` | 2000 | fp16 | 0 | 9:42 | 3.84 | 0 / 0 | none (scaler 32 -> 64 at u2000) | 12.41 / 0.503 | 22.34 / 0.679 |
| `w5_compact_otf`: train_compact_otf (compact + metagan, OTF degradations, augmentations) | 1000 | bf16 | 0 | 2:41 | 7.46 | 0 / 0 | none | - | 26.49 / 0.585 |

Validation scores the EMA. A diagnostic re-scoring of the saved states (same 7 images, neosr
`calculate_psnr`) separates the three weight sets:

| State | live y | SF x | EMA (= validated / exported) |
|---|---|---|---|
| esrgan bf16 @1000 / @2000 | 23.51 / 26.22 | 23.94 / 25.89 | 6.55 / 19.16 |
| esrgan fp16 @1000 / @2000 | 23.04 / 26.15 | 23.01 / 25.91 | 12.41 / 22.35 |
| compact otf @1000 | 22.09 | 21.15 | 26.49 |

Trajectories (window means every 100 updates):

| u | bf16 l_g_gan | bf16 out_d_real / fake | bf16 gnorm g / d | fp16 l_g_gan | fp16 out_d_real / fake | fp16 gnorm g / d | otf l_g_gan | otf out_d_real / fake | otf gnorm g / d |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 0.380 | 0.125 / -0.868 | 101 / 18.3 | 0.207 | 0.313 / 0.040 | 150 / 11.2 | 0.196 | 0.134 / 0.118 | 43.7 / 4.9 |
| 200 | 0.428 | 2.619 / -0.455 | 61.8 / 36.4 | 0.229 | 0.187 / -0.098 | 48.3 / 4.1 | 0.303 | 0.450 / -0.488 | 19.9 / 12.5 |
| 300 | 0.296 | 0.459 / -0.384 | 81.1 / 11.2 | 0.228 | 0.077 / -0.089 | 88.1 / 3.6 | 0.301 | 0.274 / -0.417 | 16.7 / 24.4 |
| 400 | 0.240 | 0.153 / -0.134 | 109 / 10.8 | 0.229 | 0.129 / -0.100 | 124 / 5.6 | 0.226 | 0.060 / 0.051 | 40.8 / 18.1 |
| 500 | 0.239 | 0.104 / -0.112 | 109 / 13.8 | 0.223 | 0.085 / -0.065 | 139 / 5.6 | 0.314 | 0.551 / -0.307 | 48.1 / 31.8 |
| 600 | 0.246 | 0.151 / -0.157 | 102 / 7.4 | 0.294 | 0.477 / -0.375 | 125 / 18.1 | 0.222 | 0.040 / -0.028 | 36.0 / 8.9 |
| 700 | 0.290 | 0.220 / -0.154 | 95.1 / 46.8 | 0.367 | 0.730 / -0.632 | 111 / 49.3 | 0.211 | 0.037 / -0.013 | 43.5 / 1.1 |
| 800 | 0.286 | 0.261 / -0.231 | 92.7 / 24.9 | 0.297 | -0.046 / -0.219 | 90.3 / 60.2 | 0.238 | 0.132 / -0.118 | 42.1 / 12.5 |
| 900 | 0.340 | 0.614 / -0.533 | 81.4 / 29.2 | 0.305 | -0.099 / -0.356 | 86.8 / 18.4 | 0.229 | 0.198 / 0.008 | 42.4 / 24.6 |
| 1000 | 0.300 | 0.000 / -0.314 | 80.0 / 33.5 | 0.334 | 0.718 / -0.282 | 78.3 / 34.3 | 0.268 | 0.318 / -0.150 | 34.5 / 9.8 |
| 1200 | 0.315 | -0.723 / -0.157 | 63.4 / 40.8 | 0.246 | 0.455 / 0.061 | 70.2 / 32.7 | | | |
| 1400 | 0.253 | 0.305 / 0.213 | 62.8 / 252 | 0.174 | 0.056 / 0.784 | 64.9 / 49.3 | | | |
| 1600 | 0.235 | 0.143 / -0.079 | 50.7 / 22.4 | 0.239 | 0.222 / -0.043 | 55.7 / 10.4 | | | |
| 1800 | 0.262 | -0.174 / 0.183 | 42.9 / 41.1 | 0.249 | 0.038 / 0.041 | 49.5 / 9.1 | | | |
| 2000 | 0.263 | 0.233 / -0.236 | 38.8 / 17.7 | 0.297 | -0.473 / -0.404 | 46.9 / 3.4 | | | |

l_g_total, esrgan: 21.3 -> 9.71 (bf16) and 20.7 -> 9.74 (fp16). OTF: 25.2 -> 13.9.

## 3. Conclusions

- **Continuation is exact.** Weights, buffers, EMA, optimizers, schedulers, scalers, counters, best
  metrics and every rank-0 RNG stream continue bit-identically after a resume:
  - on CPU and on CUDA fp16/bf16, through an fp16 scaler skip;
  - through a validation + SF export between updates, whose `net_g` artifact is SF x;
  - under 2-rank DDP with accumulation, GAN and SF.
  The stop flag all-reduce stops both ranks at the same update. A removed D on resume is a `ConfigError`.
- **The AMP paths are stable in real runs.**
  - Three runs (5000 updates in total) had no skipped step, no non-finite value and no abort. All
    exited 0 with the expected states, exports and `_latest`, and no `.tmp` left behind
    (checked on the esrgan runs).
  - fp16 (with `fast_matmul`, i.e. `allow_fp16_accumulation`) never overflowed: the scaler only grew.
  - fp16 is about 5% faster than bf16 on this esrgan config, and the two converge alike (live y
    26.15 vs 26.22 dB at 2000).
- **The GAN is balanced.** l_g_gan stays at 0.17-0.43 and out_d_real/out_d_fake stay within ±1 except
  one bf16 window (out_d_real 2.6 at u200). D never collapses or runs away. The D view switch at
  `wavelet_init=500` gives a short D transient at u600-700 (grad_norm_d 7 -> 47 in bf16, 18 -> 49
  in fp16), which settles. bf16 grad_norm_d spikes to 252 at u1400 (window mean, so clipping at 1.0
  bounds the step) and recovers by u1500. G grad norms (clipped to 1.0) fall steadily from about 100
  to about 40.
- **Early EMA validation is not a regression signal for esrgan.** With `ema=0.999`, the update-1 copy
  still weighs 0.999^999 ≈ 37% at u1000. For esrgan that average scores 6.6-12.4 dB while the live
  weights score 23-24 dB. EMA PSNR (= validation and the exported `net_g_<u>.pth`) only becomes
  meaningful after several thousand updates. For compact, the EMA is already the best of the three
  (26.5 vs 22.1). This is the documented "copy on first update" semantics (T1/T2 D6), not a defect.
  It is a usability caveat for short runs and for comparing early validations.

## 4. Defects

None found in product code. No product file was changed.
