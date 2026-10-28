# R3GAN in neosr: A/B results (stage 3 R4-R6)

Brief: `TASK_R3GAN.md`; design and acceptance gate: `DESIGN.md` section 4. Runs, configs and the harness live outside git
in the worktree's `experiments/r3gan_ab/` (`harness/` = runner with per-image metric capture, config generator, launcher,
parity and visual scripts; `cfg/`; `runs/<arm>/{train,metrics,per_image}.jsonl`, `experiments/<arm>/{models,visual}`).

## 0. Round 0: data and judges (fixed before any A/B run)
| item | value |
|---|---|
| pool | the owner's 10,607 x4 pairs (`/home/korvin/apps/data/{GT,LQ}`, GT 256 / LQ 64 tiles, names are content hashes) |
| source recovery | content hashing: every 16x16 block of every tile looked up against all tiles' stride-16 grid blocks (exact integer hashes, non-flat blocks): 488 overlapping pairs; plus 18 confident edge adjacencies (mutual best border continuation, ratio < 0.3); union -> 10,144 source groups (385 multi-tile groups hold 848 tiles, largest 8) |
| split (by group, seed 2026) | `split/dev.txt` 300 tiles (290 groups), `split/conf.txt` 300 tiles (290 groups, untouched until the final check), `split/train.txt` 10,007 tiles; excluded from every run incl. the pretrain |
| duplicates vs `val8` | none (no shared block between any tile and the 8 val8 GT photos) |
| limit | these are **overlap-recovered groups**, not verified source photographs: tile names carry no source, and tiles of one photograph that share no exact non-flat 16x16 block and no border (or were re-encoded/transformed) are not found. Same-photo leakage between train and held-out may remain; it can favour objectives that memorise differently and can make the grouped CIs too narrow, so source independence (and with it that part of the gate) is **unverified**. Mitigation: a fuzzy cross-correlation search of every held-out tile against train (R5 addendum below) drops matched held-out tiles from evaluation |
| R5 addendum: fuzzy leak check (before any A/B run) | every held-out tile vs every train tile (and vs each other): FFT cross-correlation of high-passed grey images over all offsets with >= 1/16 overlap, top-5 candidates verified by the normalised correlation inside the overlap (random pairs: max 0.19; exact-overlap pairs: median 0.997). **164 of the 600 held-out tiles (81 dev, 83 conf) have a train tile at NCC >= 0.5**: shifted, re-encoded crops of the same photographs that exact hashing missed (checked by eye at several score levels). They are dropped from evaluation (`split/exclude.txt`, with 2 conf tiles linked to dev tiles); 45 held-out pairs are linked into the CI groups. Evaluated: **219 dev tiles (211 groups), 215 conf tiles (207 groups)**. The pretrain had already seen the dropped tiles' content, which is why they are excluded rather than re-split. Detector limits: rescaled, rotated or flipped copies are not searched |
| `val8` | 8 real photos, 512 px GT / 128 px LQ: domain sanity set |
| visual set (`split/visual.txt`) | 16 dev tiles + 4 real-LQ 512 px crops of `val8/Anna Ralphs - SHOOTS - 0068.jpg` (nose skin, upper lip, tongue, fingernail). The owner's data are close-up photos of people, so the design's foliage/sky/text categories were replaced by what the domain has: eyes, faces with hair, teeth/lips, eyebrow, hair, pores/freckles, tattoo lettering, rings, patterned fabric (non-explicit tiles only) |
| judges | TOPIQ-FR `dataset = "pipal"` (higher better), LPIPS alex (lower), DISTS (lower), PSNR (fidelity sanity); none is a training loss in any arm (FDL is dinov2); EMA weights every 2k updates |

## 1. Predeclared margins and decision rules (written before round 1)
- Unit of comparison: the 219 kept dev tiles at 20k GAN updates (EMA weights); per-image paired differences arm -
  reference, 95 % CI by a bootstrap over their 211 groups (10,000 resamples). val8 is reported, never decisive.
- Practical margins: TOPIQ-pipal +0.005, LPIPS -0.003, DISTS not worse than +0.002, PSNR not below A0 - 0.3 dB.
  "Better than X" on a metric = the mean paired difference beats the margin and its CI excludes 0 (superiority over zero
  with a practically sized mean; it does not claim the CI clears the margin). After round 3 the margin is the larger of the
  above and X's seed-to-seed spread on that metric (|seed 7 - seed 2| of the dev means). Non-inferiority: DISTS upper CI
  bound <= +0.002 (or the seed spread), PSNR lower CI bound >= -0.3 dB.
- Acceptance evidence (gate items 2-4): the selected arm, A0 and A1 at 20k, both seeds, on dev; then once on the untouched
  confirmation set (300 tiles) with the same rules. "Agree" = both seeds give the same verdict per metric on dev, and the
  confirmation set gives the same sign with a CI excluding 0. A dev win not reproduced on confirmation is "inconclusive".
- Round 1 -> round 2: the best R arm is the one with the largest TOPIQ-pipal gain over A1 among R arms that pass PSNR
  and stability; ties (CIs overlapping) go to LPIPS, then to cost. If no R arm beats A1, round 2 still runs on the best R
  (the question then is whether any setting makes the GAN useful) and the outcome is likely "inconclusive" or "no".
- Stability (diagnostic, not a quality score): no skipped updates, finite `r1`/`r2`/penalties, `out_d_margin` without
  monotone runaway, G's adversarial gradient share non-zero (`scripts/term_grads.py` at 0/10k/20k).
- Cost: `scripts/bench.py` median ms/update within +30 % and peak VRAM within +1.5 GB of A0, else the owner decides.
- Analysis guards: every comparison requires the full manifest (300 dev / 300 conf / 8 val8), unique finite
  observations and complete group membership, else it stops; a run counts only with its final validation logged.
- Outcomes allowed: replace, do not replace, inconclusive. The owner's visual sign-off and an 80k+ run are required before
  any template default changes (proposed, not run here).

## 2. R4 qualification tolerances (predeclared; `harness/qualify.py`)
bf16 stays the penalty default of the recipe only if all hold, else the template sets `penalty_fp32 = true`:
- identical probe signs (one CUDA seed per batch for both paths), 10 real training batches (GT 256, B 8), at init (patch
  and scalar256) and at a trained patch D (2k GAN updates): max |raw R1/R2 relative error| <= 3 %; penalty-only
  D-parameter gradient cosine >= 0.999 per batch, norm ratio in [0.97, 1.03], every parameter tensor's cosine >= 0.99;
- the CUDA double-backward tests (bf16, channels_last, both heads) pass;
- 2k GAN updates from the pretrain, bf16 vs `penalty_fp32`, same seed, plus a bf16 replicate (cuDNN autotuning makes
  same-config runs differ): dev |dPSNR| <= 0.1 dB, |dTOPIQ| <= 0.005, |dLPIPS| <= 0.003, |dDISTS| <= 0.002 at 2k, each
  widened to twice the replicate gap if that is larger; mean raw R1/R2 over updates 1k-2k within a factor 0.8-1.25; no
  skipped update.
Reported but not gating: one-probe gradient variance (16 draws, 2 batches) and fixed-probe clip incidence (EMA G for the
trained probe, unaugmented batches, gamma 1). The clip incidence that matters is each arm's logged `clipped_d`.

## 3. R4 qualification results (2026-10-28, RTX 4090, frozen code 7e8989e, bf16 AMP, GT 256, B 8)
### Cost (`scripts/bench.py`, 3 x 100 updates after warm-up, median ms/update; peak VRAM allocated in training)
| arm | eager ms/update | compile ms/update | peak VRAM | vs A0 eager |
|---|---|---|---|---|
| A0 template (metagan, bce, adan_sf D) | 97.6 | 84.4 | 3.27 GB (compile 3.10) | - |
| A1 no D | 84.3 | - | 3.09 GB | -14 % |
| r3gan patch, rpgan, gammas 0 (ordinary update) | 105.2 | - | 3.30 GB | +8 % |
| **R1 r3gan patch, R1/R2 every update** | **719.2** | **700.6** | 3.30 GB (compile 3.18) | **+637 %** |
| R4 r3gan_scalar256, R1/R2 every update | 725.4 | - | 3.31 GB | +643 % |
- **The cost gate fails by far**: the penalty pass costs ~614 ms per update (719 - 105) against an estimate of +0-10 ms;
  memory is fine (+0.03 GB with bf16; the fp32 penalty peaks at 5.61 GB vs 3.22 GB in the 2k runs). The 2k runs took
  0.71 s (bf16) and 0.71 s (fp32) per update: precision is not the bottleneck. Hypothesis (unverified): the double
  backward of grouped convs (group width 16) and of the depthwise FIR runs a slow path (e.g. a per-group loop) on CUDA.
  Consequence: a 20k R arm takes ~4 h instead of ~40 min; round 1 as planned would take ~21 GPU-hours.
### bf16 vs fp32 penalty with identical probe signs (10 real batches; tolerances in section 2)
| state | max raw R1/R2 rel. error | min D-grad cosine | norm ratio | min per-tensor cosine | one-probe rel. variance |
|---|---|---|---|---|---|
| r3gan patch, init | 0.52 % | 0.99997 | 1.0029-1.0031 | 0.99991 | 0.00035 |
| r3gan_scalar256, init | 0.44 % | 0.99996 | 0.9965-0.9966 | 0.99976 | 0 (P = 1, no draw) |
| r3gan patch, after 2k GAN updates | 0.23 % | 0.99996 | 0.9971-0.9983 | 0.99990 | 0.003-0.005 |
Fixed-probe clip incidence (gamma 1, EMA G, unaugmented): 0 at init and trained; the trainer's own `clipped_d` is the
relevant number (below). CUDA double-backward tests (bf16, channels_last, both heads, MATH SDPA): 6 passed.
### 2k GAN updates from the pretrain, R1 settings (patch, gamma 1, w 0.3), seed 7: bf16 vs `penalty_fp32`
| run | dev PSNR | dev TOPIQ-pipal | dev LPIPS | dev DISTS | val8 PSNR / TOPIQ / LPIPS / DISTS | mean upd. 1k-2k: margin, raw r1, r2, `clipped_d`, `l_g_gan` |
|---|---|---|---|---|---|---|
| pretrain P0 (20k, no GAN) | 30.399 | 0.6394 | 0.1854 | 0.1608 | 28.99 / 0.6354 / 0.2192 / 0.1489 | - |
| Q4_bf16 | 29.233 | 0.6526 | 0.1737 | 0.1606 | 27.95 / 0.6546 / 0.1901 / 0.1379 | 0.078, 0.0182, 0.0164, 0.49, 0.221 |
| Q4_fp32 | 29.228 | 0.6511 | 0.1715 | 0.1608 | 27.94 / 0.6538 / 0.1875 / 0.1391 | 0.095, 0.0222, 0.0198, 0.53, 0.224 |
(dev means over all 300 dev tiles, as logged; the 219-tile evaluation applies to the A/B.) No skipped updates. The bf16
replicate (run-to-run noise) was **not run** (owner's wrap-up decision): plain tolerances applied; raw R1/R2 bf16/fp32
ratio 0.82-0.83 is close to the 0.8 bound and cannot be separated from run-to-run divergence without it.
**Verdict: PASS, bf16 stays the penalty default** of `options/train_realplksr_r3gan.toml` (unchanged).
Indicative only (no A1 control at 2k): versus the pretrain the GAN start costs 1.17 dB dev PSNR and gains +0.013
TOPIQ-pipal and -0.012 LPIPS; D is not at chance (margin 0.08-0.09 > 0.01), but D's gradient is clipped in about half
of the updates (template max-norm 1.0) and G's in ~90 %.
