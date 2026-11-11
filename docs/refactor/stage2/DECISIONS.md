# Stage 2 decisions

Evidence: `analysis/*.md` (audit), `PROGRESS.md` (A/B numbers). A/B protocol: the owner's paired x4 data
(`/home/korvin/apps/data`, 10540 train / 67 held-out pairs), seeded init and data order, EMA eval weights,
PSNR/SSIM (and DISTS for GAN runs) every 500-1000 updates; one seed per arm, so differences below ~0.05 dB
count as ties.

## Principles (owner)
1. Stability comes from correctness (exact math, AMP-safe numerics, live gradients, a sane GAN schedule), never
   from automatic damping (no dynamic loss weighting, adaptive LR/batch automation or always-on normalisation
   tricks of the kind traiNNer-redux uses).
2. Exact rewrites ship as defaults behind parity tests; objective changes ship opt-in until an A/B decides.

## Optimizer
3. `adan_sf` schedule-free branch replaced by a sound SF-Adan (derivation by the optimizer lane (Fable),
   verified line by line by Codex): Adan's first moment is dropped (the SF interpolation is the momentum), `y`
   and `z` take the same preconditioned step, decay at `y`. `b2 = 0` is exactly `adamw_sf(b1, b3)` with 2
   state buffers. Old schedule-free `adan_sf` `.state` files are refused (`STATE_VERSION = 2`, scoped to SF
   mode); plain-mode states and every `net_*.pth` keep loading.
4. Templates train the generator with AdamW (lr 2e-4, betas (0.9, 0.99), no decay; the fixed `adan_sf` with
   `betas = [0.98, 0, 0.99]` is the documented schedule-free option). GAN template A/B @8k (FDL 0.02, L1, no aug,
   restore): AdamW 29.36/0.721/0.1787 vs SF-Adan(b2=0, lr 5e-4) 29.24/0.718/0.1890 (PSNR/SSIM/DISTS); with FDL
   0.05: 28.58/0.1829 vs ~28.8/0.1896; SF-Adan at lr 2e-4: 28.83/0.1950 (not an LR effect). On L1 alone they tie
   (29.87 vs 29.85) and the gradient-difference term (b2 = 0.92) costs 0.34 dB. Discriminators keep `adan_sf`
   `(0.98, 0.92, 0.99)`, lr 1e-4 (D betas b2 = 0 vs 0.92: tie at 6k, slower early).
5. `grad_clip = true` (max-norm 1.0) stays the default: it clips almost every update but is near scale-invariant
   for Adam-family optimizers (AdamW clip 29.87 vs off 29.85); the opt-in `"spike"` mode gave no gain (29.43 vs
   29.51 for SF-Adan B).

## Trainer
6. `train.clamp` defaults to `"restore"` (forward clamp to [0, 1], inward gradients pass): +4.2 dB at 500
   updates over the hard clamp, tie at 5k. `true` keeps the old hard [1/255, 1] clamp.
7. A non-finite micro-batch is a skipped update (buffers restored, EMA untouched, logged as `nonfinite_*`), not
   a dead run; the run stops at `max_skip_streak` consecutive skips or on non-finite weights. A non-finite loss
   in an applied update still raises (T1 D15).
8. `manual_seed` no longer implies deterministic kernels (`deterministic = true` does); `fast_matmul` no longer
   implies fp16 GEMM accumulation (`fp16_accumulation = true`).

## Template objective (A/B on the shipped realplksr + metagan template, 8k updates)
9. FDL(dinov2) weight 0.75 -> 0.02: the DINOv2 FDL score has no 0.01 scale (VGG/ResNet FDL do), so at 0.75 it
   held ~98-100% of the generator gradient. PSNR/DISTS @8k: shipped template 27.23/0.2024 (8.39 dB at 1k);
   FDL 0.1 28.17/0.1905; 0.05 28.69/0.1830 (AdamW, old code); 0.02 vs 0.05 with the new code: +0.4 dB at equal
   or better DISTS (SF-Adan), 29.36/0.1787 vs 28.58/0.1829 (AdamW).
10. An L1 term (`pixel_opt`, weight 1.0) joins the template objective (FDL 0.1: 28.34/0.1879 vs 28.17/0.1905).
11. Batch augmentation off in the templates (kept as a commented example): it touched 73% of batches; DISTS
    0.1862 vs 0.1879 at 8k with equal PSNR.
12. GAN weight stays 0.3 and starts at update 1 (0.1: mixed; start 1000 + ramp 1000: 28.69/0.1900 vs
    28.81/0.1896, no gain). `start_iter`, `ramp_iters`, `rpgan`, R1/R2 remain opt-in.
13. Net effect on the realplksr GAN template (same data, seed, 8k updates): 27.23 -> 29.36 dB PSNR,
    0.627 -> 0.721 SSIM, 0.2024 -> 0.1787 DISTS; 8.39 -> 28.84 dB at 1k updates. One architecture and one seed:
    the shared mechanisms (FDL dominance, clamp, augmentation share) apply to every template, the optimizer
    choice is the least general part.

## Session-9 screens (2026-11-11, 8k updates, R5 protocol)
Protocol (stricter than the one above; tool `scripts/ab/`, `scripts/ab/README.md`): the owner's 10,607 x4 pairs
split by recovered source group (R5, `../stage3/r3gan/RESULTS.md` section 0); every arm starts from the 20k
non-GAN pretrain `P0_pretrain` with the shipped realplksr + metagan template (bf16, seed 7), 8000 updates, EMA
weights. Judges on the 219 kept dev tiles (211 source groups): TOPIQ-pipal, LPIPS-alex, DISTS, PSNR. Per-image
paired differences against B0 with a 95 % bootstrap CI over source groups (10,000 resamples). Floor per metric =
max(|B0 - B0s2| (seed 2), 0.05 dB PSNR, 1 % of B0 for TOPIQ/LPIPS/DISTS): today PSNR 0.05, TOPIQ 0.0065, LPIPS
0.0019 (seed), DISTS 0.0015. A metric is better or worse only beyond its floor with a CI excluding 0; an arm is
LOSS if a perceptual metric is worse, WIN if at least 2 of 3 are better and none worse, else TIE. 8k is a screen:
a WIN on a GAN default needs a second seed and a >= 20k run before it flips (Fable consult, 13:55).
8k dev means (PSNR/TOPIQ/LPIPS/DISTS): B0 30.407/0.6548/0.1577/0.1493, B0s2 30.402/0.6527/0.1596/0.1507.

14. `grad_clip = true` stays the template default (B1, `grad_clip = "spike"`): TIE at 6k and 8k (8k paired: PSNR
    +0.0008, TOPIQ -0.0007, LPIPS +0.0015 [+0.0011, +0.0020] under its 0.0019 floor, DISTS 0.0000). The fixed clip
    is near idle in this regime (clipped_g ~2 %, clipped_d 0 % after 300 updates), so the screen cannot separate
    the two; per the pre-agreed rule no clip arm flips today, a WIN would have gone to a second seed and the
    >= 20k list. Where clipping is active (R3GAN: G clipped in 95-100 % of the first 2k updates) the question stays
    open (`../STATUS.md`, "Needs the Linux machine" item 7).
15. LDL (`[train.ldl_opt]`, B2 without it): TIE at 6k and 8k (8k paired: PSNR +0.0122, TOPIQ -0.0005, LPIPS
    +0.0014, DISTS -0.0003), -4.8 % ms/update (104.6 vs 109.9), LDL's G-gradient share 0.000 % at B0's 8k weights.
    Pre-agreed rule: a TIE or WIN without a LOSS flips the templates provisionally. Held by a Fable consult (14:41)
    on new evidence: more late G-gradient spikes without LDL (`grad_norm_g` window means > 1.2 after update 2000:
    B0 0, B0s2 4 (max 3.87), B2 6 (max 10.61)). Criterion fixed before the run: the seed replicate B2s2 is clean
    with at most 4 such windows and a max <= 7.7, then the flip (prepared as bd54d64b: 85 templates, Studio
    presets, training guide) merges as provisional; else LDL stays and the >= 20k list records "metrics TIE, more
    late G-gradient spikes without LDL". <<PENDING: B2s2 windows / max, verdict, the decision and its commit>>
16. B3 (`fdl_opt.fp32_backbone = false`) was not screened: it is not faster on this setup (116.5 vs 109.9
    ms/update; it saves ~160 MiB), so a quality tie could not flip the default. The docs that promised "~2x faster
    FDL" now give the measurement.

## Kept opt-in pending an A/B
`tight_canvas` (OTF), `aug_antialias`, CRAFT `pad_extra_window = false`, span_fast `fan_in_init`, `"spike"`
clip, `gan_type = "rpgan"`, R1/R2, `fdl_opt.checkpoint`.
Not switches (corrected 2026-11-10): realplksr norm init (A-07) and the RGT train/eval depth (W-09) have no key
in the code; both are arch-specific and deferred by the owner's goal (`../DECISIONS.md`, 2026-11-10).
