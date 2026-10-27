# Stage 2: speed, reliability, memory, training efficiency

Started 2026-10-27 (session 3). Input: `analyze.md` (Codex audit F01-F25) and the 10-lane stage-2 audit in
`analysis/` (167 findings; ids T-, O-, L-, D-, W-, MX-, A-, MT-, ST-, GP-; `MX` = archs_mixed, `MT` = metrics).
Branch `refactor/stabilize`; each lane works in its own worktree `.worktrees/s2-<lane>` on branch `s2/<lane>`,
then merges back here. Owner rules of `../DECISIONS.md` stay binding.

## Principles
- Stability from correctness (exact math, AMP-safe numerics, live gradients, sane GAN schedule), never from
  automatic damping (no dynamic loss weighting, no adaptive LR/batch automation).
- Exact rewrites (bit-identical or rounding-level, with parity tests) ship as defaults.
- Objective/dynamics changes ship as opt-in config values first; a default or template flips only after an
  A/B on the real data (`/home/korvin/apps/data`, held-out split in the session scratchpad).

## Lanes and file ownership
| lane | files (exclusive unless noted) | items |
|---|---|---|
| I1 trainer | models/{image (not validate/_img_path, not the net_d_adv block), phase, base (not wrap), otf, checkpoint}.py, train.py, runtime.py, config.py, cli.py, utils/{logger,stop,dist_util,rng}.py, data/prefetch_dataloader.py, losses/base.py (only `Loss.active`) | T-01 clamp modes, T-02/GP-02 clip modes, ST-02 + T-04 skip transaction, T-03 save on validation failure, T-06, T-08/O-10/F19, T-09/F20/ST-11/MT-14/M-06, T-10/A-05 census, T-13/F23, T-14, ST-09/T-15, ST-12, ST-10 warmup, T-16, D-13/GP-08 data wait, T-12 keep_exports, ST-05 Phase idle, O-06 Phase version, O-09, O-08 eval_weights, D-03 multi_prob plumbing |
| I2 optimizers | optimizers/*.py | O-01 sound SF-Adan (after Codex math review), O-03 b2=0 path, O-04, O-05, O-06 STATE_VERSION, O-07, O-08 |
| I3 losses+GAN | losses/* (not base.Loss.active), archs/vgg_arch.py, base.wrap, image.__init__ net_d_adv block | L-02, L-04, L-05, L-06, GP-01/ST-08 one D forward + compile D, GP-03, L-10, L-17, ST-05 GAN start/ramp, ST-06 lazy R1/R2, ST-07 rpgan, ST-01 term_grads tool |
| I4 data | data/* (not prefetch_dataloader), utils/{diffjpeg,color_util,file_client,lmdb_util,misc}.py, img_util read side, create_lmdb.py | D-01, D-02, D-04, D-05, D-06, D-07, D-08, D-09/F15, D-10/F16, D-11/F14, D-12/F13, D-14, D-15, D-17, D-18, D-19, D-20 |
| I5 window archs | swinir, swinirdense, hat, drct, dat, rgt, atd, srformer, craft, hitsrf; arch_util (append window_sdpa/shift_mask helpers only) | W-01..W-15 |
| I6 mixed archs | hma, mfghmoe, asid, ditn, grformer, catanet, dct, dctlsa, lmlt, omnisr, esc, microsr | MX-01..MX-18 |
| I7 cnn archs + D | esrgan, compact, span, spanplus, realplksr, plksr, mosrv2, moesr, aether, cugan, rcan, krgn, safmn, eimn, man, msdan, ninasr, sebica, cfsr, hasn, plainusr, ea2fpn, desrgan, flexnet, unet, dunet, patchgan, metagan, resgan; arch_util DySample | A-01, A-02/MT-03/MX-13, A-03, A-05, A-06, A-08, A-09, A-10/F12, A-11, A-13, A-14, A-15, MT-01, MT-09, MX-14 (flexnet) |
| I8 metrics/export | metrics/*, image.validate + _img_path, models/tiling.py, img_util.tensor2img, test.py, convert.py | MT-02, MT-04, MT-05, MT-07, MT-08, MT-10, MT-11/GP-05, MT-12, MT-13, MT-15/F21, MT-16, MT-17, T-03 OOM fallback |

Arch lanes add their tests in new files (`tests/test_s2_<lane>.py`); nobody rewrites another lane's tests.

## Cross-lane interfaces
1. `Loss.active(self, it: int) -> bool` (default True, I1). Phase.backward drops inactive terms; a phase with
   no active term is idle for that window: no backward, `step()` zero-grads, steps schedulers, returns None
   (no skip, no streak); EMA updates only after a True G step. I3 overrides `active` for the GAN pair.
2. Phase state records `"version": getattr(type(opt), "STATE_VERSION", 0)`; load compares it like the type (I1).
   I2 sets `adan_sf.STATE_VERSION = 2`.
3. `train.clamp`: true (legacy hard [1/255,1], code default) | false | "restore" (forward [0,1]; backward passes
   inward gradients, blocks outward ones). `train.grad_clip`: true (legacy max-norm 1.0) | false | float | "spike".
4. D update: one forward on `cat([real, fake])` (two forwards when D has BatchNorm); then `wrap()` may compile D.

## Experiments (main thread, GPU)
R-set (running): realplksr L1 5k updates: clamp hard/off, grad_clip on/off, adan_sf vs adamw vs adan, aug off.
P-set (after wave A): the shipped GAN template with the fixes: FDL weight, clamp restore, fixed adan_sf, GAN
start/ramp, rpgan/R1. Decide template defaults from these; record numbers in `PROGRESS.md`.

## Acceptance
Full suite green; lane parity tests; GPU benchmark of the five profiling workloads before/after
(`analysis/gpu_profile.md` harness); a short training run per model type (image, otf) end to end.
