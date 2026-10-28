# Stage 2 progress (newest last)

## 2026-10-27/28 session 3 (Opus 5.5)
- Baseline at f360edd: full suite 1192 passed / 188 skipped / 5 xfailed (6:44).
- Audit workflow (10 lanes; Fable for optimizers, Opus elsewhere; GPU profiling lane on the 4090): 167 findings,
  reports in `analysis/`. Headlines: hard train clamp freezes 34-84% of output pixels at init; adan_sf SF recurrence
  unsound (F01 confirmed, the planned G4.M2 formula rejected as double momentum); FDL(dinov2) carries 98-100% of the
  template generator gradient; grad_clip=1.0 clips 100% of updates; D does two forwards (one cat forward: -14-16%);
  cuDNN exhaustive autotune per validation shape (+2.4-3.7 s, up to 4.8 GB); one bad image kills a run; one
  non-finite micro-batch aborts the run although the update was skipped.
- A/B infrastructure: held-out split (10540 train / 67 val) as symlinks in the scratchpad (`ab/data`), runner
  `ab/run_ab.py` (seeded init/data, fast kernels), configs `ab/mkcfg.py`, launcher `ab/launch.py`; frozen baseline
  worktree `ab/base_repo` at f360edd.
- R-set started (realplksr, L1 only, 5k updates, EMA eval): early PSNR at 500/1000 updates: shipped adan_sf + hard
  clamp 13.90/23.86; adan_sf no clamp 21.89/26.28; adamw 2e-4 no clamp 28.57/29.55; adamw no clamp no clip 28.11/29.00.
- Codex (gpt-6.1-sol, xhigh) reviewing the proposed sound SF-Adan recurrence before lane I2 starts.
- Implementation lanes launched in worktrees `.worktrees/s2-<lane>`: trainer, losses, metrics, data, window, mixed,
  cnn (Opus implements -> Codex reviews -> Opus fixes). Merge into refactor/stabilize after all return.
- Codex (gpt-6.1-sol xhigh) verified the SF-Adan derivation of the optimizer lane (wrapper identities, swap formulas,
  kernel line by line; b2=0 reduces bit-for-bit to SF-AdamW). Corrections adopted: clipping does not bound the z
  step (doc claim removed); stamp sf_version=2 in every fresh SF group and refuse any other version; test formulas
  use gamma_1 (not nominal lr) for decay. Decision: variant B + b2=0 fast path; template betas decided by A/B
  ((0.98,0.92,0.99) vs (0.98,0,0.99)). Phase records the state version for schedule-free optimizers only, so old
  non-SF adan_sf states stay loadable. Optimizer lane launched.
- R-set final @5k (PSNR/SSIM, 67 held-out, EMA): shipped adan_sf+hard clamp 29.78/0.7411 (13.90 @500); adan_sf no
  clamp 29.68; adan (SF off) 29.87; adamw 2e-4 29.87/0.7448 (28.57 @500, 29.55 @1000); adamw no clip 29.85; adan_sf
  no clip 29.69; adan_sf no aug 29.71. => current SF branch is worse than plain Adan; clip 1.0 harmless; aug ~neutral.
- 2026-10-28 01:08: the session ended; all 8 lane implementers were interrupted mid-way (partial work kept in their
  worktrees: commits and uncommitted changes), P-set killed at ~700 updates. Resumed 01:12: lanes relaunched with a
  RESUME note (inspect and continue the partial work, commit in small steps); P-set restarted from scratch.
- Resume handles (if interrupted again: Workflow({scriptPath, resumeFromRunId}); script
  ~/.claude/projects/-home-korvin-apps-neosr-dev--worktrees-refactor/c1b40b02-f454-4498-9d89-be0afa4e2973/workflows/scripts/neosr-stage2-implement-wf_df07b9bc-879.js):
  wf_5328fd80-7fc (trainer, losses, metrics), wf_eef6ea89-eac (data, window, mixed), wf_4bffd2bf-3da (cnn, optim).
  P-set running P0-P3 (P4 noaug deferred: 5 runs exhausted 24 GB VRAM).
- P-set @8k (realplksr+metagan template, PSNR/SSIM/DISTS): shipped 27.23/0.627/0.2024 (8.39 dB @1000); adamw no clamp
  FDL 0.75 26.27/0.561/0.2087; FDL 0.1 28.17/0.673/0.1905; FDL 0.1 + L1 28.34/0.683/0.1879. FDL(dinov2) has no
  0.01 scale (cb9b92f) unlike VGG/ResNet FDL, so 0.75 is ~75x the VGG-era weight. Follow-ups P4b (no aug), P5
  (FDL 0.05), P6 (GAN 0.1) running on top of P3.
- Lanes trainer, losses, metrics, cnn, optim done (each Codex-reviewed and fixed; optim: no findings). Merged into
  refactor/stabilize (readme/base.py conflicts resolved); integration: Phase version only for SF optimizers,
  sf_version skipped in the resume diff, idle-G test forces a D-only window, tests/test_s2_integration.py.
  CPU suite on the trainer+losses+metrics merge: 1267 passed after the one test fix. Frozen A/B snapshot
  ab/s2_m2 at be010fe; F-set launched (clamp restore/hard, fixed adan_sf B vs b2=0, lr 1e-3, spike clip, adamw warmup).
- F-set @5k (realplksr L1, s2_m2 code; PSNR; R-set refs adamw no clamp 29.87, plain adan 29.87): adamw restore 28.60
  @500 / 29.87; adamw hard 24.38 @500 / 29.91; fixed adan_sf B (0.98,0.92,0.99) lr 5e-4 29.51; b2=0 29.85;
  B + spike clip 29.43; B lr 1e-3 29.79; adamw + warmup 1000 29.85. => restore wins early (+4.2 dB @500), ties
  late (single seed); the Adan difference term hurts (b2=0 is the SF default candidate); spike clip no gain.
- P4b/P5/P6 @8k (on P3): no aug 28.31/0.1862; FDL 0.05 28.69/0.1830 (best); GAN 0.1 28.43/0.1854.
- Q-set launched (s2_m2 code, restore, L1, no aug, FDL 0.05, G adan_sf b2=0): D betas B vs b2=0, GAN 0.1,
  FDL 0.02, GAN start 1000 + ramp 1000, adamw control.
