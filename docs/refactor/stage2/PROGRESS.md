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
