# Stage 2 lane: training-stability mechanisms (competitor study and design for neosr)

Date 2026-10-27. Repo `refactor/stabilize` (= master, `f360edd`), torch 2.9.1+cu128, Python 3.13. All probes ran on CPU
(`CUDA_VISIBLE_DEVICES=`) on a machine shared with other lanes (load 25-125), so CPU timings are indicative only.
Probe scripts and raw outputs: `/tmp/claude-1000/-home-korvin-apps-neosr-dev/c1b40b02-f454-4498-9d89-be0afa4e2973/scratchpad/stability/`
(`p*.py`, `out/`). Real data: `/home/korvin/apps/data` (the same 10607 pairs both projects train on; neosr reads the
16-bit `GT`, the competitor the 8-bit `GT8` copy; `val` = `val8` minus `ar.png`).

Evidence tags: **reproduced** (probe shows the failure), **measured** (probe gives the number), **source-verified**
(read, not run), **hypothesis**.

## 0. Summary

The refactor made neosr's GAN/AMP path mechanically sound: the acceptance runs (2000 updates, bf16 and fp16) had no
skip, no NaN, D logits within about ±1, and resume is exact. This lane found no remaining NaN mechanism in the default
path. The remaining problems are about *learning efficiency* and *reliability policy*. Items 1 and 3 below matter more
than the GAN-schedule questions this lane was asked to design for:

1. **The template objective is one loss.** In all 84 templates, FDL(DINOv2, weight 0.75) supplies 97.7% (random init)
   and 98.5% (partly trained) of the squared generator-gradient norm. MS-SSIM, consistency, LDL and the GAN term
   together give under 3.5%. The GAN at weight 0.3 gives 0.2-0.5% of the update and is effectively absent. The same
   backbone is also the most expensive part of an iteration for light generators (41 GFLOP/img forward vs 5 for
   compact and 1.3 for metagan). The acceptance esrgan run was 2.2 dB *below bicubic* on the validation set at
   update 2000 (live 26.2 dB vs bicubic 28.42 dB). The competitor, with an L1-led objective on the same data, was at
   28.6 dB (EMA). This is an objective question for the owner, with an A/B, not a code bug. It does mean that
   "GAN 0.3 from iteration 0" is not today's stability risk: the GAN barely acts.
2. **The training output clamp stalls the first few hundred updates.** `train.clamp` defaults to true (78 of 84
   templates). At init it zeroes the gradient of 34-84% of output pixels for 11 of 19 archs measured (span 57%,
   omnisr 54%, rcan 49%, realplksr 84%). With L1:
   - span is 9.4 dB behind at update 200 and omnisr 7.5 dB behind;
   - span catches up by about update 600 and is equal at 1000.

   So it costs roughly 500 updates at the start. That matters for short runs and for a GAN that starts at update
   0, not for long ones.
3. **One transient non-finite loss crashes the run** at the next log print (`LossMeter.read` raises), even though
   `Phase.step` has already skipped that update and the weights are finite. `max_skip_streak` only protects against
   non-finite *gradients*, not non-finite *losses*.
4. **The default optimizer's schedule-free part does almost nothing.** In adan_sf with SF on, `z` takes raw-gradient
   SGD steps: in the probe it moved 0.003 in norm, against 12 for adamw_sf. Adan-SF learned like Adan without SF
   (30.13 vs 30.17 dB), and both were slower than AdamW (31.05 dB) on the CPU toy (F01 confirmed). This belongs to
   the optimizer lane / R4.1; it is listed here because it caps how fast the templates can learn.
5. **The GAN mechanisms for S.1 are designed and probed** (section 6):
   - `gan_opt.start_iter` + `ramp_iters`, with a generic "idle phase", so no if-chain is needed;
   - lazy R1/R2 (`reg_every`, fp32, SDPA math backend). The probe shows that metagan double-backward *fails* with the
     default SDPA kernel and works with the math backend; a regularised D step costs +54%, about +3.4% amortised at
     k=16;
   - relativistic paired loss `gan_type = "rpgan"` (SR has natural real/fake pairs);
   - a one-forward D step on `cat([gt, sr])`: same outputs, 19-48% faster D step on CPU, and it removes the
     two-forwards-before-one-backward pattern that blocks compiling D.

   None of these change defaults. Template GAN/D defaults should change only after the A/B in section 7.

Rejected (owner's line, and the evidence agrees): automatic loss re-weighting, adaptive clipping, adaptive D
skipping, dynamic batch/patch, plateau LR automation, early stopping. neosr's always-on `grad_clip = 1.0` looks like
"aggressive clipping": pre-clip norms are 40-150, so every step is rescaled 40-150x. The probe shows no learning cost
for Adam-family optimizers (AdamW +0.13 dB, Adan +0.15 dB at 400 steps, from scale invariance). Keep it. Do not
add adaptive clipping.

## 1. Codex audit items in this lane (analyze.md)

| id | verdict | evidence (this lane) |
|---|---|---|
| F01 adan_sf first step uphill / z recurrence | **confirmed** | Snippet reproduced: train 1.0, eval 1.0002040863. Also measured: `z` is plain SGD on the (clipped) raw gradient (`adan_sf.py:66`), so with clip=1 it moves 0.042 vs 0.244 without clip (p1). At loss scale 1 it moves 0.003 in 800 steps vs 12 for adamw_sf (p1b). SF on ≈ SF off (30.13 vs 30.17 dB). |
| F20 EMA decay range unchecked | **confirmed** | `Ema(net, 1.5)`: live 1→2 gives EMA 0.5 (extrapolation); `ema=1.0` freezes after the copy. `validate_train` (`config.py:361-411`) checks only `ema_power`. |
| §6 table: hard output clamp `[1/255,1]` | **confirmed, quantified** | 34-84% dead-gradient pixels at init for 11/19 archs (p2). Learning cost (p2b): span -9.4 dB @200, omnisr -7.5 dB @200; transient, equal by ~600-1000 updates (§3.3). Code default true, 78/84 templates. |
| §6 table: `fast_matmul` bundles fp16 accumulation | **confirmed** | `runtime.py:96-101` sets `allow_fp16_accumulation = True` with TF32. It only affects fp16 cuBLAS GEMMs (the fp16 AMP path: metagan's Linear/SDPA projections, transformer G). The templates use bf16, so they are unaffected. |
| §6 table: CPU AMP ineffective | **confirmed** (= PLAN S.13) | `phase.py:71-74` hard-codes `device_type="cuda"`. |
| §6 table: finite-skip is not a transaction | **confirmed, harmless** | The D is always in train mode, so spectral-norm u/v (and BN/QAT buffers) advance in the forwards of a skipped update. Document it; no fix. |
| §10.3 measure loss influence by gradients | **confirmed, quantified** | p3: FDL has 97.7-98.5% of the squared G gradient (§3.1). |
| §10.4 clipping permanently active | **confirmed, refuted as harmful** | Acceptance: pre-clip G norm 40-150, D 3-250 vs threshold 1.0, so every step is clipped. p1: no learning cost for AdamW/Adan (scale invariance). The only scale-dependent consumer is adan_sf's `z` (F01). |
| §10.6 GAN balance ≠ bounded logits | **partial** | Logits are bounded (acceptance). But the GAN's share of the update is 0.2-0.5% (p3), so "balanced" partly means "almost inactive". |

## 2. Competitor stability mechanisms (catalogue)

C = `/home/korvin/apps/traiNNer-redux-dev/traiNNer`. "Rejected" = the automatic damping style the owner rejects, or broken.

| mechanism | where | what it does | evidence it helps | cost / defects | verdict for neosr |
|---|---|---|---|---|---|
| GAN `start_iter`, `target_iter/target_weight`, `disable_after`, linear/cosine/step ramps | `C/losses/iterative_loss_wrapper.py:48-232`, `C/losses/__init__.py:78-111`; used `C/models/sr_model.py:693,948` | weight schedule per loss; while inactive, D is not updated (`skip_d_update`) | own ESRGAN runs use `start_iter: 1000` (`experiments/4x_ESRGAN/ESRGAN7.yml`); no ablation in the logs | 290-line wrapper with `hasattr(loss,"loss_module")` branching in the loop; D weight multiplied by the G schedule (H3) | **port the idea** as `gan_opt.start_iter` + `ramp_iters` inside `gan_loss` (§6.1) |
| R3GAN loss | `C/losses/r3gan_loss.py:121-345` | relativistic-*average hinge* with detached means (`:234-239`), lazy R1/R2 every `reg_interval=4` D calls, weight × interval, fp32 penalty forward, 96 px random crop | logs: `r1_penalty` 1e-3 → 0.1 over 12k it, `grad_norm_d` 0.13 → 48; no NaN in 60 logs | not R3GAN (not pairwise), `.item()` sync per D step (`:245-248`) + 2 per crop (`:211-212`), silent `create_graph=False` fallback makes R1 a no-op (`:70-103`), step counter buffer in the loss | **rewrite** (H1 #10): rpgan + lazy R1/R2 keyed on `ctx.it` (§6.2-6.3) |
| `r1_penalty`, `gradient_penalty_loss` (WGAN-GP) | `C/losses/gan_loss.py:173-251` | classic penalties | unused in C | GP norm over channels only (H1) | R1 only, own code |
| hinge / softplus (non-saturating) GAN | `C/losses/gan_loss.py:42-126` | standard objectives | standard | none | next stage (cheap) |
| EMA (ema_pytorch) `ema_decay`, `ema_power`, `update_after_step`, `ema_switch_iter` | `C/models/sr_model.py:250-288,1057-1059` | decay warm-up; optional periodic copy EMA→online | warm-up: used in every C experiment (0.75) | `ema_switch_iter` resets the live model to the EMA: a damping/regulariser, no evidence | warm-up **already ported** (decision #9); skip `switch_iter` |
| LR warm-up `warmup_iter` | `C/models/base_model.py:375-396` | sets `lr = initial_lr * it / warmup` after the scheduler step | all C experiments 200-10000 | overrides the scheduler output each step; interacts badly with skipped steps (H3 #8) | **port** as a chainable `LinearLR` in `Phase.schedulers` (§6.5) |
| Schedulers: CosineAnnealingWarmup, KneeLR, WSD, all torch | `C/schedulers/*`, `C/models/base_model.py:262-319` | more schedule shapes | none in logs | KneeLR uses G's peak lr for D too | now: only the LinearLR warm-up (§6.5); other shapes next stage |
| grad clip `grad_clip` + `grad_clip_max_norm` | `C/models/sr_model.py:290-297,927-931,1029-1032` | default **off**, max norm 1.0 | C ESRGAN runs: off (`ESRGAN7.yml`), on in the resumed run with `grad_norm_g` ≈ 1.1-1.5, so it barely clips | computes the norm twice (H3 #10) | neosr keeps on at 1.0 (p1: harmless) |
| AdaptiveGradientClipping | `C/utils/training_automations.py:870-1096` | threshold starts at 0.8-1.0, ×1.2 if >10% of steps clipped, ×0.8 if mean < 20%, bounds 0.05-10 | none | **one instance fed G and D norms** (`C/models/base_model.py:1002-1015` is called with G grads at `sr_model.py:911` and D grads at `:1013`), so one shared history and threshold; `.item()` per call (`:933`); with neosr-scale norms (40-150) it would clip at its 10.0 cap forever | **rejected** |
| DynamicLossScheduler | `C/losses/dynamic_loss_scheduling.py:284-343` | multiplies weights by 1 ± rate·Δ: boosts *decreasing* losses, cuts increasing ones | none | non-stationary objective, positive feedback toward easy losses; several `.item()` per iteration | **rejected** |
| adaptive_d | `C/models/sr_model.py:745-767` | skips D when smoothed G-GAN loss > 1.02 × its own EMA | none | sync, state not saved, broken skip flags (H3) | **rejected** |
| IntelligentLearningRateScheduler / IntelligentEarlyStopping | `C/utils/training_automations.py:139-358,1099-1392` | loss-plateau LR multipliers, patience stopping | none | host float per iteration: `float(l_g_total.detach().cpu())` at `sr_model.py:893` runs **even when automations are disabled** (one forced sync per iteration) | **rejected** |
| DynamicBatchAndPatchSize / VRAM manager / OOM recovery | `C/utils/training_automations.py:360-868`, `C/utils/vram_manager.py`, `C/train.py:758-858` | shrink batch/patch on VRAM pressure or OOM, retry | none | changes the objective (patch statistics) mid-run; complexity | **rejected** |
| NaN handling | `C/models/sr_model.py:869-881` | `if not l_g_total.isfinite()`: skip the iteration, raise after >10 consecutive | 0 NaN events in 60 C logs (grep) | per-iteration host sync; skips D too; nothing for non-finite *grads* except the scaler | neosr's grad-norm skip is better; port only the "a NaN loss is a skip, not a crash" policy (§4 S-02) |
| GradScaler per net, scaler only for fp16 | `C/models/sr_model.py:238-241,933-937` | one scaler per optimizer, default init 65536 | fp16 path | bf16→fp16 fallback runs fp16 without a scaler (H3) | neosr already per-phase; init scale see S-12 |
| D update in train mode, D params frozen in the G pass | `C/models/sr_model.py:648,959` | correct GAN step | — | — | neosr already does this better (`backward(inputs=G params)`) |
| ECO teacher curriculum | `C/utils/eco.py`, `sr_model.py:600-633` | targets lerp teacher-SR→HR over 75% of the run | paper claim; no C log | needs a teacher net; changes objective | next stage, optional |
| MoA batch augment (mixup/cutmix/resizemix/cutblur) | `C/ops/batchaug.py` | augmentation | — | — | neosr has it (`data/augmentations.py`) |
| `detect_anomaly` flag | `C/train.py:273` | debug | — | slow | next stage (1 line) |

What C's logs show (`experiments/4x_ESRGAN/train_4x_ESRGAN_20260107_205351.log` from scratch, then the resumed
`..._215959.log`). The setup is AdamW G 1e-4, D 5e-6, warm-up 1000, charbonnier 1.0 + hsluv 0.1 + r3gan 0.1 from
update 1000, R1=R2=3, no clipping, EMA 0.999/0.75:

- PSNR (EMA, 8 val images): 28.05 @1k, 28.60 @2k, 29.01 @8k, a peak of 29.47 @23k, then 28.6-29.2 to 37k as D
  sharpens.
- `grad_norm_d` 0.13 @2k → 15 @6k → 48 @12k; `r1_penalty` 1e-3 → 0.1; `l_g_gan` ≈ 0.009-0.014 (weighted).
- `grad_norm_g` 9.8 @100 → 1.4 @12k.
- No NaN/skip line in any of the 60 C logs.

## 3. Measurements

### 3.1 Per-term generator gradients, esrgan template mix (p3, `p3_term_grads.py`)

The objective is neosr's own `build_losses` on `train_esrgan.toml [train]`: MS-SSIM 1, consistency 1, LDL 1,
FDL(dinov2) 0.75, GAN bce 0.3. G = esrgan, D = metagan, real LQ 64 / GT 256, batch 2. "old3236" = the owner's
pre-refactor run weights (`experiments/train_esrgan/models/net_[gd]_3236.pth`) loaded as pretrained.

| term | init: ‖g‖ | init: share of Σ‖g‖² | init: cos(g, rest) | old3236: ‖g‖ | old3236: share | old3236: cos(g, rest) |
|---|---|---|---|---|---|---|
| fdl (0.75) | 221.1 | **97.7%** | -0.07 | 69.7 | **98.5%** | 0.28 |
| consistency | 29.1 | 1.7% | 0.05 | 7.8 | 1.2% | 0.27 |
| mssim | 14.1 | 0.4% | -0.03 | 1.6 | 0.05% | 0.25 |
| gan (0.3) | 9.4 | 0.18% | 0.08 | 3.6 | 0.26% | -0.09 |
| ldl | 0.13 | 0.0% | 0.16 | 0.016 | 0.0% | 0.32 |
| total | 222.9 | | | 72.4 | | |

Projection of each term on the total: fdl 0.97/0.96, gan 0.005/-0.002. Adam-family optimizers divide each
parameter by a running RMS that this one term sets, so the update direction is essentially FDL's. FDL amplitude vs
phase split: §3.1b.

### 3.1b FDL amplitude vs phase gradient (p3b)

Same setup, two disjoint batches (b0, b1), FDL weight 0.75 split into amplitude (`phase_weight=0`) and phase:

| part | init ‖g‖ b0/b1 | init cos(b0,b1) | old3236 ‖g‖ b0/b1 | old3236 cos(b0,b1) | value ×0.75 (init / old) |
|---|---|---|---|---|---|
| FDL amplitude | 337 / 465 | 0.67 | 69.3 / 73.0 | **0.05** | 22.0 / 9.19 |
| FDL phase | 13.7 / 20.4 | -0.05 | 6.4 / 9.9 | 0.11 | 0.98 / 0.96 |
| L1 (for scale) | 19.7 / 23.1 | 0.98 | 3.6 / 6.6 | 0.99 | — |

The dominance comes from the **amplitude** term's scale, not from phase-angle noise. The scale is an upstream
choice: `fdl_loss.py:361` multiplies the score by 0.01 for VGG/ResNet/EffNet but not for DINOv2 (introduced in
`cb9b92f` "make dinov2 the default", upstream author). At the partly trained point, two batches' FDL gradients are
almost orthogonal (cos 0.05, n = 2 batches of 2, low confidence), so most of each Adam step is batch noise. This
matches the old run's `l_g_fdl` barely moving (8.98 → 8.71 over 1000 updates, `experiments/train_esrgan/*.log`).

### 3.2 Gradient clipping vs optimizer (p1, p1b)

compact (32 feat, 8 conv), L1, real crops (LQ 32), batch 8, CPU. Val PSNR on 64 held-out crops: bicubic 30.61,
nearest 29.35. Loss ×1000 puts the pre-clip norm at 20-130, neosr's measured regime.

| run (loss ×1000, lr 5e-4) | @100 | @200 | @300 | @400 | ‖g‖ |
|---|---|---|---|---|---|
| adamw, no clip | 29.36 | 29.63 | 30.14 | 30.45 | 21 |
| adamw, clip 1.0 | 29.43 | 30.01 | 30.37 | **30.58** | 31 |
| adan, no clip | 29.34 | 29.36 | 29.46 | 29.67 | 25 |
| adan, clip 1.0 | 29.36 | 29.47 | 29.57 | **29.82** | 24 |
| adan_sf (SF), no clip | 29.35 | 29.35 | 29.42 | 29.55 | z moved 0.244 |
| adan_sf (SF), clip 1.0 | 29.25 | 29.35 | 29.43 | 29.52 | z moved 0.042 |

| run (loss ×1, no clip active, 800 steps) | @200 | @400 | @600 | @800 |
|---|---|---|---|---|
| adamw 1e-3 | 29.99 | 30.77 | 30.97 | **31.05** |
| adamw 1e-3 + LinearLR warm-up 100 | 29.68 | 30.69 | 30.95 | 31.00 |
| adamw 5e-4 | 29.64 | 30.44 | 30.75 | 30.88 |
| adamw_sf 1e-3 (wu 100) | 29.47 | 30.05 | 30.37 | 30.56 (z moved 12) |
| adan 1e-3 | 29.44 | 29.88 | 30.25 | 30.47 |
| adan_sf 1e-3 SF on (wu 100) | 29.34 | 29.52 | 29.91 | 30.13 (z moved **0.003**) |
| adan_sf 1e-3 SF off | 29.35 | 29.50 | 29.88 | 30.17 |

Reading: (a) Always-on clipping does not slow AdamW or Adan; it was slightly ahead in both. (b) adan_sf's
schedule-free half is inert: `z` takes `lr * raw_grad` steps (`adan_sf.py:66`), so it stays at its init, and SF on ≈
SF off. (c) On this toy problem the Adan family is 0.6-0.9 dB behind AdamW at 800 steps. That is a hypothesis for
real SR (tiny model, short run, untuned Adan betas). It needs the GPU A/B in §7, and it is the optimizer lane's call.

### 3.3 Output clamp (p2, p2b)

Fraction of output pixels outside `[1/255, 1]` at init (default arch config, real LQ crops). Those pixels get zero
gradient from every loss while `train.clamp = true`:

| arch | dead | arch | dead | arch | dead |
|---|---|---|---|---|---|
| realplksr | **0.835** | eimn | 0.582 | span | 0.566 |
| omnisr | 0.542 | man | 0.541 | rcan | 0.493 |
| ninasr (clamp off in template) | 0.466 | flexnet | 0.409 | esc | 0.399 |
| spanplus | 0.339 | swinir | 0.184 | sebica | 0.116 |
| plksr | 0.062 | cugan | 0.040 | esrgan | 0.019 |
| compact | 0.005 | hat | 0.000 | dat | 0.000 |

Learning with and without the clamp (p2b): span at its default config, L1, AdamW 5e-4, batch 8, real LQ-32 crops.
Val PSNR on 32 held-out crops (the metric clamps to [0,1] for both arms; bicubic 30.61). `dead` = fraction of
clamped pixels in the training batch.

| update | 1 | 10 | 25 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|
| clamp off: PSNR | 6.82 | 15.04 | 19.79 | 24.07 | 25.10 | 25.76 | **28.11** |
| clamp off: dead | 0.56 | 0.003 | 0.001 | 0.054 | 0.005 | 0.003 | 0.000 |
| clamp on: PSNR | 6.94 | 12.58 | 15.15 | 17.07 | 17.34 | 17.57 | **18.66** |
| clamp on: dead | 0.56 | 0.21 | 0.11 | 0.15 | 0.11 | 0.11 | 0.07 |

Without the clamp, the out-of-range outputs are pulled back into range within 10 updates. With the clamp they stay
dead (7-21%): a pixel below 1/255 gets zero gradient whatever its target, so dark-predicted regions of bright
content only recover through weight sharing. At 200 updates the gap is **9.4 dB**.

omnisr, same setup: clamp off 25.05 / 26.92 / 28.50 vs clamp on 19.97 / 20.55 / 20.98 at updates 50 / 100 / 200
(dead stays 4.5-6%). span over 1000 updates:

| update | 300 | 400 | 500 | 600 | 800 | 1000 |
|---|---|---|---|---|---|---|
| clamp off | 28.76 | 29.43 | 29.75 | 29.48 | 30.18 | 30.43 |
| clamp on | 20.91 | 24.13 | 24.15 | 29.54 | 29.93 | 30.42 |

The cost is a **transient stall of roughly 500 updates**, not a lasting quality loss (L1, one seed, CPU). It
matters:
- for short fine-tunes from a random init;
- for the early D, which learns from stalled, partly flat outputs when the GAN starts at update 0;
- for archs with extreme init ranges (realplksr: 84% dead; not run to completion here).

`consistency_loss` already clamps its own inputs (`consistency_loss.py:160-161`), MS-SSIM is guarded
(`ssim_loss.py:146-148`), and LDL is floored (`ldl_loss.py:87-90`). The global clamp is no longer a NaN guard for the
shipped losses. It is an objective choice, with the dead-gradient cost above.

### 3.4 Transient non-finite loss (p7, reproduced)

```
update 3: step ok=False, weights finite=True      # Phase skipped it, as designed
skipped: 1 streak: 0
LossMeter.read raised: non-finite training values ['l_g_pix', 'l_g_total']; nothing was saved from this window
```

`Phase.backward` logs the micro-batch total before backward (`phase.py:85`) and the loss logs its value
(`base.py:66`). `LossMeter.add` sums them unmasked (`phase.py:237-241`), and `read` raises (`phase.py:254-257`), so
`train.py:241` dies at the next print (≤ `print_freq` = 100 updates later). The readme (l. 190-191) documents this,
but it makes `max_skip_streak` dead for loss-level NaN/inf, which is exactly what an fp16 forward overflow produces.

### 3.5 R1 double backward and the one-forward D step (p4, p4b)

metagan (default config), B=4 at 128 px, CPU:

- R1 with the default SDPA: `RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward
  is not implemented`. On CUDA, the flash and mem-efficient SDPA backward kernels have no derivative either
  (PyTorch derivative table; not run here, GPU reserved).
- With `sdpa_kernel(SDPBackend.MATH)` around the regularised forward: works, finite grads (RMSNorm, GroupNorm,
  spectral_norm and Mish all double-differentiate on CPU). D step: plain 10.0 s vs R1 15.4 s (**×1.54**). At
  `reg_every = 16` that is +3.4% of D time.
- Concat forward `D(cat([gt, sr]))` vs `D(gt), D(sr)`: max |diff| 0.0 in eval mode (GroupNorm/RMSNorm are per
  sample). D fwd+bwd 6.3 → 5.1 s and 6.2 → 3.2 s in two runs (19-48% faster, noisy CPU).
- CUDA double backward of the fused RMSNorm kernel is not run. On CPU, `F.rms_norm` decomposes, but calling the
  fused op `torch.ops.aten._fused_rms_norm` directly also double-differentiates. Its autograd formula is device
  independent, so CUDA is likely fine; the GPU lane must still confirm it before R1 ships.

### 3.6 Cost shares (p6, forward GFLOP per image, LQ 64 → 256)

esrgan 146.9, omnisr 7.1, compact 5.1, span 4.9, **metagan D 1.27**, **FDL dinov2 41.3** (run on sr, and on gt
without grad, with backward through sr). Per iteration, D work is about 3 forwards + 2 backwards ≈ 8 D-forwards ≈
10 GFLOP/img. That is ≈ 2% of esrgan's G fwd+bwd (≈ 440) but ≈ 2/3 of compact's (≈ 15), i.e. ≈ 40% of compact's G+D. FDL is ≈ 120-160 GFLOP/img,
which makes it the largest single cost for every light generator. FLOPs are not time (FDL runs fp32 by default,
`fp32_backbone=True`); the GPU lane owns the timings.

### 3.7 Learning speed vs the competitor (what can and cannot be concluded)

| | neosr acceptance (esrgan2, bf16, 7 val imgs) | competitor ESRGAN (8 val imgs incl. `ar.png`) |
|---|---|---|
| objective | FDL-dominated (98% of grad) + small MS-SSIM/consistency/LDL/GAN 0.3 from 0 | charbonnier 1 + hsluv 0.1 + r3gan 0.1 from 1000 |
| optimizer | adan_sf G 8e-4 / D 1e-4, clip 1.0 | AdamW G 1e-4 / D 5e-6, warm-up 1000, no clip |
| PSNR @1k / @2k | live 23.5 / 26.2; EMA 6.6 / 19.2 (pre-`ema_power`) | EMA 28.05 / 28.60 |
| bicubic on the neosr val set | 28.42 dB (computed here) | (same images ± `ar.png`) |
| NaN / skips | 0 / 0 (2000 updates × 3 runs) | 0 (60 logs) |
| D behaviour | logits within ±1, `l_g_gan` raw ≈ BCE 0.6-1.4 (near chance) | `grad_norm_d` rises 0.1 → 48 by 12k; PSNR drops 0.8 dB after 23k |

The pre-refactor neosr logs (`/home/korvin/apps/neosr-dev/experiments/train_esrgan/*.log`, updates 2300-3200,
same template mix) show the old failure mode for contrast: `l_g_gan` 4.9-9.2 *weighted* (raw BCE 16-30, so D was
fully saturated with a dead G-side gradient, H3/G1.1). `l_g_fdl` was flat at 8.6-9.0. The refactor removed the
saturation; the acceptance runs' D sits near chance.

Can conclude:
- both trainers are now numerically stable over these horizons;
- neosr's default objective leaves the generator below bicubic fidelity after 2000 updates, while an L1-led
  objective is above it;
- neosr's D stays near chance, which together with p3 says the GAN term is weak.

Cannot conclude:
- that neosr learns "worse" perceptually. PSNR is not FDL's target, and no DISTS/TOPIQ was logged on both;
- anything about horizons beyond 2k (neosr) or about C's D at neosr's settings;
- which factor (objective, optimizer, clamp, GAN start) causes the gap.

Hence the A/B design in §7.

## 4. Findings (ranked)

Priority is value for the owner's goals (speed, reliability, memory, effective training). "Numerics" says whether a
fix changes training and therefore needs the §7 A/B before it becomes a default.

### S-01 (P1, learning) The template objective is FDL-only in effect — measured
- Where: `options/train_*.toml` (84/84 `fdl_opt` dinov2, 83 at 0.75); `neosr/losses/fdl_loss.py:361` (no 0.01 for
  dinov2).
- Evidence: §3.1/3.1b. FDL has 97.7-98.5% of Σ‖g‖², and that comes from its amplitude term. GAN 0.3 gets
  0.2-0.5%; LDL 0%. FDL dinov2 is 41 GFLOP/img forward. The acceptance esrgan run was 26.2 dB live @2k vs 28.42
  bicubic.
- Fix:
  1. Ship the p3 probe as `scripts/term_grads.py <config> [state]` (offline; zero trainer cost). It prints per-term
     gradient share and cosine, so a weight choice is a measured, static decision rather than an automatic
     re-weighting.
  2. A/B A1/A2 (§7) before touching template weights. If FDL dominance is intended, drop the terms that cost compute
     and do nothing (LDL at 0%, maybe the GAN).
- Numerics: objective (template-only). Checkpoints: none.

### S-02 (P1, reliability) One transient non-finite loss aborts the run — reproduced
- Where: `neosr/models/phase.py:85,237-257`; `neosr/losses/base.py:66`; `train.py:236-243`; readme l. 190-191.
- Evidence: p7 (§3.4). The step is skipped, the weights stay finite, then `FloatingPointError` comes at the next
  print. `max_skip_streak` therefore only covers gradient overflow. An fp16 forward overflow or one bad batch kills
  a multi-day run and loses up to `save_checkpoint_freq` updates.
- Fix: 6.6 (masked device accumulation + `nonfinite_<k>` counts; stop only on the streak or on `check_finite`).
- Numerics: none. Test: p7 as a unit test.

### S-03 (P1, learning; cross-lane) The default optimizer's schedule-free part is inert, and the first step goes uphill (F01) — reproduced + measured
- Where: `neosr/optimizers/adan_sf.py:58-66,165-170`; all 84 templates (G and D).
- Evidence: F01 snippet reproduced. `z ← z − lr·g` uses the raw (clipped) gradient: it moved 0.003 vs 12 for
  adamw_sf over 800 steps. SF on ≈ SF off (30.13 vs 30.17 dB). On the toy, the Adan family is 0.6-0.9 dB behind
  AdamW (31.05) @800 (§3.2).
- Fix: R4.1 redesign against the reference SF recurrence, keeping a legacy mode for old states. Meanwhile A/B C1
  (AdamW + `warmup_steps`). The optimizer lane owns the math.
- Numerics: yes. Checkpoints: optimizer-state semantics (version the state).

### S-04 (P2, learning) The training output clamp stalls early learning (34-84% dead pixels at init) — measured
- Where: `neosr/models/image.py:79,223-224` (`clamp` default true); 78/84 templates inherit it.
- Evidence: §3.3: 11/19 archs ≥ 34% dead; span -9.4 dB and omnisr -7.5 dB at update 200; span equal again by
  600-1000. The shipped losses already guard their own domains, so the clamp is no longer needed as a NaN guard.
- Fix: keep the code default (backward compatible). After A/B B1, set `clamp = false` in the templates, as the six
  esrgan/cugan/ninasr templates already do. Optional later: clamp only the inputs of losses that need a bounded
  domain (they already do it internally).
- Numerics: objective → A/B.

### S-05 (P2, stability/speed) `gan_opt.start_iter` + `ramp_iters` with a generic idle phase — design (PLAN S.1)
- Where: `losses/base.py`, `losses/gan_loss.py`, `models/phase.py:82-112`.
- Evidence: the competitor uses it in every experiment, but without ablation. Here the value is (a) no D compute
  before the start (≈40% of G+D compute for a light G, §3.6) and (b) D does not fit a near-random G.
- Fix: 6.1. Defaults 0/0 are bit-identical. Resume/DDP-safe, no sync. Template change only after D1.

### S-06 (P2, stability) Lazy R1/R2 D regularisation — design + feasibility measured
- Where: `losses/gan_loss.py:53-73`.
- Evidence: p4/p4b. Double backward fails through metagan's default SDPA kernel and works with the math backend;
  ×1.54 per regularised step, +3.4% amortised at k=16. CUDA RMSNorm double backward and DDP still need checks.
- Fix: 6.2 (fp32 reg pass, `ctx.it % reg_every`, γ/2·k scaling, per-sample score). It is a correct-gradient
  stabiliser (D smoothness), not damping of G.
- Numerics: objective when enabled → A/B D2.

### S-07 (P2, stability) Relativistic paired GAN (`rpgan`) — design
- Fix: 6.3 (≈20 lines in `gan_loss`; one no-grad D forward on gt in the G pass). With R1/R2 it is the R3GAN
  baseline; it uses SR's natural pairing.
- Numerics: objective → A/B D3.

### S-08 (P2, speed) One concatenated D forward per D step — measured (CPU)
- Where: `losses/gan_loss.py:62-64`.
- Evidence: p4: identical outputs (eval), D fwd+bwd 19-48% faster on CPU. It unblocks compiling D
  (`models/base.py:136-138`).
- Fix: 6.4 (keep two forwards for D with BatchNorm).
- Numerics: not bit-identical (one SN power iteration, a different dropout draw), statistically equivalent.

### S-09 (P2, AMP numerics) `fast_matmul` silently enables fp16 GEMM accumulation — source-verified (analyze.md §6)
- Where: `neosr/runtime.py:96-101`.
- Evidence: `allow_fp16_accumulation = True` affects fp16 cuBLAS GEMMs under fp16 AMP: metagan's
  `qkv`/`proj`/`fc1`/`fc2`, transformer G's attention and MLP layers. It is irrelevant for bf16 (the templates).
- Fix: drop it from `fast_matmul`; add an explicit `fp16_accumulation = true` opt-in (top-level bool).
- Numerics: the fp16 path only (more exact). Test: flag matrix in `tests/test_runtime.py`.

### S-10 (P2, config) EMA decay is not range-checked (F20) — reproduced
- Where: `neosr/config.py:361-411`, `neosr/models/image.py:61-62`, `neosr/models/phase.py:192-210`.
- Evidence: `ema=1.5` makes the EMA 0.5 when live moves 1 → 2; `ema=1.0` freezes it; NaN disables it.
- Fix: `validate_train`: `0 <= ema < 1` and finite (0 = off).
- Numerics: none. Test: parametrised ConfigError.

### S-11 (P2, stability) LR warm-up for plain optimizers (PLAN S.2) — design + composition verified
- Where: `neosr/models/image.py:181-197`, `neosr/models/base.py:63-69`, `neosr/config.py:186`.
- Evidence: plain adam/adamw/nadam/adan have no warm-up, and `warmup_iter` was removed. The chained LinearLR is
  exact with MultiStepLR/Cosine and resumes exactly (§6.5). Toy cost: 29.68 vs 29.99 dB at 200 steps, equal by
  800.
- Fix: 6.5. Default 0.
- Numerics: only when set.

### S-12 (P3, AMP) fp16 GradScaler starts at 2^5 and grows every 2000 updates — source-verified (arithmetic)
- Where: `neosr/models/phase.py:72-74` (`init_scale=2.0**5`, default `growth_interval=2000`).
- Evidence: reaching the usual 2^16 takes 11 doublings × 2000 = 22k updates. Acceptance: still 32 → 64 at u2000. A
  mean-reduced pixel loss on 8×3×256² outputs gives 6.4e-7 per element; ×32 = 2.0e-5, below the fp16 minimum
  normal 6.1e-5, i.e. subnormal gradients (~8 bits) for the first ~20k updates of pixel-led fp16 objectives. The
  FDL-dominated templates were not affected (fp16 matched bf16 at 2k).
- Fix: `growth_interval=200` (keeps the safe start; reaches 2^16 in ~2.2k updates; at most one overflow skip per
  200 updates, far from `max_skip_streak`).
- Numerics: the fp16 path → short fp16 L1 A/B.

### S-13 (P3, keep) Always-on `grad_clip = 1.0` looks aggressive but does not slow learning — measured (refutes a suspected problem)
- Where: `neosr/models/image.py:71`, `neosr/models/phase.py:90-94`.
- Evidence: pre-clip norms are 40-150 (acceptance). p1: AdamW 30.58 vs 30.45 dB and Adan 29.82 vs 29.67 dB, clip
  vs no clip. Adam-type updates are scale-invariant; only adan_sf's `z` is scale-dependent (S-03).
- Fix: none. Do not add adaptive clipping. If a user wants spike-only clipping, a later `grad_clip = <float>` is a
  one-line schema change, but no evidence asks for it.

### S-14 (P3, stability) Abrupt D input switch at `wavelet_init` — source-verified + acceptance evidence
- Where: `neosr/losses/wavelet_guided.py:42-44`; templates `wavelet_init = 80000`.
- Evidence: the acceptance switch at 500 gave `grad_norm_d` 7 → 47 (bf16) and 18 → 49 (fp16), which settled.
  With R1 (6.2) the penalty also jumps at the switch.
- Fix (optional): a linear cross-fade `view = lerp(x, swt(x), clamp((it - start)/fade, 0, 1))`. It is a pure
  function of `ctx.it`, and the default `fade = 0` keeps today's behaviour.
- Numerics: only when set.

### S-15 (P3, doc) A skipped update is not a transaction; CPU AMP is ignored — source-verified (analyze.md §6, PLAN S.13)
- Where: `neosr/models/phase.py:71-74,96-112`.
- Evidence: D is always in train mode, so its spectral-norm u/v (and any BN/QAT buffers) advance in the forwards of
  a skipped update. Autocast is hard-coded to `"cuda"`.
- Fix: one readme sentence; `validate_train` warns on `use_amp` without CUDA.
- Numerics: none.

## 5. What not to do

- No DynamicLossScheduler-style re-weighting, no adaptive clip threshold, no adaptive D skipping, no plateau LR
  automation, no dynamic batch/patch. Each one is a non-stationary objective or a host sync per iteration, and none
  has evidence in C's own logs.
- Do not lower `grad_clip` or make it adaptive; p1 shows the current clip is neutral for Adam-family optimizers.
- Do not port `r3gan_loss.py` (wrong formulation, syncs, silent no-op fallback) or `ema_switch_iter`.
- Do not weight D's loss by the G GAN weight (C bug, H3).

## 6. Designs (fit the loss Context contract; no trainer if-chains; resume- and DDP-safe; no host syncs)

### 6.1 `gan_opt.start_iter` + `ramp_iters`, with a generic idle phase (S.1)

Contract: a term can be off at an update; a phase whose terms are all off does nothing. The trainer loop is unchanged.

```python
# losses/base.py
class Loss(nn.Module):
    def active(self, it: int) -> bool:          # a term that is off at update `it` costs nothing
        return True

# losses/gan_loss.py
class gan_loss(Loss):
    def __init__(self, gan_type="bce", real_label_val=1.0, fake_label_val=0.0, wavelet_init=None,
                 start_iter: int = 0, ramp_iters: int = 0, r1_gamma: float = 0.0,
                 r2_gamma: float = 0.0, reg_every: int = 16): ...
    def active(self, it: int) -> bool:
        return it > self.start_iter              # ctx.it is 1-based: start_iter=0 -> always on
    def evaluate(self, ctx):                     # host float: no sync, pure function of ctx.it
        w = 1.0 if self.ramp_iters <= 0 else min(1.0, (ctx.it - self.start_iter) / self.ramp_iters)
        return w * self.g_term(ctx)

class gan_d_loss(Loss):
    def active(self, it: int) -> bool:
        return self.g.active(it)                 # D trains at full strength from start_iter + 1

# models/phase.py
def backward(self, ctx):
    terms = [t for t in self.terms if t.active(ctx.it)]
    self.idle = not terms
    if self.idle:
        return                                   # no forward: D is never called before start
    with torch.autocast(**self.autocast):
        total = sum(t.compute(ctx) for t in terms)
    ...
def step(self) -> bool | None:
    if self.idle:                                # no grads, no optimizer step, no skip count
        self.opt.zero_grad(set_to_none=True)
        for s in self.schedulers:                # the schedule clock stays = update index
            s.step()
        return None
    ...
```

- `image.optimize_parameters` needs no change: `ok[self.d] = None` is falsy, so no `grad_norm_d` is logged and the
  skip count is untouched. A plain-optimizer D warm-up (§6.5) would be used up while D is idle. For a cold D,
  configure `ramp_iters` instead; that is the job of the ramp. adan_sf/adamw_sf count their own warm-up in real
  steps (`group["step"]`), so their D warm-up starts at `start_iter` automatically.
- Resume: the whole schedule is a pure function of `ctx.it` (no state). DDP: every rank skips D at the same
  updates, so the reducer stays consistent. Compile: G's graph is unchanged; the ramp is a Python float multiplied
  outside G.
- Speed: before `start_iter` the D forwards and backwards disappear: ≈2% of esrgan's G compute, ≈40% of compact's G+D
  compute (§3.6; FDL excluded).
- Defaults 0/0 are bit-identical to today. New config keys live inside the opaque `train.gan_opt` table; the
  constructor signature validates them (`utils/factory.py:38-49`). Reject `start_iter < 0`, `ramp_iters < 0`.
- Tests: (a) D mock that raises if called for `it <= start`; D optimizer state empty and schedulers advanced;
  (b) ramp weights at `start+1 … start+ramp`; (c) the E1 resume harness with `start_iter=2` across the save point;
  (d) the gloo DDP test with `start_iter=1`.

### 6.2 Lazy R1 / R2 (StyleGAN2 lazy regularisation), inside `gan_d_loss`

```python
def evaluate(self, ctx):
    g = self.g
    x = torch.cat([g.adv(ctx.gt, ctx.it), g.adv(ctx.sr.detach(), ctx.it)])   # D's input, one forward (6.4)
    reg = (g.r1_gamma > 0 or g.r2_gamma > 0) and ctx.it % g.reg_every == 0     # host ints: no sync
    if not reg:
        real, fake = ctx.net_d(x).chunk(2)
        return self.adv_loss(real, fake, ctx)
    x = x.detach().float().requires_grad_(True)
    with torch.autocast(ctx.sr.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        out = ctx.net_d(x)                                   # fp32; flash/mem-eff SDPA has no 2nd derivative
        score = out.flatten(1).mean(1)                       # per-sample score: penalty independent of map size
        (gx,) = torch.autograd.grad(score.sum(), x, create_graph=True)
        sq = gx.square().flatten(1).sum(1)                   # ||grad_x D||^2 per sample
        b = sq.shape[0] // 2
        pen = g.r1_gamma * sq[:b].mean() + g.r2_gamma * sq[b:].mean()
    real, fake = out.chunk(2)
    ctx.log("l_d_reg", pen)
    return self.adv_loss(real, fake, ctx) + 0.5 * g.reg_every * pen          # gamma/2 * k (lazy scaling)
```

- The penalty is on D's actual input (the AdvView output once `wavelet_init` passes). The regularised step runs in
  fp32 (AMP-safe: no fp16 underflow of input-gradients, no bf16 rounding of the penalty) every `reg_every` updates.
- Cost (p4b): ×1.54 on a regularised D step, so +3.4% of D time at k=16. In-graph `reg` is decided by a host int,
  so DDP ranks agree and there is no host sync. Resume-safe (`ctx.it`). Add `l_d_reg` to `RESERVED`
  (`losses/__init__.py:46`).
- Must verify before shipping: double backward of CUDA `nn.RMSNorm` (metagan `GatedCNNBlock.norm`) in torch 2.9
  (CPU passes), and the 2-rank gloo test with `autograd.grad` through the DDP-wrapped D. StyleGAN2-ADA does this
  under DDP with `broadcast_buffers=False`, as neosr does (`models/image.py:89`).
- Defaults `r1_gamma = r2_gamma = 0`: no change.

### 6.3 Relativistic paired loss `gan_type = "rpgan"` (R3GAN's RpGAN; SR has natural pairs)

- D: `softplus(fake - real).mean()` over aligned patch maps (same LQ crop, so each real/fake pair is the same
  content).
- G: `softplus(real - fake).mean()` with `real = D_adv(adv(gt))` under `torch.no_grad()`. That is one extra D
  forward without grad per update (≈ +12% D time).
- Non-saturating, and with R1+R2 it is the minimal locally convergent GAN (Huang et al. 2024).
- Implemented as one more `crits` entry plus a pair-aware `term()` inside `gan_loss`; the branching stays in the
  loss, the loop has none. `real_label_val/fake_label_val` must stay default for rpgan (ValueError otherwise).
- It changes the objective, so it needs the A/B (§7).

### 6.4 One-forward D step

`real, fake = ctx.net_d(torch.cat([real_in, fake_in])).chunk(2)` in `gan_d_loss.evaluate`.

- Outputs are identical for D without batch statistics (p4: max |diff| 0.0): metagan, unet, dunet, ea2fpn, resgan,
  and patchgan with the default `norm_type="none"`. Keep two forwards when D contains `_BatchNorm`; decide once by
  module scan (host-side, cached).
- D fwd+bwd 19-48% faster on CPU. It also removes the "two forwards before one backward" pattern that blocks
  compiling D (`models/base.py:136-138`, perf.md); compiling D becomes a GPU-lane experiment.
- Not bit-identical: one spectral-norm power iteration instead of two per D step, and one dropout mask draw of size
  2B. Checkpoints are unaffected.

### 6.5 `warmup_steps` for plain optimizers (S.2)

- In `image._phase`, for a non-SF optimizer pop `warmup_steps` from the `optim_*` table and prepend
  `LinearLR(opt, start_factor=1/w, total_iters=w)` to the phase's scheduler list. The SF optimizers already take
  `warmup_steps` as a constructor argument, so one key means the same thing everywhere.
- `Phase.step` already steps every scheduler each update (`phase.py:103-104`). LinearLR and MultiStepLR/Cosine are
  chainable, so composing them is exact. Verified: LinearLR(0.1, 4) + MultiStepLR([6], 0.5) gives
  `[0.1, .325, .55, .775, 1, 1, .5 …]`, identical after a resume at t=2; LinearLR + Cosine equals the product
  exactly.
- Resume identity: the saved scheduler-type list changes, so `Phase.load_state_dict` raises the existing clear
  `ConfigError` on a mismatch.
- Default 0: no change. Remove `train.warmup_iter`'s "use the optimizer's warmup_steps" hint gap
  (`config.py:186`): it then becomes true for every optimizer.

### 6.6 LossMeter: a non-finite loss is a skip, not a crash

- `LossMeter.add`: `v = v.float(); ok = v.isfinite(); sum[k] += torch.where(ok, v, 0); cnt[k] += ok` (device
  tensors, no sync).
- `read`: mean = `sum / cnt.clamp_min(1)`, in the existing single stack + all_reduce + tolist; report
  `nonfinite_<k>` counts.
- The run then stops only on the existing hard signals: `max_skip_streak` consecutive skipped updates
  (`phase.py:107-111`) and `check_finite` on weights before save/validation (`image.py:264-284`). A NaN loss gives a
  NaN gradient norm, so its update is already skipped.
- No numerics change. Update readme l. 190-191. Test: the p7 scenario must not raise and must report
  `skipped_g = 1`, `nonfinite_l_g_total = 1`; a 10-update NaN streak still raises.

## 7. Template defaults and the A/B that must precede them

**Change now (no numerics change):** none of the templates' objective keys. The code-only items (S-02, S-09/S-10
validation, the 6.1-6.6 mechanisms with off-by-default keys) need no A/B.

**Do not change yet:** `gan_opt.loss_weight = 0.3`, GAN from update 0, `optim_d.lr = 1e-4`, `fdl_opt.loss_weight =
0.75`, `clamp` (code default true, templates unset). Every one of them is an objective change.

A/B protocol (GPU lane, the RTX 4090):
- Data and runs: `/home/korvin/apps/data` 10607 pairs, ≥ 20k updates per arm (PLAN S.1), 2 seeds, bf16.
- Validation: the 7-image val set plus ≥ 50 held-out training crops (7 images are too few for 0.1 dB decisions).
- Metrics: PSNR/SSIM (fidelity) **and** DISTS/TOPIQ (perceptual) on EMA and live weights at 1k/5k/20k.
- Logged per arm: per-term gradient shares at 0/5k/20k (the p3 script on a saved state), `out_d_real - out_d_fake`,
  `grad_norm_d`, `l_d_reg`, skips, it/s.
- Change one factor per arm, in this order, because later arms are meaningless while the GAN has 0.3% of the update:

| arm | change vs template (compact + esrgan) | question | decide by |
|---|---|---|---|
| A1 | `fdl_opt.loss_weight` 0.75 → 0.1 (FDL grad ≈ the other terms' sum, p3: 69.7×0.13 ≈ 9 vs 8-10) | is the template FDL-starved of the other terms? | DISTS/TOPIQ not worse, PSNR ≥ bicubic by 5k |
| A2 | A1 + `pixel_opt = {type="L1Loss", loss_weight=1.0}` | fidelity anchor | same |
| B1 | `clamp = false` on span and omnisr (dead 57%/54% at init) | clamp cost on real runs | PSNR/DISTS at 1k/5k; no NaN |
| C1 | `optim_g = adamw 5e-4, warmup_steps 1000` vs adan_sf (needs 6.5; optimizer lane owns it) | optimizer speed (p1b hint) | metric at equal updates and wall time |
| D1 | (after A) `start_iter = 5000, ramp_iters = 2000` | same quality, cheaper early phase? | equal metrics at 20k, higher early it/s |
| D2 | D1 + `r1_gamma = 1, reg_every = 16` | D smoothness without slowing G | D margin bounded, DISTS better or equal |
| D3 | D2 + `gan_type = "rpgan", r2_gamma = 1` | R3GAN-style objective | DISTS/TOPIQ, artifacts |
| D4 | D3 + `optim_d.lr` = G lr | is D under-powered at 1e-4 now that it is regularised? | D margin, DISTS |

Provisional template defaults if the arms confirm the hypotheses (owner sign-off each):
- `clamp = false`;
- an FDL weight from A1/A2;
- `start_iter` ≈ 5k with a 2k ramp;
- `rpgan` with R1 = R2 = 1 every 16 updates;
- D lr from D4.

## 7b. Tests to add with the designs

`tests/test_gan_schedule.py` (6.1 a-c), an R1 test (finite penalty, CPU, metagan small, math SDPA; skips on CUDA
until the RMSNorm double backward is verified), the rpgan sign test (D loss falls when `fake - real` falls), the
concat-vs-two-forwards equality test in eval mode, `tests/test_phase.py::test_nonfinite_loss_is_a_skip` (p7), the
LinearLR chaining + resume test (6.5), an EMA range test (`ema` ∉ (0, 1) → ConfigError), and the gloo DDP case
with `start_iter` + R1.

## 8. Next-stage port list (not now)

- Hinge and softplus (non-saturating) `gan_type`s: `C/losses/gan_loss.py:42-126`, ~10 lines in `gan_loss`.
- D feature matching: the core `mean(crit(real.detach() - fake))` (H1 #11). Needs a `return_features` path in the D
  archs.
- LDL with the EMA-refined artifact map (H1 #5): give `Context` an `ema` attribute; the loss reads it.
- ECO teacher curriculum (`C/utils/eco.py`), as an optional batch transform in `image._set_batch` with a frozen
  teacher net.
- More scheduler rows: CosineAnnealingWarmup, WSD, KneeLR (`C/schedulers/*`, `C/models/base_model.py:272-294`).
  Simple table entries.
- `detect_anomaly` debug flag (`C/train.py:273`), one line in `runtime`.
- spectral_norm parametrization API for all D archs (H4 #17, with key remap). It is a prerequisite for compiling D
  after 6.4.
- Safetensors export of the EMA (H3 #16).
- OTF/data items from H2 (configurable kernel ranges, resize modes, `blur_prob`, paired previews).
- Per-arch capability metadata (PLAN S.11), so that R1/compile/channels_last support is declared per D/G arch.

## 9. Probe index

| file | what |
|---|---|
| `data_util.py` | fixed real crops (LQ 32/64) from `/home/korvin/apps/data`, PSNR (the `.pt` crop caches were deleted; rerunning rebuilds them) |
| `p1_clip_opt.py`, `out/p1_*.json` | clip × optimizer, loss ×1000 |
| `p1b_opt.py`, `out/p1b_*.json` | optimizer comparison, loss ×1, warm-up |
| `p2_clamp_init.py`, `out/p2_clamp_init.txt` | dead-gradient fraction at init for 19 archs |
| `p2b_clamp_learn.py`, `out/p2b_span_c*_{200,1000}.json`, `out/p2b_omnisr_c*_200.json` | clamp vs no clamp learning (span 1000 updates, omnisr 200; realplksr runs were stopped for CPU) |
| `p3_term_grads.py`, `out/p3_*.json` | per-term G gradient norms/cosines, esrgan template mix |
| `p3b_fdl_split.py`, `out/p3b_*.json` | FDL amplitude vs phase, batch-to-batch agreement |
| `p4_r1_concat.py`, `p4b_r1_math.py`, `out/p4*.json` | R1 double backward through metagan; concat D forward |
| `p6_flops.py`, `out/p6_flops.txt` | forward FLOPs of G, D, FDL backbone |
| `p7_nan_window.py`, `out/p7.txt` | transient NaN loss vs LossMeter |
