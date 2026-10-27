# Stage 2 analysis: trainer core and training loop

Lane scope: `train.py`, `test.py`, `neosr/models/{base,image,otf,phase,checkpoint,tiling}.py`, `neosr/runtime.py`,
`neosr/config.py`, `neosr/cli.py`, `neosr/utils/{stop,logger,dist_util,misc,rng,factory}.py`,
`neosr/data/{prefetch_dataloader,data_sampler}.py` and their tests. Checkout: `refactor/stabilize` (= master),
torch 2.9.1+cu128, Python 3.13. All probes ran on CPU (`CUDA_VISIBLE_DEVICES=`, the GPU belongs to the profiling
lane) on a machine shared with other lanes (load average 60-120 on 32 threads), so wall-clock numbers from CPU
probes are not speed claims. Probe scripts and raw outputs: `$SCRATCH/trainer/` (session scratchpad, not
committed): `harness.py`, `p_clamp_init.py`, `p_clamp_train.py`, `p_clip_mech.py`, `p_clip_train.py`,
`p_skip_txn.py`, `p_config.py`, `p_resume.py`, `p_save.py`, `p_unused.py` and their `*.out` files.

Evidence levels: **reproduced** (a probe shows the failure), **measured** (a number from a probe or a log),
**source-verified** (read in the code, not executed), **hypothesis** (mechanism plausible, needs the profiling
lane or an A/B).

## 1. Summary and priorities

The trainer's structure (Phase, EMA, atomic state-first saves, explicit accumulation, the single loss
Context) is sound and I found no accumulation, scheduler-clock, QAT-phase or EMA-math bug. The most valuable
remaining work is not in plumbing but in two **default training policies that cost learning speed**, plus a
handful of cheap reliability fixes:

| # | Pri | Finding | Kind | Evidence | Changes numerics |
|---|---|---|---|---|---|
| T-01 | P1 | Default train clamp `sr.clamp(1/255, 1)` zeroes the gradient of every out-of-range pixel: 20-84 % of output pixels at init for 11 of 19 archs; span: -5.4 dB at 150 updates (seed 0) and a stall at 19.3 vs 28.4 dB after 600 (seed 1); a restoring clamp matches no-clamp | learning | reproduced + measured | yes (A/B) |
| T-02 | P1 | Default `grad_clip` (fixed max-norm 1.0) is active on every update (pre-clip G norm 16-150). It is harmless for scale-invariant optimizers but damps `adan_sf` (z takes raw-gradient steps), the optimizer of 167/168 template sections: 2x worse loss in a mechanism probe, -1.1 dB at 100 / -0.4 dB at 400 updates on span. A spike-only relative clip keeps the protection without the damping | learning/stability | measured (mechanism + real arch) | yes (A/B) |
| T-03 | P2 | Periodic path validates **before** it saves (`val_freq == save_freq == 1000` in all 84 templates) and validation has no OOM fallback: one validation error loses the update's checkpoint and kills the run | reliability | source-verified | no |
| T-04 | P2 | A skipped update is not a transaction: BN running stats / QAT observers written by the non-finite forward stay NaN (50 buffers in the probe), the EMA inherits them on the next success, and the skipped window's NaN log aborts the run at the next print | reliability | reproduced | no (restores buffers) |
| T-05 | P2 | Validation and `test.py` inference always run FP32: no autocast on the inference path, `use_amp` in inference configs only switches the memory format | speed/memory | source-verified | metric protocol |
| T-06 | P2 | `manual_seed` implies full determinism: cuDNN benchmark off, deterministic algorithms, NaN-fill of uninitialised memory, no expandable segments | speed/memory | source-verified | no |
| T-07 | P2 | `cudnn.benchmark_limit = 0` with `benchmark = True` autotunes exhaustively for every new validation image shape | speed/memory | source-verified, cost hypothesis | no |
| T-08 | P2 | F19 resume silently restores saved optimizer hyper-parameters; objective changes (losses, accumulation, clamp, clip, ema_power) are not reported | reliability | reproduced | no |
| T-09 | P2 | F20 config invariants: 14 invalid values accepted (ema 1.5/1.0/nan/inf/-0.5, metric `better` typo inverts best tracking, negative val_freq validates every n updates, ...) | reliability | reproduced | no |
| T-10 | P2 | Trainable parameters without gradient are silent (asid 30, mfghmoe 33, ditn 4 parameters); under DDP they crash at update 2 | reliability | reproduced | no |
| T-11 | P3 | Host syncs: 2/update (bf16 GAN), 4/update (fp16 GAN), +1 sync and 1 collective per update under DDP (stop flag) | speed | source-verified | no |
| T-12 | P3 | `save()` blocks training (584 MiB state + 87 MiB exports for esrgan+metagan); exports grow 84.6 GiB per 1M updates; no best-model artifact | speed/disk | measured | no |
| T-13 | P3 | F23: rank-0 validation failure strands the other ranks at the second barrier | operational | source-verified | no |
| T-14 | P3 | DDP: no `gradient_as_bucket_view` (one extra gradient copy), compiled `net_d_adv` only under DDP | memory | source-verified | no |
| T-15 | P3 | AMP policy details: `fast_matmul` forces fp16 GEMM accumulation, BF16 check by compute capability, CPU AMP silently ineffective, GradScaler 2**5 with 2000-update growth | stability | source-verified | fp16 only |
| T-16 | P3 | Logging / fail-fast: logged lr ignores the SF warmup factor, clip activity is invisible, loaders and `_freq` are checked only after the (slow) model build | tooling | source-verified | no |
| T-17 | P3 | Final-metric bookkeeping (R1.4) and best-model selection | tooling | source-verified | no |

Why this order: the owner reports that models "learn slowly and badly" and wants stability from correctness,
not damping. T-01 and T-02 are exactly that pattern inside neosr's own defaults: a hard clamp that removes
gradient signal and a clip that normalises every step. Both have measured, large effects and both have a
correct-by-construction replacement that keeps their protective role (bounded loss inputs; spike protection).
The reliability items T-03/T-04/T-08..T-10 are small diffs that prevent lost runs. The pure speed items in this
lane are modest (T-05 is the largest; the sync work T-11 is < 1 % on GPU-bound runs, perf.md §3.4), so they rank
below.

## 2. Codex items assigned to this lane

| Codex item | Verdict | Evidence |
|---|---|---|
| F19 resume mixes saved and edited settings | **confirmed** | `p_resume.py`: config lr 1e-4 / wd 0.05 / accumulate 2 / loss_weight 5 / clamp off / grad_clip off / ema_power 0.75; after resume lr = 1e-3, wd = 0.0, and **no** warning for any of the changes (`run_record`, `image.py:398-407`, lists only nets, AMP flags, ema, world size). See T-08 |
| F20 numeric/cross-key invariants | **confirmed, wider** | `p_config.py`: 14 of 16 bad values pass (the 17th case, `keep_states = -3`, is the documented keep-all sentinel) `load_config` + `validate_train` (table in T-09). `val_freq=-5` makes `u % -5 == 0` true every 5 updates; `better="hgher"` silently selects **lower** as better (`base.py:197`) |
| F23 rank-0 validation failure strands peers | **confirmed** (source) | `base.py:177-189`: rank 0 raises out of the `try/finally` before the second `barrier()`, peers wait there until the process-group timeout or until the launcher kills them. torchrun's agent usually kills peers promptly, so impact is operational. See T-13 |
| Hard output clamp during training | **confirmed, larger than stated** | Not just "an objective choice": 20-84 % of pixels have zero gradient at init on 11 of 19 archs and span loses 5.4 dB at 150 updates (seed 0) or stalls at 19.3 vs 28.4 dB after 600 (seed 1) (T-01). The 1/255 lower bound has no remaining consumer |
| `fast_matmul` bundles FP16 accumulation | **confirmed, narrow impact** | `runtime.py:302` sets `allow_fp16_accumulation`; it affects cuBLAS fp16 GEMMs only (Linear/attention/DINOv2 under fp16 AMP), not cuDNN convs and not bf16. All 84 templates use bf16, so only `bfloat16 = false` users are exposed. See T-15 |
| BF16 capability check | **partial** | `runtime.py:239-256` branches on the major compute capability; correct for Ampere+ and Pascal, imprecise for Turing (emulated bf16 is slow, not absent). `torch.cuda.is_bf16_supported(including_emulation=False)` is the precise query. Cosmetic (warn vs raise) |
| CPU AMP request ineffective | **confirmed** (already deferred as S.13) | `phase.py:71-74` hard-codes `device_type="cuda"`; on a CPU device autocast is a no-op. Reject in `validate_train`/`setup_runtime` |
| Seed and deterministic policy coupled | **confirmed** | `runtime.py:268-296`; legacy-compatible by T3 D14/D30, but it also disables expandable segments and enables `fill_uninitialized_memory`. See T-06 |
| Exhaustive cuDNN benchmarking | **partial** | Source-verified (`runtime.py:293-294`). Training patches are fixed-size, so the exhaustive search is paid once per shape; the real exposure is validation with variable image sizes (every image a new shape) and its workspace spikes. Cost unmeasured on GPU. See T-07 |
| Final metric bookkeeping | **confirmed** (deferred R1.4) | `train.py:247-251`. Low impact; T-03 makes the periodic path behave the same way, so a common remedy is proposed in T-17 |
| Skip is not a full transaction | **confirmed, worse than "document it"** | `p_skip_txn.py` (sebica_mini, BN): one NaN pixel in `lq` → update skipped, but 50 BN buffers are NaN afterwards, they stay NaN through the next (successful) update and the EMA copies them; the run then aborts at the next log/`check_finite`. See T-04 |
| Model-export disk retention | **confirmed** | `p_save.py`: 86.7 MiB of exports per save for esrgan+metagan → 84.6 GiB over 1M updates at `save_checkpoint_freq = 1000`. Owner decision 8a (exports never pruned) stands; propose an opt-in only. See T-12 |
| Best-model selection | **confirmed, partly mitigated** | `update_best` (`base.py:191-198`) records value and update only. Because every template has `val_freq == save_freq`, the best update's export exists on disk; with different frequencies it may not. See T-17 |

## 3. Findings

### T-01 (P1, learning) Hard training clamp removes gradient from out-of-range pixels

**Where.** `neosr/models/image.py:223-224` (`sr = sr.clamp(1 / 255, 1)` under `train.clamp`, default true;
78 of 84 templates use the default, 6 set `clamp = false`).

**Mechanism.** `clamp` has zero derivative outside `[lo, hi]`. Every loss reads the clamped `sr` from the
Context, so a pixel whose raw output is below 1/255 or above 1 contributes nothing to any gradient, including
the losses that would pull it back in range. The network still moves through shared weights, but only via
in-range pixels.

**Measured at init** (`p_clamp_init.py`: 8 real LQ crops 64 px, x4, default arch options, eval mode):

| arch | pixels with zero gradient | | arch | pixels with zero gradient |
|---|---|---|---|---|
| realplksr | **83.9 %** | | atd | **48.1 %** |
| safmn | **60.3 %** | | rcan | **47.9 %** |
| span | **56.6 %** | | ninasr (template: clamp=false) | 46.5 % |
| man | **53.9 %** | | esc | **37.7 %** |
| omnisr | **53.8 %** | | spanplus | **33.5 %** |
| swinir_small | 19.7 % | | sebica | 14.0 % |
| plksr | 7.2 % | | cugan | 5.3 % |
| mosrv2 | 1.6 % | | esrgan (template: clamp=false) | 1.3 % |
| compact | 0.6 % | | dat_m, hat_m | 0.0 % |

Real LQ pixels themselves are only 0.03 % outside the range, so this is the network's output, not data.

**Measured learning cost** (`p_clamp_train.py`: span, random init, AdamW lr 5e-4, L1, batch 8 x 32 px LQ,
validation PSNR on 16 held-out 32 px crops, output clamped to [0, 1] for scoring):

| updates | hard `[1/255,1]` (today) | none | restoring `[0,1]` (proposed) |
|---|---|---|---|
| 0 | 7.23 | 7.23 | 7.23 |
| 50 | 17.33 | 20.70 | 20.70 |
| 100 | 18.01 | 22.28 | 22.29 |
| 150 | **19.37** | **24.77** | **24.76** |

Out-of-range fraction at update 149: 8.2 % (hard) vs 2.2 % (none).

Second seed, 600 updates (same setup, seed 1):

| updates | 50 | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|
| hard `[1/255,1]` | 16.01 | 16.59 | 18.76 | 19.05 | 19.21 | 19.25 | **19.26** |
| none | 20.58 | 22.13 | 24.94 | 26.83 | 27.77 | 28.12 | **28.36** |
| restoring `[0,1]` | 20.58 | 22.15 | 25.03 | 26.80 | 27.44 | 28.18 | **28.32** |

With the hard clamp this seed **stalls** at ~19.2 dB from update 300 on (a 9.1 dB gap at 600); the out-of-range
fraction of the last training batch stays 3.4x higher (5.4 % vs 1.6 %). Pixels that end up outside the range
have no force pulling them back, so the trap persists. Diagnosis (`p_clamp_diag.py`, same seed, 300 updates,
PSNR 19.05 reproduced): on the val crops 6.3 % of R and 6.4 % of B outputs sit below 1/255 (raw down to -0.53)
while their targets are far brighter; no channel collapsed, these are dead regions with large, frozen errors. The restoring clamp tracks no-clamp within 0.1-0.3 dB on
both seeds. (Small CPU probe: one arch, L1 only, 2 seeds. It shows the mechanism and its size, it is not the
template A/B; realplksr/compact runs were stopped for CPU time.)

**Why the bounds.** History (`git log -G clamp`): the clamp entered with the consistency loss (b1a1d82, which
needs positive inputs for its `pow`/cube root), was reverted for "unexpected consequences" (4770771), re-added
"due to showing better stability on net init" (98627da), and made optional "due to kaiming init" (de60639).
Today `consistency_loss` clamps its own inputs to `[1/255, 1]` (`consistency_loss.py:160-161`), `msswd` clamps
inside its colour transform, `ldl`/`mssim`/`ncc` have their own eps guards, so **no loss needs the 1/255 lower
bound from the trainer** any more; it only makes exact black unreachable. The forward clamp does have a real
job: it keeps D and the perceptual backbones on in-range inputs (consensus G1.M1).

**Fix.** Keep the forward clamp (bounded inputs for D/perceptual losses), change the backward and the bound:

```python
class _RestoringClamp(torch.autograd.Function):
    """Forward clamp(x, lo, hi). Backward: identity inside; outside, only the component that points back in."""
    @staticmethod
    def forward(ctx, x, lo, hi):
        ctx.save_for_backward(x); ctx.lo, ctx.hi = lo, hi
        return x.clamp(lo, hi)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        outward = ((x < ctx.lo) & (g > 0)) | ((x > ctx.hi) & (g < 0))
        return g.masked_fill(outward, 0), None, None
# image.py:223-224
if self.clamp:
    sr = _RestoringClamp.apply(sr, 0.0, 1.0)
```

A straight-through clamp (`x + (x.clamp(0,1) - x).detach()`) is not enough: a perceptual or GAN gradient that
asks a clamped pixel to go darker would push the raw output further out with no restoring force (unbounded
drift); masking the outward component prevents that. The function is a plain autograd.Function (dynamo traces
it) and sits outside the compiled G anyway. Cost: one boolean mask and one `masked_fill` over `sr` in backward.

**Compatibility / test.** Changes the objective (gradient of out-of-range pixels, black reachable) → A/B on the
E and O workloads (and one transformer, e.g. omnisr/atd) before it becomes the default behind `clamp = true`;
keep the old behaviour available (e.g. `clamp = "hard"`) only if the owner wants to reproduce old runs. No
checkpoint impact. Tests: forward equals `clamp(0, 1)`; backward passes inward gradients and blocks outward ones
(hand-built `x`, `g`); a training-step test where an all-below-range output still receives a gradient; the span
probe as a slow test.

### T-02 (P1, learning/stability) Fixed max-norm-1.0 clipping is per-step normalisation; it damps adan_sf

**Where.** `image.py:71` (`grad_clip: 1.0 if tr.get("grad_clip", True)`), `phase.py:89-94`
(`clip_grads_with_norm_` after `unscale_`), `adan_sf.py:60-68` (schedule-free branch).

**Is it always active? Yes.** acceptance.md §2: window-mean pre-clip G norm 101 → 39 (esrgan bf16), 150 → 47
(fp16), D 3-252; the compact OTF run had `grad_clip = false` and G 16-48. 82 of 84 templates use the default
(only `train_compact{,_otf}.toml` set `false`). The old `experiments/train_esrgan/*.log` predate the refactor
and log no grad norm. With norms 16-150 against a threshold of 1.0 every update is scaled by 1/16-1/150, so the
clip is a normaliser `g / ||g||`, not a spike guard. The competitor defaults `grad_clip: false`; only 38 of its 654 option files
enable it.

**What it does to each optimizer** (`p_clip_mech.py`: N = 20 000 ill-conditioned noisy quadratic, raw norm
>> 1, 1500 steps; final loss, lower is better):

| optimizer | none | always (today) | spike-only k=3 | none + spikes | always + spikes | spike + spikes |
|---|---|---|---|---|---|---|
| adamw | 31.5 | 31.3 | 31.5 | 81.1 | 31.3 | **30.6** |
| adan | 13.0 | 13.0 | 13.0 | 123.7 | 13.0 | **13.7** |
| adamw_sf | 3.14 | 3.04 | 3.14 | 546 | 3.04 | **3.26** |
| adan_sf | 17.1 | **35.2** | 17.1 | 2.7e6 | 35.2 | **18.1** |

("spikes": every 97th gradient x50.) Reading:

* Adam, AdamW, NAdam, Adan and AdamW-SF divide by `sqrt(v)`: a slowly varying clip factor cancels, so always-on
  clipping does **not** reduce their effective learning rate (the hypothesis "constant damping" is refuted for
  them). It only reweights steps (each step counts as a unit-norm direction). Epsilon is irrelevant: at norm 1
  over N parameters the per-element RMS is `1/sqrt(N)` (2.4e-4 for esrgan's 16.7M, 1.3e-3 for compact),
  four orders above `eps = 1e-8`.
* `adan_sf` with `schedule_free = true` (167 of 168 template optimizer sections) updates its base sequence with
  the **raw** gradient, `z -= lr * grad` (`adan_sf.py:68`), not with `m / sqrt(v)` as `adamw_sf` does
  (`adamw_sf.py:180-192`). That path is not scale-invariant, so the clip shrinks every z step by 16-150x;
  the probe loses 2x, the real-arch run below 0.4-1.1 dB. (The z recurrence itself is F01, owned by the optimizer lane; whatever the redesign is,
  the clip must not be tuned around the current bug.)
* Without clipping, spikes hurt every optimizer (inflated `v` damps the next ~1/(1-beta2) = 100 steps; adan_sf's
  raw z step explodes). A spike-only clip keeps that protection and leaves normal steps untouched.

**Real arch** (`p_clip_train.py`: span, template optimizer `adan_sf` SF, betas (0.98, 0.92, 0.99), wd 0.01,
warmup 100, lr 1e-3, loss 50 x L1 so the raw norm sits in the template regime: median 17.7, p99 40.6; val PSNR):

| updates | 100 | 200 | 300 | 400 |
|---|---|---|---|---|
| none | 20.27 | 22.98 | 25.15 | **26.35** |
| always max-norm 1.0 (today) | 19.20 | 22.89 | 24.95 | **25.93** |
| spike-only k=3 (clipped 1 of 400 updates) | 20.27 | 22.98 | 25.15 | **26.35** |

Today's clip costs 1.1 dB at 100 and 0.4 dB at 400 updates on a real SR net with the default optimizer; the
spike clip is indistinguishable from no clip when there are no spikes (one seed, L1 only; template A/B still
required).

**Fix: spike-only relative clipping, sync-free.** Threshold = `k` times a running EMA of the (clipped) norm;
the EMA is a device scalar, the threshold a 0-d tensor (`clip_grads_with_norm_` accepts it: it computes
`clamp(max_norm / (norm + 1e-6), max=1)` on device; checked). The EMA is updated only after a successful step,
which `Phase.step` already knows on the host from its existing `isfinite` sync, so no extra sync:

```python
# Phase.__init__: self.spike, self.norm_ema = spike_factor, None   (e.g. 3.0; None = off)
# Phase.step, after norm = get_total_norm(...):
if self.grad_clip:                                   # legacy fixed max-norm
    clip_grads_with_norm_(self.params, self.grad_clip, norm, foreach=True)
elif self.spike and self.norm_ema is not None:
    thr = self.spike * self.norm_ema
    clip_grads_with_norm_(self.params, thr, norm, foreach=True)
ok = bool(norm.isfinite())                           # the existing sync
if ok:
    ...step...
    if self.spike:
        n = norm.detach() if self.norm_ema is None else torch.minimum(norm, self.spike * self.norm_ema)
        self.norm_ema = n.clone() if self.norm_ema is None else self.norm_ema.lerp_(n, 1 - 0.98)
```

Defaults `k = 3`, EMA 0.98 (50-update horizon): the acceptance D transients (7 → 47 at the wavelet switch,
→ 252 at u1400) get clipped to 3x the running norm; G's slowly falling norm is never clipped; a genuine regime
change is followed within ~60 updates (the EMA can grow 4 %/update). Log the clipped fraction as a device mean
(`meter.add({"clip_g": (norm > thr).float()})`, no sync) so clipping is visible (Codex §10 item 4). Save
`norm_ema` in `Phase.state_dict` (optional key; absent → re-initialised at the first update) so resume stays exact.
This is not the competitor's AdaptiveGradientClipping: that one bounds the threshold to an absolute [0.1, 10]
(with norms 16-150 it normalises every step, like today) and costs `.item()` syncs.

**Migration.** `train.grad_clip` keeps `true` (= legacy fixed 1.0, old runs reproducible) and `false`; add
`"spike"` (+ optional `train.grad_clip_factor`, default 3.0). Step 1: ship the option and the `clip_*` log.
Step 2: A/B `true` vs `"spike"` on E (esrgan+metagan, adan_sf) and O (compact OTF) for >= 20k updates,
compare live/SF-x/EMA PSNR and skip counts. Step 3: if positive, make `"spike"` the code default and drop
`grad_clip` from the templates. Record the policy in `run_record` (T-08). Changes training numerics → A/B
mandatory; no checkpoint impact. Tests: no clipping below `k * ema`; a x50 spike is cut to `k * ema`; EMA
untouched by a skipped step; zero host syncs added (`set_sync_debug_mode("warn")` count stays 1 per phase,
CUDA test); state round trip.

### T-03 (P2, reliability) Periodic validation runs before the save and has no OOM fallback

**Where.** `train.py:256-265`: `check_finite` → `validation` → `save`. Every template has `val_freq = 1000`
and `save_checkpoint_freq = 1000`, so both are due at the same update. `image.validate` (`image.py:337-350`)
runs the whole val image through G with no handling of `torch.OutOfMemoryError`.

**Impact.** Any exception in validation (OOM on a large val image or a large val image after training memory
growth, a metric weight download failure, disk full while writing PNGs) propagates out of `train()` → exit 1,
and the checkpoint of this update was never written: up to `save_freq` updates lost (1000 updates ≈ 1.5-4.6 min
on the measured E/O workloads: 10.8, 4.97 and 3.64 updates/s). The final path already does state first (`train.py:247-251`, T2 D20).

**Fix.** (1) Swap the order: `check_finite` → `save()` → validation (the saved `best` dict then lags one
validation, exactly as on the final path; see T-17). (2) In `validate`, catch `torch.OutOfMemoryError` per image:
`torch.cuda.empty_cache()`, warn once, retry with `infer_tiled(net, lq, scale, tile or 256, tile_overlap)`; a
second OOM re-raises. The competitor does the same fallback (`sr_model.py:1244-1258`; it also calls
`empty_cache()` after every val image, which is wasteful and not recommended). No numerics change for training;
tiled metrics differ slightly from full-image metrics (log it). Tests: monkeypatch `net.forward` to raise
`OutOfMemoryError` once for a large input → validation completes, warns, metrics close to full; a validation that
raises after `save_due` leaves `<u>.state` on disk.

### T-04 (P2, reliability) A skipped update is not a transaction; a skipped window can still kill the run

**Where.** `image.py:215-255` (forward in train mode writes buffers before `Phase.step` decides), `phase.py:88-112`,
`phase.py:237-258` (`LossMeter.read` raises on any non-finite value, design T1 D15), `phase.py:207-210` (EMA copies
live buffers on success).

**Reproduced** (`p_skip_txn.py`, sebica_mini with BatchNorm, EMA 0.999): one NaN pixel in one `lq` batch →
`optimize_parameters` returns True with `skipped_g = 1` (weights, optimizer, EMA untouched, as designed), but
**50 BN running-stat buffers are NaN** afterwards. The next update succeeds (train-mode BN uses batch stats), the
NaN running stats stay, and `Ema.update` copies them into the EMA (`_foreach_copy_(db, sb)`), so validation and
exports would be NaN. The run then aborts at the next print/`check_finite` with "non-finite training values
['l_g_pix', 'l_g_total']" — from the very window that was skipped. The same holds for QAT observer min/max.
`max_skip_streak = 10` therefore only protects backward-only overflow (fp16 scaler); a single non-finite
forward (bf16 overflow, a degenerate degradation, a corrupt batch) ends a BN/QAT run, and ends any run at the next
print.

**Fix (sync-free, small).** At the start of an update window snapshot the floating buffers of G and D
(`torch._foreach_copy_` into preallocated clones; BN stats, SN u/v and QAT observers are KiB-MiB); if
`Phase.step` returns False, copy them back (host `ok` is already known). Buffer the window's `ctx.logs` and add
them to the meter only when the window's steps succeeded; a skipped window is reported through `skipped_*`. The
abort guarantees stay: streak >= `max_skip_streak` raises in `step`, and a non-finite objective with finite
gradients (D15's "broken loss" case, `test_nan_objective_with_finite_grads_raises_at_log`) still reaches the
meter and raises. DDP: buffers are restored on every rank because `ok` is rank-identical. No numerics change
for finite runs. Tests: the probe as a unit test (buffers bitwise equal to the pre-window snapshot after a skip,
EMA buffers finite, `get_current_log` does not raise, the next good update trains); the D15 test unchanged.

### T-05 (P2, speed/memory) Validation and inference ignore AMP

**Where.** `image.py:342-350`: `with torch.inference_mode(): sr = net(lq)` (or `infer_tiled`) without
`torch.autocast`; the EMA module is an fp32 deepcopy. In inference configs `use_amp` only selects
`channels_last` (`image.py:46`); `options/test_*.toml` carry `#use_amp = true`, which suggests an effect it does
not have.

**Impact (hypothesis, needs the profiling lane).** Training forwards run in bf16 (all templates); validation and
`test.py` run every conv/GEMM in fp32 (TF32 with `fast_matmul`). On an RTX 4090 bf16 tensor-core throughput is
about 2x TF32, activations take half the memory, and SDPA can only pick the flash kernel for fp16/bf16, so
transformer archs on large val images fall back to the memory-efficient or math kernel. Expect validation and
inference 1.5-2.5x faster on tensor-core-bound archs and roughly half the activation peak (which also reduces
the T-03 OOM risk).

**Fix.** Build an inference autocast dict once in `image.__init__` (before the `is_train` return):
`{"device_type": device.type, "dtype": bf16 if opt.get("bfloat16") else fp16, "enabled": use_amp and cuda}` and
wrap the forward/`infer_tiled` call in it; `tensor2img` already converts to float. For `test.py` honour
`use_amp`/`bfloat16` (it is explicitly requested there). For validation make it opt-in (`val.use_amp`, default
off) because bf16 output rounding shifts reported PSNR slightly and breaks comparability with old logs; switch the
default after measuring the metric delta. Training numerics unchanged. Tests: CUDA test that the val forward runs
under autocast when enabled (dtype of an intermediate via a hook) and the PSNR delta vs fp32 is within a stated
tolerance; CPU path unchanged.

### T-06 (P2, speed/memory) `manual_seed` switches on full determinism

**Where.** `runtime.py:268-296`. A seed sets `deterministic = True`, which disables `cudnn.benchmark`, sets
`cudnn.deterministic`, `use_deterministic_algorithms(True, warn_only=True)`,
`fill_uninitialized_memory = True` and skips `PYTORCH_ALLOC_CONF=expandable_segments:True`. T3 D14/D30 kept this
for compatibility.

**Impact.** All 84 templates ship `#manual_seed = 1024`; a user who uncomments it to make a run reproducible
gets heuristic cuDNN algorithms (benchmark off), deterministic (often slower) kernels, a NaN fill on
uninitialised allocations, and loses expandable segments (more fragmentation after validation, higher OOM risk),
while `warn_only=True` still does not guarantee determinism (e.g. bicubic/bilinear upsample backward has no
deterministic CUDA kernel). The competitor separates `deterministic` from `manual_seed`. Cost on GPU unmeasured
(hypothesis; profiling lane: same config with/without seed).

**Fix.** New root key `deterministic: bool` (default false). `manual_seed` only seeds. `deterministic = true`
keeps today's strict branch. Set `expandable_segments` regardless of determinism (allocation layout does not
change results). Log one line when a seed is set without `deterministic`. Compatibility: users who relied on
seed ⇒ deterministic must add `deterministic = true` (state it in the changelog); no checkpoint impact; tests in
`test_runtime.py` for the four combinations.

### T-07 (P2, speed/memory) cuDNN exhaustive autotuning on every new validation shape

**Where.** `runtime.py:292-294` (`benchmark = True`, `benchmark_limit = 0` = try every algorithm).

**Impact (source-verified mechanism, cost hypothesis).** Training patches are fixed-size, so the search is paid
once. Validation with full images (`val.tile <= 0`, the template default) gives a new input shape per image for
every conv layer; with the limit at 0 cuDNN times every algorithm for each new (layer, shape) pair and allocates
the workspace each algorithm asks for. The cache persists, so the first validation pays (and every validation
for datasets whose sizes vary per image, e.g. Urban100/DIV2K-style sets), with workspace spikes that fragment the
allocator before training resumes.

**Fix (measure first).** Profiling lane: time and peak-reserved of the first and second validation with
limit 0, the torch default 10, and `cudnn.flags(benchmark=False)` inside `validate`. Expected choice: drop the
`benchmark_limit = 0` line (torch default 10, also for training) and, when `val.tile <= 0`, run validation under
`torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=torch.backends.cudnn.deterministic)`.
Also call `torch.cuda.empty_cache()` once after validation (after `release_metric_nets`) so val-shaped blocks do
not stay reserved, which matters most without expandable segments (T-06). No numerics change beyond cuDNN
algorithm choice.

### T-08 (P2, reliability) F19: resume restores optimizer hyper-parameters silently and misses objective changes

**Where.** `phase.py:153` (`self.opt.load_state_dict` restores saved `param_groups`: lr, betas, eps,
weight_decay, warmup_steps, schedule_free), `image.py:398-407` (`run_record`), `base.py:290-291` (diff warning).

**Reproduced** (`p_resume.py`): see §2. Restoring saved hyper-parameters is right for exact continuation; the
defect is that it is silent and that objective changes are invisible.

**Fix.** (1) Before `opt.load_state_dict`, compare each saved group's scalar hyper-parameters with the
configured ones and log `net_g: restored lr 1e-3 from the state (config 1e-4); start a new run with
pretrain_network_g to change it`. (2) Extend `run_record` with `train.optim_g/optim_d/scheduler`, every
`*_opt` loss section, `clamp`, `grad_clip`, `ema_power`, `datasets.train.{accumulate,batch_size,patch_size}`
and `qat_*`; `dict_diff` already flattens one level and warns per key. Old states lack the new keys → one
"changed: None -> ..." warning per key on the first resume (acceptable; or skip keys absent from the saved
record). No numerics change. Test: `test_model.py` resume with edited lr/loss weight asserts the warnings.

### T-09 (P2, reliability) F20: missing range and enum checks

**Reproduced** (`p_config.py`, all pass `load_config` + `validate_train`):

| value | consequence |
|---|---|
| `train.ema` = 1.5 / -0.5 / inf / nan / 1.0 | 1.5 and -0.5 extrapolate (`Ema.update` lerps with weight `1 - decay`), inf poisons, nan disables the `ema > 0` branch silently, 1.0 freezes the EMA at its first copy |
| `val.metrics.*.better = "hgher"` | treated as lower-is-better: best tracking silently inverted (`base.py:197`) |
| `val.val_freq = -5` / `0` | `u % -5 == 0` validates every 5 updates; 0 disables validation silently |
| `num_worker_per_gpu = "four"` | fails later inside the DataLoader |
| `dataset_enlarge_ratio = 0` | ConfigError only after model and loss construction |
| `mean` without `std` | accepted (F13 domain issue belongs to the data lane) |
| `logger.print_freq = 0`, `total_iter = 0` | rejected by `train._freq` only after `build_model` (DINOv2 load ~50 s) |
| `use_amp` without CUDA | accepted, silently fp32 (S.13) |

**Fix.** In `validate_train` (pure, runs before anything heavy): `0 < ema < 1` (finite; 0/absent = off),
`better in {"higher", "lower"}`, `val_freq >= 1` when present, `num_worker_per_gpu` int >= 0 or `"auto"`,
`dataset_enlarge_ratio >= 1`, `mean`/`std` both or neither with matching length and `std > 0`,
`total_iter/print_freq/save_checkpoint_freq >= 1` (move `_freq` here), `keep_states` int (<= 0 = keep all stays).
Reject `use_amp` in `setup_runtime` when the device is CPU. Tests: extend `test_validate_train_errors`
parametrisation with each row.

### T-10 (P2, reliability) Parameters that never receive a gradient are silent

**Reproduced** (`p_unused.py`, one forward/backward at 64 px): asid 30 of 360 trainable tensors have
`grad is None` (canorm, qkv, temperature: F02), mfghmoe 33 of 935 (router and `mfg.*`: F03), ditn 4 of 122
(temperature: F11); compact and span 0. Single-GPU training proceeds silently (those parameters are not even
weight-decayed). Under DDP with the default `find_unused_parameters = false` the reducer raises at the second
update.

**Fix.** At the first boundary in `Phase.step` (host-only, no sync: `p.grad is None` is a host attribute),
collect `[n for n, p in zip(self.names, self.params) if p.grad is None]`; log a warning with the count and the
first names; under DDP raise `ConfigError(... set find_unused_parameters = true ...)` before the reducer error.
Run the check once (or also at the GAN/wavelet switch updates). No numerics change. Test: a toy net with a dead
branch triggers the warning; compact does not. This also gives the architecture lane a cheap regression net for
F02/F03/F11-class bugs.

### T-11 (P3, speed) Host syncs per update

Counted from source (perf.md measured 2/update on E and 20 → 4 on O after the DiffJPEG fix):

| site | per update | note |
|---|---|---|
| `Phase.step` `bool(norm.isfinite())` (`phase.py:96`) | 1 per phase (G, D) | design D13 |
| `GradScaler.step` `found_inf.item()` (`torch/amp/grad_scaler.py:356`) | +1 per phase, fp16 only | redundant: `ok` already implies `found_inf == 0` |
| `StopFlag.requested` (`stop.py:22-27`) | DDP: 1 all_reduce + 1 `.item()` every update | 0 single-process |
| `LossMeter.read` | 1 per `print_freq` | fine |
| `check_finite` | 2 per val/save | fine |
| prefetcher, timer, EMA, `LossMeter.add` | 0 | verified |

**Fix.** (1) Compute G's and D's norms first (D's gradients are complete before G steps), read both with one
`torch.stack([n_g, n_d]).isfinite().tolist()`: 2 → 1 sync. (2) When `ok`, call `self.opt.step()` directly and
then `self.scaler.update()` (`update` only needs the `found_inf` recorded by `unscale_`; checked in the torch
2.9 source): fp16 4 → 1. (3) DDP stop flag: `all_reduce(async_op=True)` and read it at the next boundary (the
work is long finished by then). Expected gain small: perf.md §3.4 measured ≈ 0 for a sync on the GPU-bound E
workload; on the launch-bound O workload maybe 0.1-0.5 ms/update (< 1 %). No numerics change. Test:
`test_host_syncs_per_update` expects 1 per update for every precision, G+D.

### T-12 (P3, speed/disk) Blocking save, export growth, no async path

**Measured** (`p_save.py`, esrgan + metagan, adan_sf state, EMA): state 583.6 MiB (G 63.7, D 23.0, EMA 63.7,
adan_sf G 318.5 = 5x G, D 114.8), exports 86.7 MiB per save. Serialising + fsync to tmpfs: 0.61-0.72 s for the
state, 0.11-0.14 s for the exports; on the SSD under `/home` add the write time (~1-2 s per save). At
`save_checkpoint_freq = 1000` and 4-11 updates/s that is roughly 0.3-1 % of wall time. Exports are never pruned
(owner decision 8a): 84.6 GiB per 1M updates for this pair.

**Options.** (a) Async save: on the training thread copy the state into a reused pinned CPU buffer
(`non_blocking=True`, one stream sync, ~30 ms for 0.6 GiB), then a single background worker runs today's
`atomic_save(state)` → exports → `prune_states` in that order; the next `save()` and the end of `train()` first
wait for the previous job (re-raising its error), so atomicity and the state-first rule hold and at most one
snapshot (+0.6 GiB host RAM) exists. Gain ~1-2 s per save; serialisation in a Python thread can contend for the
GIL with a launch-bound loop, so measure. (b) Opt-in `logger.keep_exports` (default 0 = keep all, preserving 8a)
and a startup log line with the predicted export disk use. Low priority against T-01..T-10. No numerics change.

### T-13 (P3, operational) F23: rank-0 validation failure under DDP

**Fix.** In `base.validation`, catch the rank-0 exception into a local, replace the second `barrier()` by
`all_reduce(status, MAX)` of a 0/1 tensor (also a barrier), then every rank raises (rank 0 re-raises the original;
others raise `RuntimeError("validation failed on rank 0")`); `release_metric_nets` stays in `finally`. Test: the
existing 2-rank gloo harness with a validation that raises on rank 0 finishes within the timeout with exit 1 on
both ranks.

### T-14 (P3, memory) DDP wrapper options

`base.py:149-154`: add `gradient_as_bucket_view=True` (gradients become views of the all-reduce buckets: saves
one gradient-sized buffer, 64 MiB for esrgan G and more for large transformers, and the bucket copy kernels;
compatible with `zero_grad(set_to_none=True)`, `clip_grads_with_norm_` and `backward(inputs=...)`). Keep
`static_graph=False` (D runs two forwards before one backward and accumulation uses `no_sync`). Replace the
DDP-only `torch.compile(self.net_d)` for `net_d_adv` (`image.py:93-95`) by the bare `self.net_d`, as perf.md §5.4
already recommends (D is eager everywhere else). DDP is best-effort (decision 8c); test with the gloo suite.

### T-15 (P3, stability) AMP policy details

* `fast_matmul` (`runtime.py:297-302`) also sets `allow_fp16_accumulation`: fp16 GEMMs accumulate in fp16
  (overflow above 65504, error ~ sqrt(K) * 2^-11). Templates use bf16, where it has no effect; for fp16 users it
  silently changes numerics. Split it into an explicit `fp16_accumulation` opt-in, or drop it.
* `GradScaler(init_scale=2**5)` with the default `growth_interval = 2000`: reaching 2**16 takes ~22k updates
  (acceptance: 32 → 64 at u2000). Activation gradients of a mean loss over 16x3x256x256 outputs are ~3e-7, x32 ≈
  1e-5, inside fp16's subnormal range (min normal 6.1e-5), so early fp16 backward may lose precision
  (hypothesis; the fp16 acceptance run matched bf16 at 2000 updates). If fp16 matters, use
  `growth_interval = 200` so the scale finds its level within ~2k updates; fp16-only.
* Use `torch.cuda.is_bf16_supported(including_emulation=False)` in `_check_gpu`; reject CPU + `use_amp` (S.13).

### T-16 (P3, tooling) Logging and fail-fast

* `current_lr()` (`image.py:286-287`) logs `group["lr"]`; `adan_sf`/`adamw_sf` apply their warmup internally
  (`lr * step / warmup_steps`), so during the 1600/600-update template warmups the log shows the target lr, not the
  effective one. Log `lr * min(1, step / warmup_steps)` for SF groups.
* Log clip activity (T-02) and the pre-clip norm already logged; both are needed to judge stability.
* `train.py:202-203` builds the model (loss backbones, DINOv2 ~50 s) before the loaders; build the loaders first
  so a wrong dataset path fails in seconds (no RNG is drawn by dataset construction; `restore_rng` runs right
  before the loop anyway). Move `_freq` into `validate_train` (T-09).

### T-17 (P3, tooling) Best-model bookkeeping

`save()` writes `best` before the validation that may improve it (final path today, periodic path after T-03).
Rather than reordering, write a small `models/best.json` (dataset → metric → value, update, export file name)
atomically after each validation on rank 0; optionally (`val.save_best = true`) also copy the evaluated
weights to `models/net_g_best.pth` when the chosen metric improves, which also covers `val_freq != save_freq`.
No state-format change; no numerics change.

## 4. Hypotheses checked and refuted / no action

| Hypothesis | Result |
|---|---|
| Fixed clip is constant damping for every optimizer | refuted for adam/adamw/nadam/adan/adamw_sf (scale-invariant, probe within 1-3 %); confirmed for adan_sf SF (T-02) |
| Adam eps interacts with normalised gradients | refuted: per-element RMS 1e-4-1e-3 vs eps 1e-8 |
| Schedule-free averaging weights depend on the clip | refuted: `ckp1` depends on lr and step only; only adan_sf's raw z step does |
| Prefetcher, timer, EMA, `LossMeter.add` sync | refuted (device-side `wait_stream`/`record_stream`, host clocks, foreach ops, in-place adds) |
| Accumulation / scheduler clock / QAT step bugs | none found: loss scaled once by `1/accumulate`, schedulers on attempted updates (D20), QAT phase a pure function of the update, `test_accumulate_equals_full_batch` covers equivalence |
| Train last-batch recompiles | refuted: `drop_last=True` for train loaders |
| EMA / SF swap costs | negligible: EMA path swaps nothing; SF without EMA clones the parameters once per validation/save |
| Memory leak in the loop | none: per-micro-batch Context, detached logs, meter reset per window; transient peaks only (SF snapshot 1x params, validation activations) |

## 5. Competitor notes (traiNNer-redux)

* Defaults `grad_clip: false` (38 of 654 option files enable it), `grad_clip_max_norm = 1.0`; its AdaptiveGradientClipping
  bounds the threshold to [0.1, 10] (absolute) and reads `.item()` per update: the damping style to avoid.
* Does not clamp the SR output before the losses (`sr_model.py`, `self.output = self.net_g(self.lq)`).
* Validation OOM → retries the image tiled at 256 (useful, T-03); `empty_cache()` after every val image (not
  useful).
* `deterministic` is its own option, separate from `manual_seed` (T-06).
* Skips on a non-finite loss with a host check every step and aborts after 10 (similar to neosr, with more
  syncs); no buffer restore either.

## 6. Suggested order of work

1. T-03, T-04, T-09, T-10, T-08 (small, no numerics change, prevent lost runs).
2. T-01 and T-02 as options + logging, then one combined A/B plan (E, O, one transformer; >= 20k updates;
   live/SF-x/EMA PSNR, skip counts, clip fraction, out-of-range fraction) before switching defaults. Coordinate
   T-02 with the optimizer lane's F01 decision for adan_sf.
3. T-05, T-06, T-07 with profiling-lane numbers.
4. T-11..T-17 as cleanup.
