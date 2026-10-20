[codex] Assistant message captured: Read-only review completed. No files were changed, no commits made, and no training runs laun...
[codex] Turn completed.
Read-only review completed. No files were changed, no commits made, and no training runs launched. Verification comprised source inspection and small, in-memory CPU probes using `.venv/bin/python -B`.

**1. Top 10 most important issues**

1. **Generator GAN loss has no gradient:** discriminator inference disconnects the adversarial objective from G. **A1**
2. **Discriminator training uses evaluation mode:** spectral normalization and dropout do not operate as intended. **A2**
3. **Gradient accumulation advances twice per batch:** even accumulation factors produce too many, underscaled updates. **A3**
4. **Checkpoints cannot faithfully resume:** EMA replaces live weights; scaler, RNG and other state are omitted. **A4–A5**
5. **Adan-SF warmup does not reach the update kernel:** directly affects the supplied experiment. **A7**
6. **Valid slow runs terminate successfully:** throughput below 0.5 updates/sec triggers `sys.exit(0)`. **A8**
7. **LDL, NCC and MS-SSIM have reproducible non-finite cases.** **A12–A14**
8. **FSAM shares a stateful G+D closure and has separate, uncheckpointed optimizer state.** **A9**
9. **DDP is broken at loader construction and loss reduction; rank-zero validation can mismatch collectives.** **A17–A18**
10. **OTF configuration reads incompatible namespaces:** ordinary degradation configurations can fail immediately. **A20**

The logs support a training-dynamics problem, rather than proving a particular crash. At iteration 1,200, GAN loss is **11.127**, real/fake discriminator outputs are **39/−37**, and discriminator loss is approximately **3.75e−14** ([first log:42](/home/korvin/apps/neosr-dev/experiments/train_esrgan/train_train_esrgan_20251230_115003.log:42)). That is consistent with a discriminator separating samples while G receives no adversarial gradient. The second log explicitly resumes G/D from iteration 2,200 and reconstructs EMA ([second log:18](/home/korvin/apps/neosr-dev/experiments/train_esrgan/train_train_esrgan_20251230_120333.log:18)). Neither log contains evidence establishing NaN as the termination cause. Wavelet activation at 80,000 was not reached.

**2. Findings grouped by A/C/D**

Paths below are relative to `/home/korvin/apps/neosr-dev`. Confidence: **H** = directly established from code; **P** = additionally confirmed by an isolated probe; **M** = execution-dependent risk.

**A — Stability**

| ID | Severity | Source | Defect and failure scenario | Fix | Confidence |
|---|---|---|---|---|---|
| A1 | critical | [image.py:610](/home/korvin/apps/neosr-dev/neosr/models/image.py:610) | `inference_mode()` around `D(sr)` removes the G gradient. Logged GAN loss increases without teaching G. | Freeze D parameters while retaining autograd through D’s input. | P |
| A2 | high | [image.py:654](/home/korvin/apps/neosr-dev/neosr/models/image.py:654); `archs/metagan_arch.py:119,175` | Both discriminator training forwards run in eval mode. Spectral-normalization estimates remain stale as weights change; dropout is disabled. Applies also to SN UNet/PatchGAN/DUNet/ResGAN. | Explicit G/D phase mode policy; train D during its update. | H |
| A3 | critical | [image.py:502](/home/korvin/apps/neosr-dev/neosr/models/image.py:502), `:735` | Counter increments in both `optimize_parameters` and `closure`. Accumulate=2 steps every batch; 4 steps every two batches, while losses still divide by 2/4. | One counter, one boundary decision, outside closures; normalize once. | P |
| A4 | high | [train.py:600](/home/korvin/apps/neosr-dev/train.py:600), `:454`; `utils/logger.py:143` | Saves update count, resumes it as microbatch count, then divides again for display. With accumulation, resume repeats work and shifts schedules; logging divides an already adjusted count. | Persist separately named `micro_step` and successful optimizer-update counters. | H |
| A5 | critical | [image.py:992](/home/korvin/apps/neosr-dev/neosr/models/image.py:992); `base.py:348,462` | EMA-only G/D saves are paired with live optimizer moments; EMA count, scalers, RNG, sampler position, loss buffers and partial accumulation are absent. Resume changes the optimization trajectory. | Versioned checkpoint containing live networks, EMA state, optimizers, scalers and counters; save at completed boundaries. | H |
| A6 | high | [base.py:343](/home/korvin/apps/neosr-dev/neosr/models/base.py:343), `:353`; `image.py:851,889` | SF weights are copied to CPU **before** optimizer `.eval()`. Validation sets `is_train=False` before testing `sf_optim_g and is_train`, so the SF switch never occurs. | Define training/export representations; transition before export snapshots and restore with `finally`. Preserve original validation state. | H |
| A7 | high | [adan_sf.py:260](/home/korvin/apps/neosr-dev/neosr/optimizers/adan_sf.py:260), `:315` | Warmed-up `lr` is calculated, but `_multi_tensor_adan` receives `group["lr"]`. Configured 1,600/600-step warmups do not scale actual G/D updates. | Pass the intended effective rate; verify numerical traces against a documented formulation. | H |
| A8 | high | [train.py:581](/home/korvin/apps/neosr-dev/train.py:581) | Slow models, compilation startup or large accumulation can trigger “Interrupted” and exit code 0 without user interruption. | Delete throughput-based termination. | H |
| A9 | high | [image.py:405](/home/korvin/apps/neosr-dev/neosr/models/image.py:405), `:695,746`; `optimizers/fsam.py:109` | FSAM owns another base optimizer, absent from `self.optimizers`. Its second closure also accumulates D gradients and changes bookkeeping. SAM backward bypasses scaling, but D still uses scaler step; D scaler is not updated in this branch. | One optimizer owner; pure G replay closure; independently update D once. Explicitly reject unsupported AMP/SF combinations until verified. | H |
| A10 | high | [image.py:477](/home/korvin/apps/neosr-dev/neosr/models/image.py:477) | ECO overwrites `self.gt`; SAM’s second pass blends an already blended target. Its preliminary training-mode forward can also update running statistics twice. | Preserve original batch; construct ECO targets once per microbatch with an explicit buffer/mode policy. | H |
| A11 | high | [image.py:540](/home/korvin/apps/neosr-dev/neosr/models/image.py:540), `:611,657`; `losses/wavelet_guided.py:386` | After activation, D trains on high-frequency bands while G queries D with RGB. Both wavelet call sites prohibit gradients. Fixing A1 alone leaves incompatible adversarial domains. | Shared differentiable adversarial transform; detach only the D fake branch. | H |
| A12 | high | [ldl_loss.py:92](/home/korvin/apps/neosr-dev/neosr/losses/ldl_loss.py:92) | Differentiating `variance**0.2` at zero produces non-finite gradients. Identical images give finite zero loss but non-finite gradients. | Detach artifact weights if they are weighting masks, or regularize the differentiable formulation; test zero residuals. | P |
| A13 | high | [ncc_loss.py:22](/home/korvin/apps/neosr-dev/neosr/losses/ncc_loss.py:22) | Constant inputs divide by zero; `.view(C,-1)` fails for channels-last and mixes batch/channel grouping for N>1. | Flatten spatial dimensions per N,C; specify aggregation and stabilize variance denominators. | P |
| A14 | high | [ssim_loss.py:155](/home/korvin/apps/neosr-dev/neosr/losses/ssim_loss.py:155) | Negative SSIM/contrast factors raised to fractional powers produce NaN; anticorrelated checkerboards reproduce it. | Stabilize factors before fractional powers; validate minimum spatial size and channel count. | P |
| A15 | high | [vgg_perceptual_loss.py:47](/home/korvin/apps/neosr-dev/neosr/losses/vgg_perceptual_loss.py:47), `:149,176`; `archs/vgg_arch.py:128` | Patch/IPK additions have no gradients and divide by unprotected norms. Cached VGG loading calls `.load_state_dict` on a constructor function; non-VGG19 variants receive VGG19 weights. | Differentiable patch extraction with stable norms; instantiate the correct backbone/weights pair. | H |
| A16 | medium | [msswd_loss.py:177](/home/korvin/apps/neosr-dev/neosr/losses/msswd_loss.py:177); `consistency_loss.py:138`; `resnet_perceptual_loss.py:76`; `archs/resnet_arch.py:50` | MS-SWD retains only the final scale. L*’s low branch multiplies by Y twice. ResNet features remain trainable/training-mode; zero perceptual weight leaves its result undefined, and `loss_weight` is swallowed. | Accumulate inside the scale loop; correct L*; freeze/eval feature nets; normalize weight contracts. | H |
| A17 | high | [data/__init__.py:99](/home/korvin/apps/neosr-dev/neosr/data/__init__.py:99); `models/base.py:550` | DDP leaves `dataloader_args` undefined. Later, reduction stacks mixed scalar/[1] tensors and assigns the `None` return of `distributed.reduce` to `losses`. | Common loader argument construction; scalarize logs; reduce in place or use `all_reduce`. | H |
| A18 | high | [image.py:823](/home/korvin/apps/neosr-dev/neosr/models/image.py:823); `base.py:169` | Rank zero validates through a DDP wrapper while other ranks proceed with training. Buffer broadcasts can mismatch or hang. EMA copies the wrapped model too. | Validate an unwrapped inference model; synchronize ranks around validation, or implement distributed validation. | M |
| A19 | high | [base.py:163](/home/korvin/apps/neosr-dev/neosr/models/base.py:163), `:229,347` | Compilation happens before loading; unwrapping handles DP/DDP but not `_orig_mod`. Ordinary pretrained weights fail against compiled keys; compiled checkpoints lack eager/export portability. | Load and checkpoint canonical unwrapped modules; apply execution wrappers afterward. | H |
| A20 | high | [otf.py:146](/home/korvin/apps/neosr-dev/neosr/models/otf.py:146), `:154,207,231`; `train.py:132` | Degradations are merged into `datasets.train`, but Poisson/JPEG branches read root keys. Another JPEG branch ignores `jpeg_compress=False`. | One validated degradation configuration object used by every branch. | H |
| A21 | high | [otf.py:33](/home/korvin/apps/neosr-dev/neosr/models/otf.py:33); `data/otf_dataset.py:176`; `models/simple_otf.py:88` | Queue rounding uses configured per-GPU batch, not actual DataParallel batch; zero-sized queues fail. Fixed 512 preprocessing can be too small. Legacy models pass HR `gt_size` into an LR-size crop API. | Validate actual batch/crop geometry; use explicit LR/HR size names; derive queue capacity from runtime batch. | H |
| A22 | high | [prefetch_dataloader.py:104](/home/korvin/apps/neosr-dev/neosr/data/prefetch_dataloader.py:104) | Waiting on the copy stream does not record subsequent use on the consumer stream. Released batch storage can be reused while GPU consumers remain active. | Record every transferred tensor on the consuming stream; handle nested batches. | M |
| A23 | high | [paired_dataset.py:77](/home/korvin/apps/neosr-dev/neosr/data/paired_dataset.py:77); `data_util.py:122`; `otf_dataset.py:84,156` | Meta-info checks/read keys disagree; paired LQ path is a directory; OTF divides a string path by a filename. Retry sampling includes out-of-range `len(dataset)`; paired retries do not repeat reads. | Unified path resolver and bounded read/decode retry; preserve failing paths in exceptions. | H |
| A24 | high | [utils/rng.py:11](/home/korvin/apps/neosr-dev/neosr/utils/rng.py:11); `archs/arch_util.py:16`; `data/__init__.py:55` | Imports repeatedly parse CLI, reseed global state and potentially initialize distributed processing. Module NumPy Generators ignore worker seeding, correlating fixed-seed workers; spawned DDP workers can initialize process groups. | Parse once at entry; inject config and worker/rank-specific generators. | H |
| A25 | high | [soap_sf.py:594](/home/korvin/apps/neosr-dev/neosr/optimizers/soap_sf.py:594), `:629,678,692,904` | State uses `(Parameter,index)` keys, then mapping is cleared; SF eval looks up the original parameter and misses `z`—confirmed. Tuple keys also defeat ordinary optimizer parameter remapping. Decay changes `grad`, but updates consume `grad_projected`. | Store view state beneath canonical parameter keys; implement explicit serialization and correct update/decay ownership. | P/H |
| A26 | high | [arch_util.py:141](/home/korvin/apps/neosr-dev/neosr/archs/arch_util.py:141); `ea2fpn_arch.py:250` | DySample requests pinned memory for a CUDA tensor; pinning is for CPU allocations. EA2FPN reaches this helper during forward. | Remove pinning from device-local tensors; allocate grids explicitly on input device. | H |
| A27 | high | [image.py:717](/home/korvin/apps/neosr-dev/neosr/models/image.py:717), `:767`; `base.py:268` | Only G-loss NaN is checked, after backward. Inf, D failures and finite-loss/non-finite-gradient cases escape. AMP skips still advance EMA/schedulers; unscaled training can corrupt weights. | Check phase loss and unscaled gradients; track successful steps; advance associated EMA/scheduler only on success. | H |
| A28 | medium | [image.py:783](/home/korvin/apps/neosr-dev/neosr/models/image.py:783), `:793,972`; `metrics/calculate.py:163` | Reflect-padding fails when tile padding ≥ image dimension; output allocation assumes input/output channels match; no tile halo causes seams. Empty validation leaves `_idx` undefined. Nonpositive SSIM becomes “best possible” infinity. | Validate tiling, infer output channels, use context overlap; reject empty sets; preserve valid negative SSIM. | H |
| A29 | high | [train.py:431](/home/korvin/apps/neosr-dev/train.py:431); `image.py:78,768` | QAT preparation occurs after optimizer construction. `_init_ema()` replaces `AveragedModel` with an ordinary network; the next EMA update calls a missing method. | Prepare model structure before optimizer/EMA/wrapper creation; keep one EMA contract. | H |
| A30 | medium | [augmentations.py:41](/home/korvin/apps/neosr-dev/neosr/data/augmentations.py:41), `:241,261,287` | Mixup blends LQ with another **GT**, introducing clean target content. Multi-augmentation selection ignores weights; “none” can still resample LQ; omitted probabilities cause `len(None)`. | Define paired augmentation semantics; make identity exact and probabilities validated. | H |
| A31 | high | [default.py:299](/home/korvin/apps/neosr-dev/neosr/models/default.py:299), `:355,455,538,587` | Legacy hierarchy requires a scheduler, unpacks ten wavelet outputs from a two-output function, backpropagates a number when G updates are skipped, prints nonexistent `self.loss_dict`, and indexes empty/exhausted GAN schedules. | Migrate its supported behavior into one trainer; do not maintain another optimization implementation. | H |
| A32 | high | [pancam_otf.py:39](/home/korvin/apps/neosr-dev/neosr/models/pancam_otf.py:39), `:56`; `data/pancamsingle_dataset.py:50` | `linear_blur` is undefined without `kernel1final`. Pancam inference discards the last channel before extracting distortion, taking a color channel instead. | Initialize branch state; separate RGB and distortion before slicing; preserve metadata through transforms. | H |
| A33 | medium | [adamw_win.py:188](/home/korvin/apps/neosr-dev/neosr/optimizers/adamw_win.py:188); `adamw_sf.py:56,84` | Win divides by learning rate, so a scheduler reaching zero crashes. AdamW-SF ignores `schedule_free=False`; beta1=0 makes eval conversion divide by zero. | Validate optimizer domains and capabilities; avoid ratios that divide by scheduled LR. | H |

**C — Performance**

| ID | Severity | Source | Defect and scenario | Fix | Confidence |
|---|---|---|---|---|---|
| C1 | medium | [base.py:559](/home/korvin/apps/neosr-dev/neosr/models/base.py:559); `consistency_loss.py:189`; `ff_loss.py:102`; `diffjpeg.py:542` | Every microbatch synchronizes each logged loss; tensor-valued conditions add syncs; JPEG loops over GPU quality scalars. | Accumulate detached device logs, transfer together at reporting intervals; vectorize quality conversion and tensor predicates. | H |
| C2 | medium | [train.py:320](/home/korvin/apps/neosr-dev/train.py:320); `data_sampler.py:40`; `data/__init__.py:141`; `image.py:445` | Global CUDA default hides placement dependencies; sampler generates GPU indices only to copy them to Python. Prefetching is unconditional, pinning/persistent workers default off, and input layout does not follow AMP model layout. | Explicit devices; CPU sampling; configurable pinned/persistent loading and prefetch; benchmark channels-last independently of AMP. | H |
| C3 | medium | [image.py:787](/home/korvin/apps/neosr-dev/neosr/models/image.py:787), `:807,914`; `metrics/calculate.py:201,238` | Tiling shuttles data GPU→CPU→GPU and empties cache per tile; validation empties it per image. Learned metrics rebuild/load networks per image after CPU image conversion. | Reuse metric instances; tensor-based metrics; selectable CPU stitching for memory limits; remove routine cache flushing. | H |
| C4 | medium | [otf.py:70](/home/korvin/apps/neosr-dev/neosr/models/otf.py:70); `wavelet_guided.py:398`; `degradations.py:932` | Entire queues are shuffled/copied each batch; unnecessary source clones accompany assignments. Wavelet modules are rebuilt each call; Poisson unique-count loops synchronize repeatedly. | Indexed queue replacement, cached filter buffers, profiled/vectorized degradation kernels. Preserve required dequeue copies. | H |
| C5 | medium | [default.py:452](/home/korvin/apps/neosr-dev/neosr/models/default.py:452), `:533`; `dists_loss.py:96` | Legacy running sums retain graph chains. DISTS alpha/beta are created trainable after the freeze loop, so unused gradients accumulate. BF16 unnecessarily enables scalers in `image.py:149`. | Detach statistics; freeze all loss parameters after construction; use scaling where the selected precision requires it. | H |

**D — Structure, error handling and configuration**

| ID | Severity | Source | Defect and scenario | Fix | Confidence |
|---|---|---|---|---|---|
| D1 | high | [base.py:363](/home/korvin/apps/neosr-dev/neosr/models/base.py:363), `:371,485`; `train.py:235` | Direct, separate checkpoint writes permit incomplete sets. A successful third retry is reported as failure. Resume exceptions silently start fresh training. | Atomic checkpoint transaction/manifest; success-driven retries; explicit resume failure. | H |
| D2 | high | [base.py:414](/home/korvin/apps/neosr-dev/neosr/models/base.py:414), `:423,445`; `convert.py:91,126` | Requested parameter keys are overridden by guessed EMA priority; broad exceptions conceal selection/deployment failures. Failed loading exits with success. | Explicit key precedence and structured errors; one shared checkpoint reader for training/testing/export. | H |
| D3 | medium | [archs/__init__.py:24](/home/korvin/apps/neosr-dev/neosr/archs/__init__.py:24); `models/__init__.py:23`; `losses/__init__.py:20` | Selecting one component imports every implementation and its optional dependencies/CLI side effects. An unrelated architecture can prevent all startup. | Explicit lazy registry mapping; import only requested implementations. | H |
| D4 | medium | [image.py:159](/home/korvin/apps/neosr-dev/neosr/models/image.py:159); `train.py:154,329`; `base.py:253`; `data/__init__.py:135` | Conflicting total-iteration locations/defaults; custom experiment paths are overwritten; warmup without a scheduler lacks `initial_lr`; dead loader configuration follows an unconditional raise. | Normalize and validate configuration once; reject unsupported combinations and unknown keys. | H |
| D5 | medium | [convert.py:198](/home/korvin/apps/neosr-dev/convert.py:198), `:239,260` | Export relies on implicit device/default-exporter behavior; CPU allocation context does not move an existing CUDA model. Disabled flags succeed without requested FP16 output or fail only after exporting. | Explicit exporter/device contract; reject unavailable options before work; verify multiple dynamic shapes. | M/H |

**3. Loss pipeline design proposal — B**

Use one immutable batch record and one **per-forward context**. It holds `sr`, original `gt`, effective target GT after ECO, `lq`, step counters, and references to D/EMA G. Lazy cached properties provide upsampled LQ, adversarial views and optional EMA predictions. Create a fresh context for each SAM forward; never mutate the original batch or retain graphs across iterations.

Each loss implements `evaluate(context) -> Result`. Existing two-tensor `forward` methods can remain as numerical kernels through a shared `PairTerm` mixin. Configuration determines metadata once; the training loop never inspects loss names or types.

GAN configuration expands into separate generator and discriminator terms. A generic phase controller owns train/eval mode, parameter freezing, autocast, backward, accumulation, clipping and stepping. Losses neither step optimizers nor toggle model modes.

Illustrative core, excluding the generic optimizer controller:

```python
from dataclasses import dataclass, field
from typing import Callable
import torch
from torch import Tensor, nn

@dataclass
class Context:
    sr: Tensor
    gt: Tensor                    # effective target for this forward
    original_gt: Tensor
    lq: Tensor
    step: int
    discriminator: nn.Module | None
    ema_generator: nn.Module | None
    providers: dict[str, Callable]
    cache: dict[str, Tensor] = field(default_factory=dict)

    def get(self, name):
        if name not in self.cache:
            self.cache[name] = self.providers[name](self)
        return self.cache[name]

@dataclass
class Result:
    loss: Tensor
    logs: dict[str, Tensor]

class Term(nn.Module):
    def evaluate(self, ctx: Context) -> Result:
        raise NotImplementedError

class PairTerm(Term):
    # Existing loss classes inherit this and retain forward(sr, target).
    target_name = "gt"
    log_key = "l_g_pix"

    def evaluate(self, ctx):
        target = ctx.gt if self.target_name == "gt" else ctx.get(self.target_name)
        value = self(ctx.sr, target).reshape(())
        return Result(value, {self.log_key: value.detach()})

class GanGenerator(Term):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def evaluate(self, ctx):
        logits = ctx.discriminator(ctx.get("adv_sr"))
        value = self.criterion(logits, True, is_disc=False)
        return Result(value, {"l_g_gan": value.detach()})

class GanDiscriminator(GanGenerator):
    def evaluate(self, ctx):
        real = ctx.discriminator(ctx.get("adv_gt"))
        fake = ctx.discriminator(ctx.get("adv_sr").detach())
        lr = self.criterion(real, True, is_disc=True)
        lf = self.criterion(fake, False, is_disc=True)
        total = lr + lf
        return Result(total, {
            "l_d_real": lr.detach(), "l_d_fake": lf.detach(),
            "l_d_total": total.detach(),
            "out_d_real": real.detach().mean(),
            "out_d_fake": fake.detach().mean(),
        })

def evaluate_terms(terms, ctx):
    results = [term.evaluate(ctx) for term in terms]
    total = torch.stack([r.loss.reshape(()) for r in results]).sum()
    logs = {k: v for r in results for k, v in r.logs.items()}
    return total, logs

# The same execution path handles each configured phase.
for phase in phases:
    with phase.activate(ctx), phase.precision():
        total, logs = evaluate_terms(phase.terms, ctx)
    phase.backward(total / accumulation)
    phase.finish_if_boundary()
```

`adv_sr` is identity or a cached differentiable wavelet transform; `adv_gt` uses the same domain. D receives detached fake data. EMA predictions use `no_grad()` and a frozen eval model. D must remain frozen until G backward completes. The phase controller must preserve accumulated D gradients while freezing its parameters.

Keep `pixel_opt`, `mssim_opt`, etc. through a compatibility loader that produces ordered term specifications. Preserve type aliases and existing component logging keys. Apply each weight exactly once. Reject non-scalar training reductions rather than silently changing their meaning. Normalize ResNet’s `perceptual_weight` explicitly.

The current D backward optimizes `real + fake`, while logging half that sum. Preserve the optimization scale initially; make the corrected logging convention explicit. Log undivided microbatch objectives and aggregate over the reporting window.

| Current loss / source | Context inputs | Existing key → logging |
|---|---|---|
| `L1Loss`, `MSELoss`, `HuberLoss`, `chc_loss` — `basic_loss.py:25,57,89,133` | SR, effective GT | `pixel_opt` → `l_g_pix` |
| `mssim_loss` — `ssim_loss.py:66` | SR, GT; stable FP32 kernel | `mssim_opt` → `l_g_mssim` |
| `ncc_loss` — `ncc_loss.py:8` | SR, GT | `ncc_opt` → `l_g_ncc` |
| `kl_loss` — `kl_loss.py:9` | SR, GT; stable log-softmax; explicitly defined spatial normalization | `kl_opt` → `l_g_kl` |
| `fdl_loss` — `fdl_loss.py:253` | SR, GT; frozen backbone/projection buffers owned by loss | `fdl_opt` → `l_g_fdl` |
| `consistency_loss` — `consistency_loss.py:15` | SR, GT or cached upsampled LQ | `consistency_opt` → `l_g_consistency` |
| `msswd_loss` — `msswd_loss.py:116` | SR, GT or upsampled LQ | `msswd_opt` → `l_g_msswd` |
| `vgg_perceptual_loss` — `vgg_perceptual_loss.py:58` | SR, GT; differentiable optional patches | `perceptual_opt` → `l_g_percep` |
| `ResNetPerceptualLoss` — `resnet_perceptual_loss.py:11` | SR, GT; frozen feature extractor | `perceptual_opt` → `l_g_percep` |
| `dists_loss` — `dists_loss.py:45` | SR, GT; frozen coefficients/features | `dists_opt` → `l_g_dists` |
| `ldl_loss` — `ldl_loss.py:15` | SR, GT; detached/stabilized artifact map | `ldl_opt` → `l_g_ldl` |
| `ff_loss` — `ff_loss.py:9` | SR, GT; FP32 FFT | `ff_opt` → `l_g_ff` |
| `gan_loss` — `gan_loss.py:7` | D, adversarial SR/GT views | `gan_opt` → separate G/D terms and existing component keys |

Current LDL does **not** use EMA; do not silently change its algorithm during migration. The context supports an EMA-dependent variant separately. Wavelet guidance is a view provider, not another scalar loss. Legacy `gw_opt`, `color_opt`, and `luma_opt` branches lack corresponding registered implementations here; explicitly reject or migrate them rather than inventing equivalence.

**4. Competitor ideas worth porting — E**

Paths here are under `/home/korvin/apps/traiNNer-redux-dev`.

| Rank | Concrete idea and source | Value / cost |
|---|---|---|
| 1 | Separate live resume weights from EMA exports: [sr_model.py:1200](/home/korvin/apps/traiNNer-redux-dev/traiNNer/models/sr_model.py:1200). Save/restore scaler and EMA progress: `base_model.py:731,812`. | Very high / medium |
| 2 | One explicit accumulation decision supplied to optimization: [train.py:596](/home/korvin/apps/traiNNer-redux-dev/train.py:596), `sr_model.py:452,690`. Track scaler-skipped updates and gate EMA at `:845`. | Very high / low–medium |
| 3 | Strict typed configuration rejecting unknown fields: [redux_options.py:8](/home/korvin/apps/traiNNer-redux-dev/traiNNer/utils/redux_options.py:8), `options.py:91`. Keep TOML; port validation principles, not its entire schema. | High / medium |
| 4 | Standard GAN retains input gradients through frozen D: `sr_model.py:463,554`; BF16 disables GradScaler at `:188`; SF evaluation switches explicitly at `:925`. | High / low |
| 5 | Initialize the actual NumPy Generator in workers: [data/__init__.py:168](/home/korvin/apps/traiNNer-redux-dev/traiNNer/data/__init__.py:168). Parse configuration once: `utils/config.py:19`. Prefer injected objects over copying singleton restrictions. | High / low–medium |
| 6 | Common DDP/non-DDP loader construction and configurable prefetch/pinning: `data/__init__.py:92,99,148`. | Medium / low |
| 7 | Overlap-aware tiled inference: `sr_model.py:849`. Port the concept with positive boundary weights and bounded-memory stitching. | Medium / medium |

Do not copy its loss-name/type branching. Its tiled blend has zero boundary weights (`sr_model.py:874–878`), its save retry pattern shares the final-attempt flaw (`base_model.py:777`), and its DISTS metric still constructs a network per call (`metrics/dists.py:18`).

**5. Deletion list**

- Delete unimported backup [image.py.helaman:1](/home/korvin/apps/neosr-dev/neosr/models/image.py.helaman:1) and empty `data/otfgray_dataset.py:1`.
- Delete unused threaded `PrefetchGenerator`/`PrefetchDataLoader` (`prefetch_dataloader.py:9,43`); producer exceptions otherwise leave consumers waiting.
- Delete unreachable loader configuration (`data/__init__.py:135`), duplicate metric imports (`metrics/__init__.py:12`), discarded `.detach()` calls (`image.py:623,688`), duplicate logger closing (`train.py:656`), and commented export implementations (`convert.py:16,243,263`).
- Remove the throughput kill switch and advertised-but-disabled export flags.
- **After migration**, delete `models/default.py`, duplicated optimization/save/validation implementations, and duplicate OTF queues. Preserve simple/equirectangular/pancam behavior as data/degradation strategies.
- After migrating its last caller, remove duplicated artifact-map code in `losses/loss_util.py:1`.
- Replace and then remove CLI-reading `rng()`/`net_opt()` mechanisms. Do not delete shared architecture helpers merely because their callers are outside this review’s scope.
- Remove discriminator EMA unless an explicit supported use remains; it currently costs memory and replaces the discriminator needed for resume.

**6. Proposed refactor plan**

| Order | Work and parallel boundaries | Verification |
|---|---|---|
| 1 | Fix GAN autograd/modes, accumulation, Adan warmup and automatic termination. | GAN-only G gradient is nonzero; frozen D receives none; accumulation 1/2/3/4 matches a reference update; warmup affects update magnitude. |
| 2 | Build canonical checkpoint/optimizer lifecycle before architectural cleanup. | Interrupted/uninterrupted equivalence; SF representation checks; scaler/EMA restoration; injected write failures; eager/compiled/DP key round trips. |
| 3 | Parallel, disjoint work: numerical loss kernels; config/registry/RNG; dataset/degradation/prefetch repairs. Agree interfaces first. | Constant/identical/anticorrelated loss inputs and gradients; import without CLI side effects; distinct reproducible worker streams; every OTF branch and metadata path. |
| 4 | Integrate context-based losses and phase controller in `image.py`; migrate legacy models. | Legacy configuration/log-key compatibility; isolated G/D updates; ECO target immutability; SAM replay and optimizer serialization. |
| 5 | Parallel after checkpoint interfaces settle: validation/metrics; export/wrapper handling. | Small/odd/large tiled images, seam comparison, output-channel changes, empty datasets, SF mode restoration, multi-shape export parity. |
| 6 | Profile the corrected implementation, then remove obsolete code. | FP32/FP16/BF16, channels-last and compile comparisons; GPU synchronization/transfer traces; peak memory; two-rank DDP smoke tests. |

Treat FP16 overflow, SF+EMA, SAM+GAN, wavelet activation, resume and validation as explicit integration cases. Keep unsupported combinations rejected until those checks pass.

**7. Open questions / unverified areas**

- No GPU training, DDP, compile, export, downloads or checkpoint writes were executed. Performance benefits remain unmeasured.
- Numerical probes confirmed A1/A3/A12–A14; a constructor-free SOAP probe confirmed its missing eval-state lookup. A full optimizer probe was blocked because PyTorch imports attempted temporary-directory creation under the read-only sandbox.
- The logs establish saturation and resume activity, but not their exact source revision or termination cause.
- The intended Adan-SF mathematical specification needs an agreed reference beyond fixing the demonstrably unused warmup rate.
- Confirm which legacy panoramic/equirectangular behaviors, QAT support and GAN scheduling options must survive migration.
- Exact data-order resume requires a deliberate policy for worker prefetch, sampler position and OTF queue/RNG state. Ordinary weight-and-optimizer resume is insufficient.
