# Stage 3, lane "r3gan-audit": the owner's R3GAN experiments vs the paper and the reference code

Read-only audit. Sources: the paper (Huang et al., NeurIPS 2024, `/home/korvin/apps/r3gan/r3gan_2501.05441v1.pdf`, read as text),
the official repo brownvc/R3GAN (fetched read-only: `R3GAN/Trainer.py`, `R3GAN/FusedOperators.py`,
`training/networks.py`, `training/training_loop.py`, `train.py`; `Networks.py`/`Resamplers.py` from memory, see section 9),
the owner's code and 42 train logs under `traiNNer-redux-dev/experiments/4x_ESRGAN*`. Two CPU probes (torch 2.9.1, 4 GB cap):
`scratchpad/r3gan_audit/probe.py`, `probe2.py`; log parser `parse_logs.py`, output `logs.txt` (scratchpad = `/tmp/claude-1000/-home-korvin--claude-spares-spare-KgM9rZ/d7c995a9-c812-4ec5-ba57-eb99120e9524/scratchpad`, session-temporary).

Tags: [paper] [ref] (official repo code) [code] (owner code) [log] [probe] [reasoning].
Paths: `r3gan/` = `/home/korvin/apps/r3gan/`, `alt` = `r3gan/alternative/r3gan_loss.py` (= live
`traiNNer-redux-dev/traiNNer/losses/r3gan_loss.py`), `trn/` = `/home/korvin/apps/traiNNer-redux-dev/`, `exp/` = `trn/experiments/`.

## 0. Verdict

| rank | piece | what it really is | use it? | cost of one D update vs adversarial-only (owner D, GT 256) [probe] |
|---|---|---|---|---|
| 1 | `r3gan/v2/r3gan_loss.py` (minimal, 90 lines; v1 differs in the gamma units) | the official `Trainer.py` math almost line by line: pairwise RpGAN softplus, R1 + R2 every step on the same forward, `gamma/2`, `create_graph=True` | **base for the loss**, after 6 fixes (section 6) | 1.86x (2 D forwards, 2 penalty graphs) |
| 2 | `trn/traiNNer/archs/rpgan_r3gan_scalar_discriminator_arch.py` (owner D) | a faithful port of the reference `Discriminator` mechanics (blocks, fix-up init, FIR, basis head), with SR-sized widths and a dense high-res stage | **base for the D**, after fixes (cardinality, dtype, FIR, head) | 15.4 GFLOP/img forward at GT 256, 1.73 M params |
| 3 | `alt` (488 lines, the loss that actually ran) | **not R3GAN**: relativistic-*average* hinge with detached means, lazy penalties on random 96 px crops, separate fp32 penalty forwards | discard; keep 2 ideas (penalty on un-augmented input, fp32 option) | 1.26x on a GP step with crop, 2.86x without crop; 1.07x amortised at k=4 |
| 4 | `r3gan/sr_model3.py:641-651` | non-relativistic GAN + lazy R1 only, hard-coded gamma 10, k=4 | discard | +1 fp32 forward and 1 double backward every 4th step |
| 5 | `r3gan/sr_model2.py` | calls `generator_forward`/`discriminator_forward`, which no audited loss has | discard (dead code) | n/a |

Glue (`r3gan/sr_model*.py`, `/home/korvin/apps/sr_model.py`, `trn/traiNNer/models/sr_model.py`, the three patches): reference only;
nothing to port.

Top defects (details in sections 2-5):
1. **Every logged r3gan run trained with the GAN weight squared.** `start_iter` wraps the loss in `IterativeLossWrapper`
   (`trn/traiNNer/losses/__init__.py:99-108`), which multiplies by the weight (`iterative_loss_wrapper.py:212-219`), and the model
   multiplies again by `abs(loss.loss_weight)` (`trn/traiNNer/models/sr_model.py:845,865`; wrapper keeps `loss_weight`,
   `iterative_loss_wrapper.py:71`). The logged `l_g_gan` equals w^2 x (1 + D margin) for all six weights used: 0.05->0.0025,
   0.1->0.0100, 0.08->0.0064, 0.04->0.0016, 0.01->1.0e-4, 0.001->1.0e-6 [log]. "r3gan 0.1" was 0.01, "0.001" was 1e-6 (no GAN).
   The minimal path squares too: `generator_penalty` multiplies by `loss_weight` (`r3gan/r3gan_loss.py:60`) and so does
   `r3gan/sr_model.py:491` [code].
2. **The "alternative" loss is relativistic-average hinge, not RpGAN.** Its D loss is exactly invariant to shuffling the fakes
   (2.136424 paired vs 2.136424 permuted; pairwise RpGAN 1.492 vs 1.483) [probe]; the reference means are detached for D
   (`alt:233-239`). The paper's convergence and mode-coverage arguments are about pairwise RpGAN with a smooth f [paper].
3. **The penalty fallback cannot fire, and if it did it would be a silent no-op.** `autograd.grad(create_graph=True)` does not
   raise for an op without a double backward; the error comes at the outer `.backward()` [probe: grid_sample];
   the owner's MUNet run died exactly there (`exp/4x_ESRGAN_archived_20260106_222310/train_4x_ESRGAN_20260106_215836.log:277-287`,
   `sr_model.py:774`, SDPA efficient-attention backward) [log]. The `create_graph=False` branch (`alt:94-100`) would return a
   penalty with no path to D parameters while logging non-zero R1/R2.
4. **Map-output penalties are off by a factor H'W' in both directions across versions.** Mean-before-grad (`alt:65-67`,
   `r3gan/r3gan_loss.py:65`) scales R1 by 1/(H'W')^2 vs the reference `Critics.sum()` [probe: ratio 1.526e-5 = 1/256^2]; the earlier
   sum-version flattened MUNet (D margin 0.0016, penalties 10^3) [log]. gamma is not transferable between D types or patch sizes.
5. **The minimal loss crashes on the owner's scalar D** (`critics.mean([1,2,3])` on shape (B,): IndexError [probe]) and its gamma
   schedule is a Python counter with a unit bug (`max_img * 32`, `r3gan/r3gan_loss.py:52`: 1.2 M D steps at batch 8 to reach the
   end value) that resets on resume.
6. Owner D: `stage_dtype=float32` is a no-op under autocast (convs run bf16, output bf16 [probe]) but adds casts and an fp32
   residual stream; dense 3x3 at the two highest resolutions is 71% of its FLOPs (paper group size 16 would be 3.2x cheaper);
   its FIR fallback reshapes to (B*C,1,H,W), 17x slower than an equivalent grouped conv on CPU [probe].
7. Lazy regularisation (k=4 in the main run) and the random penalty crop are what made the alternative cheap; the paper reports
   lazy R1/R2 as worse on real data and divergent on toy data [paper, App. E].

## 1. What ran when [log] [code]

All 42 logs: ESRGAN (RRDBNet) 4x from scratch, LQ 64 / GT 256, AMP bf16, channels_last, TF32 on, no accumulation, EMA 0.999,
charbonnier 1 + HSLuv 0.1 + GAN, `gan_type: r3gan` routed to `R3GANLoss` (`trn/traiNNer/losses/__init__.py:69-75`), traiNNer
commit e0a9234c "(dirty)". The loss and D files were edited between launches (crash signatures below), so the code that ran is
inferred from the logged keys:

| period | loss version (inferred) | evidence |
|---|---|---|
| 2026-01-05 20:44 .. 01-06 01:01, 01-06 22:23 .. 01-07 19:12 | committed e0a9234c version: every-step R1/R2, no crop, gradient of the **sum** over the D map | no `gp_applied` key; `git show e0a9234c:traiNNer/losses/r3gan_loss.py` lacks lazy/crop/mean |
| 01-05 23:52, 01-06 00:04, 00:14, 00:17 | lazy versions, k ~ 8, 1, ~16-20, ~16-20 | `gp_applied` = 0.12-0.13, 1.0, 0.05-0.06 |
| 01-07 20:33 onward (incl. the 43k main run) | `alt` as committed 01-13: k = 4, `gp_crop` 96, mean over map | `gp_applied` = 0.25 at every print |

The minimal loss (`r3gan/r3gan_loss.py`, `v2`) and `r3gan/sr_model*.py` have **no log**: true pairwise R3GAN was never run.

## 2. Per implementation

### 2.1 Minimal: `r3gan/r3gan_loss.py` (v1) and `r3gan/v2/r3gan_loss.py` (v2)

| aspect | finding |
|---|---|
| loss | G: `softplus(-(D(sr) - D(hr)))` (`:56-60`); D: `softplus(-(D(hr) - D(sr)))` (`:77-78`). Pairwise, matched (SR_i vs its own HR_i), element-wise on maps. Matches `Trainer.py` [ref]. |
| R1/R2 | `autograd.grad(critics.mean([1,2,3]).sum(), samples, create_graph=True)`, `.square().sum([1,2,3])` (`:62-67`): per-sample squared norm, batch mean later, `create_graph=True`, on the **same** forward as the adversarial term (2 D forwards per D step [probe]). Deviation: `mean` over the map instead of the reference `Critics.sum()` = 1/(H'W')^2 scaling [probe]. |
| scalar D | `mean([1,2,3])` raises IndexError on (B,) outputs [probe]; works only with 4D maps. |
| broadcasting | `r1[..., None, None]` is (B,1,1); added to a (B,1,H',W') map gives (B,B,H',W') [probe]. The mean is still correct (0.704675 vs 0.704675) because mean of an outer sum = sum of means, but it allocates B x the map. Bug in form, not in value. |
| gamma | `gamma/2 * (R1+R2)` (`:83`) as in the reference; cosine from `start_gamma` 150 to `end_gamma` 15 copied from the FFHQ-256 preset [ref]. |
| state | `self.batch` Python int, `+= hq.size(0)` per D call (`:82`), starts at `current_img * batch_size_chek` (`:51`). Not in any state_dict; resets on resume unless `current_img` is hand-set; per-rank on DDP (schedule k x slower on k GPUs); counts micro-batches; does not advance on skipped D steps. Not a function of the iteration. |
| unit bug (v1) | `max_img * 32` (`:52`) = 9.6 M images by default: 1.2 M D steps at batch 8, 0.6 M at batch 16 [probe]. v2 drops the x32 and sets 2e6 images (`v2:43,52`) but defaults `batch_size_chek=32` (`v2:45`), so `current_img` = iteration overcounts 4x at batch 8. |
| dtype | inherits the caller's autocast: in `r3gan/sr_model.py:548-563` the D step and both double backwards run in bf16/fp16 autocast with `GradScaler` for fp16; no fp32 option. bf16 is what the paper used [paper]; fp16 is not. |
| weight | `generator_penalty` returns `mean * loss_weight` (`:60`) and `r3gan/sr_model.py:491` multiplies by `abs(loss.loss_weight)` again: w^2. |
| compile/AMP/DDP | not compile-safe (double backward); AMP bf16 fine; DDP: no special handling (see 7.6). |
| cost | G step 2 D forwards; D step 2 forwards + 2 first-order graphs + 1 double backward, every step [probe]. Cheapest correct layout apart from concatenation. |

### 2.2 Alternative: `alt` (= live traiNNer loss; `/home/korvin/apps/r3gan_loss.py` is the same with `reg_interval=8`)

| aspect | finding |
|---|---|
| loss | D: `relu(1 - (D(r) - mean(D(f)).detach()))` + `relu(1 + (D(f) - mean(D(r)).detach()))`, x0.5 (`alt:233-239`; the names `real_mean`/`fake_mean` are swapped but the formula is RaHinge). G: RaHinge without detach (`alt:321-335`). Relativistic **average**: permutation-invariant [probe]. The docstring's "relativistic hinge (R3GAN)" (`alt:8,119,200`) is wrong. |
| R1/R2 | separate forwards on `images.detach().clone()` cast to fp32 with CUDA autocast off (`alt:52-57`); map reduced by **mean** before grad (`alt:65-67`); `(grads.norm(2,dim=1)**2).mean()` (`alt:105-106`); x weight x k, then `0.5*(r1+r2)` (`alt:284-285,310`): gamma = `r1_weight`, lazy scaling exact in expectation. |
| crop | `gp_crop` 96: one random window per batch, different for R1 and R2 (`alt:287-294`). For the scalar D the penalty is taken at D(crop), a point the adversarial term never evaluates; for a map D it is a sub-region. Changes the objective. |
| lazy | `_gp_step` buffer `+1` per D call, applied when `% k == 0` with `.item()` (`alt:241-248`). Buffer is not checkpointed by traiNNer (losses are not saved): phase resets on resume (harmless). |
| fallback | `try/except` around `autograd.grad` (`alt:70-103`) cannot catch the missing-double-backward error (raised later in `.backward()`) [probe] [log]; the `create_graph=False` branch would be a silent no-op of R1/R2. |
| dtype | adversarial term in autocast (bf16), penalties fp32, cast back to `images.dtype` (`alt:109-111`): R2 becomes bf16 when the G output is bf16. |
| syncs | `.item()` every D step (`alt:247`) and 2 per GP step (`alt:211-212`); `warnings.warn` + `logger.warning` every GP step when un-augmented images are missing (`alt:262-273`). |
| dead code | `_skip_discriminator` never passed (`alt:347,353`); `MultiScaleR3GANLoss` refuses `r3gan` (`alt:458-462`); vanilla/lsgan/wgan/hinge branches duplicate `GANLoss` (`alt:149-197`). |
| patches | `/home/korvin/apps/aaa.diff` = `diff.patch` (identical): turns the every-step, sum-reduced version into this one (adds mean-reduce, lazy k=8, crop 96). `s3.diff`: adds the `IterativeLossWrapper` dispatch to the D call (present in `/home/korvin/apps/sr_model.py:747-756`). |

### 2.3 Integration glue

| file | GAN path | defects |
|---|---|---|
| `r3gan/sr_model.py` | minimal loss: G `generator_penalty` (`:455-458`), D `discriminator_penalty` under autocast (`:548-563`), D every iteration | w^2 (`:491` + loss `:60`); D step runs from iteration 0 (no GAN start); `r1/r2` logged ratio fine. |
| `r3gan/sr_model2.py` | `loss.generator_forward` / `cri_gan.discriminator_forward` (`:466,604`) | API exists in no audited loss -> AttributeError; `adaptive_d` skips D updates when the G GAN EMA rises (`:478-492`), contrary to every-step R3GAN; also replaces the perceptual loss by **minus** the distance to a lagging copy of G (`:495`, `model_neg` every 100 its `:672`) and silently resizes GT to the output size (`:430`). Out of R3GAN scope, but a red flag. |
| `r3gan/sr_model3.py` | any `cri_gan` + lazy R1 every 4 iterations, fp32, separate forward (`:641-651`) | R1 only (paper: RpGAN/GAN + R1 alone diverges [paper, Fig. 1]); gamma 10 and k=4 hard-coded (`:647`); `+ 0 * real_pred[0]` is a numeric no-op; mean over map before grad (`:644`). `apply_paired_cuts_upscale` maps HQ masks to LQ by `// 2` (`:73-74`): wrong for the 4x runs. |
| `/home/korvin/apps/sr_model.py`, live `trn/.../sr_model.py` | `alt` via kwargs; D step only on `apply_gradient` (`:733` / live `:956`) | w^2 through the wrapper (`:642,662` / live `:845,865`); with `accum_iter > 1` D sees one micro-batch per update; `gt.clone()` + `output.clone()` every step (`:490-491`) even without augmentation. |

### 2.4 The owner's discriminator (`trn/traiNNer/archs/rpgan_r3gan_scalar_discriminator_arch.py`)

Faithful to the reference [ref + memory of `Networks.py`]: 1x1 -> grouped 3x3 -> 1x1 residual block with two biased leaky ReLU
0.2 (`:220-264`); MSR init with `Gain * VSP^(-1/4)` (fix-up, L^-0.25 [paper]) and zero-init last conv (`:241-255`); forward gain 1
(`:61-86`; the reference `bias_act(..., gain=1)` [ref], the sqrt(2/(1+0.2^2)) gain is init-only — the brief's "scaled by sqrt(2)" is
not what the reference does); FIR [1,2,1] -> [1,3,3,1]/8 stride 2 (`:101-154`); 4x4 depthwise basis + linear head (`:291-309`);
no normalization, no spectral norm; projection conditioning kept (`:465-489`); `reshape` instead of the reference `view`
(`:153-154,491`), which is what crashed the 01-07 18:30 run under channels_last [log].

| deviation | effect |
|---|---|
| defaults widths (32,64,128,256,256), cardinality (1,1,2,8,8), 1 block per stage (`:381-383`) | 1.73 M params. Group size = 64/128/128/64/64 channels, not the paper's 16 [paper]: stages 0-1 are dense 3x3 at full and half resolution = **71% of FLOPs** [probe]. Cardinality (4,8,16,32,32) gives 0.87 M params, 4.83 vs 15.36 GFLOP at GT 256 [probe]. Paper: 2 blocks per stage. |
| 4 downsamples + `adaptive_avg_pool2d(4,4)` inside the basis (`:305-306`) | GT 128 -> 8x8, GT 256 -> 16x16 pooled to 4x4; the reference instead adds stages down to 4x4. `force_4x4` (`:397,481-482`) is redundant (the basis always pools). Scalar output per patch. |
| `stage_dtype=float32` (`:388,352-357,473-474`) | under autocast the convs still run bf16 and the output is bf16 [probe]; the `x.to(fp32)` per stage only adds an upcast and makes the residual stream fp32 (`x + y`, `:264`). The reference sets bf16 per stage **without** autocast [ref]. |
| fused ops (`:52-58,93-98`) | `torch_utils.ops` is absent in traiNNer and neosr -> pure PyTorch fallback (fine for double backward, 2 extra kernels per activation). |
| FIR fallback (`:150-154`) | `reshape(B*C,1,H,W)` forces a contiguous copy of a channels_last tensor and returns NCHW; same result as a grouped conv (max diff 0.0), 10.2 vs 0.6 ms on CPU [probe; GPU not measured]. FIR is 0.2% of FLOPs: the cost is layout traffic, not arithmetic. A plain strided 3x3 conv C->C in its place would cost 9C MACs per output element instead of 16 (18x more at C=32) and is not anti-aliased (paper principle d). |

History [log]: 01-07 18:44 `mat1 and mat2 shapes cannot be multiplied (8x86528 and 512x1)` (= 512 ch x 13x13: an earlier
512-wide version applying the 4x4 basis to a 16x16 map); three OOMs at 512 MiB allocations (18:49-18:55); the YAML passes no D
arguments, so the main run used whatever defaults the dirty file had at 20:53; I cannot confirm they equal today's.

### 2.5 The owner's note (`r3gan/r_3_gan_для_sr_restoration_...md`)

Correct: RpGAN + R1 + R2 as the core; R1 alone unstable; fp16 broke, bf16 worked; lazy regularization hurt; matched SR/HR pairing
is the natural SR choice; R2 adds cost [paper]. Not followed by the code: the note says "better not lazy" and RpGAN softplus,
the code that ran is lazy RaHinge. Inaccurate or incomplete: rule 1 needs **double** differentiability (DySample/grid_sample,
fused SDPA and some custom ops break it, the note recommends DUnet/attention ideas); its pseudocode never sets
`x_real.requires_grad_(True)` and re-runs D for R1 (an extra forward), and `reduce_logits` (mean over the map) gives the
1/(H'W')^2 scaling; "GP only on one scale" of a multi-scale D changes the objective.

## 3. Cost (resource efficiency)

### 3.1 Per implementation

| | minimal (v1/v2) | alt (as run) | sr_model3 lazy R1 | reference |
|---|---|---|---|---|
| D forwards per G update | 2 (fake with grad to input, real without graph) | 2 | per `cri_gan` (1-2) | 2 |
| D forwards per D update, no penalty / with penalty | n/a / 2 (every step) | 2 / 4 (2 adv bf16 + 2 penalty fp32) [probe: 2,2,2,4] | 2 / 3 | 2 / 2 |
| double backward per D update | 1 (both penalties in one outer backward) | 1 every k-th step | 1 every 4th | 1 |
| real and fake share one forward | no (2 calls, same graph) | no | no | no |
| penalty dtype | caller's autocast (bf16 in practice) | fp32 | fp32 | bf16 in the stages listed in `FP16Stages`, fp32 elsewhere, no autocast, TF32 off [ref] |
| extra tensors alive | 2 first-order graphs until backward; (B,B,H',W') broadcast map | `detach().clone()` + fp32 copy per penalty input (crop-sized), 2 extra graphs, glue clones of GT and output | `gt.requires_grad_` in place | 2 first-order graphs |
| Python-side syncs | none (numpy gamma on host) | `.item()` every D step + 2 per GP step; warnings | none | none |
| compile | D cannot be compiled with double backward (any variant) | same | same | not compiled |
| channels_last | fine | fine | fine | n/a |

### 3.2 Measured (owner D, CPU, fp32) [probe]

FLOPs per D update, batch 2, GT 256 (forward + backward, incl. double backward):

| variant | GFLOP | vs adversarial-only |
|---|---|---|
| adversarial only (2 fwd + bwd) | 215 | 1.00x |
| reference / minimal, R1+R2 every step, shared forward | 399 | 1.86x |
| alt GP step, crop 96 | 271 | 1.26x |
| alt GP step, no crop | 614 | 2.86x |
| alt amortised, k=4, crop 96 | — | 1.07x |
| exact lazy k=4, shared forward, no crop | — | ~1.21x |

One R1 term (forward + `grad(create_graph)` + backward) = 6.5x a forward. Saved-for-backward memory (activation proxy), batch 2,
GT 256: adversarial-only 640 MiB, every-step R1+R2 1662 MiB (2.6x), concatenated single forward 1649 MiB, alt GP step crop 96 897 MiB,
alt GP step no crop 2303 MiB.

Discriminators at the same input (batch 1):

| D | params | fwd GFLOP GT 128 / 256 | output | R1/R2 double backward on CPU |
|---|---|---|---|---|
| owner RpGAN scalar (defaults) | 1.73 M | 3.84 / 15.36 | (B,) | works |
| same, group size 16 (card 4,8,16,32,32) | 0.87 M | — / 4.83 | (B,) | works |
| neosr `metagan` | 6.02 M | 0.32 / 1.27 | (B,1,2,2) / (B,1,4,4) | **fails**: `_scaled_dot_product_flash_attention_for_cpu_backward` has no derivative |
| neosr `dunet` | 3.23 M | 11.67 / 46.67 | full-res map | **fails**: `grid_sampler_2d_backward` (DySample) |

Neither neosr D can run R1/R2 as is; metagan probably can with the SDPA math backend (not verified), dunet needs DySample replaced.

### 3.3 Cost-saving ideas: legitimate vs objective-changing

| idea | verdict |
|---|---|
| penalties on the same forward as the adversarial term (reference) | legitimate, saves 2 forwards vs `alt` |
| one concatenated forward for real+fake (no normalization in D, so per-sample independent) | legitimate; same FLOPs and memory (1649 vs 1662 MiB), fewer kernel launches, one `autograd.grad` |
| lazy R1/R2 every k steps with weight x k | unbiased in expectation, but the paper measured worse FID and toy-data divergence [paper App. E]; offer only as an opt-in deviation |
| fp32 penalty pass | precision choice; the paper ran bf16; costs 2 extra forwards unless the D itself runs fp32 |
| penalty on a random crop | changes the objective (penalised points differ from evaluated points; scalar D sees a different input size) |
| relativistic average instead of pairwise | changes the objective (no pairing; outside the paper's analysis); not cheaper |
| mean over the map before grad | changes the penalty scale by 1/(H'W')^2; not a saving |
| `stage_dtype=float32` under autocast | pure waste (casts, fp32 residual stream, no precision gain) |
| dense 3x3 at high resolution (cardinality 1) | the main D cost; paper-style group size 16 is 3.2x cheaper |

## 4. Training evidence [log] (extends `competitor/recipes_evidence.md` section 2)

"margin" = logged `l_g_gan` / w^2 - 1 = mean D(real) - mean D(fake) in the linear part of the RaHinge (hinge margin is 1).
"pen" = logged `r1_penalty` (already x gamma x k, averaged over the print window including zero steps).

| run (log) | D, D lr (AdamW 0.9/0.99) | GAN w -> effective | R1=R2, penalty version | what happened |
|---|---|---|---|---|
| `..20260105_204414` | metagan3, 1e-4, batch 16 | 0.05 -> 0.0025 | 0 | margin 0-6.4, `grad_norm_d` 0.1 -> 151 (unregularised D runs away), PSNR 28.63 @2k -> **27.42 @3k** -> 28.53 @5k |
| `..20260105_215003` (+6 resumes to 14.7k) | metagan3, 1e-4 | 0.05 then 0.1 -> 0.0025 / 0.01 | 0.5 then 10, every step (sum), later lazy | pen 0.69/5.6 @100 then **1e-8 from 2.8k**: D collapsed to an input-independent constant (`l_g_gan` = 0.0025024 exactly, `:324,381,438`); GAN was a no-op; PSNR rose monotonically 28.43 -> 29.39 @14k |
| `..20260106_001744` .. `_010101` (5 fresh starts) | metagan3, 1e-4 | 0.1 -> 0.01 | 10 | near-constant D from 400-500 its in 3 of 4 runs with data (pen 1e-7 .. 5e-9, margin ~0); `..002823` kept margin <= 0.17 to 2.6k |
| `..210458`, `..210955` | munet | 0.1/0.08 | 10 | OOM at start (penalty graphs, batch 8, GT 256) |
| `..215836` | munet | 0.08 | 10 | crash: no double backward for SDPA efficient attention, raised in the outer `backward()` |
| `..222310`, `..015009`, `..015602` | munet, 1e-4 | 0.08 -> 0.0064 | 10, every step (sum over the output map) | pen 5.9e3 -> 0.13, `grad_norm_d` 7.9e4 -> 4.7, margin 0.0016: **penalty-dominated D**, flat on real vs fake |
| `..020134` | munet, 1e-5 | 0.04 -> 0.0016 | 0.5 | same picture at 10x lower lr (pen 224 -> 4, margin ~0) |
| `..183025`, `..184450` | RpGAN scalar, 1e-5 | 0.04 | 0.5 | crashes: `view` on channels_last; 8x86528 vs 512x1 head |
| `..184917` .. `..185520` | RpGAN scalar, 1e-5 | 0.04/0.01 | 0.5/3/10 | OOM at the first D step (earlier, larger D) |
| `..191219` | RpGAN scalar, 1e-5 | 0.001 -> **1e-6** | 0.2, every step | effectively no GAN; PSNR 29.17 @7k |
| `..203327` | RpGAN scalar, 5e-6 | 0.01 -> 1e-4 | 3, lazy k=4, crop 96 | effectively no GAN (margin 0.0-0.42 at weight 1e-4); 29.16 / 0.191 @7k |
| main `4x_ESRGAN/..205351` | RpGAN scalar, 5e-6 | 0.1 -> 0.01 | 3, lazy k=4, crop 96 | margin 0.0-0.43; `grad_norm_d` 0.2 -> 65-77; PSNR 29.01 @8k -> **28.24 @11k**; DISTS 0.188 -> 0.174 @9k |
| `..215959` (18k-37k) | RpGAN scalar, 5e-6 | 0.083 -> 0.0069 | 3, same | margin 0.0-0.5; `grad_norm_d` 0.05 -> 121; PSNR 29.47 @23k -> **28.62 @31k** -> 29.17 @36k |
| `..225708` (37k-43k) | RpGAN scalar, 5e-6 | 0.03 -> 0.0009 | 3, same | margin 0.34-0.48; `grad_norm_d` 96-208; GAN term ~1% of `l_g_total` |

The cleanest A/B in the set: `..203327` vs main `..205351` differ only in the GAN weight (effective 1e-4 vs 1e-2; same D, D lr,
gamma, start 1000, batch, data). PSNR/DISTS: 1k 28.16/0.203 vs 28.05/0.205; 3k 28.73/0.204 vs 28.73/0.204; 5k 29.01/0.195 vs
28.98/0.194; 7k 29.16/0.191 vs 28.99/0.188; the main run then dips (28.24 @11k) while DISTS improves to 0.174 @9k.

The logs prove: weights were squared (six weights, exact); with metagan3 + any penalty the D collapsed to a constant (GAN no-op);
with MUNet and a sum-reduced full-resolution-map penalty the D was penalty-dominated (no margin); the scalar RpGAN D kept a small
margin (< 0.52) with gamma 3 / k 4 / crop 96 while its parameter-gradient norm grew from ~0.2 to 100-200 over 43k updates at lr 5e-6; no NaN in any
run; ~0.17 dB PSNR cost at 7k for effective weight 0.01 (one seed, val8 = 7 images).
They cannot prove: what code ran in each launch beyond the inferred table in section 1; any causal link between the D-gradient growth
and the PSNR dips (phases were edited mid-run, no repeat); anything about pairwise R3GAN (never run); D dtypes at runtime; why
metagan3 collapsed (metagan3 has spectral norm + GroupNorm; not investigated).

## 5. B: what the reference does that none of the variants does

| reference feature [ref]/[paper] | minimal | alt | matters for SR? [reasoning] |
|---|---|---|---|
| pairwise RpGAN softplus | yes | no (RaHinge) | **yes**: matched SR/HR pairs make the relativistic term compare the same content |
| R1 + R2 every step on the same forward | yes | lazy, separate fwd, crop | **yes** (paper App. E); cost 1.86x of the D step |
| `Critics.sum()` before grad (per-sample norm of the gradient of the summed logits) | mean | mean (earlier: sum) | **yes** for map outputs: needs an explicit H'W' convention (7.2) |
| gamma, lr, beta2, EMA half-life, augment p on cosine burn-in as functions of images seen; `cur_nimg` from the loop | Python counter | none | partly: gamma/lr as a function of the iteration, yes; the large-to-small burn-in no (7.3) |
| Adam betas (0, beta2), beta2 0.9 -> 0.99, eps 1e-8, lr 2e-4 -> 5e-5 for G and D | not handled | AdamW (0.9, 0.99) in all runs | **yes for D**: beta1 = 0 (principle b "violating ... often leads to training failures" [paper]); G keeps its SR optimizer |
| D phase then G phase each iteration | G then D | G then D | no |
| bf16 per stage, no autocast, no GradScaler, TF32 off | autocast (bf16 or fp16+scaler) | autocast + fp32 penalty | yes: forbid fp16 for the GAN path |
| no DDP: manual all-reduce of flat grads | — | — | yes if neosr trains D on >1 GPU (7.6) |
| D: 2 blocks per stage, group size 16, down to 4x4, ~23 M params | — | owner D: 1 block, dense high-res, pooled head, 1.7 M | width/size: SR D can be small; group size: yes (cost) |
| fix-up init, zero-init last conv, no normalization, leaky ReLU 0.2, FIR resampling | — | owner D: yes | yes (double-differentiable, no batch coupling) |
| non-leaky augmentation on D inputs, penalty through the augmentation | — | penalties on un-augmented images | low: SR has paired data and no D overfitting signal so far; keep off by default |
| EMA of G with half-life schedule | — | traiNNer EMA 0.999 | neosr already has G EMA; schedule not needed |

## 6. C: recommended base, fixes, junk

Base: **minimal v2 math + owner D**, rewritten as one neosr loss (no Python counter, no kwargs API).

Must fix / add:
1. Weight applied once (the factory owns `loss_weight`; the loss returns the unweighted term). Document that traiNNer weights map to
   neosr as w^2.
2. Map handling: per-position pairing `softplus(-(D(hr) - D(sr)))` averaged over (B, H', W'); penalty
   `||grad_x sum_hw D_hw||^2 / (H'W')` per sample (equals the reference for a scalar D); works for (B,) and (B,1,H',W').
3. Correct broadcasting (`[:, None, None, None]` or reduce per sample first).
4. gamma (and optional D lr/beta2) as a stateless function of the iteration from config, resume- and DDP-safe; no `max_img*32`.
5. Concatenate real and fake in one D forward and one `autograd.grad` (same FLOPs, fewer launches).
6. Every-step penalties by default; `reg_interval` as an opt-in with x k scaling, logged as a deviation.
7. Precision: never fp16 for the GAN path; D either under bf16 autocast end to end (simplest, paper-compatible) or outside
   autocast with an explicit per-stage dtype; drop `stage_dtype=float32` under autocast.
8. Fail fast on a D without double backward (a one-off check at build: a tiny R1 on a 32 px input), not a silent fallback.
9. Owner D: cardinality from a group size (default 16), optional 2 blocks/stage, grouped-conv FIR, extra stages or a 1x1 patch head
   instead of adaptive pooling (7.1), `reshape` everywhere.
10. D optimizer Adam(0, 0.99), constant or mildly scheduled lr; no schedule-free for D (7.7). D excluded from `torch.compile`.
11. Log margin (mean D(real) - D(fake)), R1, R2 separately (raw, before gamma) and their ratio; these diagnosed every failure in section 4.

Junk: RaHinge/relativistic-average, detached means, random penalty crop, `SafeGradientPenalty` fallback, the non-r3gan branches inside
`R3GANLoss`, `MultiScaleR3GANLoss`, `IterativeLossWrapper` double weighting, `adaptive_d` skipping, `0 * real_pred[0]`, hard-coded
gammas, `sr_model2` APIs and negative perceptual loss, `apply_paired_cuts_upscale` (`// 2`), `force_4x4`, `stage_dtype` under autocast.

## 7. D: SR-specific judgement [reasoning unless tagged]

1. **Scalar vs patch output.** Both fit the theory if each output position is treated as a sample. In SR, SR_i and HR_i are aligned
   pixel for pixel, so per-position pairs compare the same content: a denser, local signal than one scalar per 256 px patch. Recommend a
   patch head at 1/16 resolution (16x16 for GT 256) with the scalar basis head as an option; neither variant was ever A/B'd (scalar
   RpGAN D was only run with the wrong loss and w^2).
2. **Penalties on maps.** With `sum` the penalty grows ~H'W' (MUNet: D flattened, margin 0.0016 [log]); with `mean` it shrinks
   1/(H'W')^2 [probe]. `||grad sum D||^2 / (H'W')` is exact for non-overlapping receptive fields and keeps gamma comparable across D
   heads and patch sizes.
3. **gamma magnitude.** Reference data is in [-1,1]; for the same D in [0,1] units gamma_01 = gamma_pm / 4, so FFHQ-256's 150 -> 15
   is 37.5 -> 3.75 [ref + arithmetic]. The paper's gammas are not a clean function of resolution (FFHQ-64 2 -> 0.2, FFHQ-256
   150 -> 15, CIFAR-32 0.05 -> 0.005 [paper Table 9]) and its large initial gamma is justified by "p_theta and p_D are far apart early"
   [paper App. D]. In SR they are close from the start (pixel loss), so the D's adversarial signal is weak and a large gamma flattens D:
   metagan3 collapsed at gamma 0.5-10 (sum), MUNet was penalty-dominated at 10 (sum); the scalar D stayed alive at 3 with k 4 and a
   96 px crop [log]. Start with a constant gamma in [0.1, 1] (per-position convention, [0,1] inputs), no burn-in decay, sweep
   {0.1, 0.3, 1, 3} watching margin and R1/R2.
4. **Every-step vs lazy.** Paper: every step; lazy worse/divergent [paper]. Cost of every step with the owner D at GT 256: D update
   1.86x, ~+0.86 D-steps of compute per iteration; with group size 16 the whole D is 3.2x cheaper, which pays for it.
5. **bf16 / compile.** bf16 double backward is what the paper ran; fp16 + GradScaler is excluded. `torch.compile` has no double
   backward: compile G only, keep D eager.
6. **DDP.** The reference avoids DDP (manual all-reduce [ref]). A DDP-wrapped D used in several forwards with `create_graph=True` is
   fragile (reducer hooks fire per backward); use `no_sync` + manual all-reduce or `static_graph=True`. Not verified.
7. **D optimizer.** Adam (0, 0.99) per the paper; all owner runs used AdamW (0.9, 0.99) with lr 1e-4 .. 5e-6 [log]. Schedule-free
   optimizers evaluate gradients at an interpolation of iterates (momentum-like, beta1 0.9 by default) and switch weights between
   train/eval: no evidence for a GAN D; keep them for G only.
8. **EMA.** G EMA is orthogonal; no D EMA in the paper. Validate on EMA weights as now.

## 8. Probes [probe]

`scratchpad/r3gan_audit/probe.py` (~40 s) and `probe2.py` (~20 s), CPU, `systemd-run --user --scope -p MemoryMax=4G`, the owner's files
loaded with a stub registry: forward/R1 FLOPs and params of the owner D, neosr `metagan`, `dunet`; autocast dtypes; minimal-loss
scalar crash, broadcast shapes and value, mean-vs-sum ratio, counter units; pairing permutation test; fallback test with
`grid_sample`; forward counts per update; FLOPs per D update; saved-tensor bytes; FIR reshape vs grouped conv.

## 9. Not verified

- `Networks.py` and `Resamplers.py` of the reference were not fetched; block/init/FIR/basis statements rely on memory and agree with
  the owner's port. `FusedOperators.py`, `Trainer.py`, `training/networks.py`, `training/training_loop.py`, `train.py` were fetched.
- The reference applies `FP16Stages = [-1,-2,-3,-4]` (bf16) to `MainLayers` of both G and D [ref]; in D those are the lowest-resolution
  stages, while the paper text says "4 highest resolutions". I did not run it to resolve this.
- GPU timings, GPU memory, and the cost of the FIR reshape on CUDA; FLOPs and saved-tensor bytes are CPU fp32 proxies.
- Which loss/D source ran in each launch (dirty tree), the D defaults of the main run, and why metagan3 collapsed.
- Whether neosr `metagan` supports R1/R2 with the SDPA math backend; DDP behaviour with double backward.
- `/home/korvin/Downloads/` holds further variants (`r3gan_loss_lazy_crop*.py`, `rpgan_..._fixed*.py`, `munet_arch_r3gan.py`); out of
  scope, not read.
- Any claim that pairwise R3GAN improves SR quality: never run by the owner.
