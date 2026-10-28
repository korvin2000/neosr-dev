# R3GAN in neosr: gap table, design, efficiency and test plan (lane "r3gan-neosr-fit")

Opus 5.5, 2026-10-28, `refactor/stabilize` at 527d16f (read-only lane; other lanes had uncommitted A3/A8 work in the tree).
Tags: [code] `path:line` in this worktree; `C/` = `/home/korvin/apps/traiNNer-redux-dev/traiNNer/`; `ref/` = github.com/brownvc/R3GAN
(main, fetched 2026-10-28); [paper] Huang et al., NeurIPS 2024; [probe] CPU probes of this lane (torch 2.9.1, Ryzen 5950X
without bf16 hardware, every job capped at 4 GB; scripts and logs in the session scratchpad `.../scratchpad/r3/`, listed in
section 8); [log] stage-2 A/B run logs; [reasoning]. The probe D in `r3/r3d.py` is a measuring prototype, not the deliverable.
The sibling lane's audit of the owner's code is `audit.md` (same folder); its findings are not repeated here.

## 0. Decisions in one screen
1. **No new loss class.** R3GAN in neosr = `gan_type = "rpgan"` (exists) + R1 and R2 (exist) at `reg_every = 1` + two new
   `gan_loss` kwargs for the gamma schedule (`gamma_decay`, `gamma_iters`; a pure cosine function of `ctx.it`) + a new D.
   Three small fixes in the penalty path: a per-position penalty for map outputs, the penalty pass under bf16 autocast when the
   phase runs bf16, per-penalty logs. Defaults (`bce`, gammas 0, metagan template) do not move.
2. **New D `neosr/archs/r3gan_arch.py`**, class `r3gan`: pure-torch port of the owner's scalar D with the paper's group
   width 16, FIR as a grouped conv, in-place biased leaky ReLU, 4-D output, two heads. `ARCHS` rows: `r3gan` (patch head,
   any GT multiple of 16), `r3gan_scalar128`, `r3gan_scalar256` (paper head, exact 4x4 basis).
3. **Head:** A/B the patch head first (local, per-position pairing of the same content; gamma independent of the patch size;
   cheaper in launches and parameters); the scalar head is the paper-faithful control.
4. **Every-step R1+R2** (`reg_every = 1`) is the R3GAN setting. Lazy regularisation is not an exact saving under Adam and the
   paper found it worse (audit.md, App. E citation); it stays a measured option, not the recipe.
5. **Efficiency** (section 4): per update, D-related work is 9 D-forward units without penalty and 15 with every-step R1+R2
   (exact FLOP counts); the penalty update is 2.0x a plain D update in FLOPs, 3.1-3.4x in CPU wall time, and keeps 1.6x the
   activations alive. bf16 halves its memory and (on Ada) doubles its tensor throughput versus TF32 at a 1-2 % penalty error;
   that, the in-place activation (-24 % graph memory, exact) and the group-width-16 D (3.2x fewer FLOPs than the owner's
   default widths) are the savings worth taking. Reusing G's no-grad `D(gt)` in the D phase is impossible (no graph); the
   reverse reuse would need a phase-order change, saves 6.7 % and is not bitwise. Not recommended.
6. **Recipe arm** `options/train_realplksr_r3gan.toml` (not default): pretrained G, `r3gan` D, Adam (0, 0.99) lr 2e-4, rpgan
   0.3, R1 = R2 = 10 per position, `reg_every = 1`, `wavelet_guided = false`. A/B on one GPU in three rounds (~15 GPU-hours).
7. **Biggest risk:** the incumbent "GAN" baseline is inert in the stage-2 A/B logs (metagan sits at chance, `out_d_real` ~=
   `out_d_fake` ~= 0, `l_g_gan` = 0.3 ln 2), so every R3GAN arm is in effect compared with no GAN; the matrix therefore
   includes a no-GAN control, and gamma has to be found by a grid because it does not transfer between D types.

## 1. Gap table

### 1.1 What neosr already has (verified in code)
| R3GAN piece | where | status |
|---|---|---|
| pairwise RpGAN, G side: `softplus(D(real) - D(fake))`, real side under `no_grad` | `neosr/losses/gan_loss.py:88-92` | correct ([paper] eq. RpGAN; `ref/R3GAN/Trainer.py` same math); unit test `tests/test_s2_losses.py:540` |
| pairwise RpGAN, D side: `softplus(fake - real)` per element, margin log | `gan_loss.py:160-163` | correct for scalar and map outputs (elementwise pairs = same image, same position); test `:558` |
| R1 (real) + R2 (fake) zero-centred penalty `gamma/2 * E||grad_x D||^2`, per sample | `gan_loss.py:144-155` | correct scale for scalar heads (reference: `Gradient.square().sum([1,2,3])`, `(Gamma/2)*(R1+R2)`, `ref/R3GAN/Trainer.py`) |
| one forward serves the adversarial loss and both penalties | `gan_loss.py:143,156` | exact saving, already as the reference |
| real and fake in one `cat` forward unless D has batch statistics | `gan_loss.py:118-125`, `:11-13` | exact for per-sample Ds |
| penalty every `reg_every` updates, scaled by `reg_every`, decided by host ints of `ctx.it` | `gan_loss.py:130,155` | `reg_every = 1` is the R3GAN setting; no sync, rank- and resume-safe |
| double backward safety: eager D (`eager()`), `sdpa_kernel(MATH)` | `gan_loss.py:16-18,139-143` | needed: compiled graphs have no double backward [probe: "torch.compile with aot_autograd does not currently support double backward"] |
| D trains exactly while the G term is on (`start_iter`, `ramp_iters`) | `gan_loss.py:76-77,95-98,115-116` | usable for a GAN start after pretraining |
| D phase: autocast, scaler (off for bf16), `backward(inputs=D params)` | `neosr/models/phase.py:127-135,158-166` | the double-backward graph only reaches D params; inputs get no `.grad`; test `tests/test_s2_losses.py:476` compares param grads with a reference |
| D compiled when it is not batch-coupled; DDP wrappers; G's pass without reducer | `neosr/models/base.py:154-173`, `neosr/models/image.py:168-171` | the R3GAN D compiles (no SN, no dropout, no mutating buffers) |
| DDP: regularised rpgan through the DDP-wrapped D | `tests/test_s2_losses.py:637` (2-rank gloo) | covered |
| exact resume: optimizer/scheduler/EMA state, RNG streams saved | `phase.py:258-340`, `neosr/models/checkpoint.py:254-293` | a random penalty probe (1.3) stays resume-exact |
| Adam with `betas = [0, 0.99]`; cosine LR; per-net scheduler targeting | `base.py:33-44`; `neosr/config.py:503-510` (A2 `nets`) | R3GAN's D optimizer and LR decay are config-only |
| G EMA with a warm-up schedule (`ema_power`) | `phase.py:349-371` | the neosr analogue of R3GAN's EMA ramp |
| validation metrics DISTS, TOPIQ (kadid/pipal), LPIPS (A3, uncommitted in the tree) | `neosr/metrics/__init__.py:20,25` | E0 judges available |

### 1.2 Missing
| piece | proposal | size |
|---|---|---|
| R3GAN discriminator | `neosr/archs/r3gan_arch.py` + 3 `ARCHS` rows (section 2) | ~110 lines |
| gamma schedule | `gamma_decay`, `gamma_iters` on `gan_loss`; cosine from `start_iter` (section 3.2) | ~8 lines |
| per-penalty logs | `l_d_r1`, `l_d_r2` (weighted, sum = `l_d_reg`), added to `RESERVED` (`neosr/losses/__init__.py:47-56`) | ~6 lines |
| per-position penalty for map outputs | Rademacher probe (section 2.5); scalar outputs unchanged | ~5 lines |
| bf16 penalty pass | follow the phase autocast when it is bf16; fp32 otherwise (section 3.4) | ~4 lines |
| every-step path | none: `reg_every = 1` uses today's code; its cost is in section 4 | 0 |
| D optimizer / template | `options/train_realplksr_r3gan.toml` (section 5.1) | config |
| early refusal of a batch-coupled D with a penalty | `ConfigError` at the first D call (section 3.5) | ~3 lines |
| beta2 schedule (`ref/train.py` beta2 0.9 -> 0.99) | **not ported** (fixed 0.99, the paper's end value; section 3.6). Owner decision | 0 |

### 1.3 What is wrong or risky in today's code for this use
1. **A scalar `(B,)` output crashes the penalty.** `out.flatten(1)` (`gan_loss.py:149`) raises `IndexError: Dimension out of
   range` on a 1-D tensor [probe]; the owner's D returns `x.reshape(x.shape[0])` (`C/archs/rpgan_r3gan_scalar_discriminator_arch.py:491`)
   and `resgan` returns shape `(1,)` at batch 1 (`neosr/archs/resgan_arch.py:60`). Fix: `out.reshape(len(out), -1)`; the new D
   returns 4-D anyway.
2. **Map outputs: the penalty is on the mean critic, so gamma is not per position and does not transfer.** `score =
   out.flatten(1).mean(1).sum()` (`:149`) penalises `||grad mean_p D_p||^2`. Measured against the per-position penalty
   `mean_p ||grad D_p||^2` (one backward per position) [probe p5]: R3GAN patch D (P = 64 at GT 128) **0.031-0.047x**; metagan
   0.24x at P = 4 and 0.066x at P = 16 (~1/P). So a map head gets a 20-30x weaker R1 at the same gamma, and changing GT 128 ->
   256 changes it again (P x4). In the toy (5.2) mean-score gamma 10 behaved like per-position gamma ~1.
3. **The penalty pass always runs fp32 eager** (`:139-143`), also when the phase is bf16 and D is compiled. With
   `reg_every = 1` every D update loses bf16 and compile: 2x activation memory, TF32 instead of bf16 tensor throughput
   (section 4.3). The R3GAN reference ran its four high-resolution D stages in bf16 with R1/R2 through them
   (`ref/train.py` `FP16Stages`, `ref/training/networks.py` sets `torch.bfloat16`).
4. **`reg_every > 1` is not an exact saving under Adam and the shared clip.** The scaled kick `k * grad(penalty)` lands once
   per k updates: Adam's second moment absorbs it (step ~sqrt(k) instead of k times the per-step displacement; the inflated
   `v` also slows the adversarial updates in between), and with `grad_clip = true` (one setting for both phases,
   `image.py:138-148`; max-norm at `phase.py:180-182`) the regularised update is clipped as a whole [reasoning]. Equivalent
   only for plain SGD without clipping. The paper reports lazy R1/R2 worse (audit.md section 0 item 7); the owner's own note
   says the same (audit.md 2.5).
5. **metagan cannot take a zero-centred penalty at any sane gamma.** Its input gradient on real tiles is huge at init:
   per-position `||grad D_p||^2` = 2.5e7 (GT 128) and 8.4e5 (GT 256) [probe p5] vs 2.6e-4 for the R3GAN patch D. With
   gamma 1 the penalty starts ~1e6x the adversarial loss. This is the A-04 GroupNorm amplification
   (`docs/refactor/stage2/analysis/archs_cnn.md:171-210`). A "metagan + rpgan + R1/R2" arm needs gamma ~1e-6 and says little.
6. **dunet cannot take R1/R2 at all:** DySample uses `grid_sample` (`neosr/archs/arch_util.py:130`), whose backward has no
   derivative: `derivative for aten::grid_sampler_2d_backward is not implemented` [probe on dunet]. The error appears at the
   outer backward of the first regularised update.
7. **Batch statistics:** with `patchgan(norm_type="batch")` the split forwards (`:123-124`) make the per-sample penalty depend
   on the other samples through BN; today this runs silently. Refuse it (3.5).
8. **Dropout / DropPath in D** (metagan_arch.py `:156-157,179`) gives real and fake of a pair different masks: noise in the pairing
   and in the penalty. The R3GAN D has none.
9. **`wavelet_init` switches D's input** (`neosr/losses/wavelet_guided.py:42-48`, template `wavelet_init = 80000`): the
   penalty is on D's input, so gamma's meaning jumps at the switch and D restarts on a new input. R3GAN arms set
   `wavelet_guided = false` (template: until an A/B says otherwise).
10. **`eager()` under DDP + compile** returns the DDP wrapper (compile wraps DDP, `base.py:157-173`), so the penalty forward
    goes through the reducer; the 2-rank test (`tests/test_s2_losses.py:637`) covers DDP without compile only.
11. **`backward(inputs=...)` with `create_graph=True`:** sound. `autograd.grad(score, xs, create_graph=True)` computes only input
    gradients (weight grads pruned); the outer `backward(inputs=D params)` (`phase.py:166`) runs the forward graph's and the
    gradient graph's backward once, accumulating only into D params; the leaves `xs` get no `.grad`; the graph is freed after
    the outer backward. Under fp16 the scaler multiplies the outer loss only, not the inner `autograd.grad` (one reason to keep
    the fp16 penalty in fp32).
12. **Logs:** only `l_d_reg` (sum of both penalties) is logged, and only on regularised updates (window mean over those).

## 2. Discriminator spec: `neosr/archs/r3gan_arch.py`

### 2.1 Structure (ports `C/archs/rpgan_r3gan_scalar_discriminator_arch.py` = `ref/R3GAN/Networks.py` mechanics)
```
r3gan(in_ch=3, widths=(32, 64, 128, 256, 256), group_width=16, expansion=2, blocks=1, head="patch")
  extraction            Conv2d(in_ch, w0, 1, bias=False)                     # no activation (as ref)
  stages[i].blocks[j]   x + lin3(act2(lin2(act1(lin1(x)))))                  # i < n, j < blocks
      lin1  Conv2d(c, e*c, 1, bias=False)         act1 = bias + leaky_relu(0.2, inplace=True)
      lin2  Conv2d(e*c, e*c, 3, padding=1, groups=e*c // group_width, bias=False)   act2 = same
      lin3  Conv2d(e*c, c, 1, bias=False)  zero init
  stages[i].down        (i < n-1) FIR: F.conv2d(x, k.expand, stride=2, padding=1, groups=c), k = outer([1,3,3,1])/64,
                        non-persistent buffer (c,1,4,4); then proj Conv2d(c, w[i+1], 1, bias=False) if channels change
  head "patch":   head  Conv2d(w_last, 1, 1, bias=False)                     -> (B, 1, H/2^(n-1), W/2^(n-1))
  head "scalar":  basis Conv2d(w_last, w_last, 4, groups=w_last, bias=False) on an exact 4x4 map, head as above -> (B,1,1,1)
```
- **Init (fix-up):** MSR `normal(0, gain/sqrt(fan_in))`, bias 0 (`ref` `MSRInitializer`); `lin1`, `lin2` gain
  `sqrt(2/(1+0.2^2)) * (blocks*n)^(-1/4)` (`ref` ResidualBlock, L = 3 linear layers -> exponent -1/(2L-2)); `lin3` gain 0;
  extraction, proj, basis, head gain 1. Every block is the identity at init, so D is linear at init: real and bicubic inputs get
  identical input gradients (to 4 digits) and logits have std 0.03-0.15 [probe p5 init].
- **Activation gain:** forward gain is 1; `sqrt(2/(1+0.2^2))` is init-only (`ref/R3GAN/FusedOperators.py`: `bias_act(...,
  gain=1)`, verified). The brief's "leaky ReLU scaled by sqrt(2)" describes the init, not the forward (same finding in audit.md 2.4).
- **Group width 16** is the paper's: FFHQ-256 D widths `[96, 192, 384, 768, 768, 768, 768]`, cardinality `[12, 24, 48, 96,
  ...]`, expansion 2 -> 2*96/12 = 16 per group in every stage (`ref/train.py` presets, reversed for D). The owner's default
  cardinality `(1,1,2,8,8)` gives group widths 64-128 (dense 3x3 at the two highest resolutions): 7.68 vs 2.42 GMAC/img at
  GT 256 for the same widths and memory [probe p1].
- **Fixes vs the owner's file:** drop the `torch_utils.ops` imports (`:52-58,93-98`) and the `use_fused/use_cuda` toggles;
  drop `stage_dtype` (`:326,353-354,388,473-474`: a no-op under autocast; the loss decides the penalty dtype); one 4x4 rule
  instead of the two adaptive pools (`:305-306,481-482`); FIR as a grouped conv instead of `reshape(B*C,1,H,W)`
  (`:150-154`, a layout copy on channels_last; audit.md: 17x slower on CPU, same values); no `Convolution` wrapper casting
  weights (`:211-213`, autocast does it); no conditioning embedding (`:465-469,485-489`); output 4-D instead of `(B,)` (`:491`).
- **In-place leaky ReLU** on the fresh `x + bias` tensor saves its result instead of its input: -39 % forward saved bytes,
  -24 % penalty graph, values identical, `gradgradcheck` passes [probe p4 inplace].

### 2.2 `ARCHS` rows (`neosr/archs/__init__.py`, scale kwarg `None` like the other Ds)
| row | head | widths | for GT | params | GMAC/img fwd GT 128 / 256 | output at GT 256 |
|---|---|---|---|---|---|---|
| `r3gan` (class) | patch | (32, 64, 128, 256, 256) | any multiple of 16 | 0.87 M | 0.60 / 2.42 | 16x16, RF 108x108 px, stride 16 |
| `r3gan_scalar128` | scalar | (32, 64, 128, 256, 256, 256) | 128 | 1.21 M | 0.61 / - | (B,1,1,1) |
| `r3gan_scalar256` | scalar | (32, 64, 128, 256, 256, 256, 256) | 256 | 1.55 M | - / 2.44 | (B,1,1,1) |
The scalar head raises `ValueError("r3gan_scalar head: the final map is HxW, needs 4x4: train with GT <size> or use r3gan")`
on a wrong input size (static shape, fine under compile) rather than silently pooling. Recommended size per patch size: GT
256 (the owner's tiles at LQ 64) and GT 128: `r3gan` as listed; if VRAM-bound, `widths = (16, 32, 64, 128, 256)` (half the
activation memory, section 4.5). `blocks = 1` (owner's default; the paper uses 2, an A/B option at 2x the block cost).

### 2.3 Heads, pairing and penalties on a map
- **Pairing:** the D loss is `mean_{i,p} softplus(D_p(fake_i) - D_p(real_i))`: GT and SR are aligned crops of the same LQ, so
  position p of pair i is the same content. Already what `adv_loss` does elementwise (`gan_loss.py:162`); the G term likewise
  (`:92`). Not relativistic-average: shuffling the fakes changes the loss (audit.md section 0 item 2 shows the average variant
  does not).
- **Penalty:** each position is a critic, so the natural R1/R2 is `gamma/2 * mean_p ||grad_x D_p||^2` per sample. One backward
  gives it in expectation with a Rademacher probe: `score_i = sum_p eps_ip D_p(x_i) / sqrt(P)`, `eps = +-1`, since
  `E||grad sum_p eps_p D_p||^2 = sum_p ||grad D_p||^2` (cross terms vanish). Measured [probe p5]: 16-draw mean within 0-5 % of
  the exact per-position value for the R3GAN patch D and metagan; the deterministic alternative `sum_p D_p / sqrt(P)` is 2-3x
  too high for the R3GAN patch D (overlapping receptive fields, positively correlated gradients) and ~1x for metagan.
  `P = 1` (scalar head) draws nothing and is bitwise today's penalty. One draw per regularised update (device RNG, saved by the
  checkpoint: exact resume). Same cost as today's single `autograd.grad`.
- **Which head first for SR:** patch. Reasons: (a) SR errors are local (blur, ringing, false texture) and the per-position
  pairing compares like with like; (b) per-position gamma is independent of the patch size (RF 108 px is fixed), the scalar
  head's gamma scales with it (paper: 2 at 64 px vs 150 at 256 px); (c) fewer launches (865 vs 1800 aten ops per penalty
  update) and parameters at equal FLOPs (section 4); (d) Real-ESRGAN-style per-pixel/patch Ds are the SR norm. The scalar head
  stays as the paper-faithful control arm. [reasoning; the A/B decides]

### 2.4 Double-backward-safe ops (torch 2.9.1, `torch.autograd.gradgradcheck`, float64, CPU) [probe p3]
Pass: grouped conv 3x3, depthwise strided conv (the FIR), `leaky_relu` (also in-place on a fresh tensor), `adaptive_avg_pool2d`,
`avg_pool2d`, `F.interpolate` bilinear / bilinear+antialias / bicubic+antialias, `group_norm`, `rms_norm`, `mish`, SDPA (math
path). Fail: `grid_sample` (`derivative for aten::grid_sampler_2d_backward is not implemented`) -> DySample, dunet. Known CUDA
fails (stage 2, `stability.md:235-249`): flash / mem-efficient SDPA backward. `torch.compile` (aot_autograd) + double backward:
fails by design. The R3GAN D uses only conv2d, add, leaky_relu: no CUDA-specific risk, but the double backward of cuDNN grouped
convs in bf16 is unverified on the GPU (first GPU test in 7).

### 2.5 channels_last, bf16, compile
- channels_last: the trainer moves D to channels_last under AMP (`image.py:116,178`); the grouped-conv FIR keeps NHWC (no
  reshape copy). cuDNN NHWC grouped 3x3 with 16 channels per group in bf16 has no measurement here: grouped convs with small
  groups are often far below dense-conv efficiency [reasoning]. `scripts/bench.py` must measure `r3gan` vs metagan
  (channels_last as the trainer runs it); fallback if slow: `group_width = 32` (+48 % FLOPs, 0.89 vs 0.60 GMAC at GT 128).
- bf16: every op autocasts cleanly (convs to bf16, leaky_relu follows); no fp32 islands are needed in the D.
- compile: no SN power iteration, no dropout, no buffer writes, static shapes -> `wrap()` compiles it (`base.py:154-156`);
  the compiled graph serves G's adversarial pass and non-regularised D updates; regularised updates use `eager()`.

### 2.6 State-dict keys and loading
Plain module paths (`extraction.weight`, `stages.0.blocks.0.lin1.weight`, `.act1.bias`, `.lin2.weight`, `.lin3.weight`,
`stages.0.down.proj.weight`, `basis.weight`, `head.weight`), parameters only (the FIR kernel is a non-persistent buffer), so
`net_d_*.pth` exports load strictly and carry `network_d` (stage 3 A1). spandrel does not load discriminators; no legacy layout
exists to remap. The owner's traiNNer D weights would map key-by-key (`main_layers.i.layers.j.lin1.layer.weight` ->
`stages.i.blocks.j.lin1.weight`, `basis.linear.weight (1,C)` -> `head.weight (1,C,1,1)`) only with his group widths; not worth
code (D is retrained).

## 3. Loss spec (inside `gan_loss` / `gan_d_loss`)

### 3.1 kwargs (all default to today's behaviour)
| kwarg | default | meaning |
|---|---|---|
| `gan_type = "rpgan"` | `"bce"` | exists |
| `r1_gamma`, `r2_gamma` | 0 | exist; R3GAN uses one gamma for both (`ref/R3GAN/Trainer.py`) |
| `reg_every` | 16 | exists; R3GAN: 1 |
| `gamma_decay` | 1.0 | new: final/initial gamma ratio, in [0, 1] (R3GAN: 0.1 in every preset, `ref/train.py`) |
| `gamma_iters` | 0 | new: cosine length in updates counted from the first GAN update; 0 = constant |

### 3.2 gamma schedule: a pure function of `ctx.it`
`t = clamp(it - start_iter - 1, 0, N)`, `m(it) = d + (1 - d) * (1 + cos(pi * t / N)) / 2` (`N = gamma_iters`, `d =
gamma_decay`; `m = 1` when `N = 0`), applied to both gammas: the cosine shape of `ref/train.py` `cosine_decay_with_warmup`
without warm-up, counted from the GAN start (the reference counts from image 0 of a from-scratch run). Host float, no state,
no sync: resume-exact and rank-identical. Not adaptive (owner rule).

### 3.3 `gan_d_loss.evaluate` on a regularised update (diff against `gan_loss.py:133-158`)
```python
m = g.gamma_scale(it)                                    # host float
dev = ctx.sr.device.type
bf16 = torch.is_autocast_enabled(dev) and torch.get_autocast_dtype(dev) is torch.bfloat16
with (nullcontext() if bf16 else torch.autocast(dev, enabled=False)), sdpa_kernel(SDPBackend.MATH):
    real, fake = self.scores(eager(ctx.net_d), *xs)      # xs: fp32 leaves as today
    on = [...]                                           # as today: (out, x, gamma, "l_d_r1" | "l_d_r2")
    grads = torch.autograd.grad(sum(probe(out) for out, *_ in on), [x for _, x, *_ in on], create_graph=True)
terms = {key: 0.5 * g.reg_every * m * gm * gx.float().square().flatten(1).sum(1).mean()
         for (_, _, gm, key), gx in zip(on, grads, strict=True)}
# log each term, l_d_reg = sum (kept for older log readers); return adv_loss(real, fake) + l_d_reg

def probe(out):  # E_eps ||grad||^2 = mean over positions of ||grad D_p||^2; P = 1: today's score, no RNG draw
    o = out.float().reshape(len(out), -1)
    p = o.shape[1]
    return (o if p == 1 else o * torch.empty_like(o).bernoulli_(0.5).mul_(2).sub_(1) / p**0.5).sum()
```
Also replace `flatten(1)` by `reshape(len(out), -1)` (1.3 item 1). G side unchanged.

### 3.4 Precision
Penalty pass under the phase's bf16 autocast, fp32 accumulation of the squared norms; fp16 or no AMP: fp32 as today. Evidence:
bf16 vs fp32 per-sample `||grad D||^2` max relative error 2.3 % (patch) / 1.2 % (scalar), penalty parameter-gradient cosine
1.0000, norm ratio 0.990 / 0.999 [probe p5 bf16, CPU-emulated bf16 rounding]; input gradients stay fp32 (fp32 leaf, autocast
casts inside the convs). The reference trained D's high-resolution stages in bf16 with R1/R2 every step; the owner saw fp16 fail
and bf16 work (audit.md 2.5). StyleGAN2 (TF) used fp32 and StyleGAN2-ADA fp16 with clamping in its top layers [memory, not
re-checked]. fp16 stays fp32 because the inner `autograd.grad` is not loss-scaled (squared input gradients of 1e-4..1e-6 fall
below fp16's range). Behaviour change of an opt-in path: `tests/test_s2_losses.py:506` (pins fp32) becomes a tolerance test.

### 3.5 Validation (constructor and first call; `gan_opt` is opaque to `config.SCHEMA`, `neosr/config.py:180-186`)
- `gamma_decay` in [0, 1]; `gamma_iters >= 0`; `gamma_iters > 0` with both gammas 0 -> `ValueError` (a schedule without a
  penalty is a config mistake).
- First D call: `batch_coupled(D)` and a gamma > 0 -> `ConfigError` (1.3 item 7).
- No check for grid_sample Ds (dunet): the autograd error at the first regularised update is explicit; document it.
- Nothing for `validate_train`: no new cross-section rule is needed.

### 3.6 The rest of R3GAN's schedules, mapped
| reference (`ref/train.py`) | neosr | decision |
|---|---|---|
| lr cosine 2e-4 -> 5e-5 (G and D) | `[train.scheduler] type = "cosineannealing", T_max, eta_min = 5e-5, nets = ["g","d"]` | config only; constant LR inside the A/B horizon; decay after E2 |
| gamma cosine x0.1 | 3.2 | new kwargs |
| Adam beta2 0.9 -> 0.99 (kwargs `betas=[0,0]`, beta2 set by the scheduler) | none | not ported: fixed 0.99. The low early beta2 serves from-scratch generation over 2e7 images; SR starts from a pretrained G [reasoning]. Porting needs a beta scheduler class (the LR scheduler API cannot do it) |
| G EMA half-life ramp to 500k images | `ema`, `ema_power` | keep the template's 0.999 / 0.75 (measured in stage 2) |
| ADA augmentation 0 -> 0.3-0.55 | none | not ported (paired SR; a D-input augmentation would have to be identical for both members of a pair) |
| no gradient clipping | `grad_clip` global (`image.py:138-148`) | template value in round 1, ablated in round 2 (5.3) |

## 4. Efficiency (owner criterion: memory and speed)

### 4.1 Exact per-update cost in D passes
Unit U = one D forward over the B = 8 images of one side (no grad). Counts from `FlopCounterMode` on meta tensors with a
grouped-conv-correct backward formula [probe p2]; they are the same for every D (pure conv algebra), metagan +-3 %.

| path | G phase (gan_loss) | D phase (gan_d_loss) | total U | r3gan GT 256 GFLOP | metagan GT 256 GFLOP |
|---|---|---|---|---|---|
| bce, no penalty (template) | 1 fwd(B) + 1 input-only bwd = 2 | 1 fwd(2B) + 1 bwd (weights + activations) = 6 | 8 | 309 | 83 |
| rpgan, no penalty | + 1 no-grad fwd(B) on gt = 3 | 6 | 9 | 347 | 94 |
| rpgan + R1/R2 every update | 3 | 1 fwd(2B) + 1 input bwd with `create_graph` + 1 outer bwd through both graphs = 12 | **15** | **579** | 160 |
| rpgan + lazy k = 4 | 3 | 6 + 6/4 = 7.5 | 10.5 | 405 | 109 |
| rpgan + lazy k = 16 | 3 | 6 + 6/16 = 6.4 | 9.4 | 363 | 98 |

U = 38.6 GFLOP for `r3gan` at GT 256 (9.7 at GT 128), 10.4 for metagan. For scale: realplksr forward = 483 GFLOP per update
(B 8, LQ 64) [probe p6], so G fwd+bwd ~1.45 TFLOP and FDL(dinov2) ~1.3 TFLOP (41.3 GFLOP/img forward, `stability.md:253`):
the every-step R3GAN D adds ~20 % to the update's FLOPs, metagan+bce 3 %. The G-phase input-only backward really skips D's
weight gradients (`backward(inputs=G params)`: 3.0 U measured, not 4). CPU wall ratio of a penalty D update to a plain one
(fp32, GT 128, B 4) [probe p4]: r3gan 3.40x, owner-default D 3.08x, metagan 5.74x (FLOP ratio 2.0: the gradient graph adds
elementwise traffic). Eager launches (aten ops on meta, a proxy) [probe p6]:

| D (GT 256) | G phase | D plain | D penalty |
|---|---|---|---|
| r3gan (patch) | 145 | 107 | 865 |
| r3gan_scalar256 | 198 | 145 | 1800 |
| metagan | 1874 | 1525 | 5376 |
Stage 2 measured ~7.5 us of host time per launch (`gpu_profile.md:113`): the R3GAN D every step (~1010 launches) is cheaper
in host time than metagan with no penalty (~3400).

### 4.2 Sharing the G phase's real-side forward with the D phase
- Order per micro-batch (`image.py:326-349`): G forward -> G-phase backward (`D(gt)` under `no_grad`, `gan_loss.py:89-90`;
  `D(sr)` with graph, `:91`) -> D-phase backward (`D(cat[gt, sr.detach()])`, `:131` or `:143`). Optimizer steps only at the
  window boundary, G (`image.py:357`) then D (`:369`). So **both phases see the same D parameters**, also across the micro-batches
  of a window.
- Same function value? For `r3gan`: yes (no buffer a forward mutates; the FIR kernel is constant). For SN Ds (metagan, unet,
  patchgan): no, every train-mode forward advances the power iteration; metagan also draws new dropout/drop-path masks.
- Same numbers? No: G phase runs B images (compiled when `compile = true`), the D phase 2B images, under bf16 (plain) or fp32
  eager (penalty, today): equal only to rounding.
- Reusable? **Not from G to D:** G's logits have no graph, the D phase needs `d/dtheta` and, for R1, `d/dx`. **D to G**
  (`real.detach()` for the G term) needs the D phase's forward before G's backward: a trainer order change plus a loss writing
  to `Context` (which is read-only by contract, `neosr/losses/base.py:34`). It saves 1 U of 15 (6.7 %) every step, 1 of 9.4 at
  k = 16, ~35 launches; exact in exact arithmetic for `r3gan`, not bitwise, not even in value for SN/dropout Ds.
  **Recommendation: no.** Revisit only if `bench.py` shows D above ~20 % of the step.

### 4.3 The fp32 eager penalty pass: cost and the bf16 alternative
- FLOPs: identical in fp32 and bf16 (12 U per D update). Throughput: `fast_matmul` sets `cudnn.allow_tf32` (`runtime.py:106-110`),
  so "fp32" convs run TF32 on the GPU; on Ada (RTX 4090) dense TF32 tensor throughput is half of bf16-with-fp32-accumulate
  [Ada whitepaper, memory]. Memory: fp32 keeps 2x the bytes alive (4.4) and moves 2x the bytes.
- Eager vs compiled: the penalty pass cannot be compiled (aot_autograd, [probe]); compile only helps plain D updates and G's
  adversarial pass. For `r3gan` the eager launch count is small (107 plain), so losing compile on the D update costs little;
  for metagan (1525) it costs much more.
- Estimate for `r3gan` at GT 256, B 8, every step [reasoning from 4.1/4.4, not measured]: bf16 penalty ~15-25 ms of GPU time
  for all D work per update, fp32 penalty ~25-40 ms; metagan+bce today ~14 ms (31 ms with two D forwards minus the 16.6 ms
  dcat saving, `gpu_profile.md:170-173`). Template step ~95-100 ms eager. So every-step R3GAN costs about +0-10 ms (bf16) or
  +10-25 ms (fp32) per update. `bench.py` decides.
- Verdict: a bf16 penalty pass is acceptable (3.4); keep fp32 for fp16 runs.

### 4.4 Memory kept alive by `create_graph=True` (saved tensors, B = 8 per side, 2B = 16 images) [probe p4]
| D | GT | forward graph fp32 / bf16 MiB | forward + gradient graph fp32 / bf16 MiB |
|---|---|---|---|
| r3gan (patch) | 128 | 627 / 327 | 1010 / 520 |
| r3gan (patch) | 256 | 2508 / 1267 | 4040 / 2039 |
| r3gan, in-place leaky | 256 | - / 779 | - / 1551 |
| r3gan widths (16..256) | 256 | 1282 / 649 | 2068 / 1048 |
| r3gan_scalar256 | 256 | 2521 / 1284 | 4061 / 2060 |
| owner default widths (group 64-128) | 256 | 2508 / 1281 | 4041 / 2053 |
| metagan | 256 | 932 / 613 | 2831 / 1701 |
The gradient graph adds 0.6x the forward graph (the per-layer `grad_output`s and activation masks); everything is freed after
the outer backward. Transient gradient buffers come on top (not counted). The template's peak (3.2 GiB allocated,
`gpu_profile.md:75`) is set in the G phase (FDL 1.2 GB); the D phase runs after G's graph is freed, so a bf16 in-place `r3gan`
penalty graph (~1.55 GB) should stay near today's peak, while the fp32 path (4.0 GB) would raise it by ~1-2 GB
[reasoning]. G's adversarial pass adds `r3gan`'s forward graph for B images (~0.4-0.6 GB bf16) to the G-phase peak.

### 4.5 D sizing for SR patches [probe p1]
| D | params | GMAC/img GT 128 | GT 256 | output GT 256 |
|---|---|---|---|---|
| owner default (32..256, group widths 64-128, pooled scalar) | 1.73 M | 1.92 | 7.68 | (B,1,1,1) |
| paper FFHQ-256 D (96..768, gw 16, 2 blocks) | 23.0 M | 7.21 | 28.8 | (B,1,1,1) |
| **r3gan** (32..256, gw 16, patch) | 0.87 M | 0.60 | 2.42 | 16x16 |
| r3gan slim (16..256) | (half the activations) | ~0.3 | ~1.2 | 16x16 |
| r3gan gw 32 | 1.08 M | 0.89 | 3.57 | 16x16 |
| r3gan with a learned 4x4 stride-2 conv instead of FIR | 2.56 M | 1.04 | 4.18 | 16x16 (aliasing) |
| r3gan_scalar128 / r3gan_scalar256 | 1.21 / 1.55 M | 0.61 | 2.44 | (B,1,1,1) |
| metagan | 6.02 M | 0.16 | 0.63 | 4x4 |
| dunet | 3.23 M | 5.83 | 23.3 | 256x256 |
| unet (SN) | 4.38 M | 6.48 | 25.9 | 256x256 |
| patchgan | 2.76 M | 0.75 | 3.15 | 30x30 |
(metagan 0.634 GMAC = 1.27 GFLOP/img matches `stability.md:253`.) The FIR is ~0.2 % of the FLOPs (16 MACs per output channel
pixel); the strided-conv alternative costs +73 % FLOPs and 3x the parameters and is not anti-aliased. The cost sits in the
full-resolution stage: halving `widths[0]` halves memory.

### 4.6 Every step vs lazy: cost and what the A/B must measure
| setting | D-related U per update | GFLOP (r3gan GT 256) | peak memory | exactness |
|---|---|---|---|---|
| every step (R3GAN) | 15 | 579 | penalty graph every update | the reference objective |
| lazy k = 4 | 10.5 | 405 | same peak (the regularised update sets it) | not exact (1.3 item 4) |
| lazy k = 16 | 9.4 | 363 | same peak | not exact |
| no penalty | 9 | 347 | forward graph only | different objective (no convergence guarantee) |
Lazy saves average time, never peak VRAM. Each arm records with `scripts/bench.py` (3 trials x 100 updates, warm-up, one GPU job
at a time): median ms/update, peak allocated VRAM, and for lazy arms the ms of a regularised vs a plain update (run bench at
`reg_every = 1` and with gammas 0).

### 4.7 Which savings are exact
Exact: one forward for loss + penalties (today); the `cat` forward for per-sample Ds (today; changes SN power-iteration counts
and dropout draws for SN/dropout Ds); in-place leaky ReLU; FIR as a grouped conv instead of reshape (same math); compile of the
plain D updates. Exact in expectation only: the Rademacher per-position estimator (unbiased, adds variance). Not exact
(change the training dynamics): lazy regularisation with `xk` scaling (Adam, clip), bf16 penalty (1-2 % per-sample error),
reusing real logits across phases (not bitwise). Change the objective: penalty on random crops (traiNNer's `alt` loss),
relativistic-average instead of pairwise, mean-score vs per-position penalty on maps, scalar vs patch head, grad clipping.

## 5. Recipe and A/B plan on the owner's data

### 5.1 Template arm `options/train_realplksr_r3gan.toml` (opt-in; the metagan template stays the default)
`train_realplksr.toml` with:
```toml
[path]
pretrain_network_g = 'experiments\pretrain_g.pth'   # E5: a non-GAN run of this template; D and optimizers start fresh

[network_d]
type = "r3gan"          # patch head (GT multiple of 16); paper head: "r3gan_scalar256" (GT 256) / "r3gan_scalar128"

[train]
ema = 0.999
ema_power = 0.75
# wavelet_guided off: the R1/R2 penalty acts on D's RGB input

[train.optim_d]
type = "adam"
lr = 2e-4               # = G lr, as R3GAN
betas = [ 0.0, 0.99 ]
eps = 1e-8
weight_decay = 0

[train.gan_opt]
type = "gan_loss"
gan_type = "rpgan"
loss_weight = 0.3
r1_gamma = 10.0         # per position (patch head); scalar head: see the A/B grid
r2_gamma = 10.0
reg_every = 1
#gamma_decay = 0.1      # R3GAN: cosine to 0.1x over gamma_iters updates after the GAN start
#gamma_iters = 200000
```
Data: the owner's x4 pairs are 256 px GT tiles with 64 px LQ (10,607; `/home/korvin/apps/data/{GT,LQ}`), so LQ 64 = whole
tiles (GT 256) and LQ 32 = random GT 128 crops. The LQ files are an area-like downscale of GT (PSNR(LQ, area(GT)) 43.5-56.4 dB vs
31-34 dB for bicubic) [probe], so "real photographs" here means real photo content with a synthetic degradation.

### 5.2 Gamma anchors (toy probe p7: D-only, GT 128 crops of real tiles vs bicubic-upsampled LQ, Adam (0, 0.99) lr 2e-4, B 8, 120 updates)
| D, penalty | gamma | margin D(r)-D(f) | per-position `||grad D||^2` | G-side `|grad adv| / |grad L1|` per unit weight |
|---|---|---|---|---|
| patch, none | 0 | 127 and rising | 2e4 | 13,500 |
| patch, per-position | 1 | 0.5-1.0 | 0.05-0.2 | 18-27 |
| patch, per-position | 10 | 0.02-0.19 | 1e-3-4e-3 | 1.3-3.7 |
| patch, per-position | 100 | ~0 (frozen) | 1e-5 | 0.07-0.24 |
| patch, today's mean-score | 10 | 0.3-0.9 | 0.04-0.05 | 9.6-12.9 (~ per-position gamma 1) |
| scalar 128 | 1 | 1.0-1.7 | 0.08-0.29 | 54-87 |
| scalar 128 | 10 | 0.2-0.5 | 5e-3-1e-2 | 10-14 |
| scalar 128 | 100 | 0.01-0.07 | 1e-4 | 0.3-1.3 |
Bicubic fakes are far easier than SR outputs, so these are upper bounds on margin and G gradient. Paper anchors (`ref/train.py`):
CIFAR-10 0.05, ImageNet-32 0.5, ImageNet-64 1, FFHQ-64 2, FFHQ-256 150 (start values; x0.1 at the end), all for scalar Ds much
wider than these. Grid: patch head {1, 10} (+3 if a third arm fits), the same at GT 128 and 256; scalar head {10, 100} at GT 128,
{15, 150} at GT 256 (the paper's FFHQ-256 end/start values).

### 5.3 A/B matrix (one RTX 4090, one job at a time, capped as in stage 2)
Common: realplksr template objective (L1 1, mssim, consistency, ldl, FDL dinov2 0.02, EMA eval), x4, LQ 64 / GT 256, batch 8,
seed 7, 20k GAN updates after a shared 20k-update non-GAN pretrain (`pretrain_network_g`, fresh D/optimizers), template
`grad_clip` in round 1. Harness: the stage-2 runner (`scratchpad/ab/{run_ab,mkcfg,launch}.py` of session c1b40b02, per-run
memory cap) plus `scripts/bench.py` for speed/VRAM and `scripts/term_grads.py` for per-term G gradient shares at 0/10k/20k.

| round | arm | change | question |
|---|---|---|---|
| 1 | A0 | template GAN (metagan, bce 0.3, adan_sf D) | baseline; confirm whether D sits at chance |
| 1 | A1 | no D (continue the pretrain) | does any GAN help at all |
| 1 | A2 | metagan + rpgan, no penalty | the loss alone with the incumbent D |
| 1 | R1 | `r3gan` patch, rpgan, R1 = R2 = 10, every step, Adam (0, 0.99) 2e-4 | the R3GAN recipe |
| 1 | R2 | R1 with gamma 1 | gamma bracket |
| 1 | R3 | `r3gan_scalar256`, gamma 150 (then 15 if D frozen) | the paper head |
| 2 | best R + lazy 16, best R + lazy 4 | cost/quality of lazy (fresh gamma not tuned) |
| 2 | best R + gamma cosine 0.1 over the 20k | the R3GAN schedule |
| 2 | best R + `grad_clip = false` | the reference has no clip |
| 2 | best R + D lr 1e-4 / + `loss_weight = 0.1` | balance |
| 2 | best R + fp32 penalty (if 3.4 is implemented: a 10-batch parity check suffices instead of a run) | precision |
| 3 | top two arms + A0, second seed; best arm 80k+ | E0 power, long-run stability |
Per arm ~40-50 min (20k updates at ~100-130 ms + validation); round 1 ~5 h, round 2 ~4 h, round 3 ~6 h. GT 128 (LQ 32) only as a
cost check of the recommended GT-128 size, not as a quality arm.

### 5.4 Metrics and visual check
E0 protocol (`CONSENSUS.md` Phase 0): a fixed held-out set of a few hundred tiles (the 67-tile split if E0 is not built yet),
EMA weights, every 2k updates: PSNR/SSIM (fidelity sanity), DISTS, TOPIQ-FR with `dataset = "pipal"` (trained on SR/GAN outputs),
LPIPS alex (A3); none of them is a training loss in these arms (FDL is dinov2). Real photos: `val8` (8 photos, 512 px GT, LQ
128) with the same FR metrics, and 4 fixed 512 px crops of `val8/Anna Ralphs - SHOOTS - 0068.jpg` (3840x5758 camera JPEG, no
GT) run as real LQ for a no-reference visual check. Visual set saved at 5k/10k/20k per arm, side by side: val8 + 16 fixed
held-out tiles chosen for foliage, hair, fabric, text, faces, sky gradients. Logged per arm: `out_d_margin`, `l_d_r1`, `l_d_r2`,
`grad_norm_d`, `clipped_d`, skips, term gradient shares.

### 5.5 Acceptance (an R3GAN arm replaces the template GAN only if all hold)
1. Two seeds at 20k and one run past 80k agree.
2. TOPIQ-FR (pipal) and LPIPS better than A0 by more than A0's seed-to-seed spread; DISTS not worse beyond that spread.
3. Better than A1 (no GAN) on TOPIQ/LPIPS: the GAN has to do something.
4. PSNR on the held-out tiles at most 0.3 dB below A0.
5. Owner's visual sign-off on the visual set (no new checkerboards, ringing, colour shifts, false texture on faces/text).
6. Stability: no skipped updates; `out_d_margin` bounded (no monotone runaway) and not at chance (margin > ~0.01) in most
   windows; `l_d_r1`/`l_d_r2` finite.
7. Cost: median ms/update at most +30 % and peak VRAM at most +1.5 GB vs A0 under `bench.py`, or the owner accepts the cost.

### 5.6 Budget estimate per arm (RTX 4090, eager, GT 256, B 8) [reasoning from 4.1-4.4 and `gpu_profile.md`]
| arm | ms/update | peak VRAM |
|---|---|---|
| A0 template | 95-100 (measured, stage 2) | 3.2 GiB (measured) |
| A1 no D | ~85 | ~3.1 GiB |
| A2 metagan rpgan | ~100 | ~3.2 GiB |
| R every step, bf16 penalty | ~100-110 | ~3.5 GiB |
| R every step, fp32 penalty (today's code) | ~110-125 | ~4.5-5 GiB |
| R lazy 16 | ~95-105 | as every step |

## 6. Risks
1. Inert baseline (section 0 item 7): stage-2 Q8 at 8k updates logged `out_d_real` -0.009 / `out_d_fake` -0.013, `l_d_real`
   0.698 / `l_d_fake` 0.687, `l_g_gan` 0.210 = 0.3 ln 2 for most windows [log: `scratchpad/ab/runs/Q8_adamw_fdl002/train.jsonl`
   of session c1b40b02]; the shipped template went to chance by ~1.7k updates (`gpu_profile.md:204-213`).
2. Gamma is D-, head- and resolution-specific; a wrong gamma either freezes D (toy: 100 for the patch head) or lets it run away
   (0: margin 127). Grid in round 1, and per-penalty logs.
3. cuDNN performance of group-width-16 NHWC bf16 convs and their double backward on CUDA: unmeasured.
4. The per-position Rademacher penalty is a design choice not in the paper (the paper has scalar critics only).
5. Lazy regularisation and the global grad clip interact with Adam (1.3 item 4); kept out of the recipe.
6. Changing the meaning of `r1_gamma` for map Ds and the precision of the penalty pass are behaviour changes of an opt-in,
   never-A/B'd path: no template uses them; record in DECISIONS.
7. `wavelet_guided` arms crossing 80k would restart D on a new input; R3GAN arms keep it off.
8. Memory: fp32 penalty at GT 256 keeps 4 GB of D activations alive; implement 3.4 and the in-place activation before the GPU A/B.

## 7. Tests for the implementation (CPU unless marked)
1. Arch: golden state-dict key list and parameter counts per row (0.87 M / 1.21 M / 1.55 M); no persistent buffers; output shapes
   (patch `(B,1,H/16,W/16)`, scalar `(B,1,1,1)`); scalar head on a wrong size -> `ValueError` naming the right GT size.
2. Init: `lin3` weights zero; D linear at init (`D(a x + b y) = a D(x) + b D(y)` within fp32 tolerance).
3. FIR: grouped conv equals the reshape reference and the explicit `[1,3,3,1]x[1,3,3,1]/64`, stride 2, pad 1 formula; channels_last
   output equals contiguous.
4. `torch.autograd.gradgradcheck` of a tiny `r3gan` (float64, widths (4, 8), group width 2) incl. in-place leaky ReLU.
5. `torch.compile(backend="aot_eager")` forward equals eager; a regularised update with a compiled `ctx.net_d` uses `eager()`.
6. Pairing order: permuting the fake batch changes the rpgan D loss; swapping positions inside a map changes it (per-position).
7. Penalty on a known function: linear per-position critics with disjoint receptive fields give exactly `mean_p ||w_p||^2`
   for any eps draw; with overlapping fields the average over all 2^P sign patterns (P = 2, 3) equals the exact per-position
   penalty (deterministic, exhaustive).
8. Scalar head / `P = 1`: penalty bitwise equal to today's path and the RNG state unchanged.
9. No-op when both gammas are 0: D term bitwise the plain path, no `l_d_r*` logs, RNG state unchanged.
10. Lazy scaling: at `reg_every = k` a regularised update's penalty is exactly k x the `reg_every = 1` penalty for the same
    batch and `it`; other updates bitwise plain (extends `tests/test_s2_losses.py:476`).
11. Gamma schedule: `m = 1` at the first GAN update, `(1+d)/2` at `N/2`, `d` at and after `N`, constant at `N = 0`; independent
    of call order (pure function); constructor rejects `gamma_decay` outside [0, 1], negative `gamma_iters`, a schedule with
    zero gammas.
12. Precision: under bf16 autocast the penalty is within 3 % of fp32 and the D-parameter gradient cosine > 0.999; under fp16
    autocast the penalty pass runs fp32 (replaces `tests/test_s2_losses.py:506`).
13. Double backward under `sdpa_kernel(MATH)`: extend `tests/test_s2_losses.py:522` with `r3gan`, `r3gan_scalar128`; a dunet case
    asserting the grid_sample error (documents the refusal).
14. Batch-coupled D with a gamma -> `ConfigError` at the first D call.
15. Logs: `l_d_r1 + l_d_r2 == l_d_reg`; keys in `RESERVED`; present only on regularised updates.
16. A 1-D `(B,)` D output works (reshape fix).
17. Resume parity: a tiny CPU trainer with `r3gan`, rpgan, every-step R1/R2, a gamma schedule crossing the resume point and a map
    head (RNG draws): N updates straight vs N/2 + resume + N/2 -> bitwise equal weights, optimizer state and logs.
18. DDP 2-rank gloo regularised rpgan with the `r3gan` patch head (extends `tests/test_s2_losses.py:637`).
19. `adam` with `betas = [0.0, 0.99]` builds and resumes (Phase state identity); the new template validates (template sweep test).
20. [CUDA] bf16 channels_last double backward of `r3gan` finite; bf16 vs fp32 penalty parity on 10 real batches; `bench.py`
    numbers recorded for `r3gan` vs metagan.

## 8. Probe log (scripts in `/tmp/claude-1000/-home-korvin--claude-spares-spare-KgM9rZ/d7c995a9-c812-4ec5-ba57-eb99120e9524/scratchpad/r3/`)
- `r3d.py`: the measuring prototype (structure of 2.1; `zero=False` randomises `lin3` to mimic a trained D).
- `p1_size.py`: params and forward GMAC (FlopCounterMode, batch 1) -> 4.5.
- `p2_cost.py`: FLOPs per phase on meta tensors (B 8), units of U -> 4.1. torch 2.9.1's `conv_backward_flop`
  (`torch/utils/flop_counter.py:247-255`) counts a grouped conv's weight gradient as dense (`t(x_shape)`/`t(grad_out_shape)`
  ignore groups): it reported 13.5 U for a plain D update of the gw-16 D; the probe replaces it with `2*B*out_spatial*prod(w)` per
  requested gradient. `bench.py` does not use the FLOP counter; anyone who does should know.
- `p3_ops.py`: gradgradcheck table (2.4), `flatten(1)` IndexError on `(B,)`, aot_eager double-backward refusal, `eager()` of a
  compiled module returns the bare module.
- `p4_mem_time.py`: dunet R1 failure; saved-tensor accounting via `saved_tensors_hooks` (unique storages, parameters
  excluded, batch 1 per side scaled x8) -> 4.4; CPU timing (fp32 only: the 5950X has no bf16 units) -> 4.1; in-place leaky.
- `p5_map.py`: per-position penalty estimators (exact = one backward per position), receptive field, init statistics on 8 real
  tiles, bf16 vs fp32 penalty -> 1.3, 2.3, 3.4.
- `p6_ops.py`: aten op counts per phase (TorchDispatchMode on meta, views excluded), realplksr forward FLOPs -> 4.1.
- `p7_gamma.py`, `p7s_gamma.py`, logs `p7.log`, `p7s.log`: the gamma toy -> 5.2.
- Stage-2 D logs: `/tmp/claude-1000/-home-korvin-apps-neosr-dev/c1b40b02-f454-4498-9d89-be0afa4e2973/scratchpad/ab/runs/{Q8_adamw_fdl002,P0_shipped,Q5_sf0g_dB_ganstart}/train.jsonl`.

## 9. Not verified here / open questions
- Any GPU number: ms/update, VRAM, cuDNN grouped-conv speed, CUDA double backward of the D in bf16 (section 7 item 20).
- The 4090 TF32 : bf16 throughput ratio is from memory of NVIDIA's Ada whitepaper; StyleGAN2-ADA's R1 precision is from memory.
- The paper's App. E statement on lazy R1/R2 is cited through audit.md, not re-read.
- Gamma anchors come from a 120-update CPU toy with bicubic fakes, not from SR training.
- Owner decisions: beta2 schedule not ported (fixed 0.99); per-position penalty semantics for map Ds; bf16 penalty pass when
  the run is bf16; `wavelet_guided = false` in R3GAN arms; GAN start from a pretrained G (E5) for every GAN arm.
