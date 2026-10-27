# Stage 2 audit: losses lane

Scope: `neosr/losses/*.py`, `neosr/archs/vgg_arch.py`, `tests/test_losses*.py`, `tests/test_gan.py`, the 84 shipped
`options/train_*.toml` templates (all: `mssim 1.0 + consistency 1.0 + ldl 1.0 + fdl(dinov2) 0.75 + gan(bce) 0.3`,
metagan D, adan_sf, bf16 AMP, batch 8, gt 256 px in 58 templates and 128 px in 26; `train_catanet` uses fdl 0.5), and
the competitor `traiNNer-redux-dev/traiNNer/losses/*`.

Branch `refactor/stabilize` (= master), torch 2.9.1+cu128. **All probes ran on CPU** (`CUDA_VISIBLE_DEVICES=`; the GPU
is reserved for the profiling lane). The machine was shared: load average rose from ~8 to ~125 during the session,
so absolute CPU times are indicative only; ratios come from interleaved min-of-N runs. Inputs are real data: the owner's
paired 256/64 px 16-bit tiles (`/home/korvin/apps/data/{GT,LQ}`) and the 512 px validation GT. "Trained" networks are
the owner's old-format `experiments/train_esrgan/models/net_g_3236.pth` (esrgan, 3236 updates, 25.3 dB on the probe
batch) and `net_d_3236.pth` (metagan); both load through `read_weights` with all keys matched. Probe scripts and outputs:
`/tmp/claude-1000/-home-korvin-apps-neosr-dev/c1b40b02-f454-4498-9d89-be0afa4e2973/scratchpad/losses/p*.py|.out`.

Evidence labels: **measured** (probe on CPU), **reproduced** (a failure triggered), **source-verified**,
**hypothesis**. "Numerics" says whether a fix changes the training objective or numerics (then it needs an A/B
before it becomes a default).

---

## 1. Summary (most valuable first)

1. **The template objective is ~99 % FDL(dinov2) by generator gradient** (measured). At the template weights, the
   FDL term's parameter gradient is 61.7 of a total of 61.8 on the trained esrgan (projection share 0.99), and
   100 % on randomly initialised compact and span. MS-SSIM + consistency + LDL contribute 1-2 %. The FDL direction is
   nearly orthogonal to pixel fidelity (cos with the L1 gradient +0.03 on esrgan, **-0.16 on compact at init**, so
   the template pushes compact *away* from the target at the start). FDL does not fade near convergence: at 54 dB
   its gradient w.r.t. sr is about 3600 times L1's. Given the 1.0 global clip (pre-clip G norms 40-150 in
   acceptance.md), the reconstruction terms barely steer training. This is the biggest learning-efficiency lever in
   this lane. It is a static-weight A/B question, not an automatic balancer (section 5 has the protocol).
2. **consistency_loss runs its blur and its luma matmul in bf16** under AMP. It is the only template term without
   `fp32 = True`. Its gradient is off by 37-60 % (relative L2) against fp32 at 27-40 dB (measured). It is also the
   largest reconstruction term in parameter space. Fix: `fp32 = True`, one line, exact.
3. **FDL is also the most expensive term.** At the default fp32 backbone it costs an estimated 25-50 % of step GPU
   time (from perf.md numbers). Its saved activations are 717 MiB at batch 4 fp32 (measured, CPU). Per-block
   activation checkpointing of the sr branch is bitwise identical, keeps 195 MiB instead of 717 (-73 %) and costs
   about +20 % of FDL time (measured). The bf16 backbone (exists, opt-in) and compiling the backbone (deferred S.12,
   +8 %) are the speed knobs, and they belong in the same A/B as item 1.
4. **Three exact kernel rewrites** (same objective up to fp32 rounding):
   - consistency: separable 21-tap blur instead of torchvision's 441-tap 2D kernel, which is rebuilt on every call
     and also draws twice from the global RNG per call. Loss x1.55 faster on CPU.
   - LDL: local variance from box-filter moments instead of `unfold`. x3.7 faster on CPU, and it removes a 49x
     backward intermediate of about 0.2-0.4 GiB at batch 16.
   - MS-SSIM: separable 11-tap window. x2.2 faster on CPU.
5. **LDL is dead weight at template settings** (measured): 0.04 % of the gradient, 1e-7 of the loss near
   convergence. neosr dropped the EMA "refined artifact map" that makes LDL meaningful.
6. **The GAN path is correct** (non-saturating BCE, fresh D forwards, live adversarial term). The next stability step
   (deferred S.1: RpGAN plus R1/R2) is feasible, but only with the **math SDPA backend**: double backward through
   metagan's attention fails with the default kernels (reproduced).
7. **Codex items:** FF-loss syncs, KL semantics and "heavy perceptual objectives" are confirmed. The precision smoke
   test was too coarse: it found no non-finite values, but it missed the consistency precision loss in item 2.

---

## 2. Measurements

### 2.1 Cost per loss (CPU fp32, 4x3x256x256, 8 threads, load ~8; p1_cost)

sr = bicubic(gt) as a leaf; "saved" = unique storages saved for backward (a `saved_tensors_hooks` pack hook).

| loss | fwd (no_grad) ms | fwd+bwd ms | saved MiB | in templates |
|---|---:|---:|---:|---|
| L1Loss | 0.2 | 1.9 | 3.0 | - |
| chc_loss | 0.3 | 3.1 | 6.0 | - |
| ncc_loss | 0.4 | 0.8 | 6.0 | - |
| kl_loss | 1.1 | 1.2 | 9.0 | - |
| ff_loss | 3.0 | 5.8 | 6.2 | - |
| ldl_loss | 19.4 | 37.4 | 16.0 (+49x transient in backward, 2.4) | yes |
| mssim_loss | 33.9 | 50.0 | 48.0 | yes |
| consistency_loss | 46.0 | 61.4 | 51.2 | yes |
| gan_loss G term (metagan, train mode) | - | 126.8 | 192.8 | yes |
| msswd_loss | 169.5 | 276.6 | 127.3 | - |
| dists_loss | 562.7 | 978.9 | 729.3 | - |
| vgg_perceptual_loss | 701.6 | 1134.2 | 707.4 | - |
| fdl_loss (dinov2) | 690.3 | 1513.0 | 717.4 | yes |
| fdl_loss (vgg) | 1222.0 | 2365.5 | 1140.9 | - |

GPU context (perf.md, historical, measured by lane C6 on the owner's E workload = esrgan b16 and O = compact b8 OTF):
FDL forward was 61.5 ms of 226 ms kernel time (E) and 36.4 of 79.6 ms (O), with the fp32 backbone and 12 blocks. The
default is still the fp32 backbone (the bf16 backbone was made opt-in afterwards), with blocks 8-11 dropped. So the
current FDL forward is roughly 40-55 ms (E) and 25-34 ms (O), plus a backward of similar size through the sr branch.
That is an **estimated 25-40 % (E) and 35-50 % (O) of step GPU time**. MS-SSIM forward was 2.2 ms (E). consistency
and LDL were not broken out; at batch 16 the 21x21 depthwise blur is about 4.2 GMAC (fwd sr+gt, bwd sr), against
0.4 GMAC separable.

### 2.2 Host syncs per loss (fwd+bwd, CPU profiler, p6_syncs)

Counted `aten::_local_scalar_dense` / `nonzero` / `masked_select` / `unique`.

| loss | count | real device syncs on CUDA |
|---|---|---|
| ff_loss | 3 | **2**: the two `assert weight_matrix.min()/max().item()` at ff_loss.py:100,103. The third is the CPU scalar value in `x[mask] = 0.0` (index_put on a scalar becomes masked_fill, no nonzero, no device sync). |
| consistency_loss | 2 | 0: torchvision `GaussianBlur.get_params` calls `torch.empty(1).uniform_().item()` on a **CPU** tensor. No device sync, but it draws from the global torch CPU RNG twice per call. |
| all other losses, gan_loss G+D | 0 | 0 (the MS-SSIM `prod` sync was already removed in C6) |

### 2.3 Generator-gradient composition at template weights (p3_paramgrad, p3b_reweight)

Batch 4 at 64 -> 256 px, fp32. "share" = projection of each term's parameter gradient on the total
(`<g_total, g_k> / |g_total|^2`). D = the owner's trained metagan, which comes from the old code (its logits were
saturated in that run).

| G (state) | PSNR | mssim | consistency | ldl | fdl 0.75 | gan 0.3 | total | share fdl | cos(total, g_L1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| esrgan @3236 (trained) | 25.3 | 1.33 | 7.48 | 0.022 | **61.7** | 3.46 | 61.8 | **0.99** | +0.12 |
| compact (random init) | 25.2 | 0.33 | 0.18 | 0.0003 | **36.6** | 0.60 | 36.5 | **1.00** | **-0.15** |
| span (random init) | 8.4 | 0.70 | 0.36 | 0.036 | **111** | 1.29 | 111 | **1.00** | +0.12 |

The amplitude part of FDL carries almost all of it (amplitude-only `|dθ|` 61.0 vs 61.7). FDL(vgg), whose score is
multiplied by 0.01 in the kernel (fdl_loss.py:361), is 11x smaller than FDL(dinov2), which has no such factor.

Gradient composition vs a static FDL weight (esrgan @3236; "+L1" adds `L1Loss` at 1.0):

| fdl weight | recon share | fdl share | cos(total, g_L1) | with +L1: cos |
|---|---:|---:|---:|---:|
| 0.75 (template) | 0.02 | 0.99 | +0.12 | +0.18 |
| 0.25 | 0.13 | 0.88 | +0.30 | +0.44 |
| 0.1 | 0.50 | 0.51 | +0.55 | +0.73 |
| 0.05 | 0.80 | 0.20 | +0.69 | +0.84 |
| 0.02 | 0.95 | 0.03 | +0.74 | +0.87 |

For compact at init FDL still holds 96 % at 0.05 and 70 % at 0.02. The balance depends on the architecture: compact
starts as bicubic plus a near-zero residual, so its reconstruction gradients are tiny.

### 2.4 sr-space gradients per term (p2_srgrad)

Norm of d(wL)/d sr; cos with (sr - gt): > 0 means descent moves sr toward gt.

| sr candidate | PSNR | L1 (w=1) | mssim | consistency | ldl | fdl 0.75 (cos) | gan, trained D (cos) |
|---|---:|---:|---:|---:|---:|---|---|
| random | 8.8 | 1.1e-3 | 5.6e-3 | 1.5e-3 | 6.0e-4 | 1.57 (+0.004) | 0.030 (-0.005) |
| bicubic | 26.9 | 1.1e-3 | 6.6e-3 | 7.9e-4 | 2.7e-5 | 10.4 (+0.030) | 0.049 (-0.002) |
| esrgan @3236 | 25.3 | 1.1e-3 | 5.7e-3 | 1.5e-3 | 3.2e-5 | 4.54 (+0.000) | 0.049 (-0.006) |
| gt + N(0, 0.01) | 40.0 | 1.1e-3 | 1.6e-3 | 1.1e-3 | 1.1e-7 | 3.89 (+0.065) | 0.044 (+0.002) |
| gt + N(0, 0.002) | 54.0 | 1.1e-3 | 3.4e-4 | 1.1e-3 | 2.4e-9 | 4.08 (+0.087) | 0.044 (+0.002) |

FDL value at 54 dB is still 1.45 (weighted). It has an L1-like V-shaped minimum in feature space, so its gradient
stays around 4 while the residual goes to zero. On flat or smooth content it is 3-5x larger than on natural content
(25.9 vs 4.7 at noise 0.01), and the phase part reaches 34-43 % of its gradient energy at noise 0.001 (p12). It
stays finite in fp32 (max 1.6 per pixel).

### 2.5 Precision under bf16 autocast for terms without `fp32 = True` (p4, p4b)

`Loss.compute` through a Context, fp32 sr (isolates loss-internal precision), vs the same in fp32:

| loss | point | loss rel. err | grad rel. L2 | `fp32 = True` |
|---|---|---:|---:|---|
| **consistency_loss** | bicubic (~27 dB) | 3.6e-3 | **0.42** (cos 0.91) | exact (0 / 0) |
| **consistency_loss** | gt + N(0.03) | 7.8e-3 | **0.37** | exact |
| **consistency_loss** | gt + N(0.01) | 2.8e-2 | **0.60** (cos 0.81) | exact |
| ldl, L1, chc, ncc, kl, mssim | 40 / 50 dB | 0 | 0 | (promotion to fp32 via gt, or already fp32) |

The mechanism is `self.blur` (conv2d, 21x21) and `img @ self.luma` (matmul), consistency_loss.py:135,165-166. Both
are in the lower-precision autocast list on CUDA as well as on CPU. The blurred sr is rounded to bf16, a step of 2^-8
(1/256) in [0.5, 1), which is about the residual of a converging model.

---

## 3. Findings

### L-01 (P1, learning) FDL(dinov2) dominates the generator gradient at template weights
- **Where:** all 84 `options/train_*.toml` (`[train.fdl_opt] loss_weight = 0.75`); fdl_loss.py:361 (dinov2 score is
  not scaled; the CNN backbones get `* 0.01`); fdl_loss.py:345 (unnormalised `fftn`).
- **Evidence:** measured, see 2.3 and 2.4. Projection share 0.99 to 1.00 across one trained and two random-init
  generators. cos(total, g_L1) is +0.12 / -0.15 / +0.12. Pre-clip G norms of 40-150 in acceptance.md match the FDL
  norm (61.8 here at batch 4). FDL's relative weight also grows with the patch side (L-12).
- **Impact:** reconstruction terms are effectively switched off, and fidelity is learned only through FDL's
  feature-distribution match. Near convergence FDL keeps injecting a gradient of norm about 4 whose cosine with the
  residual is only 0.07-0.09, so it is almost entirely orthogonal to it. With the 1.0 global clip and Adan's per-parameter normalisation, the update
  direction is FDL's. That plausibly explains slow PSNR progress and texture or noise drift, and it is the opposite
  failure mode from traiNNer's damping. The fix is not normalisation but a sane static weight.
- **Fix (objective change, A/B):** keep the kernel unchanged (checkpoint and config compatible). Compare template
  arms with `fdl loss_weight` 0.75 / 0.1 / 0.05, optionally with an `L1Loss` (or `chc_loss`) pixel term at 1.0, on
  identical data, seeds and update budgets; protocol in section 5. If an arm wins, change the 84 templates' weight
  and document "FDL(dinov2) at 0.75 is ~99 % of the gradient". Do not add automatic loss balancing (owner rule). An
  optional one-shot diagnostic (per-term `|dθ|` of the first update, printed once) would let users check other
  architectures. It is cheap: one extra backward per term, once.
- **Test:** an A/B run (section 5). A CPU unit test that the reported per-term shares sum to 1 if the diagnostic is
  added.

### L-02 (P1, correctness/stability) consistency_loss is computed in bf16 under AMP
- **Where:** consistency_loss.py:14 (no `fp32 = True`); :49/:165-166 (torchvision blur conv); :135 (matmul).
- **Evidence:** measured (2.5). Gradient relative L2 error is 0.37-0.60 at realistic operating points; with
  `fp32 = True` it is exactly 0. It is the largest reconstruction term in parameter space (7.5 vs mssim 1.3 on
  esrgan), so after L-01 it becomes the main fidelity driver, and its precision matters.
- **Fix:** add the class attribute `fp32 = True` (as mssim, ff, dists and fdl have). The cost is negligible, about
  10 small fp32 elementwise kernels plus the blur (cheaper still with L-04).
- **Numerics:** changes AMP runs only, restoring the fp32 objective (fp32 runs are bitwise unchanged). This is a
  precision fix, so it does not need a quality A/B to become the default. Run the standard bf16 and fp16 acceptance
  smoke.
- **Test:** `test_fp32_disables_autocast`-style: `compute()` under CUDA autocast bf16 equals fp32 within 1e-6
  (values and sr.grad). Add consistency to the parametrisation of `tests/test_losses_pipeline.py::test_fp32_disables_autocast`.

### L-03 (P1, speed) FDL is the dominant loss cost; tie the precision/compile choice to the L-01 A/B
- **Where:** fdl_loss.py:262-287 (`fp32_backbone=True` default), :330-339; perf.md §2/§3.1/§5.2; PLAN S.12.
- **Evidence:** source-verified, plus historical GPU measurements (2.1). The bf16 backbone gave E +23 % it/s (FDL
  forward 61.5 -> 30.2 ms), but was rejected as the default because the pixel gradient differs by up to 60 %. That
  concern scales with FDL's share of the total gradient. Today the share is 99 %, so a 60 % FDL error is a ~60 %
  error of the whole update. At fdl 0.05-0.1 it would be a 10-30 % error of a term that is 20-50 % of the update.
- **Fix:** in the L-01 A/B, add the arm "winning weight + `fp32_backbone = false`" and keep the faster one if quality
  ties. Then wire S.12 (compile the DINOv2 backbone under `compile = true`, +8 % measured in perf.md §5.2). Expected
  value: +20-30 % it/s on GAN templates with no loss of quality.
- **Numerics:** the bf16 backbone changes numerics (it is opt-in today). Compile changes fp32 rounding only.

### L-04 (P2, speed) consistency blur: torchvision 2D Gaussian -> fixed separable kernel
- **Where:** consistency_loss.py:5, 48-49, 164-166. torchvision `_functional_tensor.gaussian_blur` builds the
  21x21 kernel on the device in the input dtype on **every call**. `GaussianBlur.forward` draws a sigma from the
  global CPU RNG (`torch.empty(1).uniform_(3, 3).item()`) on every call, twice per micro-batch.
- **Evidence:** measured (p5, p5b). Separable reflect-padded 1D kernels give a max blur difference of 1.4e-6, loss
  relative difference 1e-7 and grad relative L2 4.6e-4 (fp32 accumulation order). The whole loss is x1.55 faster on
  CPU, and the blur uses 42 instead of 441 MACs per pixel per channel.
- **Fix:** register a non-persistent buffer `k = gauss1d(21, 3)` and use
  `blur(x) = conv2d(conv2d(pad(x, 10, 'reflect'), k_row, groups=3), k_col, groups=3)`. Drop the torchvision import.
- **Numerics:** fp32 rounding only. It removes 2 global-RNG draws per micro-batch, so for the same seed the global
  torch CPU stream stays where it was before this loss ran. Resume exactness is unaffected.
- **Test:** in-test reference (torchvision `gaussian_blur`) vs new: values and grads within 1e-5 relative; a
  no-RNG-consumption check (`torch.get_rng_state()` unchanged across `forward`).

### L-05 (P2, memory/speed) LDL local variance: `unfold` -> box-filter moments
- **Where:** ldl_loss.py:44-68 (`unfold(2,7,1).unfold(3,7,1)` + `torch.var`).
- **Evidence:** measured. Box filter
  `m1 = avg_pool2d(pad(r), 7, 1); m2 = avg_pool2d(pad(r)^2, 7, 1); var = (m2 - m1^2) * 49/48` (clamped >= 0) gives a
  loss relative difference of 0 and grad relative L2 of 4e-7. The loss is x3.7 faster on CPU (the variance op alone
  x2.3 at b16). `VarBackward0` allocates a contiguous 7x7-unfolded gradient (200 MiB at 16x1x256x256) plus
  same-size `sub`/`mul` temporaries. That is about **0.2-0.4 GiB of transient memory** during the loss backward,
  while G's activations are still alive.
- **Fix:** replace `get_local_weights` with the box-moment version. Rounding (m2 - m1^2) is safe here: the residual
  is fp32 in [0, 3], and the clamp handles the zero-variance case (the existing "constant/offset" test cases).
- **Numerics:** fp32 rounding only. **Test:** in-test unfold reference vs new on the `CASES` of test_losses.py
  (random, identical, constant, offset, anticorrelated, channels_last) at rtol 1e-5, values and grads.

### L-06 (P2, speed) MS-SSIM: separable Gaussian window
- **Where:** ssim_loss.py:9-62 (`GaussianFilter2D`, 11x11 2D depthwise conv, 5 calls x 5 scales), :123-128.
- **Evidence:** measured. Separable `(1,11)` then `(11,1)` zero-padded convs give a loss relative difference of
  1.2e-5 (absolute 3e-7) and grad relative L2 of 6e-5. x2.2 faster on CPU, 22 instead of 121 MACs per pixel. Do not
  fuse all five moments with one `cat`: that pulls the gt-only moments into the graph and doubled saved memory
  (48 -> 96 MiB). Keep the per-moment calls, or fuse only `[x, x*x, x*y]`.
- **Numerics:** fp32 rounding. `test_b1_w1_matches_pre_b1[mssim_loss]` pins MS-SSIM to the pre-B1 kernel at
  rtol 1e-6. That test guards B1 mechanics, not this rewrite. Give the rewrite its own reference test at 1e-4, and
  keep the B1 test against an in-test copy of the 2D kernel. Expected GPU gain is small (mssim forward 2.2 ms at E).

### L-07 (P2, learning/speed) LDL contributes nothing at template settings
- **Where:** ldl_loss.py:70-96; templates `[train.ldl_opt] loss_weight = 1.0`.
- **Evidence:** measured. Weighted value 3e-4, `|dθ|` 0.022 of 61.8 (0.04 %) on esrgan; 0.0003 on compact; value
  1.4e-7 at 40 dB. The artifact map is `var(residual)^(1/5) x local_var(residual)`, which scales as
  residual^2.4. neosr dropped the original refinement (`weight[residual_sr < residual_ema] = 0`, BasicSR and the
  competitor's `loss_util.get_refined_artifact_map`). That refinement needs an EMA forward that the Context does not
  have.
- **Fix (objective change, tiny):** drop `ldl_opt` from the templates in the L-01 A/B. Expect no quality change and
  save its cost (L-05 makes it cheap anyway). Restoring the paper's formulation would need `net_g_ema(lq)` per step
  (+1 G forward), which is not justified by these numbers.

### L-08 (P2, stability, next phase) RpGAN + R1/R2 for the D: feasibility and spec (PLAN S.1, deferred)
- **Where:** gan_loss.py:57-73 (`gan_d_loss`); metagan_arch.py:54 (SDPA attention).
- **Evidence:** reproduced (p10). The R1 double backward through train-mode metagan fails on the default SDPA
  kernel: `derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward is not implemented`. The CUDA
  flash and mem-efficient kernels have no double backward either. Under `sdpa_kernel(SDPBackend.MATH)` it works
  (R1 = 24 at init).
- **Spec (when S.1 is scheduled):** `gan_type = "rpgan"`: `L_D = softplus(D(f) - D(r))`, `L_G = softplus(D(r) - D(f))`
  with D(r) under no_grad in the G pass, read from ctx. Add `r1_gamma` / `r2_gamma` and `reg_every = k` inside
  `gan_d_loss.evaluate`, using `ctx.it % k` (Python int, no sync), `gamma * k / 2 * |∇_x D(x)|^2` on the unaugmented
  `adv(gt)` / `adv(sr.detach())`, fp32, under the math SDPA backend. This is pure math, no loss scheduling or
  clipping, and fits the Context contract (no trainer if-chains). Do **not** port the competitor's
  `SafeGradientPenalty`: on a missing double-backward it falls back to `create_graph=False`, which silently disables
  the penalty. Its R3GAN also calls `.item()` twice per call (crop offsets, `gp_step`) and is a relativistic-average
  hinge, not R3GAN's RpGAN.
- **Numerics:** a new opt-in objective. Needs the >= 20k-update acceptance run that PLAN S.1 already demands.

### L-09 (P3, speed) One batched D forward for the D term
- **Where:** gan_loss.py:62-64 (two `ctx.net_d(...)` calls).
- **Evidence:** hypothesis. perf.md §6 says workload O is launch-bound, with "three D forwards with spectral-norm
  hooks". `cat([adv(gt), adv(sr.detach())])` gives one forward, then split. GroupNorm is per sample, so outputs match
  except dropout masks and **one SN power iteration per D step instead of two** (u is updated once). It is also the
  canonical one-forward-one-backward DDP pattern.
- **Numerics:** small (SN u schedule). Run the GAN acceptance check, and let the profiling lane measure ms/update on O.

### L-10 (P3, sync) ff_loss: two tautological `.item()` asserts
- **Where:** ff_loss.py:100-105 (also :28 `ave_spectrum=True`, :67 `fft2`).
- **Evidence:** measured (2.2), confirms Codex. Two device syncs per micro-batch when ff is enabled. The matrix is
  clamped to [0, 1] two lines above, and `evaluate` never passes a predefined matrix.
- **Fix:** delete both asserts (or keep a check only on the user-matrix path). Optional: `rfft2` with doubled weights
  for the non-self-conjugate columns is exact and halves the FFT. Note for docs: `ave_spectrum=True` (the upstream
  FFL default is False) compares batch-mean spectra, so per-sample errors cancel (value 5x smaller: 0.0038 vs 0.0186,
  p8). ff is not in any template.
- **Numerics:** none for the asserts. **Test:** CUDA `set_sync_debug_mode("error")` over fwd+bwd, like
  `test_mssim_backward_does_not_sync`.

### L-11 (P3, correctness/docs) FDL magnitude depends on patch size
- **Evidence:** measured (p7). Same validation content: FDL is 16.3 / 19.5 / 29.2 at 128 / 256 / 512 px. The
  per-pixel gradient relative to L1 is 6.4e3 / 1.2e4 / 2.0e4x, i.e. x1.7-1.9 per doubling of the patch side. The
  unnormalised FFT DC bin grows with the token count.
- **Fix:** document that FDL weights do not transfer across `patch_size` (26 templates train at 128 px, 58 at
  256 px). `norm="ortho"` would over-correct (0.79 / 1.02 / 1.63 after rescaling) and is not proposed.

### L-12 (P3, semantics) KL loss: shift invariance and H*W scaling (confirms Codex)
- **Where:** kl_loss.py:9-20.
- **Evidence:** measured. `KL(gt + per-pixel offset, gt) = 6e-6` for offsets up to 0.3, so it does not constrain
  brightness. Value 0.0445 at 256 px vs 0.0116 at 128 px (x3.8): `batchmean` sums over C, H, W. On a 1-channel input
  it is identically 0.
- **Fix:** docs. Optionally `F.kl_div(F.log_softmax(x,1), F.log_softmax(gt,1), log_target=True)`, which removes the
  `log(p + 1e-8)` bias (numerics: negligible). A per-pixel mean reduction would be an objective change behind an
  option. Not in templates.

### L-13 (P3, robustness) Image-domain assumptions in losses (F13/F14 from the loss side)
- **Evidence:** reproduced (p13). On 1-channel (`color = 'y'`) inputs, `consistency_loss` raises (`size mismatch ...
  vec (3)`), `fdl_loss` dinov2 raises (patch-embed expects 3 channels) and `kl_loss` returns 0. `mssim_loss`
  silently computes 3 identical channels, and FDL(vgg) and msswd broadcast gray to RGB. consistency, msswd, FDL, VGG
  and DISTS assume sRGB in [0, 1] (clamps at consistency_loss.py:160-161, ImageNet normalisation), so the dataset
  mean/std normalisation of F13 breaks them.
- **Fix:** let `validate_train` reject `color='y'` with consistency, fdl(dinov2) or kl, and reject non-identity
  dataset mean/std when any RGB-domain loss is configured. This fits the data lane's F13/F14 repair.

### L-14 (P3, simplification) consistency cosine gate is always on and adds ~1e-5
- **Where:** consistency_loss.py:186-192.
- **Evidence:** measured (p8). The cosine term is 4e-6 at bicubic quality and 1.4e-5 at 40 dB, so the
  `torch.where(cosim < 1e-3, ...)` gate is ON for any non-random output. The comment "early iterations are
  unaffected" is wrong in practice, and the term is negligible next to the 4e-3 main term. The luma cosine is taken
  along H (`(B,H,W)`, dim=1: column profiles), and the chroma cosine is near 1 because of the +0.5 offset.
- **Fix:** none needed for training. If touched, document it or make `cosim = False` the default after the A/B
  (objective change ~1e-5). Do not port the competitor's version: its luma cosine over a size-1 channel is always 0.

### L-15 (P3, hypothesis) FDL gradient carries the fixed 14-px ViT grid
- **Evidence:** measured (p11). Mean |d FDL/d sr| varies 1.41x with the position inside the 14-px patch grid (the
  reflect pad of 5 px is constant), against 1.09x for an MS-SSIM control. It is a weak spectral peak, and a possible
  source of grid artifacts.
- **Fix (objective change, experiment):** pick a random split of the 10-px pad per call (or a random offset in
  0..13 px with extra reflect pad) to decorrelate the grid. Only after L-01, and only if artifacts are seen.

### L-16 (P3, cross-lane: trainer/arch) bf16-quantised generator output
- **Evidence:** measured. With a bf16 sr (archs whose last op runs under autocast), L1's sr-gradient relative L2 vs
  fp32 sr is 0.31 at 40 dB and 0.55 at 50 dB, and mssim's is 0.08 / 0.26. Mostly these are sign flips on
  sub-quantum residuals. The losses cannot recover the precision. Running the generator's last layer in fp32
  (autocast off for the tail) is the fix and belongs to the trainer/arch lane. Low impact at 25-35 dB.

### L-17 (P3, memory) FDL sr-branch activation checkpointing (opt-in)
- **Where:** fdl_loss.py:168-252 (`dinov2`).
- **Evidence:** measured (p9). Wrapping each DINOv2 block in `torch.utils.checkpoint(..., use_reentrant=False)`
  when grad is on gives bitwise-equal loss and gradient (max |diff| 0), saved 717 -> 195 MiB (b4 fp32, CPU math
  SDPA), and +20 % FDL fwd+bwd time. On GPU (mem-efficient SDPA) the estimate is ~1.0-1.3 GiB saved at batch 16 fp32
  (about 10-13 % of E's 10.1 GiB peak), for one extra sr-branch backbone forward.
- **Fix:** `fdl_opt.checkpoint = false` by default; true wraps the blocks (only when `torch.is_grad_enabled()`; the
  gt branch is under no_grad). Exact, so no A/B. Ranked below speed by the owner's priorities, hence opt-in.
- **Test:** checkpoint on/off give bitwise-equal values and grads on CPU (VGG-free, a small dinov2 stub is fine).

### L-18 (P3, tests) Gaps
- No test checks consistency precision under autocast (L-02). Add consistency to `test_fp32_disables_autocast`.
- No sync test for ff (L-10).
- Exact-rewrite parity tests for L-04, L-05 and L-06 against in-test references.
- An R1 math-SDPA feasibility test when S.1 lands (L-08).
- The existing B1 tests compare kernels to a pre-B1 git revision via `git show`. Keep them, but give the L-04 to L-06
  rewrites their own in-file reference implementations, so the tests do not depend on git history.

### Verified OK (no change)
- All loss modules are frozen and in eval mode (`build_losses`: `nn.ModuleList(terms).to(device).requires_grad_(False).eval()`).
  FDL gt features are under `no_grad`. VGG and DISTS gt branches build no graph (frozen params, detached gt). Frozen
  `nn.Linear` saves no input; frozen convs do, but their inputs are shared with ReLU outputs.
- The guards are real fixes, not NaN hiding: MS-SSIM `clamp_min(1e-6)` before the fractional pow, LDL
  `clamp_min(1e-12) ** 0.2`, consistency `clamp_min(knee)` in the unused `where` branch, chc fp32 promotion, ncc
  denominator clamp, ff NaN -> 0 for identical patches (masked_fill, no sync), DISTS `sqrt(out + 1e-12)` (zero
  gradient at zero input). `torch.angle` returns a zero gradient at exactly 0.
- GAN: non-saturating BCE (BCEWithLogits is autocast-fp32), D term on fresh forwards of `adv(gt)` /
  `adv(sr.detach())`, the G term through train-mode D with `backward(inputs=G params)`. `AdvView` is fp32 and
  identical for real and fake. The ~5.5 MB/sample cost of not freezing D in the G pass is documented in
  `test_metagan_inputs_equals_freeze_toggle`.
- FDL(VGG) and DISTS stage boundaries do not corrupt saved features through in-place ReLU (the next stage starts
  with maxpool or L2pooling). `VGGFeatureExtractor` sets `inplace=False`.

---

## 4. Codex items (analyze.md) for this lane

| item | verdict | note |
|---|---|---|
| FF-loss synchronisation (`.item()` twice) | confirmed | 2 device syncs from the asserts (ff_loss.py:100,103); the NaN mask assignment is masked_fill with a CPU scalar, no sync (p6) |
| Heavy perceptual objectives | confirmed | quantified: FDL is ~99-100 % of the G gradient (L-01) and an estimated 25-50 % of step GPU time (L-03) |
| FDL DINO precision (fp32 default; BF16 opt-in) | partial | default fp32 confirmed (fdl_loss.py:283-287); under `fast_matmul` those GEMMs are TF32 (perf.md). The precision concern scales with FDL's gradient share, so decide it in the same A/B as the weight (L-03) |
| KL objective semantics | confirmed | offset invariance (6e-6) and x3.8 H*W scaling reproduced; 0 on 1-channel input (L-12) |
| 33/33 finite precision probes, "no generic finite failure" | partial | finite, but consistency loses 37-60 % gradient accuracy under bf16 autocast; the smoke test did not compare gradients (L-02) |
| Retain LDL/MS-SSIM guards, detached targets, frozen weights | confirmed | every guard checked is a real fix (section 3, "Verified OK") |
| Recommendation 3: measure loss influence by gradients | confirmed | done (2.3); the result drives L-01 |
| F13 normalisation vs domains (loss side) | confirmed | consistency, msswd, FDL, VGG and DISTS assume sRGB [0,1] (L-13) |
| F14 grayscale OTF (loss side) | confirmed | consistency and fdl(dinov2) also crash on 1-channel; kl returns 0 (L-13) |
| DINOv2 hub code not pinned (fdl_loss.py:180) | confirmed (source) | supply-chain item for the tooling lane; `facebookresearch/dinov2:<sha>` pins it |
| DISTS downloads beside the source (dists_loss.py:107-115) | confirmed (source) | tooling lane |

---

## 5. A/B protocol for the objective changes (L-01, L-03, L-07)

Fixed: one architecture from each family (esrgan as the heavy CNN, compact or span as the light one), `train_*_otf`
and paired data, the same seeds (>= 2), batch 8, gt 256, adan_sf, EMA 0.999/0.75, bf16, `wavelet_init` unchanged,
20k-50k updates, validation every 2k on the fixed 7-image val set: PSNR, SSIM, DISTS (lower is better) and TOPIQ,
plus it/s and peak memory. Also log per-term `|dθ|` once at u1, u5k and u20k.

| arm | fdl weight | fdl backbone | ldl | pixel term |
|---|---|---|---|---|
| A (template) | 0.75 | fp32 | 1.0 | - |
| B | 0.1 | fp32 | 1.0 | - |
| C | 0.05 | fp32 | - | - |
| D | best of B/C | bf16 (`fp32_backbone = false`) | as winner | - |
| E | best of B/C | as winner | as winner | `L1Loss` 1.0 |

All arms include L-02 (consistency fp32). It is a precision fix and lands before the A/B, because otherwise the arms
measure bf16 noise. Accept an arm if DISTS/TOPIQ are no worse than A and PSNR/SSIM are better at equal updates. Accept
the bf16 backbone (D) if it ties its fp32 twin. This uses only static weights: no scheduling, no normalisation.

---

## 6. Competitor (traiNNer-redux) comparison

| competitor | verdict |
|---|---|
| `r3gan_loss.R3GANLoss` (R1/R2 + relativistic) | take the **idea** (L-08), reimplemented. Its "relativistic average hinge" is not R3GAN's RpGAN. `SafeGradientPenalty` silently disables the penalty on missing double-backward. `.item()` x2 per call (crop offsets, gp interval). Hard-coded `autocast(device_type="cuda")` |
| `ms_ssim_l1_loss.MSSSIML1Loss` (0.1 MS-SSIM + 0.9 Gaussian-weighted L1) | take the **principle**: a pixel-anchored reconstruction term (arm E). Do not port the code: 15-channel 33x33 2D kernels, `cs.prod(dim=1)` backward sync, device picked at construction |
| `ldl_loss` (keeps the EMA-refined artifact map, fp32 cast) | the original LDL semantics. neosr cannot use it without an EMA forward per step; drop LDL instead (L-07) |
| `consistency_loss` | not better: the luma cosine over a size-1 channel is always 0 (dead term), and `brightness = 0.9` darkens the target by default. Its fp32 is via custom_fwd; neosr gets the same with L-02 |
| `mssim_loss` (Y channel only, no luminance term) | a different objective; nothing to fix in neosr |
| `dists_loss`, `perceptual_loss` | equivalent to neosr (fp32 cast, gt under no_grad) |
| `dynamic_loss_scheduling`, adaptive gradient clipping, training automations | **do not take** (owner rule); the static re-weighting in L-01 covers the legitimate part |

---

## 7. Cross-lane notes
- Trainer: `grad_clip = 1.0` clips every update (pre-clip G norm 40-150), and FDL sets that norm. After L-01 the norm
  drops about 5-7x but stays above 1. Flat-content batches raise the FDL norm 3-5x (p12) and thereby shrink the
  reconstruction share further within that batch.
- Trainer/arch: L-16 (fp32 generator tail).
- Profiling lane: please measure per-term GPU fwd+bwd for consistency (blur), LDL (unfold) and MS-SSIM before and
  after L-04/L-05/L-06, FDL with `checkpoint = true`, and the batched D forward (L-09) on workload O.
- Data/config lane: L-13 validation rules for color 'y' and mean/std.

## 8. Probe index (scratchpad `losses/`)
`common.py` (data and helpers), `p1_cost` (2.1), `p2_srgrad` (2.4), `p3_paramgrad` + `p3b_reweight` (2.3),
`p4_precision` + `p4b_consistency` (2.5), `p5_kernels` + `p5b_timing` (L-04/L-05/L-06), `p6_syncs` (2.2),
`p7_fdl_scale` (L-11), `p8_semantics` (L-12, L-14, the LDL transient, ff `ave_spectrum`), `p9_fdl_ckpt` (L-17),
`p10_r1` (L-08), `p11_fdl_grid` (L-15), `p12_fdl_flat` (2.4), `p13_gray` (L-13).
