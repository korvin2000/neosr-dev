# Performance pass (lane C6, T4 §2)

> **Main-thread decision (Opus, after review):** the DINOv2 bf16 backbone is **opt-in** (`fp32_backbone = false`),
> not the default. Even at 256 px the pixel gradient differs from fp32 by up to 60% (generator-gradient cosine ~0.98), and no
> training-quality A/B exists yet, so the reference default keeps the old fp32 numerics. The bitwise-identical speedups (unused
> DINOv2 blocks dropped, gt features under no_grad, vectorised Poisson noise, MS-SSIM sync removal) stay on by default; the
> +23% esrgan figure below needs `fp32_backbone = false`.


Measured on 2026-10-21 against `refactor/stabilize` HEAD `14a1078` plus the uncommitted edits other W6
lanes had in the worktree at the time (EMA warmup, factory.build, wavelet_guided shrink, docs). Machine:
RTX 4090 and 32 CPU threads, both shared with other agents (see method), torch 2.9.1+cu128.

## 1. Method

- **Workloads**: the `train.py` hot path on real data, bf16, no `manual_seed` (cuDNN benchmark on,
  as the owner runs).
  - **E** = `options/train_esrgan2.toml` from the main checkout: esrgan G + metagan D, batch 16, lq 64 ->
    gt 256, `adan_sf` for G and D, EMA 0.999, losses mssim + consistency + ldl + fdl(dinov2) + gan
    (bce; wavelet view from update 80000), paired data `/home/korvin/apps/data/{GT,LQ}` (10,607 pairs).
  - **O** = `options/train_compact_otf.toml`: compact G + metagan D, batch 8, OTF degradations on the
    same GT folder, 5 augmentations, the same loss template (`ema_power` removed until the EMA lane lands).
- **Harness** (scratchpad, not committed): builds through `load_config`, `validate_train`,
  `setup_runtime`, `build_model`, `train.build_loaders` and `CUDAPrefetcher`, then runs the train.py
  loop (`feed_data`, `optimize_parameters`, `get_current_log` every `print_freq` = 100 updates).
  30 warm-up updates (60 with `compile`), then 200 updates timed between `torch.cuda.synchronize()`.
  Host syncs: `torch.cuda.set_sync_debug_mode("warn")` over 5 updates, counting only the
  "synchronizing operation" warnings, attributed to their neosr call site. GPU split: `torch.profiler`
  over 5 (E) or 10 (O) updates with a `record_function` per loss term (forward only).
- **Before/after**: two frozen copies of the worktree that differ only in the four C6 source files
  (`before` = those files at HEAD). Interleaved rounds (E-before, E-after, E-after+compile,
  O-before, O-after, O-after+compile), one process per run, median over the rounds. Each run waited
  until `nvidia-smi` showed no other compute process and logged `nvidia-smi pmon` and the load average
  every 5 s. Runs during which another process used the GPU are excluded (one O-before run, 7.49 it/s).
  O is CPU/launch-bound (§6), so other agents' CPU load moves it by a few percent between runs.

## 2. Result

| Workload | Variant | it/s per run | median it/s | ms/update | vs HEAD | syncs/update | peak alloc |
|---|---|---|---|---|---|---|---|
| E | HEAD | 4.009, 4.044, 4.041 | **4.04** | 247.5 | | 3 | 10.66 GiB |
| E | C6 | 4.972, 4.997, 4.974 | **4.97** | 201.0 | **+23 %** | 2 | 10.12 GiB |
| E | C6 + `compile = true` | 5.272, 5.300, 5.159 | **5.27** | 189.7 | **+30 %** | 2 | 9.56 GiB |
| O | HEAD | 8.523, 8.493, 8.427, 8.040, 8.515 | **8.49** | 117.7 | | 46 | 2.22 GiB |
| O | C6 | 10.818, 10.810, 10.482, 10.420, 10.799 | **10.80** | 92.6 | **+27 %** | 20 (4 with §5.1) | 1.95 GiB |
| O | C6 + `compile = true` | 10.736, 10.580, 10.551 | 10.58 | 94.5 | +25 % | 20 | 1.95 GiB |

On HEAD, `compile = true` does not run at all for a GAN config (§3.5), so there is no HEAD compile row.
Data loading is not a factor: the loop waited 0.2-0.3 ms/update in `CUDAPrefetcher.next()` in every run
(4 workers, pinned, persistent).

Attribution: single runs of intermediate trees, same protocol, ms/update (run-to-run noise ±3 %):

| Step | E | O |
|---|---|---|
| HEAD (median above) | 247.5 | 117.7 |
| + §3.3 Poisson without syncs, §3.4 MS-SSIM backward without sync | 246.1 | 104.9 |
| + §3.1 FDL DINOv2 backbone under autocast | 207.7 | 95.4 |
| + §3.2 DINOv2 blocks 8-11 dropped (= C6) | 201.0 | 92.6 |
| + `compile = true`, G only (§3.5) | 189.7 | 94.5 |

Profiler, kernel time per update (the profiler's own CPU overhead inflates wall time, so only GPU time
is shown):

| | E HEAD | E C6 | O HEAD | O C6 |
|---|---|---|---|---|
| all kernels | 226.3 ms | 182.0 ms | 79.6 ms | 54.5 ms |
| FDL forward (both branches + head) | 61.5 ms | 30.2 ms | 36.4 ms | 18.7 ms |
| OTF `feed_data` (degradations, pool, augmentations) | | | 15.2 ms | 13.4 ms |

## 3. Accepted changes

### 3.1 FDL: DINOv2 backbone under autocast, FFT / sort head in fp32 (`losses/fdl_loss.py`)
Profile: after G itself, FDL was the largest item (61.5 of 226 ms of kernel time on E). `fdl_loss.fp32 =
True` ran the whole loss outside autocast, so the ViT-B/14 ran in fp32 (TF32 `s1688gemm` GEMMs and the
fp32 memory-efficient attention kernel near the top of the table) for the sr and the gt branch. Now
the backbone runs under the trainer's autocast (bf16 GEMMs, flash attention) and `_score` (FFT,
`angle`, sorted projections) runs with autocast disabled on `.float()` features; the gt branch runs
under `no_grad`. This applies only to patches whose shorter side is at least `fdl_loss.amp_min_size`
= 256 px (both profiled templates train on 256 px gt); smaller patches run the whole loss in fp32, as on
HEAD (size table below). `fp32_backbone = true` in `[train.fdl_opt]` forces the old all-fp32 loss at
every size. The CNN backbones (`vgg`, `resnet`, `effnet`) keep the all-fp32 path (§4). Without AMP
nothing changes.

Numerics: 3 sets x 8 real training pairs against a strict-fp32 loss (TF32 off, `highest` matmul
precision). "G gradient" = d loss / d theta of a trained esrgan (owner's `net_g_3236.pth`, bf16 forward
as in training); "sr gradient" = d loss / d sr for 3 sr candidates (esrgan output, bicubic lq,
gt + noise):

| Variant | loss rel. error | G gradient cos | sr gradient cos |
|---|---|---|---|
| strict fp32 rerun (non-deterministic kernels) | 0 | 1.000 | 1.000 |
| HEAD as the owner runs it (`fast_matmul`: fp32 loss on TF32 GEMMs) | <= 3.7e-4 | 0.991-0.996 | 0.91-0.97 |
| **C6, bf16 AMP: bf16 backbone, fp32 head (gt 256 px)** | **<= 2.5e-3** | **0.975-0.987** | 0.83-0.92 |
| C6, fp16 AMP (x1024 loss scale; 1 set) | <= 7.2e-4 | 0.995 | 0.94-0.96 |
| scale reference: strict-fp32 G gradient of a *different* batch of 8 | | 0.59-0.64 | |

Stated tolerance: loss relative error < 1e-2, G-gradient cosine >= 0.97 against strict fp32, and no
gradient outlier (sr-gradient relative L2 < 1). The pixel-level FDL gradient is intrinsically
precision-sensitive (the sort routes gradients through near-ties: TF32 alone moves it to cos 0.91), but
the parameter-gradient deviation from bf16 (1 - cos <= 0.025) is ~15x smaller than minibatch noise
(1 - cos >= 0.36).

Smaller patches fail the gradient criteria (review fix pass). 3 x 8 real pairs, centre crops
at the given gt size (lq = gt / 4) through the trained esrgan, bf16 backbone against strict fp32; the
TF32 column is HEAD's all-fp32 loss under `fast_matmul`:

| gt px (token grid) | bf16 loss rel. error | bf16 G-gradient cos | bf16 sr-gradient rel. L2 | TF32 G-gradient cos |
|---|---|---|---|---|
| 64 (5x5) | <= 6.3e-3 | 0.62-0.77 | <= 0.96 | 0.92-0.96 |
| 96 (7x7) | <= 3.1e-3 | 0.65-0.87 | **<= 1.59** | |
| 112 (8x8) | <= 4.9e-3 | 0.83-0.91 | <= 0.95 | |
| 128 (10x10) | <= 2.5e-3 | 0.53-0.83 | **<= 1.51** | 0.90-0.97 |
| 160 (12x12) | <= 2.7e-3 | 0.22-0.90 | **<= 20.8** | 0.96-0.99 |
| 192 (14x14) | <= 2.1e-3 | 0.967-0.972 | <= 0.63 | 0.988-0.994 |
| 256 (19x19) | <= 2.5e-3 | **0.976-0.987** | <= 0.60 | 0.991-0.996 |

The loss value stays within 1e-2 at every size (the earlier 2.3e-2 at 64 px was a smooth synthetic
pair), but on the coarse token grids bf16 rounding moves near-zero FFT bins whose `angle` gradient
explodes, as with VGG (§4). Only 256 px meets all three criteria, so `amp_min_size = 256` and smaller
patches keep the HEAD numerics; the gain is unchanged for the profiled workloads, which train on 256 px
gt. Larger patches were not measured (the dataset tiles are 256 px); the trend from 64 to 256 px is
toward smaller drift. `tests/test_perf.py::test_fdl_dinov2_amp_policy_within_tolerance` checks the
three criteria for the default policy at 64, 128 and 256 px against a strict-fp32 loss (TF32 off),
through a fixed generator with bf16 output and 1/f synthetic images; with the fallback removed it fails
at 64 and 128 px (G-gradient cos 0.96 and lower).

### 3.2 DINOv2: blocks after the deepest requested layer are dropped (`losses/fdl_loss.py`)
`get_intermediate_layers(n=[0..7])` runs all 12 ViT blocks and keeps outputs 0-7, so blocks 8-11 were
computed for both branches and thrown away (a third of the backbone forward; no backward, since nothing
consumes them). `dinov2.__init__` deletes `blocks[max(layers) + 1:]` (when the hub model is not
chunked). Features are bitwise identical in fp32 and under bf16 autocast
(`test_fdl_dinov2_drops_unused_blocks_bitwise`). E -6.7 ms, O -2.8 ms per update.

### 3.3 OTF Poisson noise without per-sample syncs (`data/degradations.py`)
`generate_poisson_noise_pt` counted the distinct 8-bit levels of each sample with
`len(torch.unique(img[i]))`: one host sync per sample, per colour and gray pass, plus a `new_tensor`
copy (26.6 of the 46 syncs/update on O). `_poisson_levels` gets the same counts from a (b, 256)
occupancy scatter and the same `2 ** ceil(log2(n))` in float64, on the device. The "any gray sample"
gate (`torch.sum(...) > 0`, read by two `if`s, 2 syncs) is now read once as `bool(gray_noise.any())`,
also in `generate_gaussian_noise_pt`. The noise is bitwise identical for the same RNG state, on CPU
and CUDA, at gray probability 0 / 0.5 / 1 (`test_poisson_noise_is_bitwise_unchanged`). O -13 ms/update.

### 3.4 MS-SSIM backward without a sync (`losses/ssim_loss.py`)
The backward of `torch.stack(components).prod(0)` checks its input for zeros with a host sync (1 per
micro-batch, inside `backward()`). A multiplication chain (`math.prod`) gives the same product (the
components are clamped >= 1e-6): `test_b1_w1_matches_pre_b1[mssim_loss]` still holds at rtol 1e-6 in
fp32 and bf16, and `test_mssim_backward_does_not_sync` runs the backward under
`set_sync_debug_mode("error")`. Throughput effect ≈ 0 (E: 247.5 -> 246.1 ms), as T4 found for syncs on
GPU-bound runs; it was the only sync left in the loss terms.

### 3.5 `compile = true` compiles the generator only (`models/base.py`, `wrap`)
On HEAD, `compile = true` crashed every GAN config at the first discriminator backward (E and O):
`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace
operation: [torch.cuda.FloatTensor [1]] is at version 3; expected version 2`. The D update runs D on gt
and then on sr before one backward. In training mode each `torch.nn.utils.spectral_norm` layer updates
its `weight_u` in place; eager spectral norm then clones u and v exactly so that this GAN pattern can
backpropagate (comment in `torch/nn/utils/spectral_norm.py`), but the compiled graph saves the buffer
itself (the size-1 tensor is the u of metagan's last, 1-channel conv). `wrap()` now applies
`torch.compile` (mode `compile_mode`, default `"default"`, `dynamic=False`, as before) to `net_g` only;
D, and G's adversarial pass through D, stay eager (`test_compile_wraps_the_generator_only`). Gain on top
of C6: E +6 % (201.0 -> 189.7 ms), O none (launch-bound, §6). With a warm inductor cache compile adds
~4 s to the start (E: 54 s for 60 warm-up updates against ~50 s eager); a cold cache took 109 s (§4).

## 4. Measured and rejected, or left unchanged

| Candidate | Evidence | Decision |
|---|---|---|
| bf16 backbone for FDL `vgg` (and the other CNN backbones) | VGG, gt + noise candidate: sr-gradient relative L2 **2.1e5**, cos -0.02 (bf16 rounding leaves near-zero FFT bins whose `angle` gradient explodes); otherwise loss error <= 7.5e-4 and G-gradient cos 0.9993 | rejected (outlier criterion); `resnet`/`effnet` not measured, kept fp32 like VGG |
| bf16 for `dists_loss`, `vgg_perceptual_loss`, `ff_loss`, `mssim_loss` (still `fp32 = True`) | not in either profiled template; mssim forward is 2.2 ms/update | unchanged: not shown by this profile |
| more DataLoader workers or prefetch | prefetcher wait 0.2-0.3 ms/update at 4 workers on E (16 x 256 px PNG pairs) and O (8 x 256 px + kernels) | unchanged |
| remove the gray-noise gate sync (≤ 2/update on O) | computing gray noise unconditionally reorders the `torch.poisson` / `randn` draws, so degradations would no longer match HEAD for a given seed | kept: one sync per noise call |
| the 2 per-update syncs of `Phase.step` (one `isfinite` per G and D update) | designed (T2); a sync costs ≈ 0 on GPU-bound runs (T4, and §3.4 here) | unchanged (`phase.py` is not C6) |
| compile D | crashes (§3.5) | rejected |
| `compile_mode = "max-autotune-no-cudagraphs"` | with `fast_matmul = true` (the owner's setting) inductor fails while autotuning: `InductorError: ... PyTorch is checking the matmul precision without a specific backend name ... used mix of the legacy and new APIs`; with `fast_matmul = false` autotuning runs, but the run hit a CUDA OOM because another agent held 15 GiB, so no number | not a default; §5.3 |
| `compile = true`, mode `"default"`, `fast_matmul = false` | E 4.89 it/s, warm-up 109 s with a cold cache | informational |

## 5. Needed outside C6 (measured with harness prototypes, not in the code)

1. **`neosr/utils/diffjpeg.py`, `DiffJPEG.forward`**: the per-element `factor[i] = quality_to_factor(factor[i])`
   loop compares a 0-d CUDA tensor in Python: 8 syncs per JPEG call, **16 of the 20 syncs per update on
   O**. Vectorised with the same arithmetic (bitwise equal factors), for tensor `quality`:
   `factor = torch.where(quality < 50, 5000.0 / quality, 200.0 - quality * 2) / 100.0`.
   Prototype on the state before §3.2: O 98.6 -> 96.5 ms/update (+2 %), syncs 20 -> 4.
2. **Compile the frozen loss backbones under `compile = true`** (wiring in `build_losses` or `image.py`,
   e.g. `t.model = torch.compile(t.model, dynamic=False)` for `fdl_loss` terms). Prototype on the state
   before §3.2: E 214 -> 198.5 ms/update (+8 %), O 98.6 -> 91.4 ms/update (+8 %), ~20 s more warm-up.
   §3.2 removed part of what it compiles, so expect somewhat less now.
3. **`neosr/runtime.py`, `fast_matmul`**: it mixes `torch.set_float32_matmul_precision("medium")`
   with `allow_tf32`; inductor's autotuner then refuses (§4). Use one precision API, otherwise
   `compile_mode = "max-autotune*"` cannot be used with `fast_matmul = true`.
4. **`neosr/models/image.py`**: under DDP + `compile`, `net_d_adv` is still `torch.compile(self.net_d)`.
   It works (one D forward before G's backward), but with D eager everywhere else it can simply be
   `self.net_d`.
5. **Docs**: `[train.fdl_opt] fp32_backbone` (default `true` after main-thread review: the bf16 backbone is opt-in, see note at the top) belongs in the readme loss options,
   with the rule that the DINOv2 backbone uses AMP only for gt patches of 256 px and larger.

## 6. Where the time goes now

- **E** (201 ms/update eager) is GPU-bound: 182 ms of kernel time per update. G (RRDB) forward +
  backward dominates (`convolution_backward` ~59 ms, cuDNN forward ~24 ms, shared with D), then the
  FDL forward (30 ms) and backward, D (22.7 ms for the gt + sr forwards, plus backward), the two
  `adan_sf` steps (11.5 ms) and the autocast weight casts (`aten::copy_`, 11.8 ms). The C6-owned
  remainder is §5.2.
- **O** (92.6 ms/update) is launch-bound: 54.5 ms of kernel time per update. The CPU time goes to
  Python and kernel launches in metagan (three D forwards with spectral-norm hooks), the DINOv2 blocks,
  the autograd engine and the degradation chain. Compiling G does not help; §5.2 does (+8 %), and a
  compilable D would.
- Peak allocated memory dropped 0.54 GiB (E) and 0.27 GiB (O): no fp32 ViT activations, gt branch under
  `no_grad`, 4 fewer blocks.

## 7. Against the T4 draft

| T4 item | T4 expectation | Measured here |
|---|---|---|
| P0 loss backbones under autocast | +35-45 % on E | +23 % on E (FDL only: §3.1 + §3.2); FDL forward 61.5 -> 30.2 ms |
| P7 compile G, mode default | +17 % | +6 % on top of P0, after fixing the D crash |
| P11 vectorised Poisson / JPEG | up to -70 ms/update on OTF | Poisson -13 ms (O, batch 8); JPEG -2 ms (prototype, §5.1) |
| P1 syncs | ≈ 0 % | confirmed for E; on O the gain comes with the removed `unique` loops |
| P3 loader | small | already landed (pinned, persistent, 4 workers); no wait measured |
