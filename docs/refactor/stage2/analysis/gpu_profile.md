# Stage 2 lane: empirical GPU profiling of real training

Measured 2026-10-27/28 on `refactor/stabilize` (= master, `f360edd`), RTX 4090 (450 W, driver default clocks),
32 CPU threads, torch 2.9.1+cu128, Python 3.13. No repository code was edited: every variant below is a
monkeypatch inside the harness (`scratchpad/gpu_profile/h.py`), every output (configs, experiments/,
traces, logs) lives in the scratch directory. GPU left idle at the end.

## 0. Summary (owner priorities: speed, reliability, memory, training stability/efficiency)

1. **The step is launch-bound, not math-bound.** Every GAN template runs 6.4k-19.5k CUDA kernels per update;
   the GPU is idle 16-24 % of the wall time on the CNN templates and 60 % on DAT. The single biggest launch
   consumer is the metagan discriminator: about 4,300 kernels per update (52 % of all launches in the
   realplksr template) at an average of 3-4 µs each, and it costs **31 ms of a 116 ms step (27 %)**.
2. **One D forward on `cat([gt, sr])` instead of two** (GP-01) cuts the realplksr step by 14 % and the
   compact-OTF step by 16 % in eager mode. It also makes `torch.compile` of D work: today that crashes because
   two spectral-norm forwards run before one backward. With all the compile switches the step is 26-30 %
   faster (a: 116 to 85 ms, c: 160 to 117 ms, e: 95 to 67 ms).
3. **Default `grad_clip` = 1.0 is active on 100 % of updates.** Pre-clip G norms have medians of 597 (a),
   92 (c) and 135 over 2,500 updates (a). So clipping acts as a gradient normaliser, not as a spike guard.
   This corroborates trainer.md T-02 with real-template GPU numbers.
4. **FDL (DINOv2) is the largest objective cost.** It takes 20 % of the wall time (a), 1.2 GB (37 % of
   peak allocated memory in a, 57 % in e), and 50-100x the gradient norm of mssim, consistency and ldl.
   Recomputing a constant bicubic position embedding twice per update costs 4.3 ms. Caching it (GP-03) is
   bitwise-identical and saves 3.2-3.8 % of the step.
5. **First validation pass:** cuDNN benchmark autotuning (benchmark_limit = 0) takes 2.4-3.7 s per new shape
   and briefly raises peak allocated memory to **4.8 GB** (a; 0.64 GB steady state). With the benchmark
   disabled, validation is just as fast and the spike is gone (GP-05).
6. **GAN from update 1 on a fresh G** (a): D saturates by update 250-350 (logits +-8.8, l_d 0.0009). Its
   gradient norm spikes to 778 around update 1,050, and by update ~1,700 it sits at chance level (l_d =
   2 ln 2). The esrgan template stays balanced. This supports doing the deferred S.1 (GAN start / pretrain)
   step next.
7. **Measured and not worth changing:** the per-update `isfinite` host sync (0 to 0.3 %), the switch to fused
   AdamW (slower here), channels_last (keep it: it is 3-9 % faster), checkpoint save (0.3 s) and the EMA
   (0.12 ms).

## 1. Method

- **Harness.** It builds the trainer the way `train.py` does: `load_config(is_train=True, root=SCRATCH)`,
  then `validate_train`, `setup_runtime`, `train.prepare_run`, `build_model`, `train.build_loaders` and
  `CUDAPrefetcher`. It then runs `feed_data` + `optimize_parameters` with `get_current_log` every 100
  updates, as train.py does. Workers use the spawn start method (as train.py), 4 workers, pinned and
  persistent. `cudnn.benchmark` is on (no manual_seed).
- **Timing.** Warmup of 30-60 updates (60 with compile), then 3 trials of 100 updates (50 for DAT), each
  bracketed by `torch.cuda.synchronize()`. I report the median with min-max. All variance figures are
  within-process.
- **CPU contention.** Other lanes ran CPU probes at the same time. During the first hour the load average
  was 60-137 on 32 threads, and those timings came out 1.6-3x too slow. They were discarded
  (`results_polluted.jsonl`). All reported wall times come from runs that started when the 1-minute load
  average was below 12. Most ran at a load of 1-3, and the load is logged per trial. Kernel-time numbers
  (profiler) do not depend on host load.
- **Profiler.** `torch.profiler` (CPU+CUDA) over 10 updates with a `record_function` around G forward, each
  loss `compute`, each `backward`, `Phase.step`, `opt.step`, the EMA, augmentation, degradation and the
  pool. Each kernel is attributed to the innermost range that encloses its launch call (cuda_runtime +
  cuda_driver correlation). Host syncs are counted as `cudaStreamSynchronize`, `cudaDeviceSynchronize`,
  `cudaEventSynchronize` and DtoH memcpys in the trace, and are also read from
  `torch.cuda.set_sync_debug_mode("warn")` with neosr call-site attribution. A CUDA-only trace gives the
  kernel counts and the gap histogram.
- **Workloads.** Templates are used as shipped; only the dataroots are changed. Data: paired x4
  `/home/korvin/apps/data/{GT,LQ}` (10,607 PNG, GT 256 px) and val (7 pairs, GT 512 px).
  - **a** `train_realplksr.toml`: realplksr + metagan, mssim + consistency + ldl + fdl(dinov2) + gan bce
    0.3, adan_sf, bf16, batch 8, LQ 64.
  - **b** `train_compact.toml` with only `L1Loss` and no D (overhead-bound case).
  - **c** `train_esrgan.toml`.
  - **d** `train_dat.toml` as shipped (`dat_m`, LQ 32, batch 8). The full model fits, so no light variant
    was needed.
  - **e** `train_compact_otf.toml` (OTF degradations, compact + metagan, full loss set,
    `grad_clip = false`).

## 2. Per-workload results (steady state, quiet machine)

| | a realplksr | b compact L1 | c esrgan | d dat_m | e compact OTF |
|---|---|---|---|---|---|
| ms/update (median, min-max) | **116.2** (116.1-116.2); r3 118.0 | **6.3** (6.2-6.3) | **159.5** (159.4-159.6) | **268.7** (268.6-269.2) | **95.2** (94.8-95.3); r2 96.4 |
| images/s | 69 | 1279 | 50 | 30 | 84 |
| peak allocated / reserved MiB | 3273 / 3624 | 197 / 226 | 5841 / 6226 | 5659 / 5824 | 2137 / 2904 |
| data wait in `prefetcher.next()` ms/update | 0.19 | 0.22 | 0.18 | 0.20 | 0.20 |
| GPU kernel time ms/update (profiler) | 97.5 | 2.7 | 120.5 | 108.1 | 75.4 |
| GPU idle share of the wall time | 16 % | 57 % | 24 % | 60 % | 21 % |
| kernels / update | 8,348 | 323 | 11,402 | 19,492 | 6,419 |
| main-thread CPU ms/update (`thread_time`; backward runs on the autograd thread) | 63 | 3.4 | 83 | 139 | 58 |
| host syncs / update on the training thread | 2 (`phase.py:96` x2) | 1 | 2 | 2 | 4 (`phase.py:96` x2, `degradations.py:642/720` gray gate) |
| syncs flagged in the pin-memory thread (not blocking the step) | 4 | - | - | - | 8 |

Kernel GPU time per stage (ms/update; `bwd_g` includes the backward through FDL and through D's adversarial
pass):

| stage | a | c | d | e | b |
|---|---|---|---|---|---|
| G forward | 10.9 | 17.0 | 27.8 | 0.6 | 0.6 |
| FDL forward (sr + gt branch + head) | 27.8 | 26.4 | 8.1 | 28.2 | - |
| G backward (`bwd_g`) | 41.1 | 57.1 | 53.5 | 22.5 | 1.6 |
| D: adversarial fwd / D-loss fwd / D backward | 1.8 / 3.5 / 6.8 | 1.7 / 3.3 / 6.5 | 1.4 / 2.6 / 4.4 | 1.9 / 3.6 / 7.0 | - |
| mssim / consistency / ldl forward | 1.2 / 1.2 / 0.1 | 1.1 / 1.2 / 0.1 | 0.6 / 0.5 / 0.1 | 1.2 / 1.3 / 0.1 | - |
| adan_sf steps (G + D) | 2.5 | 5.2 | 7.6 | 1.1 | 0.15 |
| EMA lerp | 0.12 | 0.26 | 1.05 | 0.01 | 0.01 |
| augment / degrade + pool | 0.03 | 0.05 | 0.02 | 0.04 / 6.8 | 0.04 |

Kernel count per stage for (a) (`kcount.py traces/a_base2.json.gz`) shows where the launches come from:

| range | kernels/update | GPU ms | avg µs/kernel |
|---|---|---|---|
| bwd_g | 2,648 | 40.75 | 15.4 |
| bwd_d | 1,687 | 6.73 | **4.0** |
| loss:gan_d_loss (D on gt and sr) | 1,101 | 3.43 | **3.1** |
| G_fwd | 964 | 10.85 | 11.3 |
| loss:fdl_loss | 673 | 27.70 | 41.2 |
| loss:gan_loss (D on sr, G's term) | 608 | 1.79 | **2.9** |
| loss:consistency / mssim | 238 / 181 | 1.22 / 1.16 | 5.1 / 6.4 |
| adan_sf (2 steps) | 162 | 2.50 | 15.4 |

D accounts for 1,687 + 1,101 + 608 kernels in its own ranges, plus about 900 of the `bwd_g` kernels (G's
gradient flowing through D). That is roughly 4,300 of 8,348 launches for about 13 ms of GPU work. At 3-4 µs
per kernel, D is below the per-kernel launch cost of about 7 µs (63 ms main-thread CPU / 8,348 kernels),
so the GPU waits for the host. DAT shows the same pattern (19.5k kernels at an average of 5 µs).

Top kernels (self GPU time, ms/update, % of kernel time, launches/update):

| a realplksr | ms | % | n |
|---|---|---|---|
| sm80_xmma_wgrad_implicit_gemm (bf16 conv wgrad) | 5.53 | 5.7 | 119 |
| cutlass s1688gemm 128x256 (fp32/TF32, DINOv2) | 5.51 | 5.7 | 32 |
| direct_copy_kernel (autocast weight casts, `.float()`) | 5.36 | 5.5 | 695 |
| cutlass s1688gemm 64x256 (DINOv2) | 5.16 | 5.3 | 16 |
| fmha_cutlassB f32 (DINOv2 attention backward) | 4.68 | 4.8 | 8 |
| **upsample_bicubic2d_out_frame (DINOv2 pos-embed interpolation)** | **4.26** | **4.4** | **2** |
| sm80_xmma_dgrad / fprop implicit gemm | 4.01 / 3.93 | 4.1 / 4.0 | 32 / 28 |
| fmha_cutlassF f32 (DINOv2 attention forward) | 3.26 | 3.3 | 16 |

| d dat_m | ms | % | n |
|---|---|---|---|
| direct_copy_kernel (elementwise) | 13.79 | 12.8 | 1,638 |
| reduce_kernel | 6.04 | 5.6 | 979 |
| BinaryFunctor elementwise | 5.19 | 4.8 | 1,048 |
| LayerNorm GammaBetaBackward | 4.32 | 4.0 | 254 |
| direct_copy (unrolled) / bf16 copy | 3.77 / 3.66 | 3.5 / 3.4 | 1,714 / 2,257 |

The raw `key_averages` tables (top 30 by CUDA total) are in `scratchpad/gpu_profile/traces/*_top.txt`, and
the full gzip traces sit next to them.

## 3. Targeted experiments

All timings are medians of 3 x 100 updates on a quiet machine. "base" is the matching row in section 2.

| # | experiment | workload | base ms | variant ms | delta | notes |
|---|---|---|---|---|---|---|
| 1 | `Phase.step` without `bool(norm.isfinite())` (always steps) | b | 6.3 | 6.3 | 0 % | interleaved r2 |
| 1 | same | a | 116.2 | 115.9 | -0.3 % | |
| 1 | same | c / e | 159.5 / 95.2 | 159.5 / 95.4 | 0 % | |
| 2 | adan_sf to `AdamW(fused=True)`, optimizer GPU time | a | 2.50 ms | 3.73 ms | **+49 %** | FusedAdamMathFunctor 246 µs per launch, grid of only 45 blocks |
| 2 | same | d | 7.57 ms | 12.84 ms | +70 % | optimizer is 2.6-7 % of GPU time either way |
| 3 | `compile = true` (G only, as shipped) | b | 6.3 | 6.3 | 0 % | warmup +3 s |
| 3 | same | a | 116.2 | 112.0 | -3.6 % | 60-update warmup 56 s vs 26 s for 30 eager updates (warm inductor cache) |
| 3 | same | c | 159.5 | 148.5 | -6.9 % | warmup 91 s |
| 3 | same | d | 268.7 | 193.5 | **-28 %** | warmup 234 s; peak allocated 5659 to 4590 MiB |
| 4 | contiguous instead of channels_last | a | 116.2 | 120.0 | +3.3 % | channels_last stays |
| 4 | same | c | 159.5 | 175.5 | +10 % | |
| 5 | validation, 7 images, first pass (benchmark on) | a | | 2.54 s, peak 4,794 MiB | | 2nd pass 0.175 s / 637 MiB |
| 5 | validation with benchmark off from the start | a | | 0.21 s, peak 637 MiB | | later passes 0.18-0.20 s |
| 5 | first pass / benchmark off | c | | 3.96 s, 2,974 MiB / 0.26 s, 1,072 MiB | | steady 0.23 s / 1,072 MiB |
| 5 | validation under bf16 autocast | a | 0.175 s | 0.167 s | -5 % | metric numerics change; not recommended |
| 6 | `model.save()` | a | | 0.29-0.30 s | | `.state` 336 MiB, `net_g` 28 MiB, `net_d` 23 MiB |
| 7 | train loader alone, 4 workers | b | step 6.3 ms | 6.2 ms/batch (161/s) | | **98 % of loader capacity** |
| 7 | 8 workers | b | | 3.4 ms/batch (293/s) | | step unchanged (6.4 ms) on a quiet machine |
| 7 | 4 workers | e | step 95 ms | 11.5 ms/batch | | ample |

Additional experiments (beyond the brief) that locate the cost:

| experiment | workload | ms | delta vs base | peak alloc MiB |
|---|---|---|---|---|
| no D, no GAN term | a | 85.2 | **-26.7 %** (D = 31 ms) | 3070 |
| no FDL term | a | 92.7 | **-20.2 %** (FDL = 23.5 ms) | **2058** (-1,215) |
| no FDL term | e | 78.1 | -18 % | **919** (-1,218) |
| **dcat: one D forward on `cat([gt, sr.detach()])`** | a | **99.6** | **-14.3 %** | 3276 |
| dcat | e | 79.7 | -16.3 % | 2138 |
| dcat + compile D (`net_d_fwd = net_d_adv = torch.compile(net_d)`) | a | 96.3 | -17 % | 3221 |
| compile D **without** dcat | a | crash | `RuntimeError ... modified by an inplace operation: [FloatTensor [1]] is at version 3; expected 2` (perf.md section 3.5) | |
| fdlpos: cache DINOv2 `interpolate_pos_encoding` | a | 114.2 vs r3 118.0 | -3.2 % | same |
| fdlpos | e | 92.7 vs r2 96.4 | -3.8 % | same |
| dcat + fdlpos (eager, no compile) | a / c | 97.6 / 140.7 | **-16 % / -12 %** | |
| compile each DINOv2 block (S.12) | a | 114.2 | -1.7 % | |
| compile each metagan `GatedCNNBlock` | a / e | 119.7 / 98.2 | **+3 % (worse)** | |
| stack: dcat + compile G + compile D + compile FDL blocks + fdlpos | a | **85.4** | **-26.5 %** | 3111 |
| stack | c | **117.4** | **-26.4 %** | 5504 |
| stack | e | **66.8** | **-30 %** | 2087 |
| stack, CUDA-only trace | a | kernels 8,909 to 6,599; busy 93.5 to 77.4 ms | | |
| DAT: bounded cache for `calculate_mask` (LQ 32 != img_size 64 recomputes it each forward) | d | 265.9 | -1 % | |

## 4. Gradient norms and training dynamics

Pre-clip total norms recorded per successful update (`Phase.grad_norm`), starting from update 1:

| workload (clip) | net | n | median | p10-p90 | max | share > 1.0 |
|---|---|---|---|---|---|---|
| a (clip 1.0) | G | 400 | 597 | 325-871 | 1,121 | **100 %** |
| a (clip 1.0) | D | 400 | 0.13 | 0.013-17.8 | 21.6 | 34 % (first 50: median 19; last 100: 0.016, 0 %) |
| a long (clip 1.0) | G | 2,500 | 135 | 38-684 | 1,632 | **100 %** (last 100: median 35) |
| a long (clip 1.0) | D | 2,500 | 34 | 0.65-259 | 2,422 | 89 % |
| c (clip 1.0) | G | 400 | 92 | 57-228 | 424 | **100 %** |
| c (clip 1.0) | D | 400 | 11.6 | 6.2-21.8 | 39 | **100 %** |
| e (`grad_clip = false`) | G / D | 400 | 29 / 6.0 | 9.8-56 / 1.7-36 | 121 / 136 | 100 % / 100 % (would be clipped) |

The window logs of a-long (every 50 updates; `out_d_*` are mean logits) look like this:

| update | l_g_fdl | l_g_gan | out_d_real / out_d_fake | l_d_total | grad_norm_d |
|---|---|---|---|---|---|
| 50 | 42.7 | 0.22 | -0.09 / -0.03 | 1.45 | 13 |
| 250 | 20.3 | 2.10 | **6.86 / -7.01** | **0.0051** | 0.12 |
| 350 | 18.4 | 2.61 | **8.61 / -8.72** | **0.0009** | 0.12 |
| 550 | 17.9 | 0.86 | 3.37 / -2.63 | 0.59 | 93 |
| 1,050 | 14.4 | 0.81 | -1.55 / -1.62 | **4.73** | **778** |
| 1,650 | 10.5 | 0.21 | 0.60 / 0.08 | 1.30 | 76 |
| 2,450 | 9.8 | 0.22 | -0.05 / -0.04 | **1.41 (= 2 ln 2: chance)** | 2.1 |

The dcat variant reproduces the same trajectory over 400 updates (l_d 0.0036 at 250, 0.0012 at 350, logits
+-8.8), so it does not change these dynamics. Esrgan (c) stays balanced over 400 updates (l_d 1.25-1.57,
logits within +-0.5).

Per-term gradient share for G (`autograd.grad` per term on 4 batches, 50 updates apart, after 300 updates):

| a realplksr | value | norm of grad | cos(term, total) |
|---|---|---|---|
| fdl | 17.7-19.6 | **556-1,464** | 0.66-0.997 |
| gan | 0.76-2.68 | 61-465 | -0.41 to 0.47 |
| mssim | 0.53-0.62 | 7.7-18 | -0.06 to 0.37 |
| consistency | 0.21-0.24 | 4.7-13 | -0.11 to 0.48 |
| ldl | 0.026-0.037 | 2.9-4.5 | -0.45 to 0.09 |

| c esrgan | value | norm of grad | cos(term, total) |
|---|---|---|---|
| fdl | 10.7-11.3 | 79-87 | 0.77-0.97 |
| consistency | 0.03-0.08 | 22-44 | 0.30-0.80 |
| gan | 0.18-0.23 | 11-21 | -0.56 to 0.62 |
| mssim | 0.09-0.14 | 2.1-7.0 | 0.18-0.52 |
| ldl | 0.0001-0.0004 | 0.03-0.2 | -0.57 to 0.55 |

In the realplksr template, the reconstruction terms (mssim, consistency, ldl) together contribute 1-3 % of
G's gradient norm. FDL alone sets the direction (cos up to 0.997), and since every update is clipped to norm
1, the clip normalises by FDL's magnitude. The dinov2 path of `fdl_loss._score` returns `score.mean()`
without the `* 0.01` that the CNN backbones get (`fdl_loss.py:361`), so at weight 0.75 FDL is 20-40x
the other terms in value.

## 5. Findings, prioritised

### GP-01 (P1, speed + enables compile): one D forward per D update instead of two
**Where:** `neosr/losses/gan_loss.py:62-66` (`gan_d_loss.evaluate`: `ctx.net_d(gt)` and then
`ctx.net_d(sr.detach())`).

**Evidence (measured).** Removing D entirely saves 31 ms of 116 (a). Replacing the two forwards with one:

```python
x = torch.cat([self.g.adv(ctx.gt, ctx.it), self.g.adv(ctx.sr.detach(), ctx.it)])
real, fake = ctx.net_d(x).chunk(2)
```

That gives a 116.2 to 99.6 ms (-14.3 %) and e 95.2 to 79.7 ms (-16.3 %), with unchanged peak memory. It
also removes the two-forwards-before-one-backward pattern that makes compiled spectral norm crash
(reproduced). With dcat, `torch.compile(net_d)` works for both the D update and G's adversarial pass:
99.6 to 96.3 ms.

**Numerics.** On CPU fp32, `probe_dcat.py` with converged u matches the legacy outputs to a max |diff| of
7e-7 (0.0 with SN frozen). GroupNorm and RMSNorm are per sample. Two things differ:
- one spectral-norm power iteration per D step instead of two, so real and fake now see the same W/sigma
  (legacy: different);
- the dropout/DropPath masks are drawn in a different order.

The D trajectory over 400 updates matches the baseline (section 4).

**Compatibility:** checkpoints unchanged; logs unchanged (`l_d_real` etc. are split from the chunks).
**Behaviour:** small numerics change, so a short A/B is needed. Corroborates stability.md and losses.md
(CPU).

**Test:** a unit test that the dcat D loss equals the two-forward loss in eval mode (bitwise), plus the
existing GAN tests. Once landed, consider letting `wrap()` compile D (base.py:142 compiles G only).

### GP-02 (P1, learning efficiency): `grad_clip` = 1.0 is a permanent normaliser
**Where:** `neosr/models/image.py:71` (`1.0 if tr.get("grad_clip", True)`), `phase.py:93-94`,
`readme.md:187`. Clipping is on in 82 of 84 training templates.

**Evidence (measured):** section 4. 100 % of G updates are clipped in a and c (median ratio 92-597x).
D in c is clipped on 100 % of updates, D in a on 89 % over 2,500 updates. With adan_sf the schedule-free `z`
takes raw-gradient steps (`adan_sf.py: torch._foreach_sub_(z, grads, alpha=lr)`), so the clip directly
scales z's step size to exactly `lr`. Elsewhere it reweights batches by 1/||g||.

**Fix:** let `train.grad_clip` accept a float max-norm (`true` stays legacy 1.0; `false` stays off) and log
the clipped share. A/B three arms: {1.0 legacy, off, a fixed spike-guard threshold well above the measured
p90 (for example 2-5x the p90 of the first 1k updates)}. This is not adaptive clipping, and the threshold is
explicit per template.

**Behaviour:** objective unchanged, optimisation dynamics change, so an A/B is needed. Config-compatible.
Same conclusion as trainer.md T-02 (CPU mechanism probe), now with real-template norms.

### GP-03 (P1, speed, bitwise): cache DINOv2's interpolated position embedding
**Where:** `neosr/losses/fdl_loss.py:243` calls `get_intermediate_layers`, which runs
`interpolate_pos_encoding` (hub `vision_transformer.py:182-214`). That function bicubic-resamples a
`[1,768,37,37]` permuted tensor to 19x19 on every forward, for both the sr and the gt branch.

**Evidence (measured):** `upsample_bicubic2d_out_frame` takes 4.26 ms/update (2 launches, 4.4 % of GPU time
in a). Wrapping it in a per-(tokens, w, h, dtype, device) dict cache brings a from 118.0 to 114.2 ms (-3.2 %)
and e from 96.4 to 92.7 ms (-3.8 %). The backbone is frozen (`requires_grad=False`), so the cached value is
the same tensor that is computed today.

**Fix:** wrap `self.model.interpolate_pos_encoding` in `dinov2.__init__` with a bounded cache (a few
entries; it is graph-independent, and the key includes dtype and device). This affects 84 templates.
**Behaviour:** none (bitwise). **Test:** feature equality with and without the cache in fp32 and under bf16
autocast; the cache size stays bounded across 3 sizes.

### GP-04 (P1, speed + memory, tuning): FDL dominates cost, memory and gradient
**Evidence (measured).**
- Cost: FDL costs 23.5 ms/update (20 %) in a and 17 ms (18 %) in e. It takes 27-28 ms of GPU time in a, c
  and e (fp32/TF32 ViT-B GEMMs and the fp32 memory-efficient attention).
- Memory: dropping FDL lowers peak allocated memory by 1,215 MiB in a (3,273 to 2,058) and by 1,218 MiB in
  e (2,137 to 919). That is the largest controllable memory item.
- Gradient: section 4. At update ~300 in a, FDL carries 556-1,464 of a total gradient norm of 502-1,392.

**Options, each needing an A/B:**
- (i) A/B `fp32_backbone = false`, which already exists. perf.md measured about 2x faster FDL and roughly
  half the activation memory, with G-gradient cos 0.976-0.987 at 256 px.
- (ii) Rebalance the template weights: FDL dinov2 at 0.75 without the 0.01 factor versus mssim and
  consistency at 1.0. This is an objective change.
- (iii) Optional activation checkpointing of the DINOv2 blocks for memory-limited GPUs (hypothesis: about
  1 GB less for one extra sr-branch forward; losses.md estimates similar).

Compiling the DINOv2 blocks (PLAN S.12) gives only -1.7 % now that blocks 8-11 are gone.

### GP-05 (P1 reliability / P2 speed): cuDNN benchmark during validation
**Where:** `runtime.py:91-93` sets `benchmark = True` and `benchmark_limit = 0` (try every algorithm);
`image.py:337` validates under it.

**Evidence (measured):**
- a: the first validation takes 2.54 s, and peak allocated memory reaches **4,794 MiB** against 637 MiB in
  steady state. The extra is workspace for the exhaustive autotune, added on top of the reserved training
  pool.
- c: 3.96 s and 2,974 MiB, against 0.23 s and 1,072 MiB.
- With benchmark off from the start: 0.21 s / 637 MiB (a) and 0.26 s / 1,072 MiB (c). Later passes are
  equally fast.
- The training update after validation is not re-tuned (0.12 s).

Each new validation shape repeats the 2.5-4 s autotune. A dataset with N different image sizes costs about
3N s per validation, plus an OOM risk when VRAM is nearly full.

**Fix:** in `validate()`, run under `torch.backends.cudnn.flags(enabled=..., benchmark=False,
deterministic=..., allow_tf32=...)`, passing the current values for the other flags. Do the same in
test.py's inference path. **Behaviour:** algorithm choice for validation only, with fp-level metric noise.
**Test:** a CUDA test that the benchmark flag is restored after validation; peak memory during the first
validation stays below 1.5x the steady-state value.

### GP-06 (P1 stability, needs a long run): GAN on an untrained G saturates D, then D goes to chance
**Evidence (measured, 1 seed):** section 4, a-long.
- D wins outright by update 250-350 (l_d 0.0009, logits +-8.7), and D's gradients vanish (norm 0.12).
- G's adversarial loss grows to 2.6.
- D then oscillates (grad norm 778, l_d 4.73 at around update 1,050) and from about update 1,700 sits at
  chance level (l_d = 2 ln 2, logits about 0).

Esrgan stays balanced. Templates ship with `pretrain_network_g` commented out and no GAN start.

**Fix (feature, already deferred as PLAN S.1):** `gan_opt.start_iter` (pixel/perceptual warm phase before
D trains) and an R1/R2 option. Validate with a 20k+ update run before changing any template default.

**Behaviour:** objective schedule change, so an A/B is needed. This finding raises the priority of S.1; it
does not justify adaptive damping.

### GP-07 (P1 speed): the step is launch-bound; compile is the remaining lever, best after GP-01
**Evidence (measured):** sections 2-3. Kernels per update: 8.3k (a), 11.4k (c), 19.5k (d), 6.4k (e).
- `compile = true` (G only) gives -3.6 % (a), -6.9 % (c) and **-28 % (d, DAT; peak memory -19 %)**, and 0 on
  compact.
- Compiling each metagan `GatedCNNBlock` separately is +3 % slower (guard and dispatch overhead), so it is
  rejected.
- The whole stack (dcat + compile G, D and FDL blocks + fdlpos) gives -26 to -30 % on a, c and e.

**Fix:**
- (1) GP-01 first.
- (2) Then let `wrap()` compile `net_d` as well, and drop the `net_d_adv` special case in
  `image.py:91-97` (perf.md section 5.4).
- (3) Template comments: recommend `compile = true` for the transformer templates (DAT -28 %), and state the
  warmup cost (about 4 min for DAT with a warm cache).

**Behaviour:** none beyond compile's fp-level differences. **Test:** the existing
`test_compile_wraps_the_generator_only` changes to cover D; a CUDA smoke test of 3 updates with a compiled
spectral-norm D after dcat.

### GP-08 (P2 speed/robustness, tuning): loader capacity for light models; data wait is invisible
**Where:** `neosr/data/__init__.py:53` (default `num_worker_per_gpu = 4`); `train.py` logs only `time`.

**Evidence (measured):**
- With 4 workers, b's loader delivers 6.2 ms/batch against a 6.3 ms step, so it runs at 98 % of capacity.
  With 8 workers: 3.4 ms/batch.
- Under the background CPU load of other lanes (load 5-7), b's data wait rose to 0.4-3.35 ms/update (up to
  37 % of the step). On a quiet machine it was 0.2 ms.
- Heavy templates have ample loader headroom (e: 11.5 ms/batch against a 95 ms step).

**Fix:**
- Log the data-wait share (time spent in `prefetcher.next()`) next to `time` in the log line.
- Document (or default to) `min(8, cpu_count // world_size)` workers.

**Behaviour:** none. **Test:** a unit test on the timer arithmetic.

### GP-09 (P2 info, verified): the `Phase.step` isfinite sync costs nothing measurable; keep it
**Evidence (measured):** sync-free variant: a 116.2 to 115.9 ms, b 6.3 to 6.3, c 159.5 to 159.5, e 95.2 to
95.4. Training-thread syncs are 2 per update (1 for b). The OTF gray-noise gate adds 2. The
`pin_memory.py:57` syncs (4-8 per update) happen on the pin thread. **Decision:** no change. The sync
guards against NaN updates for free.

### GP-10 (P2 info): no optimizer swap for speed
**Evidence (measured):** `torch.optim.AdamW(fused=True)` is slower than adan_sf's foreach implementation in
torch 2.9.1 for these parameter sets: 3.73 vs 2.50 ms (a), 12.84 vs 7.57 ms (d). The fused kernel launches
with a 45-block grid at 246 µs per launch. The optimizer is 2.6-7 % of GPU time. F01 (adan_sf recurrence) is
a correctness question for the optimizer lane, not a speed question.

### GP-11 (P2 info): keep channels_last for the CNN templates
Contiguous is 3.3 % slower for realplksr and 10 % slower for esrgan (measured). `image.py:46` (`mf` follows
`use_amp`) stays as it is.

### GP-12 (P3): checkpoint save cost and disk growth
**Evidence (measured):**
- `save()` takes 0.29-0.30 s, which is 0.26 % of wall time at `save_checkpoint_freq = 1000`.
- The `.state` file is 336 MiB, of which the adan_sf state (5 fp32 tensors per parameter) is about 80 %.
- Exports are 51 MiB per save (net_g 28 + net_d 23). They are never pruned (owner decision 8a), so 1M
  updates at the default frequency produce about 51 GB.

**Fix:** none in code; document the growth next to `keep_states`.

### GP-13 (P3): validation runs in fp32 (no autocast)
`image.py:341-349` (`net(lq)` under `inference_mode` only). bf16 validation is only 5 % faster in steady
state (0.167 vs 0.175 s for 7 images) and changes the reported metrics. The competitor validates under
autocast (`traiNNer/models/sr_model.py:1134`). Recommendation: keep fp32 as the reference.

### GP-14 (P3): DAT mask recomputation
`dat_arch.py:525-527`: with LQ 32 and `img_size` 64, `calculate_mask` runs on every forward for each shifted
block. A bounded cache saves only 1 % (268.7 to 265.9 ms). The real DAT lever is compile (GP-07). This is
left to the arch lanes.

## 6. Verdicts on analyze.md items in this lane

| analyze.md item | verdict | evidence |
|---|---|---|
| Perf rank 3: loss composition / backbone precision dominates light generators | confirmed | FDL is 20 % of wall time (a), 18 % (e), 28 ms of 75 ms GPU time (e), and 1.2 GB of peak memory (GP-04) |
| Perf rank 7: loader workers / threads | partial | 4 workers sit at 98 % of capacity for compact L1 (b); heavy templates have ample headroom. The default is 4, not "auto" (GP-08) |
| Perf rank 8: layout per family | confirmed (measured) | channels_last faster by 3 % (realplksr) and 10 % (esrgan) |
| Perf rank 9: compile on representative shapes | confirmed | -3.6 % (a), -6.9 % (c), -28 % (d), 0 (b); D compile blocked until dcat (GP-07) |
| Perf rank 10: checkpoint / validation disruption | partial | save 0.3 s (negligible); the first validation per shape costs 2.4-3.7 s plus a 3-4.8 GB transient (GP-05) |
| Perf rank 1: dead computation | partial (this lane) | DINOv2 pos-embed recomputation 4.3 ms/update (GP-03); arch-specific items belong to other lanes |
| Memory table: prefetcher, PairPool, loss context are bounded | confirmed | allocated memory is flat across 3 x 100 updates in every workload; the OTF pool is included in e's 2.1 GB |
| Memory table: checkpoint exports grow on disk without bound | confirmed (quantified) | 51 MiB per save, about 51 GB per 1M updates (GP-12; pruning excluded by owner decision 8a) |
| Loss obs.: heavy perceptual objectives, profile separately | confirmed | section 2 per-stage table; per-term gradient share in section 4 |
| Loss obs.: FDL DINO precision opt-in | confirmed | default stays fp32; the A/B is the remaining step (GP-04 i) |
| Loss obs.: perceptual-metric round trip / work without outputs | partial | validation is 0.18 s for 7 images; not significant at this validation size |
| Profiling protocol | applied | sections 1-3 (CPU contention handling added) |
| F22 TOPIQ attention weights | not checked | metrics lane |

## 7. Competitor notes (traiNNer-redux)

- `grad_clip` defaults to **false** (`traiNNer/utils/redux_options.py:277-282`; max norm 1.0 when enabled).
  It uses `clip_grad_norm_` per net (`sr_model.py:930`, `:1032`), with the automation-chosen threshold the
  owner dislikes. Taking only the default-off part is in line with GP-02.
- It validates under autocast (`sr_model.py:1134`). Not recommended here (GP-13).
- It makes channels_last opt-in with collate workarounds (`data/collate_functions.py`). neosr's AMP-coupled
  channels_last is measured faster and simpler; keep it.

## 8. Reproduction

All files are under `scratchpad/gpu_profile/`.
- `h.py <w> <mode> [flags]`: modes `time | profile | hosttime | gaps | syncs | gradnorm | termgrad | val |
  save | loader`; flags `--nosync --adamw --compile --contig --dcat --compile_d --compile_fdl
  --compile_dblocks --fdlpos --datmask --nod --drop <slot> --workers N --valamp`.
- `run.sh` waits for an idle GPU and a 1-minute load below 12.
- `results.jsonl` holds every clean result; `summ.py` prints it.
- `kcount.py`, `topk.py` and `krange.py` analyse the gzip traces in `traces/`.
- `probe_dcat.py` is the CPU equivalence probe.
