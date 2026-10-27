# Stage 2 analysis: metrics, validation, tiling, inference and export

Lane: `neosr/metrics/*`, `neosr/models/tiling.py`, `neosr/models/image.py` (validate, `_img_path`, `eval_weights`),
`neosr/utils/img_util.py`, `test.py`, `convert.py`, and their tests; competitor `traiNNer/metrics`, `convert_to_onnx.py`.
Checkout: `refactor/stabilize` (= master, f360edd). Probes ran on CPU (`CUDA_VISIBLE_DEVICES=`), torch 2.9.1, onnx 1.20.0,
onnxruntime 1.23.2; onnxscript is not installed. Scripts and raw output: the scratchpad `metrics/` folder (p1 to p14,
`export_results*.jsonl`).

**About the CPU timings.** Other lanes were running CPU probes at the same time, with load averages of 90 to 120 on 32
threads. Absolute CPU times are therefore upper bounds. The ratios come from interleaved runs in the same process and
are reliable. Exactness and memory results do not depend on load. GPU numbers are estimates, labelled as such, for the
profiling lane to confirm.

Baseline: `pytest tests/test_metrics.py tests/test_tiling.py` gives 48 passed, 1 skipped.

---

## 1. Priorities

| # | Finding | Kind | Sev | Verified | Numerics change? |
|---|---|---|---|---|---|
| M-01 | SPAN, span_fast and SPAN+ cannot be exported at all (static or dynamic). This regressed in 09ffccc | bug, deploy | P1 | reproduced | none |
| M-02 | convert.py verifies only the traced shape; dynamic exports are silently wrong for 10 sampled archs | bug, deploy | P1 | reproduced | none |
| M-03 | F07 DySample: the normalizer is frozen at the trace shape. Exact fix verified. Eager `torch.tensor(..., device)` adds a blocking H2D copy per forward | bug, deploy (+speed) | P1 | reproduced (sync: source) | none (bit-identical) |
| M-04 | Exact on-device validation metrics: CPU SSIM dominates per-image host work; skip unused conversions | speed | P2 | measured | none (exact to 3e-14) |
| M-05 | DISTS/TOPIQ are rebuilt at every validation event: about 0.8 to 1.0 s and a +1.07 GiB host spike (the full VGG16 classifier is loaded) | speed, memory | P2 | measured | none |
| M-06 | A val dataset without `name` crashes training at the first validation, and the save at that update is lost | reliability | P2 | reproduced | none |
| M-07 | ONNX input and output share dim_param names (and width/height are swapped); ORT buffer reuse fails | bug, deploy | P2 | reproduced | none |
| M-08 | `tensor2img` uses 3x the fp32 image in temporary host memory; in-place quantisation is bit-identical and 5x faster | memory | P2 | measured | none |
| M-09 | SPAN family eval forward refuses 20 kernels every call: 272 vs 52 compute ops (validation, test.py, every tile) | speed | P2 | measured (ops), GPU time hypothesis | none (exact) |
| M-10 | test.py ignores `use_amp`/`bfloat16` for precision (fp32 always); decode and PNG encode run in series with the GPU | speed | P2 | source + measured encode | yes, inference only (opt-in) |
| M-11 | `cudnn.benchmark` with `benchmark_limit=0` autotunes every algorithm for every new image shape | speed | P3 | hypothesis | none |
| M-12 | F22 TOPIQ `need_weights=False`: score parity 1.2e-7, but no peak-memory gain up to 4K | memory | P3 | measured | none (1e-7) |
| M-13 | DISTS metric keeps both full VGG pyramids; streaming per stage gives -28 % peak with an identical value | memory | P3 | measured | none |
| M-14 | Metric `better` is never validated; DISTS without `better` tracks the worst value | correctness (logs) | P3 | source | best bookkeeping only |
| M-15 | F21 output name collisions; the TensorBoard tag contains the iteration (a new tag per validation) | correctness (outputs) | P3 | source | output layout |
| M-16 | `-fp16` silently does nothing and exits 0; ORT's own float16 converter works; `--fulloptimization` exits 1 after writing | tooling | P3 | measured | n/a |
| M-17 | Up-front export capability gaps: about 13 archs fail with obscure errors; the up-front list names only craft and ditn | tooling | P3 | reproduced | none |

Why this order: the owner's first priority is speed, and the next stage ports features. Even so, two deployment
correctness bugs (M-01, M-02 with M-03) come first. They make the main export path produce nothing, or produce wrong
pixels, for the most-deployed light architectures (SPAN/SPAN+), and each fix is a few lines. They do not touch training
numerics. Next come the validation costs (M-04, M-05, M-08). These are pure waste paid at every validation event, and
the fixes keep values bit- or 1e-14-identical. M-06 is a cheap crash fix for long runs. Speed items that need GPU
evidence (M-09 to M-11) follow. Low-impact memory and bookkeeping items come last.

Nothing in this lane needs a training A/B. M-10 changes inference pixels only under an explicit `use_amp`, and only in
test.py. The validation metric protocol (uint8 rounding, clipping, BGR order, Y conversion, crop) is preserved exactly
by every proposal here.

---

## 2. Codex items: verdicts

| Codex item | Verdict | Evidence (this lane) |
|---|---|---|
| F07 DySample dynamic export | **confirmed** | realplksr(dysample=True) export at 20x20: max abs error 1.5e-5 at 20x20, **5.76** at 32x24 (ref std 1.77); mosrv2(upsampler=dysample) 0.087; spanplus (after M-01) 0.044. The exact fix is verified (M-03) |
| F08 transformer dynamic export | **partial** | Default variants traced at 20x20 are correct at 8x8, 12x8, 27x41 and 64x48 (swinir_small). **Batch 2 is silently wrong** for swinir_small 0.042, hat_s 0.034, grformer 0.18, srformer_light 0.058, rgt_s 0.10, dat_s 0.057, microsr 0.12. swinir_small at 1x1 gives a **NaN** output. The 12x8 failure did not reproduce for default swinir_small. Root cause for batch: `b = int(windows.shape[0] / ...)` in window_reverse (swinir_arch.py:70 and its copies) freezes B |
| F21 output name collisions | **confirmed** | image.py:339 keeps only the stem; image.py:384 omits the dataset; paired scanning is recursive (data_util.py:198). Also, the TB tag `f"{img}/{iter}"` (image.py:359) creates a new tag every validation |
| F22 TOPIQ need_weights | **partial** | Weights are computed and discarded: confirmed. Score parity on the 7 real pairs: max diff 1.2e-7. **Memory claim refuted up to 4K**: peak RSS 154 vs 150 MiB (512²), 3007 vs 3023 (2048²), 5850 vs 5827 (3840x2160). The ResNet50 pyramid and gated pooling dominate; attention only becomes the peak at roughly 6K and above |
| Perceptual metrics round trip | **confirmed, minor** | GPU to CPU uint8 to CPU tensor to GPU is real, but costs about 2 to 3 ms per image. The costly host work is numpy SSIM (48 to 59 ms per 512² image). See M-04 |
| Work with no outputs/metrics | **confirmed** | image.py:351 calls `tensor2img` unconditionally |
| DISTS/TOPIQ lifecycle | **confirmed, quantified** | Rebuilt every event: DISTS 541 to 602 ms with peak RSS +1073 MiB, TOPIQ 270 to 378 ms with +302 MiB (M-05) |
| Metric direction | **confirmed** | base.py:192-197: any value other than `"higher"` means lower is better; nothing validates it (M-14) |
| FP16 export flag | **confirmed** | convert.py:321-324 prints "temporarily disabled" and exits 0. `onnxruntime.transformers.float16` (ships with ORT) works (M-16) |
| New (dynamo) exporter | **not checked** | `torch.onnx.export(dynamo=True)` raises `ModuleNotFoundError: onnxscript`; onnxscript is not a dependency |
| Dynamic ONNX coverage | **confirmed, extended** | 48-arch matrix in §5 |
| Best-model artifact selection | **confirmed (feature)** | `best` is only recorded in `.state`; no `*_best.pth` is written. Feature for the next stage, not a fix |
| Model-export disk retention | **partial: fact true, fix refused** | Exports accumulate, but owner decision 8a says exports in models/ are never pruned. For scale: the ESRGAN+MetaGAN template writes 67 + 24 MB per save; at `save_checkpoint_freq = 1000` and `total_iter = 1000000` that is about 91 GB per run |

---

## 3. Findings in detail

### M-01 (P1): SPAN, span_fast and SPAN+ cannot be exported (refactor regression)

`Conv3XC.forward` in eval mode now computes the fused kernel on the fly
(`span_arch.py:146-150`, and the same in `spanplus_arch.py`):
`return F.conv2d(x, *self.fused_kernel(), stride=self.stride, padding=1)`. The TorchScript ONNX exporter cannot export a
convolution whose kernel is a traced value:

```
$ python convert.py --input span.pth -net span -s 4 [-static 3 32 32]
torch.onnx.errors.SymbolicValueError: Unsupported: ONNX export of convolution for kernel of unknown shape.
exit 1   (both dynamic and -static)
```

The pre-refactor `span_arch.py` (`git show 43db06a~1:...`) called `self.update_params(); out = self.eval_conv(x)`. It
exports, and its output matches at 32x24 with max error 3.3e-7. The change came in 09ffccc. The only converter test
uses `compact` (tests/test_entry.py:671), so CI did not catch it.

**Fix (exact):**

```python
def forward(self, x):
    if self.training:
        return self.conv(F.pad(x, (1, 1, 1, 1), "constant", 0)) + self.sk(x)
    if torch.onnx.is_in_onnx_export():  # deploy graph: eval_conv was refreshed by eval() (convert.load_net)
        return self.eval_conv(x)
    return F.conv2d(x, *self.fused_kernel(), stride=self.stride, padding=1)
```

Apply the same change in spanplus_arch.Conv3XC. `load_net` already calls `net.eval()` after `load_weights`, and
`Conv3XC.train(False)` then refreshes `eval_conv`.

Verified (p12): span, span_fast and spanplus(upsampler=ps) export and match ORT at 20x20, 32x24, 27x41 and 2x24x32 with
max error at most 9.5e-7. Default spanplus also needs M-03 (DySample).

Test: parametrize `test_convert_main_exports_verified_onnx_on_cpu` over `span`, `span_fast`, `spanplus` together with
M-02's multi-shape check. Numerics, checkpoint and config: unchanged.

### M-02 (P1): one-shape verification of a dynamic contract allows silently wrong exports

`to_onnx` declares batch, H and W dynamic (convert.py:260-268). `assert_verify` (convert.py:204-231) compares torch and
ORT only at the trace input, 1x3x20x20. Each arch below was exported exactly as convert.py does (opset 17,
`dynamo=False`) and compared at 20x20, 32x24, 27x41 and 2x24x32 (p6, `export_results.jsonl`):

* **Silently wrong pixels** (the trace shape passes, so convert.py reports "successfully verified"):
  * DySample variants: realplksr dysample 5.76, mosrv2 dysample 0.087.
  * 7 window transformers at batch 2: grformer 0.18, microsr 0.12, rgt_s 0.10, srformer_light 0.058, dat_s 0.057,
    swinir_small 0.042, hat_s 0.034.
  * swinir_small at 1x1 outputs NaN.
* **ORT runtime errors at other shapes**: cugan (odd sizes), plainusr (every non-trace shape), dctlsa (batch 2), hma
  (every shape, including the trace shape, traced at 32).

**Fix (convert.py, S):** verify at a small fixed set of shapes. Use the trace shape, a rectangular shape of a different
size with H≠W (for example trace + (12, 4)), and a batch-2 copy of the trace shape. Static exports keep only the trace
shape. A shape that fails in eager torch is skipped with a warning, because the arch does not support it. A shape that
passes in torch but fails or mismatches in ORT raises:

```
RuntimeError: dynamic export is wrong at 2x3x20x20 (max err ...): this arch/batch is not export-safe; use -static C H W
```

Keep the existing tolerance (rtol 1e-2, atol 1e-3) and the non-finite rejection.

Root-cause fixes stay with the arch lane. For the batch freeze, `window_reverse` should use
`windows.view(-1, h // ws, w // ws, ws, ws, c)` instead of `int(...)`, and the same in every copy. Until then the
verification turns silent corruption into a clear error.

Test: a parametrized CPU export test over about 12 representative archs (compact, esrgan, realplksr, omnisr, span,
spanplus, swinir_small, hat_s, mosrv2, safmn -static, aether_small, cugan) at 4 shapes. Mark it slow; it takes about
2 minutes on CPU in total.

### M-03 (P1): F07 DySample normalizer (exact fix), plus a per-forward blocking copy

`arch_util.py:114`: `normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device)` is a trace-time constant.
Verified fix (p13):

```python
if torch.onnx.is_in_onnx_export():   # shape-derived, stays dynamic in the graph
    normalizer = torch._shape_as_tensor(offset)[[4, 3]].to(device=x.device, dtype=x.dtype).view(1, 2, 1, 1, 1)
else:
    normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
```

Results: eager output is `torch.equal` to the current code. ORT matches at all 4 shapes for realplksr(dysample)
(at most 1.9e-5), spanplus with M-01 (8.2e-8) and mosrv2(dysample) (3.6e-7). `torch._shape_as_tensor` is what
torchvision uses for ONNX-traced shapes.

Speed side (source-verified; the GPU effect is a hypothesis): `torch.tensor(list, device=cuda)` builds the tensor on
the CPU and copies it with `non_blocking=False`, which is a stream synchronize. In eager training (`compile = false`)
that is one host sync per DySample forward. It is used by spanplus (default upsampler), realplksr/esc/moesr/mosrv2 when
DySample is selected, and esc_real. A sync-free eager form with the same bits is
`torch.stack((x.new_full((), W), x.new_full((), H))).view(1, 2, 1, 1, 1)` (a fill kernel, no H2D copy). The profiling
lane should count `cudaStreamSynchronize` per step on spanplus before and after. With `torch.compile` the constant is
folded and nothing changes.

### M-04 (P2): exact on-device metric path for validation

Current flow per image (image.py:342-370):

1. `tensor2img([sr])` copies GPU to CPU and converts to uint8 BGR, then `tensor2img([gt])`.
2. PSNR and SSIM run in CPU numpy float64.
3. `img2tensor` converts each image back to RGB float and uploads it again, once per neural metric (4 host-to-device
   copies per image).

Measured cost split, 7 pairs from `/home/korvin/apps/data/val` (GT 512², x4), CPU (p1, first and least contended run):

| stage | ms per image | note |
|---|---:|---|
| span inference (CPU, fp32) | 39.5 | GPU estimate 2 to 5 ms |
| tensor2img sr / gt | 2.1 / 0.7 | |
| PSNR (numpy) | 1.8 | |
| **SSIM (numpy, cv2.filter2D 11x11 float64)** | **59.3** | 48 to 217 ms under load; on GPU the CPU still does this work |
| DISTS (CPU net) | 585 | GPU estimate about 10 ms (2 x 80 GMAC at TF32) |
| TOPIQ (CPU net) | 259 | GPU estimate about 5 ms |
| PNG write (imwrite) | 14 | |
| build DISTS / TOPIQ (per event) | 541 / 271 | see M-05 |

GPU estimate for one validation event on this set (SPAN, RTX 4090): about 1.0 s rebuilding metric nets, 0.35 to 0.42 s
CPU SSIM, about 0.1 s PNG, under 0.1 s for everything else. That is about 1.6 s, of which roughly 85 % is waste removed
by M-04 and M-05. For a 100-image 1024² validation set, CPU SSIM alone is about 100 x 4 x 50 ms, roughly 20 s per event.
With `val_freq = 1000` at about 0.1 s per step, that is about 20 % of training wall time; on-device SSIM brings it to
about 1 s.

**Exactness proof (p3):** the proposed tensor path reproduces the protocol.

* `q = sr.float().clamp(0, 1).mul(255).round()` equals `tensor2img` exactly (asserted on 7 images with added noise and
  out-of-range values).
* PSNR in float64 on `q`: |ΔPSNR| at most 3.6e-15 dB, because integer squared errors sum exactly.
* SSIM in float64 with the same `cv2.getGaussianKernel(11, 1.5)` window, valid conv: |ΔSSIM| at most 2.5e-14.
* Separable, with the 5 statistics batched in one depthwise conv pass: |ΔSSIM| at most 3.0e-14.
* Y channel: replicate `to_y_channel`'s dtype sequence (float32 /255, float64 dot with BT.601 weights, +16, /255,
  float32, x255).

This was checked for crop_border 0 and 4, RGB and Y. On CPU the separable float64 torch SSIM (38 ms) also beats cv2
(48 ms), and `cv2.sepFilter2D` is about 1.9x faster than `filter2D` (|Δ| 9e-15, p10). So CPU-only runs improve too.

**Change (calculate.py and image.validate, M):**

* `quantize(t)` gives a float32 0..255 RGB NCHW tensor on the device of `sr`.
* `calculate_psnr`/`calculate_ssim` get tensor implementations used by `build_metric`. Keep the numpy functions as the
  reference and test the tensor path against `REFERENCE`/`SYNTHETIC_REFERENCE` in tests/test_metrics.py at rel 1e-12.
* DISTS/TOPIQ take `q / 255` directly, which is bit-identical to today's `uint8 / 255` input.
* `validate` builds `q_sr` (and `q_gt` only if metrics exist), sums metric values on device, and syncs once per image
  or once at the end. Convert to uint8 BGR on the host only when saving an image or TB (`q.to(torch.uint8)`: 1 byte per
  pixel instead of 4).
* If nothing consumes it (no `save_img`, no TB, no metrics), do not convert at all.

Numerics, checkpoint and config: none. Metric values match the logs of the current version (to 1e-14).

### M-05 (P2): metric nets are rebuilt at every validation event

`base.validation` calls `release_metric_nets()` in `finally` (base.py:187), so the next event rebuilds both nets (p2):

| build | time | peak host RSS | params kept |
|---|---:|---:|---:|
| `dists_loss(as_loss=False)`, current | 602 ms | **+1073 MiB** | 14.7 M |
| `models.vgg16(weights="DEFAULT")` alone | 784 ms | +1057 MiB | 138.4 M (124 M of them classifier, discarded) |
| VGG16 features only (meta skeleton plus `features.*` from the cached file) | 162 ms | +587 MiB | 14.7 M |
| `topiq()` | 270 to 378 ms | +302 MiB | 36.0 M |

The full classifier is randomly initialised and then overwritten with 528 MB of weights, only to be thrown away. This
happens on every validation, in the training process.

**Fix (S):**

* (a) `dists_loss.__init__`: build `vgg16(weights=None)` under `torch.device("meta")`, take `.features.to_empty("cpu")`
  and load only the `features.*` keys of the hub file. The weights are identical. This also helps the training DISTS
  loss build in the losses lane.
* (b) Lifecycle: `release_metric_nets()` moves the cached nets to CPU (`.to("cpu")`, about 200 MB host) instead of
  dropping them. `_metric_net` moves them back to the run device at the next event (about 200 MB H2D, roughly 20 ms).
  VRAM is still freed between events, which was the original intent. The rebuild and the 1 GiB spike disappear. Key
  the cache by name only and move to the device as needed.

Numerics: none. Test: extend `test_validation_releases_metric_nets` with "rebuilt 0 times across 2 events; parameters
on CPU between events".

### M-06 (P2): a val dataset without `name` crashes training after val_freq updates

`image.validate` reads `ds["name"]` (image.py:325). `name` is not required for datasets (config.py:316-321 sets only
phase and scale). `test.py:324` reads it too. Reproduced with `train.py` (CPU, 2 updates, `val_freq = 1`):
`KeyError: 'name'` at image.py:325, exit 1. With the template `val_freq = 1000`, the crash happens after 1000 updates.
Validation runs before the periodic save of the same update (train.py:258-265), so that checkpoint is lost too.
train.py:119 already logs `ds_cfg.get('name', phase)`, which shows the intended fallback.

**Fix (S):** `ds.setdefault("name", phase)` in `config._derive`. Alternatively, raise a ConfigError in `validate_train`
for val/test datasets without a name. The default is friendlier. Also reject two val datasets with the same `name`:
best tracking (keyed by dataset name) and TB scalars would otherwise merge. This overlaps Codex F20 ("missing
validation/test dataset names").

Related, owner sign-off needed (R1.4 family): any validation crash, such as an OOM on a large val image, kills the run
and loses the same-update save. Saving before validating when both are due would make that loss impossible. Do not
change this without sign-off.

### M-07 (P2): ONNX input and output share dim_param names (and the labels are swapped)

convert.py:262-265 has `"input": {0: "batch_size", 2: "width", 3: "height"}` and the same for `"output"`. The same
dim_param string tells the runtime "these dimensions are equal". That is false for scale > 1, and dim 2 of NCHW is the
height.

Reproduced (p9 vs p9b): the fp16 omnisr graph fails in ORT with

```
Shape mismatch attempting to re-use buffer. {1,3,32,24} != {1,3,128,96}. Validate usage of dim_value ... and dim_param
(all values with the same string should equate to the same size)
```

With distinct output names it runs (max error 4.3e-4). Other runtimes, such as TensorRT shape profiles and planners
that trust dim_params, are exposed to the same contract violation.

**Fix (S):** input `{0: "batch", 2: "height", 3: "width"}`, output `{0: "batch", 2: "out_height", 3: "out_width"}`.
The competitor has the same bug (`convert_to_onnx.py:425-431`).

### M-08 (P2): conversion memory for large outputs (tensor2img and the stitched SR)

`tensor2img` (img_util.py:147-174) makes clamp, normalise, transpose/cvtColor, x255 and round copies in float32. For a
4320x7680 output it measured 803 ms and **+1142 MiB** peak over the 380 MiB input. In-place
`t[0].clamp_(0, 1).mul_(255).round_().to(torch.uint8)` followed by `.permute(1, 2, 0).flip(-1)` took 162 ms with no
extra peak, and the result is `np.array_equal` to `tensor2img` (p11). `infer_tiled` stitches the fp32 output on CPU
(tiling.py:35); 8K x4 is 5.9 GiB per the docstring.

Quantisation is elementwise, so stitching quantised uint8 tile cores gives the same bytes at a quarter of the host
memory and a quarter of the device-to-host traffic. Neural metrics then use `q / 255`, which is exactly today's input.

**Fix (S or M):** use the in-place quantisation of M-04 for every image output. Optionally, `infer_tiled(...,
out_dtype=torch.uint8)` quantises each core on the device before the copy. Numerics: none.

### M-09 (P2, speed): SPAN family fuses 20 kernels on every eval forward

Profiler count of compute ops in one span eval forward (p8 follow-up): **272 with on-the-fly fusion vs 52 with the
cached `eval_conv`**. That is 5.2x the kernel launches, and the output is identical (max|Δ| 0). It applies to every
validation forward, every test.py image and every tile of tiled inference. On GPU a light SPAN at tile or val sizes is
launch-bound, so this plausibly costs 1.5 to 3x eval time (hypothesis, to be measured by the profiling lane).

The on-the-fly form exists so that weights changed in eval mode are never stale; the EMA module is permanently in eval
and is updated in place.

Exact alternative:

* Use `eval_conv` in eval forward.
* Refresh it where eval weights can have changed. `eval_weights` enters with `self.ema.module.eval()` (Conv3XC.train
  override, then `update_params`). The non-EMA branch already calls `net.eval()`. test.py and convert.py call `eval()`
  after loading.

Coordinate this with the arch lane, which owns the SPAN eval-cache decision. Test: EMA update, then validate; the
result must equal the on-the-fly result.

### M-10 (P2): test.py inference precision and pipelining

* `image.validate` never enters autocast. In test.py `use_amp` only switches the model to channels_last
  (image.py:46). The test templates carry `#use_amp = true`, which implies fp16 inference that does not happen. On a
  4090, bf16 or fp16 tensor cores give roughly 2x over TF32 conv. For transformers the gain is larger, because fp32
  matmul runs at "highest" precision unless `fast_matmul` is set (hypothesis, for the GPU lane).
* Fix: when `not self.is_train`, wrap inference in
  `torch.autocast(self.device.type, dtype=bf16 if bfloat16 else fp16, enabled=use_amp)`. Training validation stays
  fp32, so metrics stay comparable across runs. This changes inference pixels only on opt-in (fp16 can overflow in some
  transformers; bf16 is the safer default).
* Val/test loaders use 0 workers and batch 1 (data/__init__.py:51), so PNG decode, inference and PNG encode (14 ms per
  512², estimated 0.3 to 0.4 s per 4K output) run in series. Honour `num_worker_per_gpu` for the test phase, and add a
  one-thread writer (`ThreadPoolExecutor(1)`; `cv2.imencode` releases the GIL) to overlap encoding with the GPU. This
  matters for light models on large outputs (hypothesis on GPU; the encode cost is measured).

### M-11 (P3, hypothesis): cuDNN autotuning on every new shape

runtime.py:91-93 sets `cudnn.benchmark = True` and `benchmark_limit = 0`, so every algorithm is tried. Each new input
shape autotunes every distinct conv of G, DISTS (13 convs) and TOPIQ (about 60 convs). Training validation pays this
once per distinct val image size, and the cache persists in the process. test.py over a heterogeneous photo set pays it
on every image. Tiled inference uses one shape, so it is unaffected.

Measure: test.py on N distinct-size images with benchmark on and off, and with limit 0 and 10. If confirmed, set
`benchmark_limit` to the default 10, or enable benchmark in test.py only when `tile > 0`.

### M-12 (P3): F22 TOPIQ `need_weights=False`

This is a free two-line change (topiq.py:36 and :68). Parity is 1.2e-7 on the 7 real pairs. The peak memory claim does
not hold up to 4K (see the table in §2): at 3840x2160 the peak is 5.8 GiB either way. The ResNet50 pyramids of both
images are held through gated pooling, whose concat at stride 4 is 3 x 256 channels. Attention (8040 tokens, about
1 GiB per matrix) stays under that peak. A real TOPIQ memory fix would pool each level as it is produced and drop the
full-resolution features. That is exact but a medium-sized rewrite, and only worth it if 4K validation with TOPIQ is a
real use case.

### M-13 (P3): DISTS metric keeps both full pyramids

`dists_loss.forward` (dists_loss.py:135-158) runs `forward_once` twice, keeping 6 feature maps per image, then reduces.
Streaming per stage (run x and y through stage k, accumulate S1/S2, drop the maps) gives the same ops in the same order:
1080p peak 2078 to 1491 MiB (-28 %), value identical to 8 digits (p5). Use it for the metric path only (under
inference_mode). The training loss keeps activations for backward anyway. At 4K this is roughly 8.3 to 6 GiB, which
matters after tiled inference, since tiling bounds G but not the metrics.

### M-14 (P3): metric direction is unvalidated

base.py:192-197: `init = -inf if better == "higher" else inf`, so `"Higher"`, `"hihger"` or anything else means lower
is better. A DISTS metric without `better = "lower"` (the templates have it only as a comment) tracks the worst value.

**Fix (S):** a `DIRECTION = {"calculate_psnr": "higher", "calculate_ssim": "higher", "calculate_dists": "lower",
"calculate_topiq": "higher"}` lookup next to `METRICS`. `build_metric` uses it as the default, raises ConfigError for a
value other than higher/lower, and warns when the value contradicts the known direction. `update_best` reads the
resolved value. Only the logged and stored best changes; no model selection depends on it.

### M-15 (P3): F21 naming collisions and the TB tag

**Fix (S):** identity = dataset name plus `Path(lq_path).relative_to(dataroot_lq).with_suffix("")`, falling back to the
stem for LMDB keys.

* Training: `vis/<dataset>/<rel>/<stem>_<iter>.png`, LQ beside it.
* Test: `vis/<dataset>/<rel parent>/<stem>_<suffix>.png`.
* TB: tag `f"{dataset}/{rel}"` with `global_step=iter`, so the TB slider works and the number of tags stops growing.
* Reject two files with the same stem and different extensions in one directory at dataset build.

Output layout changes only. The competitor (`sr_model.py:1284-1319`) already uses dataset plus relative path for
multiple validation datasets.

### M-16 (P3): `-fp16` and `--fulloptimization`

* `-fp16` prints that it is disabled and returns 0 (convert.py:321-340). ORT ships `onnxruntime.transformers.float16`
  (ORT is already a dependency). `convert_float_to_float16(m, keep_io_types=True)` followed by
  `OnnxModel(m).topological_sort()` passes `onnx.checker` and runs on the CPU EP (p9). Max error vs the fp32 torch
  output at 32x24 (with M-07's names): compact 1.9e-5, omnisr 4.3e-4, esrgan 1.6e-3, realplksr 6.3e-3. File size is
  halved.
* Implement it with an fp16 tolerance (for example max 2e-2, mean 2e-3, at the M-02 shapes). Until then, `-fp16` must
  raise ConfigError.
* `--fulloptimization` raises ValueError after the fp32 export is written (exit 1 with a traceback). Reject it at parse
  time.

### M-17 (P3): export capability gaps (non-silent)

Failing at export with obscure errors (p6, convert.py settings):

* atd_light: `Sort` op.
* hit_srf: model larger than the 2 GiB protobuf limit (dense position-bias constants, related to F09).
* flexnet, metaflexnet: `aten::rms_norm` at opset 17. With a custom symbolic the failure moves to einops dynamic
  reshapes, so this is not a quick fix.
* esc_light, esc_fp: padding trace error.
* catanet, lmlt_tiny: fail even with -static.
* hma: ORT LayerNorm error at every shape.
* dct.
* safmn, light_safmnpp, msdan: dynamic `adaptive_max_pool2d` output sizes. **-static works** (safmn and msdan verified,
  exit 0, "successfully verified").

The up-front table `net_error = ["craft", "ditn"]` (convert.py:239) is incomplete. Extend the explicit lookup with a
reason and a hint (static-only: safmn, light_safmnpp, msdan; unsupported: the rest), without adding a registry. The
dynamo exporter could fix some of these, but it needs `onnxscript` and a per-arch evaluation.

Exports verified correct at all 4 shapes: compact, plksr_tiny, realplksr(_s), omnisr, esrgan, mosrv2, man_tiny, moesr,
eimn, sebica, ninasr_b0, cfsr, asid, hasn, krgn, aether_small, rcan. drct_s passes the tolerance at batch 2, but with a
2.1e-3 error, about 1000x the others.

---

## 4. Verified correct (no change)

* **PSNR/SSIM conventions** match BasicSR (float64; Y = BT.601 with +16 offsets, as MATLAB `rgb2ycbcr` does; SSIM
  Gaussian 11/1.5, valid window, per-channel mean). The REFERENCE tests pin them at rel 1e-12.
  * Protocol note: `crop_border` defaults to 4 for every scale. Papers use `crop_border = scale` with
    `test_y_channel = true`. This only matters for paper comparability; changing it would shift every logged value.
* **Tiling**: cores tile [0, n) exactly once, windows are fixed tile² (one cuDNN shape), and the output is exact when
  RF ≤ halo (tests). The compute overhead is (tile / (tile − 2·halo))², about 1.42x for 200/16.
  * Batching the equal-shaped windows would help launch-bound light archs on GPU, but multiplies activation memory,
    which is what tiling exists to bound. If wanted, make it opt-in (hypothesis).
* **eval_weights**: SF snapshot and restore, per-module flags, `observers_frozen`. Metric nets are eval with
  requires_grad off, and inference runs under `inference_mode`.
* **Caching key** `(name, device)` normalises `cuda` to `cuda:N` correctly.

---

## 5. Export matrix (default constructors, scale 4, trace 1x3x20x20, opset 17, legacy exporter)

Shapes are 20x20, 32x24, 27x41 and 2x3x24x32. "ok" means rtol 1e-2 / atol 1e-3.

| status | archs |
|---|---|
| ok at all shapes | compact, plksr_tiny, realplksr, realplksr_s, omnisr, esrgan, mosrv2, man_tiny, moesr, eimn, sebica, ninasr_b0, cfsr, asid, hasn, krgn, aether_small, rcan (drct_s: 2.1e-3 at batch 2) |
| silent wrong at batch 2 | swinir_small 4.2e-2, hat_s 3.4e-2, grformer 1.8e-1, srformer_light 5.8e-2, rgt_s 1.0e-1, dat_s 5.7e-2, microsr 1.2e-1 |
| silent wrong at other H/W | realplksr(dysample) 5.76, mosrv2(dysample) 8.7e-2, spanplus (after M-01) 4.4e-2; swinir_small 1x1: NaN |
| ORT error at other shapes | cugan (27x41), plainusr (all), dctlsa (batch 2), hma (all, traced at 32) |
| export fails | span, span_fast, spanplus (M-01); safmn, light_safmnpp, msdan (dynamic only); catanet, lmlt_tiny (also static); atd_light; esc_light, esc_fp; hit_srf; flexnet, metaflexnet; dct |

---

## 6. Competitor (traiNNer-redux) notes

* **Metrics**: `calculate_dists`/`calculate_topiq` build the network on every call (worse than neosr). Its
  `calculate_psnr_pt`/`ssim_pt` skip uint8 quantisation and add `mse + 1e-8`, so they do not reproduce the protocol.
  M-04 is exact; do not port theirs. It has an LPIPS metric, a candidate for the feature stage.
* **Validation**: on CUDA OOM it falls back to tile 256, and on any exception it logs and `continue`s. Skipped images
  then bias the mean and the best tracking. Do not port. If anything, convert an OOM into a clear "set val.tile" error
  (M-06 context).
* **Export**: it has dynamo, fp16/bf16 (onnxconverter-common, which is unmaintained) and onnxslim. It also verifies a
  single shape, only warns on a mismatch, and has the same shared-dim_param bug (M-07). With M-02, M-07 and M-16,
  neosr's converter would be stricter than the competitor's.

---

## 7. Suggested order of work

1. M-01, M-03 export part, M-07, M-02, as one convert.py + arch step, with a parametrized export test (none of it
   affects training).
2. M-05 and M-04 + M-08 (validation path; tests pin values against REFERENCE).
3. M-06 (config) and M-14 (`better`).
4. Profiling lane to confirm M-03 (sync), M-09, M-10 and M-11 on GPU, then implement whichever pays off.
5. M-12, M-13, M-15, M-16, M-17 as cleanup.
