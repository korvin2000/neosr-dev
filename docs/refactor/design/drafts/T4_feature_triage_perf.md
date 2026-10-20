# T4 — Feature triage and performance (design)

Repo: `/home/korvin/apps/neosr-dev` (read-only analysis). Evidence base: options/*.toml usage counts (135 files: 87 `model_type="image"`, 42 `"otf"`, dataset types only `paired`/`single`/`otf`), the three prior reviews, the two real logs in `experiments/train_esrgan/`, synthetic benchmarks run on this machine (RTX 4090, torch 2.9.1+cu128; scripts and raw output in `scratchpad/design/bench_*.py`, `bench_results*.txt`), and the parallel Codex T4 opinion (`scratchpad/design/t4_answer.txt`), used as evidence and checked against the code (§8). Note: `options/train_dctlsa.toml:88` does not parse (duplicate `type` in `[train.ldl_opt]`, missing `[train.fdl_opt]` header; verified with `tomllib`), so the census is 134 runnable configs.

Policy used for every decision: **used by a config, or correct, or cheap to keep → keep+fix; unused and broken and expensive → delete.** Deleted features fail loudly: their config keys raise `ValueError("<key> was removed in ...; see ...")` from a single `REMOVED_KEYS` table in `neosr/utils/options.py`, so no config silently changes meaning.

## 1. Decisions (feature by feature)

| Feature | Decision | Rationale (one line) | Evidence |
|---|---|---|---|
| `models/default.py` (1200 l) + `simple_otf.py`, `eqr_otf.py`, `pancam_otf.py` | **DELETE** | Second trainer hierarchy with 0 configs and known crashes; everything they do that matters (OTF queue, EMA, save/load) exists in `base.py`/`image.py`/`otf.py`. | `default.py:538-539` prints the loss dict every print_freq; Codex A31 (10-value unpack of a 2-value `wavelet_guided`); `simple_otf.py:88` passes HR `gt_size` into an LQ-size crop API; `pancam_otf.py:56` `linear_blur` undefined unless `kernel1final` present. No `model_type` other than image/otf in options/. |
| datasets `eqrotf`, `pancamotf`, `pancamsingle`, `pcpaired`, `simple_otf`; archs `eqrsrgan`, `pancam` | **DELETE** | Only consumers are the deleted models; use legacy `opt['io_backend']`, `gt_size` keys that the current config schema no longer produces. | `pancamsingle_dataset.py:50-51` slices the alpha channel *after* dropping it (takes a colour channel as distortion); 0 configs use these dataset/arch types. Re-adding a "distortion channel" later = a `extra_channels` option in `paired`/`otf` datasets, not a parallel hierarchy. |
| `models/image.py.helaman` | **DELETE** | Stale 89-line-diff copy of `image.py` with a QAT-era `optimize_parameters`; not importable (`.helaman` suffix). | `diff` shows no accumulation/AMP/GAN handling. |
| `data/otfgray_dataset.py` | **DELETE** | Empty file (1 blank line) picked up by the `_dataset.py` scan. | `cat -n` → 1 line. |
| `aether` arch (+ `aether_core.py`, `common/v4`, `common/v5`) and QAT | **KEEP arch, register variants (owned by T5 `T5_aether_qat.md`); the trainer carries no arch-specific code** | 6 configs in `options/aether/` but unrunnable today: `aether_small/medium/large` are module-level lambdas never registered; the QAT block in `train.py` is broken at three points. T4 position: `train.py`/`image.py` lose the QAT block, `_init_ema`, the synthetic calibration loader and the `common/` `sys.path` hack (T5 D2 agrees); whatever QAT contract T5 defines must live in the arch/a QAT helper, not as `if enable_qat` branches in the training loop. | `archs/aether_arch.py:645-653` lambdas, no `ARCH_REGISTRY.register`; `train.py:438` calls `calibrate(loader, iters=)` vs `aether_core.py:584 def calibrate(self, calib_loader)`; `image.py:74-86 _init_ema` replaces the `AveragedModel` with a plain net → `update_parameters` at `image.py:768` raises; `train.py:431 prepare_qat` runs after optimizers were built on the pre-swap parameters. `aether_arch.py == common/v4/aether_arch.py` (cmp), `aether_core.py` ≈ `common/v5`. `train.py:677` injects `common/` into `sys.path`. Single source `neosr/archs/aether_arch.py` (byte-identical to `common/v4`), delete `aether_core.py` + `common/`. QAT keys stay valid under T5's design, so they are *not* in the removed-key table. |
| FSAM / `sam`, `sam_init` | **DELETE** | 0 configs; base optimizer not checkpointed; the SAM closure re-runs the discriminator step and the counters; explicitly incompatible with AMP and accumulation. | `fsam.py:30` separate `base_optimizer` never in `self.optimizers` (not in `.state`); `image.py:746` `sam_optimizer_g.step(self.closure, ...)` re-enters `closure` (D backward again, `n_accumulated` again); `image.py:301-310` warns/exits for AMP/accum. `optimizers/fsam.py` (115 l) removed. |
| ECO (`eco`, `eco_schedule`, `eco_iters`, `eco_init`) | **DELETE** | 0 configs (owner has it commented out); mutates `self.gt` in place and doubles generator forwards; re-addable later as a 30-line target provider in the context-based loss design (T-loss topic). | `image.py:462-494`: `self.gt = ...` overwritten, extra `self.net_g(self.lq)` forward in `no_grad` then second forward. |
| Wavelet-guided GAN (`wavelet_guided`, `wavelet_init`) | **KEEP + FIX** | Used by 164 configs (default template) and by the owner's config; currently inconsistent (D trained on HF bands, G queried with RGB) and non-differentiable; filters rebuilt per call. | `wavelet_guided.py:386` `@torch.no_grad`; `:388-398` rebuilds `pywt.Wavelet` + `SWTForward` every call; `image.py:539-540` computed under `no_grad`, `:611` G uses `net_d(self.output)` (RGB) while `:657/672` D sees HF. Fix: an `AdvView` module built once (filters as buffers, differentiable), applied identically for G and D terms (see §3.7). |
| Schedule-free `adan_sf` | **KEEP + FIX** | 170/170 train configs use it. | Warmup never reaches the kernel: `adan_sf.py:260` computes `lr` but `:315` passes `group["lr"]`; SF `eval()` ordering in `base.py:343-356` (state copied before `optimizer_g.eval()`), `image.py:851/889` (`is_train=False` set before the `sf_optim_g and is_train` test). |
| `adamw_sf` | **KEEP** | 0 configs but 208 lines, official reference implementation, same `train()/eval()` contract; the simplest SF optimizer for users. | Fix A33 (`schedule_free=False` ignored; beta1=0 division). |
| `soap_sf` | **DELETE** | 0 configs, 917 lines, broken state keys, `empty_cache()+gc.collect()` helper, needs grad-clip off. | Codex A25 (`(Parameter,index)` state keys; SF eval misses `z`); `soap_sf.py:302-304 clean()`; `image.py:374-377` special-casing. |
| `adamw_win` | **DELETE** | 0 configs, divides by lr (crash at lr→0). | `adamw_win.py:188` (Codex A33). |
| `adan` (non-SF) | **KEEP (fix one sync)** | 0 configs, but it is the non-SF control for the dominant optimizer family, and it is *not* numerically identical to `adan_sf(schedule_free=False)` (max |Δp| = 2e-4 after 20 steps at lr 1e-3; `no_prox`/defaults differ), so an alias would silently change the algorithm. Unify only after reconciling both against the official Adan. | `adan.py:231 .item()` on the clip factor → keep as tensor. Alias-and-delete was my first choice and is rejected on that measurement (Codex T4 also argued keep). |
| `pytorch_optimizer` dependency | **DELETE** | Only `default.py` imports it. | `grep pytorch_optimizer` → `default.py:23-269` only; `pyproject.toml` dep removed. |
| Discriminator EMA (`net_d_ema`) | **DELETE** | Doubles D memory + an extra foreach-lerp per step (measured ≈1–2 %, within noise), and the EMA copy is saved as `net_d`, so resume continues from averaged D weights; no supported use. | `image.py:103-108, 769-770, 998-999`; `ema` key continues to apply to G only. |
| Augmentations `mixup/cutmix/resizemix/cutblur` | **KEEP + FIX (rewrite, 120 lines)** | 168 configs use the 5-aug template; but every LQ batch is bicubic up+down-sampled even when "none" is chosen (silent blur of the input for 168 configs), mixup mixes LQ with *GT*, multi-aug ignores weights, `aug_prob=None` crashes. | `augmentations.py:250-253` and `:287-290` unconditional round trip; `:38-41` `img_ = img_gt[rand_index]` used for `img_lq`; `:261` `random.choices(weighted)` without `weights=`; `:241 len(prob)`. New design in §3.8: boxes at native LQ/GT resolution scaled by `scale`; only cutblur resamples the pasted GT crop. |
| `PrefetchGenerator` / `PrefetchDataLoader` | **DELETE** | Never instantiated. | `grep PrefetchDataLoader` → only its definition `prefetch_dataloader.py:9-65`. |
| `CUDAPrefetcher` | **KEEP, 25-line fix** | Side-stream prefetch is reference practice and free once correct (H2D ≈ 50 MB/iter for OTF ≈ 2 ms, 1 % of an iteration); today it lacks `record_stream` (allocator reuse race) and runs on pageable memory (`pin_memory=False`), so `non_blocking` is a synchronous copy. | `prefetch_dataloader.py:97-108` no `record_stream`; `data/__init__.py:141 pin_memory` default False → `non_blocking` is a synchronous pageable copy. |
| LMDB backend (`file_client.py`, `lmdb_util.py`, `create_lmdb.py`, `data_util.py` lmdb helpers) | **KEEP as is** | Upstream datasets ship as `.lmdb` (readme:293-295); ~400 stable lines; not on the hot path. Only fix the dead retry loop in `paired_dataset.py:122-141`. | 0 configs today but the loader auto-detects by folder suffix (`paired_dataset.py:63`), so no config change is needed to use it. |
| Throughput kill-switch | **DELETE** | Exits with code 0 when `it/s/accumulate < 0.5` — large nets, compile warm-up, and accumulation all trigger it. | `train.py:577-584`. |
| `convert.py` (ONNX) | **KEEP + CLEAN** | User-facing deliverable (chaiNNer/VapourSynth); but 40% of the file is commented-out code and two flags that cannot work. | Delete `convert.py:16-67` (rt4ksr), `:243-258` (fp16), `:263-294` (onnxsim); drop `-fp16`/`-fulloptimization` from `options.py:132-146`; build+export on CPU explicitly (D5, `convert.py:85-87` loads to cuda, `:198` CPU device context cannot move it); use one shared `load_checkpoint()` with `base.load_network` (strip `module.`/`_orig_mod.`, explicit `param_key` precedence). Installed torch 2.9 already defaults `torch.onnx.export(dynamo=True)` (verified via `inspect.signature`), while `convert.py:201-208` passes legacy `dynamic_axes`/`do_constant_folding`: make the exporter explicit (`--dynamo/--no-dynamo`, `dynamic_shapes` on the dynamo path, `dynamic_axes` only on the legacy path, `verify=True`). |
| `test.py` | **KEEP (minor)** | 73 lines, used by 45 `test_*.toml`. | Remove `test.py:19 set_default_device`; pass a label instead of `current_iter=opt["name"]` (`test.py:55`). |
| MACs via `torchprofile` (`train.py:82-100, 403-412`) | **REPLACE (6 lines)** | Optional dep not installed; `torch.utils.flop_counter.FlopCounterMode` is built in. | `input_size` key keeps working. |
| `rng()` / `net_opt()` import-time CLI parsing | out of T4 scope (T-config topic) but a precondition for the dataloader start-method change below | `utils/rng.py:11`, `archs/arch_util.py:16`; 10 module-level `rng()` calls (`grep "rng()"`). |

Net deletion: ≈4,400 lines (`default.py` 1200, legacy models 425, legacy datasets 876, legacy archs 292, `soap_sf` 917, `adamw_win` 226, `fsam` 115, prefetch generator 65, `convert.py` dead blocks ≈110, trainer QAT block ≈70, ECO ≈45, D-EMA ≈15, kill-switch 8) plus ≈2,000 lines of aether duplicates (`aether_core.py` 663, `common/v4` 666, `common/v5` 695) under T5.

## 2. Performance — hot-path changes

Reference workload: `options/train_esrgan2.toml` (esrgan + metagan, bf16, batch 16, 64→256), real logs show **3.2–3.5 it/s** (`experiments/train_esrgan/train_train_esrgan_20251230_120333.log`), 10,607 images → 663 iters/epoch (`..._115003.log:9-12`).

Ranked by measured or expected effect on the owner's workload:
1. **P0** loss backbones (FDL/DISTS/VGG) under bf16 autocast + optional compile: −70…−86 ms of ≈275 ms/it (**+35–45 %**).
2. **P7** `torch.compile` of the generator done right (eager canonical module): **+17 %**, −0.5 GiB.
3. **P11** vectorised Poisson/JPEG in OTF: −35 ms per call, up to −70 ms/it (**+20–30 % for the 42 OTF configs**).
4. **P8/P9/P10** validation: −0.5 s per image per learned metric, tiling 92 → 25 ms per 1024 px image, no `empty_cache` stalls (+110 ms each) — validation 5–20× faster.
5. **P3** persistent workers / pinned memory: −1.6 s per epoch boundary; small for 10k-image sets, large for small sets.
6. **P6** channels_last: already 1.21× in the current code; inputs conversion ≈0 but makes it explicit.
7. **P1/P2/P4/P5** syncs, bf16 scaler, prefetcher `record_stream`, CPU sampler: ≈0 % throughput; correctness/clarity only (measured, see below).

Measured on this machine (synthetic, same shapes; `bench_results*.txt`):

| Variant (esrgan + metagan, bf16, batch 16, 64→256, `adan_sf`, EMA) | ms/it (best of 3 interleaved rounds) | it/s | Source |
|---|---|---|---|
| NCHW model + NCHW inputs (what `use_amp=false` gets) | 181.0 | 5.5 | `bench_results5.txt` F |
| **current code**: channels_last model (`base.py:156-159`), NCHW inputs, 16 syncs/it, GradScaler on bf16, D-EMA | 149.8 | 6.7 | `bench_results5.txt` A |
| + channels_last inputs | 150.3 | 6.7 | B |
| + no per-iteration syncs (device-side loss accumulation) | 152.5 | 6.6 | C |
| + GradScaler disabled for bf16 | 150.9 | 6.6 | D |
| + no discriminator EMA (**proposed trainer**) | 148.5 | 6.7 | E |
| proposed + `torch.compile(net_g, mode="default")` | 137.6 (vs 160.9 same-run eager) | 7.3 | `bench_results3.txt` |
| span (no D), current vs proposed vs NCHW | 16.6 / 16.4 / 18.2 | 60 / 61 / 55 | `bench_results4.txt` |

Profiler (`bench_results3.txt`): current trainer issues **16 `cudaStreamSynchronize` + 16 DtoH copies per iteration** (10 `.item()` in `reduce_loss_dict`, `isnan`, 2× GradScaler `found_inf`, 2× `adan_sf.py:223 .item()` on the global grad norm, EMA); the sync-free variant issues 2 (the two `adan_sf` ones). **Measured cost of those syncs on this machine: ≈0** for both a GPU-bound net (esrgan@16) and a small net (span@16, 60 it/s), because the CPU enqueues the next iteration faster than the GPU drains it. So the sync clean-up is justified by log quality, DDP correctness and hygiene, not by throughput — contrary to what all three prior reviews assumed (Codex C1, my_review "per-iteration GPU syncs").

**Where the owner's real time goes** (`train_esrgan2.toml`, logged 3.2–3.5 it/s = 285–310 ms/it): trainer+G+D ≈ 150 ms (above) + the configured losses ≈ 123 ms (`bench_results6.txt`: `fdl_loss(dinov2)` **117 ms**, `mssim` 3.7, `ldl` 1.7; `consistency_loss` could not even be constructed outside the global default device: `consistency_loss.py:42,132` create CPU tensors → device-mismatch error, see P5) → predicted 3.7 it/s, consistent with the log. The single largest hot-path item is therefore **FDL's DINOv2 backbone running in fp32** because `fdl_loss.py:317 @torch.amp.custom_fwd(cast_inputs=torch.float32)` wraps the *whole* forward, backbone included (`bench_results7.txt`): 127.5 ms → **58.7 ms** with the ViT-B under bf16 autocast and only the FFT/sort head in fp32 → **41.8 ms** with `torch.compile` on the backbone. That alone is −70 to −86 ms of a ≈275 ms iteration (**+30–45 % throughput for the default loss template**, which 85 configs use). The same `custom_fwd(cast_inputs=float32)`-on-the-whole-forward pattern sits on `dists_loss.py:143`, `vgg_perceptual_loss.py:42,150` (`fdl_loss(vgg)` measured 162 ms/it), `ssim_loss.py:111`, `ff_loss.py:117`, `wavelet_guided.py:355`; only the numerically sensitive parts (FFT, SSIM statistics, wavelet filters) need fp32.

Dataset side (measured in this session, not saved to a results file): the owner's GT set is 10,607 PNGs of 256×256 (3.4 GB); `paired.__getitem__` = 3.4 ms → 14 ms/batch with 4 workers, far below the 150 ms iteration. The data path is not the bottleneck for this dataset; it is for OTF (512² crops + GPU degradation with 2× `torch.unique` loops, below).

Micro-benchmarks (`bench_results.txt`, `bench_results2.txt`):

| Item | Current | Changed | Note |
|---|---|---|---|
| Poisson noise, 16×3×512² (`degradations.py:931-943` per-sample `torch.unique` loop) | 41.1 ms | 5.8 ms (histogram, no sync) | OTF calls this up to twice per iteration → up to 70 ms/iter ≈ 25% of a 285 ms iteration |
| OTF pair-pool full shuffle (`otf.py:70-78`) | 0.49 ms | 0.06 ms (random-slot swap) | small; mainly removes 4 redundant `.clone()`s and 2×141 MB gathers |
| Conv step right after `torch.cuda.empty_cache()` (`base.py:447,506,530`; `image.py:807,914`) | 171 ms | 61 ms steady | +110 ms per call; called per val image, per tile and per save |
| Spawned worker start = fresh interpreter importing `neosr.data` (`train.py:682` spawn, `data/__init__.py:142` `persistent_workers=False`) | 1.57 s per epoch boundary | 0 with persistent workers | ≈1% at 663 iters/epoch, ≈10% for a 1k-image dataset |
| `calculate_dists` builds VGG16 + loads weights **per image** (`metrics/calculate.py:201`, same for TOPIQ `:238`) | 518 ms construct + forward | forward only | 50 val images × 2 learned metrics ≈ 50 s wasted per validation |
| Tiled validation 1024 px / 256 tiles, CPU round trip + `empty_cache` per tile (`image.py:786-816`) | 91.9 ms per 1024 px image (16 tiles) | 25.2 ms on GPU, one final `.cpu()` | plus seams: no overlap today |

### P0. Loss backbones under autocast (largest single win for the default loss template)
Current: `fdl_loss.py:317`, `dists_loss.py:143`, `vgg_perceptual_loss.py:42,150` put `@torch.amp.custom_fwd(cast_inputs=torch.float32)` on the whole `forward`, so the pretrained backbone (ViT-B/14, VGG16/19) runs in fp32 for both the SR and the GT branch; the GT branch builds no graph but is not under `no_grad` either (`fdl_loss.py:320`). Change: feature extraction under the caller's autocast (bf16), GT features under `torch.no_grad()`, features cast `.float()` before the FFT/sort/statistics head, which keeps its own `custom_fwd`; optional `compile_losses=true` compiles the backbone sub-modules (`torch.compile(self.model, dynamic=False)`); same shape for `dists_loss`, `vgg_perceptual_loss`, `resnet_perceptual_loss`. Expected: FDL(dinov2) 127.5 → 58.7 → 41.8 ms/it (measured); whole iteration ≈275 → ≈190–205 ms (**+35–45 % it/s** on `train_esrgan2.toml`), peak memory −0.6 GiB. Numerics: bf16 features change the loss value slightly (ViT activations are bf16-stable; verify by comparing loss on 32 fixed pairs fp32 vs bf16: expect relative difference < 1e-2 and identical ranking). Measure: `bench_losses.py` style per-loss fwd+bwd timing plus end-to-end it/s with the real config.

### P1. Per-iteration GPU syncs (hygiene, not throughput)
Current: `base.py:559` `value.mean().item()` for every logged loss every micro-batch (10–14 syncs); `image.py:717` `torch.isnan(l_g_total)` (sync); `image.py:149-153` `GradScaler(enabled=use_amp)` also for bf16 → `grad_scaler.py:356` `found_inf.item()` twice per step; `consistency_loss.py:189` `if cosim < 1e-3`; `ff_loss.py:102,105` `.item()` asserts; `diffjpeg.py:542` per-sample `quality_to_factor(factor[i])` (`diffjpeg.py:60` python comparison on a 0-d CUDA tensor = sync × batch); `degradations.py:931-943` (sync × batch).
Change: (a) `LossMeter` (§3.4): detached losses are stacked into one device buffer, `.tolist()` once per `print_freq` → logged values become window averages (better signal than a single-iteration snapshot); (b) one bounded sync per optimizer *update* (not per micro-batch, not per loss): `torch.isfinite(grad_norm)` on the total norm that `clip_grad_norm_` already computes, skipping the step (and the EMA/scheduler advance) when it is non-finite — this also covers the inf / NaN-gradient cases that `isnan(l_g_total)` misses (Codex A27); GradScaler keeps doing it for fp16; (c) scaler enabled only for fp16; (d) vectorise poisson/jpeg (§3.9); (e) `cosim` branch → `torch.where`; (f) delete the `ff_loss` asserts.
Expected throughput gain: **≈0 % measured** (table) for esrgan@16 and span@16; the value is window-averaged logs, one place to all-reduce under DDP, no `print(losses)`/`None` bug (`base.py:551-552`), and no hidden stalls when a future small-batch/tiny-net config is launch-bound. Also make `adan_sf.py:220-223` keep `clip_global_grad_norm` as a 0-d tensor (`_foreach_mul_` accepts it) to reach 0 syncs/iteration. Measure: `bench_train.py --profile` count of `cudaStreamSynchronize`/`Memcpy DtoH` per iteration (target 0; ≤2 per `print_freq`).

### P2. GradScaler only for fp16
`image.py:149-153`. bf16 needs no loss scaling; the scaler costs two syncs and a `found_inf` kernel per step. `enabled = use_amp and amp_dtype is torch.float16`. Measured ≈0 % (150.9 vs 152.5 ms/it); it removes two syncs and a kernel per step and the `GradScaler`+bf16 combination that confuses readers.

### P3. DataLoader defaults
`data/__init__.py:93-121,141-142`: `num_workers` 4, `prefetch_factor` 2, `pin_memory` False, `persistent_workers` False; `train.py:682` `spawn`. Change: `pin_memory=True`, `persistent_workers=num_workers>0`, `prefetch_factor=4` for `otf` datasets (512² PNG decode) else 2; one argument builder for dist/non-dist (fixes the `UnboundLocalError` at `:99-104`); delete the unreachable block `:135-139` (`psutil`). Keep `spawn` until `rng()`/`net_opt()` import-time parsing is gone (it initialises CUDA in workers via `options.py:207` when `manual_seed` is set, which forbids `fork`); then switch to the platform default. Measure: time between `prefetcher.reset()` and first batch at an epoch boundary (log it), `nvidia-smi dmon -s u` for dips at epoch boundaries.

### P4. CUDAPrefetcher correctness
`prefetch_dataloader.py:97-108`: allocate on side stream, consume on main stream, no `record_stream` → the caching allocator may hand the block to the next `preload()` while the main stream still reads it. Fix in §3.2. Measure: correctness only (run 2k iters with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` vs cached and compare loss traces bitwise with a fixed seed).

### P5. Sampler on CPU, no global default device
`data_sampler.py:40-42` CUDA generator + `randperm(..., device="cuda").tolist()` exists only because of `train.py:320`/`test.py:19`/`convert.py:298` `torch.set_default_device("cuda")`. That global also hides placement bugs (verified without the global default: `metrics/calculate.py:201 dists_loss(as_loss=False)` leaves VGG on CPU and raises a device mismatch; `consistency_loss.py:42,132` build CPU tensors and the loss raises `mat2 is on cpu` — i.e. two of the default template's components only run because of `set_default_device`) and forces `.cuda()` hacks (`otf.py:57,65`, `otf_dataset.py:123-126`). Change: delete the global; `EnlargedSampler` uses `torch.Generator()` (CPU); every module takes an explicit device. Impact: clarity/correctness, ~0 time.

### P6. channels_last policy
`base.py:156-161` converts the *model* to channels_last iff `use_amp`, but inputs stay NCHW (`image.py:445-447`, `otf.py:97,252`), so the first conv of every forward transposes, and `otf.py:265` needs a `.contiguous()` hack; loss modules are converted too (`image.py:180-269`, meaningless). Change: explicit top-level `channels_last` (default = `use_amp`, i.e. unchanged behaviour); `feed_data` converts inputs with the same memory format; losses untouched. Measured: the model-side conversion the code already does is worth 1.21× on esrgan (181 → 150 ms/it) and 1.10× on span; converting the inputs too is ≈0 on esrgan (150.3 vs 149.8) — its value is removing the per-forward layout conversion inside the first conv, the grad-layout warning and the `otf.py:265` hack, and making the policy explicit. Transformer archs must be benchmarked separately (`swinir_small`, `hat_s`, `dat_s`): keep the key per-config.

### P7. torch.compile placement
`base.py:163-164` compiles *before* `load_network` and before EMA, with `mode="max-autotune", dynamic=True`. Verified on torch 2.9.1: loading plain keys into the compiled module raises `RuntimeError` (`_orig_mod.` prefix), the EMA/saved state dicts carry `module._orig_mod.*` keys that `save_network` does not strip (`base.py:347`), so `compile=true` runs produce checkpoints `test.py`/`convert.py` cannot load. Change: `self.net_g` stays the eager canonical module (load/save/EMA); `self.net_g_fwd = torch.compile(self.net_g, mode=opt.compile_mode ("default" | "max-autotune-no-cudagraphs"), dynamic=False)` is used only for the training forward; validation runs eager (shapes vary). Measured: 1.17× (160.9 → 137.6 ms/it, same run) and −0.55 GiB peak with `mode="default"`; first compile ≈1–2 min (inductor cache makes later runs seconds). Measure compile cost separately (first-iteration wall time) and exclude warm-up from it/s.

### P8. Metric network caching
`metrics/calculate.py:201,238`. §3.6: module-level cache keyed by `(metric, device)`, built once, `eval().requires_grad_(False)`, fed GPU tensors directly (quantise to uint8 on GPU to keep the current numeric protocol). Saves 0.5 s × images × learned metrics per validation (table).

### P9. Tiled validation on GPU with overlap
`image.py:772-818`. §3.5: pad guard (`pad < dim` for reflect), overlap `tile_overlap` (default 16 LQ px) with linear blend weights, output channels/dtype taken from the first tile, stitch on GPU, one `.cpu()` at the end; `nondist_validation` keeps visuals on GPU for the learned metrics and converts once for PSNR/SSIM/`imwrite`. Also drop `del self.lq/self.output` + `empty_cache` (`image.py:909-914`). Measure: validation wall time per image; seam check = PSNR(tiled, whole-image) ≥ 50 dB on a conv net.

### P10. empty_cache / gc removal
All six call sites (table). Rationale: the caching allocator already reuses blocks; `expandable_segments` is set in `options.py:201`. Keep no replacement. Measure: iteration time of the first 5 iterations after a save/validation before vs after.

### P11. OTF queue and degradation kernels
`otf.py:70-78` shuffle+gather of the whole pool and four redundant `.clone()` (`:74-78,85-89`; assignment into a slice already copies); `degradations.py:931-943`, `diffjpeg.py:542`. §3.9. Impact: table rows 1–2.

### P12. Logging
`MessageLogger` already runs only at `print_freq` (`train.py:568`); the only per-iteration cost is the sync in P1(a). After P1, logging = one `.tolist()` + `add_scalar` calls per `print_freq`. Nothing else to do.

### Measurement harness (new `scripts/bench_train.py`, ≈60 lines)
`bench_train.py -opt cfg [--synthetic] --warmup 100 --iters 300 [--profile 20]`: builds the model via `build_model`, feeds either the real loader or random pinned batches of the config's shapes, times `--iters` with `torch.cuda.synchronize()` at the boundaries, prints it/s, ms/it, `max_memory_allocated`, and with `--profile` wraps 20 iterations in `torch.profiler` and prints the count of sync events (`cudaStreamSynchronize`, `cudaDeviceSynchronize`, `Memcpy DtoH`) and total GPU-idle gap. Every P-item above is accepted only if (1) it/s does not regress on the three reference archetypes (`esrgan`, `span`, `swinir_small`) and (2) with `manual_seed` set the printed losses at the first two `print_freq` points are bitwise equal before/after for pure-performance changes (P1a, P2, P4, P5, P10, P11-queue), or within fp tolerance where the arithmetic order changes (P6, P7, P11-poisson).

## 3. Code skeletons

### 3.1 `neosr/data/__init__.py` — one loader builder
```python
def build_dataloader(dataset, dataset_opt, *, num_gpu=1, dist=False, sampler=None, seed=None):
    phase = dataset_opt["phase"]
    if phase != "train":
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    workers = dataset_opt.get("num_worker_per_gpu", 4)
    workers = min(8, os.cpu_count() or 4) if workers == "auto" else int(workers)
    mult = 1 if dist or num_gpu == 0 else num_gpu          # DataParallel multiplies, DDP does not
    rank, _ = get_dist_info()
    return DataLoader(
        dataset, batch_size=dataset_opt["batch_size"] * mult, num_workers=workers * mult,
        sampler=sampler, shuffle=sampler is None, drop_last=True,
        pin_memory=dataset_opt.get("pin_memory", True),
        persistent_workers=dataset_opt.get("persistent_workers", workers > 0),
        prefetch_factor=None if workers == 0 else dataset_opt.get("prefetch_factor", 4 if dataset_opt["type"] == "otf" else 2),
        worker_init_fn=partial(worker_init_fn, num_workers=workers * mult, rank=rank, seed=seed) if seed is not None else None,
    )
```
`worker_init_fn` additionally seeds the dataset's own `numpy.random.Generator` (`dataset.rng = default_rng(worker_seed)`), replacing the module-level `rng()` singletons (interface agreed with the T-config lane).

### 3.2 `neosr/data/prefetch_dataloader.py` — whole file after the change
```python
class CUDAPrefetcher:
    """Copies the next batch to the GPU on a side stream while the current one trains."""
    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.loader, self.device, self.stream = loader, device, torch.cuda.Stream(device)
        self.reset()
    def reset(self) -> None:
        self.it = iter(self.loader); self._preload()
    def _preload(self) -> None:
        self.batch = next(self.it, None)
        if self.batch is None: return
        with torch.cuda.stream(self.stream):
            self.batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in self.batch.items()}
    def next(self) -> dict | None:
        cur = torch.cuda.current_stream(self.device); cur.wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            for v in batch.values():
                if torch.is_tensor(v): v.record_stream(cur)
            self._preload()
        return batch
```

### 3.3 `neosr/data/data_sampler.py`
```python
def __iter__(self):
    g = torch.Generator(); g.manual_seed(self.epoch)
    idx = torch.randperm(self.total_size, generator=g).remainder_(len(self.dataset))
    return iter(idx[self.rank::self.num_replicas].tolist())
```

### 3.4 `neosr/models/base.py` — device-side loss meter (replaces `reduce_loss_dict`)
```python
class LossMeter:
    """Accumulates detached scalars on device; one host sync per read()."""
    def __init__(self, device): self.device, self.keys, self.sum, self.n = device, None, None, 0
    def update(self, logs: dict[str, Tensor]) -> None:
        vals = torch.stack([v.detach().float().reshape(()) for v in logs.values()])
        if self.keys != list(logs): self.keys, self.sum, self.n = list(logs), torch.zeros_like(vals), 0
        self.sum += vals; self.n += 1
    def read(self, dist: bool = False) -> dict[str, float]:
        s = self.sum / max(self.n, 1)
        if dist: torch.distributed.all_reduce(s); s /= torch.distributed.get_world_size()
        out = dict(zip(self.keys, s.tolist())); self.sum.zero_(); self.n = 0
        if not all(math.isfinite(v) for v in out.values()): raise FloatingPointError(f"non-finite loss: {out}")
        return out
```
`image.py` calls `self.meter.update(logs)` per micro-batch (the `logs` dict comes straight from the loss terms of the context-based pipeline) and `train.py` calls `model.get_current_log()` → `meter.read()` only at `print_freq`. `torch.isnan(l_g_total)` (`image.py:717`) is deleted; the only per-update host decision is `finite = torch.isfinite(total_norm).item()` after `clip_grad_norm_` (one sync, measured free), which gates `optimizer.step()`, the EMA update and the scheduler; `read()` additionally raises if the window average went non-finite (a sustained divergence rather than a single bad step).

### 3.5 `neosr/models/image.py` — device/memory-format/compile policy and tiled inference
```python
def _setup_execution(self):
    self.channels_last = self.opt.get("channels_last", self.use_amp)
    self.mf = torch.channels_last if self.channels_last else torch.contiguous_format
    self.net_g_fwd = torch.compile(self.net_g, mode=self.opt.get("compile_mode", "default"), dynamic=False) if self.opt.get("compile") else self.net_g

def feed_data(self, data):
    self.lq = data["lq"].to(self.device, non_blocking=True).contiguous(memory_format=self.mf)
    if "gt" in data: self.gt = data["gt"].to(self.device, non_blocking=True).contiguous(memory_format=self.mf)

@torch.inference_mode()
def infer(self, net, lq, tile=-1, overlap=16):
    if tile <= 0 or max(lq.shape[-2:]) <= tile: return net(lq)
    b, c, h, w = lq.shape; s = self.scale; step = tile - overlap
    out = weight = None
    ys = list(range(0, max(h - tile, 0) + 1, step)); xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h: ys.append(h - tile)
    if xs[-1] + tile < w: xs.append(w - tile)
    ramp = torch.linspace(0, 1, overlap * s, device=lq.device)
    for y in ys:
        for x in xs:
            o = net(lq[..., y:y + tile, x:x + tile])
            if out is None:
                out = lq.new_zeros(b, o.shape[1], h * s, w * s, dtype=o.dtype); weight = torch.zeros_like(out[:, :1])
            wgt = torch.ones_like(o[:, :1])
            if y: wgt[..., :overlap * s, :] *= ramp.view(-1, 1)
            if x: wgt[..., :, :overlap * s] *= ramp
            out[..., y*s:(y+tile)*s, x*s:(x+tile)*s] += o * wgt; weight[..., y*s:(y+tile)*s, x*s:(x+tile)*s] += wgt
    return out / weight
```
Images smaller than `tile` run whole (removes the `reflect` pad ≥ dim crash, Codex A28). `nondist_validation` keeps `sr` on GPU: learned metrics get the tensor, `tensor2img` runs once for PSNR/SSIM/`imwrite`.

### 3.6 `neosr/metrics/calculate.py` — cached learned metrics
```python
@functools.cache
def _metric_net(name: str, device: str) -> nn.Module:
    net = {"dists": lambda: dists_loss(as_loss=False), "topiq": topiq}[name]()
    return net.to(device).eval().requires_grad_(False)

@METRIC_REGISTRY.register()
def calculate_dists(img: Tensor, img2: Tensor, **_) -> float:   # img: (1,3,H,W) RGB float on GPU
    q = lambda t: (t.clamp(0, 1) * 255).round() / 255             # keep today's uint8 protocol
    with torch.inference_mode(): return _metric_net("dists", str(img.device))(q(img), q(img2)).item()
```
PSNR/SSIM stay numpy/cv2 (reference implementations, cheap); `calculate_metric` passes `metric_data = {"img": sr_gpu, "img2": gt_gpu, "img_np": ..., "img2_np": ...}` and each metric picks what it needs — same "the consumer takes what it needs" rule the owner requires for losses.

### 3.7 Wavelet-guided as a differentiable adversarial view (`losses/wavelet_guided.py`)
```python
class AdvView(nn.Module):
    """Identity until `start_iter`, then stationary-wavelet HF bands of Y; same op for G and D terms."""
    def __init__(self, start_iter: int = 0, wave: str = "sym19"):
        super().__init__(); self.start_iter = start_iter
        self.swt = SWTForward(1, _normalised(wave), "periodic")       # buffers, built once
        self.register_buffer("y_w", torch.tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1))
        self.register_buffer("band_w", torch.tensor([0.035, 0.025, 0.02]).view(1, 3, 1, 1))
    def forward(self, x: Tensor, it: int) -> Tensor:
        if it < self.start_iter: return x
        y = 16.0 + (x * self.y_w).sum(1, keepdim=True)
        return self.swt(y)[0][:, 1:4] * self.band_w
```
`SWTForward.h*_*` become buffers (not `nn.Parameter`, `wavelet_guided.py:350-353`). The GAN generator term computes `D(view(sr, it))` with grad; the discriminator term computes `D(view(gt, it))` and `D(view(sr, it).detach())`. Config keys unchanged.

### 3.8 `neosr/data/augmentations.py` — native-resolution paired augmentations (no global round trip)
```python
def _box(h, w, frac, rng):                      # LQ-space box; GT box = box * scale
    ch, cw = int(h * frac), int(w * frac); cy, cx = rng.integers(h), rng.integers(w)
    return max(cy - ch // 2, 0), max(cx - cw // 2, 0), min(cy + ch // 2, h), min(cx + cw // 2, w)

def cutmix(gt, lq, s, rng, alpha=0.9):
    p = torch.randperm(gt.size(0), device=gt.device); y0, x0, y1, x1 = _box(*lq.shape[-2:], math.sqrt(1 - rng.uniform(0, alpha)), rng)
    lq[..., y0:y1, x0:x1] = lq[p][..., y0:y1, x0:x1]; gt[..., y0*s:y1*s, x0*s:x1*s] = gt[p][..., y0*s:y1*s, x0*s:x1*s]; return gt, lq

def mixup(gt, lq, s, rng, a=(0.4, 0.6)):
    lam = rng.uniform(*a); p = torch.randperm(gt.size(0), device=gt.device)
    return lam * gt + (1 - lam) * gt[p], lam * lq + (1 - lam) * lq[p]

def cutblur(gt, lq, s, rng, alpha=0.7):          # paste *downsampled* GT crop into LQ (clean region), GT unchanged
    y0, x0, y1, x1 = _box(*lq.shape[-2:], rng.uniform(0.2, alpha), rng)
    crop = gt[..., y0*s:y1*s, x0*s:x1*s]
    lq[..., y0:y1, x0:x1] = crop if s == 1 else F.interpolate(crop, scale_factor=1 / s, mode="bicubic", antialias=True).clamp_(0, 1)
    return gt, lq

AUGS = {"none": None, "mixup": mixup, "cutmix": cutmix, "resizemix": resizemix, "cutblur": cutblur}

def apply_augment(gt, lq, *, scale, augs, prob, rng, multi_prob=0.3):
    if len(augs) != len(prob): raise ValueError("augmentation and aug_prob lengths differ")
    k = rng.integers(2, len(augs) + 1) if len(augs) > 2 and rng.random() < multi_prob else 1
    chosen = rng.choice(augs, size=k, replace=False, p=np.asarray(prob) / sum(prob))
    for name in chosen:
        if (fn := AUGS[name]) is not None: gt, lq = fn(gt, lq, scale, rng)
    return gt, lq
```
"none" is an exact identity; `resizemix` keeps its current logic with the box scaled by `s`. Semantics change only where the old code was wrong (mixup partner, unconditional resample), which the owner should know when comparing old/new runs.

### 3.9 OTF kernels (`models/otf.py`, `data/degradations.py`, `utils/diffjpeg.py`)
```python
# otf.py pool: random-slot swap, no full-queue permutation, no extra clones
def _pool(self, lq, gt):
    if self.ptr < self.size:                       # fill phase
        self.q_lq[self.ptr:self.ptr + len(lq)] = lq; self.q_gt[self.ptr:self.ptr + len(gt)] = gt; self.ptr += len(lq); return lq, gt
    idx = torch.randperm(self.size, device=lq.device)[: len(lq)]
    out_lq, out_gt = self.q_lq[idx], self.q_gt[idx]            # gathers are copies
    self.q_lq[idx], self.q_gt[idx] = lq, gt
    return out_lq, out_gt

# degradations.py: unique-value count per sample without a sync
def _vals_per_sample(img_q: Tensor) -> Tensor:                  # img_q in {0..255}/255, (b,c,h,w)
    hist = torch.zeros(img_q.size(0), 256, device=img_q.device).scatter_(1, (img_q * 255).round().long().flatten(1), 1.0)
    return torch.exp2(torch.ceil(torch.log2(hist.gt(0).sum(1).float()))).view(-1, 1, 1, 1)

# diffjpeg.py: quality → factor vectorised
factor = torch.where(quality < 50, 5000.0 / quality, 200.0 - quality * 2) / 100.0
```
All three keep the current numerics (verified equal for the poisson count: histogram over c·h·w of the quantised image = `len(torch.unique)`).

### 3.10 Optimizer table (`models/base.py`)
```python
OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "nadam": torch.optim.NAdam,
              "adan": adan, "adamw_sf": adamw_sf, "adan_sf": adan_sf}
def get_optimizer(self, kind, params, **kw):
    try: cls = OPTIMIZERS[kind.lower()]
    except KeyError: raise ValueError(f"optimizer {kind!r}; choose one of {sorted(OPTIMIZERS)}") from None
    return cls(params, **kw)
```
`schedule_free` is read from the optimizer options (default `True` for `*_sf`, forbidden for the others) instead of the `sys.exit` checks at `image.py:367-377, 423-433`.

### 3.11 Removed keys (`utils/options.py`)
```python
REMOVED = {"train.sam": "SAM/FSAM removed (0.9)", "train.sam_init": ..., "train.eco": "ECO removed", "train.eco_iters": ..., "train.eco_init": ..., "train.eco_schedule": ...}
# QAT keys are owned by T5 and stay valid; inert values (eco = false) are accepted.
for dotted, why in REMOVED.items():
    if _get(opt, dotted) is not None: raise ValueError(f"config key '{dotted}': {why}")
```
Model types `default/simple_otf/eqrotf/pancamotf`, dataset types `eqrotf/pancamotf/pancamsingle/pcpaired/simple_otf`, optimizer types `soap_sf/adamw_win` fail naturally through the registries with a clear `KeyError` message listing valid names.

## 4. Migration steps (ordered; lanes are disjoint file sets and can run in parallel)

Step 0 (serial, 1 commit): add `REMOVED` key check (`utils/options.py`) and the `bench_train.py` harness; record baseline it/s (and per-loss fwd+bwd ms via the harness `--losses` flag) and the first two `print_freq` loss lines for `train_esrgan2.toml` with `manual_seed = 1024`, for `train_span.toml` and `train_swinir.toml` (paired) and one `*_otf.toml`. Verification: harness runs, baseline file saved under `experiments/bench/`.

| Lane | Files | Steps | Verification |
|---|---|---|---|
| A — deletions | `models/default.py`, `models/{simple_otf,eqr_otf,pancam_otf}.py`, `models/image.py.helaman`, `data/{eqrotf,pancamotf,pancamsingle,pcpaired,simple_otf,otfgray}_dataset.py`, `archs/{eqrsrgan,pancam}_arch.py`, `optimizers/{soap_sf,adamw_win,fsam}.py`, `optimizers/__init__.py`, `common/`, `pyproject.toml` | 1. delete files; 2. optimizer table §3.10; 3. drop `pytorch_optimizer`; 4. `train.py`: remove `common` sys.path block, QAT block, kill-switch, torchprofile → `FlopCounterMode` | `python -c "import neosr.models, neosr.data, neosr.archs, neosr.optimizers"`; `ruff check`; dry-run (`--debug`, 16 iters) of one `image`, one `otf`, one `test_*.toml` config; a config with `type="adan"` builds; a config with `sam="fsam"` fails with the removed-key message |
| B — aether | owned by T5 (`archs/aether_arch.py`, QAT helper); T4 only removes the trainer-side QAT block, `_init_ema`, `generate_calib_data` and the `common/` sys.path hack from `train.py`/`image.py` (lanes A/D) | per T5 | per T5; from T4: `grep -n qat train.py neosr/models` returns nothing |
| C — data path | `data/__init__.py`, `data/prefetch_dataloader.py`, `data/data_sampler.py`, `data/augmentations.py`, `data/paired_dataset.py` (retry loop), `data/degradations.py`, `utils/diffjpeg.py` | §3.1, §3.2, §3.3, §3.8, §3.9 (poisson/jpeg); `pin_memory`/`persistent_workers` defaults | unit checks in `scripts/check_data.py`: sampler indices identical for a given epoch before/after; augment "none" → bitwise identity; mixup/cutmix shapes & ranges; poisson histogram == unique-count loop on 64 random batches; jpeg factor vector == loop; prefetcher: 2k-iter loss trace equal with and without `PYTORCH_NO_CUDA_MEMORY_CACHING=1`; epoch-boundary stall measured < 50 ms |
| D — trainer hot path | `models/base.py`, `models/image.py`, `models/otf.py`, `losses/wavelet_guided.py`, `losses/consistency_loss.py`, `losses/ff_loss.py`, `train.py` (log call site), `test.py`, `convert.py` | §3.4 meter + non-finite policy; GradScaler fp16-only; delete D-EMA, ECO, SAM branches; §3.5 execution policy (channels_last key, compile runner, eager canonical module); §3.5 `infer` tiling; remove all `empty_cache/gc`; remove `set_default_device`; §3.7 AdvView wiring (shared with the T-loss lane's context design — AdvView is the `adv_sr/adv_gt` provider); OTF pool §3.9; `convert.py` cleanup | bitwise-equal `print_freq` lines vs P0 baseline for pure-perf items (run with `compile=false`, `channels_last=false` first), then it/s table re-run; `compile=true` run saves a checkpoint whose keys have no `_orig_mod.` and loads in `test.py`; tiled vs whole PSNR ≥ 50 dB for `esrgan`; validation with 0 learned metrics unaffected; `convert.py -net esrgan --input <ckpt> --onnx` round-trips `assert_verify` |
| E — metrics | `metrics/calculate.py`, `metrics/__init__.py` (duplicate imports), `losses/dists_loss.py` (`as_loss=False` must `.to(device)` explicitly), `metrics/topiq.py` | §3.6 cache; tensor inputs; explicit device | DISTS/TOPIQ values on 10 val images equal (±1e-5) to the current per-image construction; validation wall time per image logged before/after |
| G — loss backbones (P0) | `losses/fdl_loss.py`, `losses/dists_loss.py`, `losses/vgg_perceptual_loss.py`, `losses/resnet_perceptual_loss.py` (interface only: `forward(sr, gt)` unchanged, so it is independent of the T-loss context work) | move `custom_fwd(cast_inputs=float32)` from `forward` to the fp32 head; GT branch under `no_grad`; `compile_losses` option compiles `self.model`; explicit `.to(device)` instead of `device="cuda"` literals (`fdl_loss.py:184,298`) | per-loss fwd+bwd timing (`bench_losses.py` pattern) before/after; loss values on 32 fixed (sr, gt) pairs: relative delta < 1e-2 fp32 vs bf16; end-to-end it/s on `train_esrgan2.toml` ≥ +30 % |
| F — docs/config | `readme.md` options table, `options/train_dctlsa.toml` (fix the duplicate `type` / missing `[train.fdl_opt]` header), `pyproject.toml` | document `channels_last`, `compile_mode`, `compile_losses`, `prefetch_factor`, `tile_overlap`, removed keys | all 135 TOMLs parse (`tomllib`); `grep` for removed keys over options/ returns nothing |

Order: Step 0 → A, C, E, G in parallel (B on T5's schedule) → D (depends on A's deletions landing to avoid conflicts in `image.py`, and on the T-loss context design for the GAN/AdvView wiring; the perf items in D that do not touch the loss loop — meter, scaler, execution policy, tiling, empty_cache, OTF pool — can land first on top of the current `if self.cri_*` loop) → F. Final: re-run the harness for the four reference configs, commit the numbers to `experiments/bench/`.

## 5. Risks
- **bf16 loss backbones (P0)** change perceptual-loss values slightly (bf16 ViT/VGG activations). Gate: fixed-pair comparison < 1e-2 relative; keep `loss_precision = "fp32"` as an escape hatch per loss.
- **Log semantics change**: losses become `print_freq`-window averages instead of the last micro-batch value; TensorBoard curves get smoother. Document; keep the key names.
- **Skipped steps on non-finite gradients** are a behaviour change (today the run crashes on a NaN loss and silently updates on inf gradients); log the skip count (`skipped_steps`) so divergence is visible, and abort after 50 consecutive skips.
- **channels_last default = `use_amp`** keeps today's behaviour for the model but now also converts inputs; for transformer archs this is a measurable but unknown delta until benchmarked (lane D measurement step covers `swinir_small`, `hat_s`).
- **Augmentation rewrite changes training data statistics** for the 168 configs using the template (no more blur of every LQ batch; mixup now mixes LQ with LQ). Expected to improve results, but it invalidates direct loss-curve comparisons with old runs.
- **Deleting D-EMA** changes which D weights are saved; old `net_d_*.pth` checkpoints (EMA weights) still load as the live D — same as today's resume behaviour, so no regression.
- **Persistent workers + `spawn`** hold 4 interpreters (~1.5 GB RSS total) for the whole run; fine on 62 GB.
- **`torch.compile` with `dynamic=False`** recompiles if the batch shape changes (e.g. the last `drop_last=False` batch — we keep `drop_last=True`), and the eager validation path means compiled and eager numerics differ slightly (inductor fusion); EMA/checkpoints come from the eager module so this is only a validation-time difference.
- **Aether QAT** is a T5 deliverable; T4's removal of the trainer QAT block must land together with T5's replacement, or the 5 QAT configs stay unrunnable (they are unrunnable today anyway: unregistered arch).

## 6. What I would NOT do
- Not port traiNNer-redux's `CUDAPrefetcher`/`RobustPrefetchGenerator` (499 lines, health monitors, auto-disable logic `prefetch_dataloader.py:431-441`); a 25-line correct prefetcher is the reference solution.
- Not add a "dynamic batch/patch size VRAM automation" (competitor `train.py:596-640`): it changes training semantics under the user.
- Not replace numpy PSNR/SSIM with GPU versions: they are the reference implementations users compare against and cost < 10 ms/image.
- Not keep `soap_sf`/`adamw_win`/FSAM/ECO "just in case": every one is unused and at least one is wrong; git history preserves them.
- Not generalise OTF degradations into a plug-in pipeline now (T-loss/T-config decide the context design first); only remove syncs and copies.
- Not switch the start method to `fork` before the import-time `parse_options` side effects are removed (CUDA init in workers would crash `fork`).
- Not enable `max-autotune`/CUDA graphs by default: minutes of compile and fragile with `requires_grad_` toggling on the discriminator; opt-in via `compile_mode`.
- Not implement distributed validation or DDP fixes here (owner trains on one 4090; DDP correctness belongs to the stability topic).

## 7. Open questions
1. Keep `adamw_sf` (0 configs) — I kept it as the minimal SF reference; delete if the owner wants exactly one SF optimizer.
2. Aether QAT scope is T5's; T4 only requires that the trainer stays arch-agnostic (no `enable_qat` branches in `train.py`/`image.py`). Confirm that T5's hook lands as an arch/QAT-helper API and not as trainer branches.
3. `channels_last` default: keep `= use_amp` (no behaviour change) or `false` until transformer archs are benchmarked?
4. Should the window-averaged logging also apply to `out_d_real/out_d_fake` (yes in this design) or should those stay last-value?

## 8. Second opinion (Codex T4, `t4_answer.txt`) — agreements and settled disagreements
Agrees on every delete/keep above except two, and on the performance list (it ranked logging syncs first "for lightweight models" without measuring; my measurement shows ≈0 even for span@16, so P0/P7/P11 lead instead).
- **Aether**: Codex would retire it as a feature; T5 keeps and repairs it. Settled by T5's probe (`T5_aether_qat.md` §0: QAT from scratch works, Q/DQ export verified) — keep.
- **`adan`**: Codex keeps it as the non-SF control; I had planned an alias to `adan_sf(schedule_free=False)`. Settled by the numerical probe (not equivalent, Δ = 2e-4 after 20 steps) — keep.
- **Finiteness checks**: Codex objects to moving all checks to `print_freq`; adopted its bounded per-update check (P1b) because the sync is measured free.
- **ONNX**: Codex points out torch 2.9 already defaults to the dynamo exporter — verified and folded into the `convert.py` row.
- **Config census**: Codex found `train_dctlsa.toml` unparseable — verified, added to lane F.
- Not adopted: Codex's "retain explicit CPU stitching for genuinely large outputs" — a 4K×4K×3 fp32 output is 200 MB on a 24 GB card; add it only if an actual OOM shows up.
