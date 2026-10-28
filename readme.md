<div align="center">
<img src="./licenses/assets/neosr.jpg" alt="NEOSR Logo" width="600"></img>

<a href="https://discord.gg/NN2HGtJ3d6"><img src="https://github.com/neosr-project/neosr/assets/132400428/4bd54b1d-4639-4940-b9c7-b3f212aea5c8?sanitize=true" width="100"></img></a><br>
Join our <a href="https://discord.gg/NN2HGtJ3d6">Discord</a>
</div>

[neosr](https://github.com/neosr-project/neosr) is an open-source framework for training single-image
super-resolution (SISR) models. It aims at realistic restoration of photos, anime/cartoons and illustrations, and at
reproducible research.

- **Broad**: 45 generator families, 6 discriminators, 15 losses, schedule-free optimizers, on-the-fly
  (Real-ESRGAN) degradations and batch augmentations, all selected from a TOML file.
- **Stable**: correct adversarial training, exact gradient accumulation, AMP (fp16/bf16) with non-finite update
  skipping, EMA, and atomic, self-contained checkpoints.
- **Reproducible**: `manual_seed` seeds every RNG; `deterministic = true` also turns on deterministic kernels. A resume is a stochastic continuation: weights,
  optimizer, EMA, scheduler, scaler and counters are restored exactly, model-side RNG on rank 0 only; data order
  and augmentations are not order-exact.
- **Simple**: plain Python, no registries or import-time magic; each package has one lookup table. Managed with
  [`uv`](https://docs.astral.sh/uv/) and [`ruff`](https://docs.astral.sh/ruff/).

The [wiki](https://github.com/neosr-project/neosr/wiki) has per-arch options. Where it disagrees with this
readme about configuration keys, this readme describes the current code.

## support the project

> [!TIP]
> Consider supporting the project on [**KoFi**](https://ko-fi.com/neosr) &#9749; or [**Patreon**](https://www.patreon.com/neosr)

## installation

Python 3.13 and an NVIDIA GPU (PyTorch CUDA 12.8 wheels, see `pyproject.toml`). CPU runs work for tests and
conversion.

**Windows**:
```ps1
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/neosr-project/neosr/refs/heads/master/install_windows.ps1 | iex"
```

**Linux**:
```bash
curl -LsSf https://raw.githubusercontent.com/neosr-project/neosr/refs/heads/master/install_linux.sh | sh
```
`neosr` is installed in the current directory. The scripts add the aliases `neosr-train` and `neosr-test`
(`uv run --isolated train.py -opt` / `test.py -opt`), `neosr-convert` (`convert.py`) and `neosr-update`.

**Manual (any platform, recommended for cloud machines)**: install [`uv`](https://docs.astral.sh/uv/), then
```bash
git clone https://github.com/neosr-project/neosr && cd neosr
uv sync --locked            # creates .venv from uv.lock
uv lock --upgrade && uv sync --locked   # later: update the dependencies
```
A conda/virtualenv Python 3.13 works too; keep using `uv` for the installs. See the
[Installation Instructions](https://github.com/neosr-project/neosr/wiki/Installation-Instructions) wiki for more.

## usage

```bash
uv run python train.py -opt options/train_esrgan.toml        # or: neosr-train options/train_esrgan.toml
uv run python train.py -opt config.toml --auto_resume        # continue from the newest training state
uv run python train.py -opt config.toml --debug              # name -> debug_<name>, log every update, val/save every 8
uv run python test.py  -opt options/test_esrgan.toml         # inference over every [datasets.test_*]
uv run python convert.py --input net_g_10000.pth -net esrgan -s 4   # ONNX export
```

- `train.py` and `test.py` take `-opt <file.toml>` (required), `--launcher none|pytorch|slurm` and `--debug`;
  `train.py` also `--auto_resume`. Any run name containing `debug` gets the debug frequencies and no TensorBoard.
- `convert.py` builds `-net` at scale `-s` (default 4) on the CPU, loads `--input` (any layout accepted by
  `pretrain_network_g`), exports `<output>_fp32.onnx` (`--output` defaults to the input path with `.onnx`) and
  checks it against PyTorch with ONNX Runtime (`-nocheck` skips). Options: `-opset` (17), `-static C H W`
  (default: dynamic batch/height/width), `-window` (sets `window_size`), `-optimize` (adds
  `<output>_fp32_optimized.onnx`). `-fp16` is temporarily a no-op, `-fulloptimization` is disabled, and `craft`
  and `ditn` cannot be exported. A QAT checkpoint exports as INT8 QDQ ONNX; `--strip-qat` instead writes its
  float deploy weights to `<input>_float.pth` (`--output` overrides), and with `--onnx` also their float ONNX
  ([docs/aether.md](docs/aether.md)).
- Templates live in [options/](options/): `train_<arch>.toml` (paired data), `train_<arch>_otf.toml`
  (on-the-fly degradations), `test_<arch>.toml` and `options/aether/`.

Exit codes (`train.py`, `test.py`, `convert.py`):

| code | meaning |
|---|---|
| 0 | finished, or (training only) stopped by the first Ctrl-C after saving |
| 1 | unexpected error; the traceback is in the log (includes non-finite training, see below) |
| 2 | configuration error (`config error: ...` on stderr) or a bad command line |
| 130 | training: a second Ctrl-C, the last committed checkpoint stands; inference/export: the first Ctrl-C |

## configuration

Configs are TOML (`.toml` only). Required keys: `name`, `model_type` (`image` or `otf`), `scale` (training: 1, 2,
3, 4 or 8), `[network_g]` and `[datasets.*]`. Training also needs `[datasets.train]` with `patch_size` and
`batch_size`, `[train.optim_g]` and `logger.save_checkpoint_freq`.

- **Validation of keys.** Every key outside the free-form tables is checked against a schema
  (`neosr/config.py`). A wrong type is an error. An unknown key is a `ConfigWarning` with a suggestion
  (`unknown key 'train.emaa' is ignored (did you mean 'ema'?)`) and the run continues. `[network_g]`,
  `[network_d]`, `[train.*_opt]`, `[train.optim_*]`, `[train.scheduler]` and `[val.metrics.*]` go to their
  constructors, which reject unknown keyword arguments as configuration errors.
- **`[degradations]`** (OTF keys) is merged into `[datasets.train]`. A key set in both warns, and
  `[degradations]` wins.
- **Cross-key rules** (errors): `network_d` needs `train.optim_d` and `train.gan_opt`, and the reverse;
  `bfloat16` needs `use_amp`; `val.tile > 2 * val.tile_overlap`; `datasets.train.accumulate >= 1`;
  `train.max_skip_streak >= 1`; `persistent_workers` needs `num_worker_per_gpu > 0`; `num_gpu > 1` needs a
  launcher; `test_*` datasets belong in an inference config. Training configs also need `0 <= train.ema < 1`,
  `val.val_freq >= 1`, `val.metrics.*.better` in `"higher"`/`"lower"`, `num_worker_per_gpu` an integer >= 0 or
  `"auto"`, `dataset_enlarge_ratio >= 1`, `logger.total_iter`/`print_freq` >= 1 and a `save_checkpoint_freq >= 1`,
  unique val dataset names, no dataset `mean`/`std` (nothing inverts a normalisation for the losses, outputs and
  metrics), and CUDA for `use_amp`. Each is checked before the model is built.
- **Paths** expand `~`. A relative `path.resume_state` is relative to the repository root. Quote Windows paths
  with single quotes.
- **Removed keys** are errors that name the replacement:

| key | replacement |
|---|---|
| `train.sam`, `train.sam_init` | none (SAM/FSAM removed) |
| `train.eco`, `train.eco_init`, `train.eco_iters`, `train.eco_schedule` | none (ECO removed) |
| `train.warmup_iter` | `warmup_steps` in `train.optim_*` (every optimizer) |
| `train.gw_opt`, `train.color_opt`, `train.luma_opt` | none (losses removed) |
| `path.ignore_resume_networks` | `pretrain_network_*` for a new run |

Top-level keys: `num_gpu` (`"auto"`), `manual_seed`, `deterministic` (default false), `use_amp`, `bfloat16`,
`fast_matmul` (TF32 and reduced matmul precision), `fp16_accumulation` (default false: fp16 GEMMs accumulate in
fp16, faster, overflow above 65504; fp16 AMP only), `compile` and `compile_mode` (training only, not on Windows; `test.py` runs eager),
`find_unused_parameters` (DDP),
`input_size`/`in_chans` (log MACs if `torchprofile` is installed), `print_network`.

### datasets

| type | use |
|---|---|
| `paired` | `dataroot_gt` + `dataroot_lq` (folders, or LMDB with `create_lmdb.py`); training and validation |
| `otf` | `dataroot_gt` only; Real-ESRGAN two-stage degradation on the GPU (`model_type = "otf"`) |
| `single` | `dataroot_lq` only; inference without metrics |

Training keys: `patch_size` (the lq patch), `batch_size` and `num_worker_per_gpu` (default 4, `"auto"`) per
process, `accumulate`, `dataset_enlarge_ratio`, `use_hflip`/`use_rot` (default on), `pin_memory` (default on with CUDA) and
`persistent_workers` (default on with workers), `queue_size` (OTF pair pool), and the batch augmentations
`augmentation = ["none", "mixup", "cutmix", "resizemix", "cutblur"]` with `aug_prob` (relative weights) and
`aug_multi_prob` (default 0.3: the chance that a batch combines several of them instead of drawing one).
Validation sets are `[datasets.val]`, `[datasets.val_1]`, ... (`name` defaults to the section key); inference
sets are `[datasets.test_1]`, ....

## losses

Every non-empty `[train.<slot>_opt]` table becomes one term of the objective. The slot name is free and names
the log key `l_g_<slot>` (`pixel_opt` logs `l_g_pix`, `perceptual_opt` logs `l_g_percep`).

```toml
[train.pixel_opt]
type = "L1Loss"
loss_weight = 1.0     # default: the loss's own default (1.0; gan_loss 0.1); 0 drops the term
[train.consistency_opt]
type = "consistency_loss"
target = "lq"         # reference: "gt" (default) or "lq" = bicubic-upscaled lq (colour reference)
```

- Keys besides `type`, `loss_weight` (alias `perceptual_weight`, not both) and `target` go to the loss
  constructor. Two slots may use the same type. At least one generator term with a non-zero weight is required;
  log keys must be unique.
- Legacy `train.match_lq_colors = true` still sets `target = "lq"` for `consistency_opt` and `msswd_opt`.
- **GAN**: `[train.gan_opt]` (`type = "gan_loss"`, `gan_type = "bce" | "mse" | "huber"`, `real_label_val`,
  `fake_label_val`, `loss_weight` for the generator term) also adds the discriminator objective, which is
  unweighted. `wavelet_init = N` makes D see the stationary-wavelet high-frequency view from update N on, for both
  terms. The legacy pair `train.wavelet_guided = true` + `train.wavelet_init` maps to `gan_opt.wavelet_init`. D is
  always in train mode, and the adversarial term reaches only the generator's parameters. Opt-in (defaults keep
  the plain GAN): `start_iter = N` keeps both GAN terms off through update N, `ramp_iters = M` then raises the
  generator term's weight linearly over M updates; `r1_gamma` / `r2_gamma` add the lazy R1 / R2 penalty
  `gamma / 2 * |grad_x D(x)|^2` on real / fake inputs every `reg_every` (16) updates; `gan_type = "rpgan"` is the
  relativistic pairing loss `softplus(D(fake) - D(real))` (no labels).
- **Log keys** (means over the logging window): `l_g_<slot>`, `l_g_total`, `l_d_total`, `l_d_real`, `l_d_fake`,
  `out_d_real`, `out_d_fake` (mean D logits), `out_d_margin` (rpgan), `l_d_reg` (R1/R2 updates), `grad_norm_g`/
  `grad_norm_d` (before clipping, successful updates only), `clipped_g`/`clipped_d` (share of clipped updates),
  `skipped_g`/`skipped_d`, and `nonfinite_<key>` (count of left-out non-finite values) when there were any. The
  performance field also shows the share of the window spent waiting for training batches (`data 12%`).
- **Which term drives the generator?** `python scripts/term_grads.py -opt config.toml [--net_g net_g.pth]
  [--batches 4]` prints, for a few training batches of the config, each generator term's gradient norm, its share
  of the summed squared norms and its cosine with the total gradient. It runs offline (no optimizer step, nothing
  saved), so loss weights can be chosen from a measurement; e.g. at the shipped weights FDL(DINOv2) holds ~99 %.

| loss | `type` |
|---|---|
| L1 / L2 / Huber | `L1Loss`, `MSELoss`, `HuberLoss` |
| CHC (clipped Huber with cosine similarity) | `chc_loss` |
| [NCC](https://github.com/neosr-project/neosr/wiki/Losses#ncc_opt-ncc_loss) (normalized cross-correlation) | `ncc_loss` |
| [Perceptual (VGG19)](https://github.com/neosr-project/neosr/wiki/Losses#perceptual_opt-vgg_perceptual_loss) | `vgg_perceptual_loss` |
| [GAN](https://github.com/neosr-project/neosr/wiki/Losses#gan_opt-gan_loss) (+ [wavelet guided](https://github.com/mandalinadagi/WGSR) view) | `gan_loss` |
| [MS-SSIM](https://github.com/lartpang/mssim.pytorch) | `mssim_loss` |
| [LDL](https://github.com/csjliang/LDL) | `ldl_loss` |
| [Focal Frequency](https://github.com/EndlessSora/focal-frequency-loss) | `ff_loss` |
| [DISTS](https://github.com/dingkeyan93/DISTS) | `dists_loss` |
| [Consistency](https://github.com/neosr-project/neosr/wiki/Losses#consistency_opt-consistency_loss) (Oklab and CIE L*) | `consistency_loss` |
| [KL divergence](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html) | `kl_loss` |
| [MS-SWD](https://github.com/real-hjq/MS-SWD) | `msswd_loss` |
| [FDL](https://github.com/eezkni/FDL) (`model = "dinov2" \| "vgg" \| "resnet" \| "effnet"`) | `fdl_loss`; `fp32_backbone = false` runs the DINOv2 backbone in bf16 for ~2x faster FDL at 256 px patches (opt-in: gradients drift, see docs/refactor/analysis/perf.md); `checkpoint = true` (DINOv2) recomputes the backbone activations in backward: same result, less memory, slower |

## training

- **Updates.** An update is one optimizer step over `datasets.train.accumulate` micro-batches (default 1; the
  loss is divided by it, and DDP reduces once per window). `logger.total_iter` (default 1000000),
  `logger.print_freq` (100), `logger.save_checkpoint_freq` and `val.val_freq` all count updates.
- **Output clamp.** `train.clamp` sets what the losses see of the generator output: `"restore"` (default)
  clamps to [0, 1] but keeps the gradient that moves an out-of-range pixel back into range; `true` is the old
  hard clamp to [1/255, 1], which zeroes the gradient of every out-of-range pixel (at init 34-84% of the output
  of most archs: training stalls for the first ~1-2k updates); `false` passes the output unclamped.
- **AMP.** `use_amp = true` runs fp16 autocast with a GradScaler; adding `bfloat16 = true` runs bf16 (no
  scaler). Under AMP the nets and inputs use channels-last.
- **Clipping.** `train.grad_clip` bounds each network's total gradient norm: `true` (default) clips to 1.0, a
  number clips to that norm, `false` turns clipping off, and `"spike"` clips only norms above 3x a running mean
  (EMA over ~50 updates) of the applied norms; the first update only starts the mean, which a resume restores.
- **Non-finite updates.** If a network's gradient norm is inf/NaN, its optimizer step is skipped (the fp16
  scaler backs off), the buffers its forwards wrote (BatchNorm statistics, spectral-norm vectors, QAT observers)
  are put back, the EMA is not updated, the skip is counted in `skipped_g`/`skipped_d`, and the scheduler still
  advances. The skipped update's non-finite losses are left out of the logged means and counted as
  `nonfinite_<key>`. `train.max_skip_streak` (default 10) consecutive skips stop the run; so do non-finite weights
  or buffers before a save or validation, and a non-finite loss of a network whose step was applied (finite
  gradients: a broken loss), before anything is written (exit 1). The fp16 scaler starts at 2^5 and doubles after 200 clean updates.
- **EMA.** `train.ema = 0.999` keeps an exponential moving average of the generator. The first successful update
  copies the generator; later updates average. Validation and the exported `net_g` use the EMA.
  `train.ema_power` (optional) ramps the decay up over the first updates (ema_pytorch-style warmup); without it
  the decay is constant. The templates set `ema_power = 0.75`.
- **Optimizers** (`[train.optim_g]`, `[train.optim_d]`, `type` is case-insensitive): `adam`, `adamw`, `nadam`,
  `adan`, and schedule-free `adamw_sf` and `adan_sf` (`schedule_free = true`). `warmup_steps = N` ramps the lr
  linearly over the first N updates for every optimizer (a trainer `LinearLR` for the plain ones, before the
  configured scheduler; the schedule-free ones count their applied steps); the logged lr includes it. With a
  schedule-free optimizer and no EMA, validation and exports use the evaluation (x) weights; training resumes
  from the training weights. `[train.scheduler]` (`multisteplr` or `cosineannealing`) applies to every optimizer.
  Schedule-free `adan_sf` is Schedule-Free AdamW with Adan's direction: `y` and `z` take the same step
  `lr * min(t / warmup_steps, 1) * sqrt(1-b3^t)` along `(g + b2/(1-b2^t) v) / (sqrt(n) + eps) + weight_decay * y`,
  where `v` is the EMA of the gradient difference
  and `n` the second moment of the extrapolated gradient `g + b2 (g - g_prev)`; Adan's first moment is replaced
  by the schedule-free interpolation (`b1`). (Earlier versions moved `z` with the raw gradient: the first step
  did not move the weights and the eval weights moved uphill.) `betas = [b1, 0, b3]` drops the difference
  term: exactly `adamw_sf` with `betas = [b1, b3]`, 2 instead of 4 state buffers per parameter.
  `schedule_free = false` is plain Adan with warmup. `max_grad_norm` (default 0) is an extra optimizer-side
  clip; the trainer already clips with `train.grad_clip`. A schedule-free `adan_sf` `.state` saved before this
  fix cannot be resumed (configuration error): start a new run with `path.pretrain_network_g` (and `_d`) set to
  its `net_g`/`net_d` `.pth`.
- **Seeds.** `manual_seed` seeds the RNGs; without it a random seed is drawn and logged. `deterministic = true`
  selects deterministic algorithms (cuDNN benchmark off, `use_deterministic_algorithms`, filled uninitialised
  memory), which is often slower. Expandable allocator segments are on unless `PYTORCH_ALLOC_CONF` is set.

## checkpoints and resume

Files under `experiments/<name>/`:

| file | content |
|---|---|
| `models/net_g_<u>.pth`, `models/net_d_<u>.pth` | `{"params": state_dict}` at update `u`; `net_g` is the EMA if enabled. Kept unless `logger.keep_exports` |
| `models/net_*_latest.pth` | copies written at the end of training or on a Ctrl-C stop |
| `training_states/<u>.state` | format 2: nets, EMA, optimizers, schedulers, scalers, counters, RNG, best metrics, run record; loads with `torch.load(weights_only=True)` |

- A save writes the state first, then the exports, each atomically. `logger.keep_states` (default 3; 0 keeps
  all) keeps the newest states; `logger.keep_exports` (default 0 = all) keeps the newest numbered exports per
  network (`*_latest` stays). The first save logs the size of one export set. If a periodic validation fails, the
  state of that update is saved before the run stops.
- **Resume**: an explicit `path.resume_state` wins; else `--auto_resume` takes the newest `<u>.state` in
  `training_states/` (none: a fresh start). A fresh run renames an existing run folder to
  `<name>_archived_<time>`; a resume never does. Resume is strict: a changed set of networks, optimizer type or
  schedule-free mode, trainable parameter list or scheduler list is a configuration error ("use
  pretrain_network_* for a new run"), and so is a changed optimizer state version. Other differences in the
  run record (network options, AMP, EMA, world size, optimizers, scheduler, loss sections, clamp, clip,
  `ema_power`, accumulation, batch and patch size, QAT keys) are warnings. The optimizer hyper-parameters come
  from the state; each one that differs from the config is reported (`restored lr 0.001 from the state (config
  0.0001)`). A resume continues stochastically, not order-exactly: nets, EMA, optimizers, schedulers,
  scalers and counters are exact; the model-side RNG (dropout, DropPath) is restored on rank 0 only (other DDP
  ranks start fresh streams); data order, augmentations and the OTF pool restart.
- **Pretrained weights**: `path.pretrain_network_g`/`_d` (ignored while resuming), `param_key_g`/`_d` (default:
  `params_ema`, `params-ema`, `params`, or a bare state dict), `strict_load_g`/`_d` (default true; false skips
  shape mismatches and logs missing and unexpected keys). Pre-refactor `net_g_*.pth` files load this way;
  pre-refactor `.state` files cannot be resumed, and a `.state` is not accepted as a pretrain.
- **Ctrl-C**: the first one stops at the next update boundary, saves state, exports and `*_latest`, and exits 0
  without the final validation. A second one exits 130 at once. Under DDP all ranks stop together, one update
  after the signal.

## validation

`[val]`: `val_freq` (updates; validation also runs after the last update), `save_img` (default true, to
`experiments/<name>/visualization/<image>/<image>_<u>.png`), `save_lq` (default true), `pbar` (default true),
`tile` (lq tile size; default -1 = whole image), `tile_overlap` (default 16; the context kept around each tile;
tiles are stitched on the CPU) and `suffix` (inference file name suffix, default the run name).

```toml
[val.metrics.psnr]
type = "calculate_psnr"   # calculate_psnr, calculate_ssim (crop_border, test_y_channel), calculate_dists, calculate_topiq
[val.metrics.dists]
type = "calculate_dists"
better = "lower"          # default "higher" ("lower" for calculate_dists); best-value tracking (a resumed best
                          # tracked the other way starts over, with a warning)
```

Validation runs on rank 0 with the EMA (or evaluation) weights and leaves the training RNG streams untouched.
`single` datasets skip the metrics. `test.py` writes `experiments/results/<name>/<dataset>/<image>_<suffix>.png`.

## QAT, multi-GPU, logging

- **QAT** (quantization-aware training, INT8 QDQ ONNX): AetherNet only, `train.enable_qat = true`. See
  [docs/aether.md](docs/aether.md).
- **DDP** (best effort: tested with gloo on one GPU, NCCL untested):
  `torchrun --nproc_per_node=2 train.py -opt config.toml --launcher pytorch`. `batch_size` is per process;
  `[dist_params]` takes `backend` and `port`. DataParallel was removed.
- **Logging**: `logger.use_tb_logger` (TensorBoard in `experiments/tb_logger/<name>`), `logger.save_tb_img`,
  `[logger.wandb]` (needs `use_tb_logger`): only `project` (enables W&B) and `resume_id` are read; the table
  is not validated, so a misspelt `project` silently leaves W&B off.

## adding an architecture

Archs are listed in one lazy table, `ARCHS` in `neosr/archs/__init__.py`. The old `@ARCH_REGISTRY.register()`
decorators were removed with the registry: they required importing every arch module (and parsing the config for
the scale) when the package was imported. Now only the module of the requested type is imported, the run `scale`
is passed explicitly under the arch's own kwarg name, and an unknown type is a configuration error with a
suggestion. `tests/test_hygiene.py` rejects `_REGISTRY` under `neosr/`.

1. Create `neosr/archs/<name>_arch.py` with a plain `nn.Module` class. Its constructor takes the scale as a
   keyword argument with a default; factories forward `**kwargs` and never set the scale themselves. No work at
   import time (no CUDA, argv, RNG or `torch.set_default_device`); create tensors on the input's device.

   ```python
   from torch import nn

   class myarch(nn.Module):
       def __init__(self, in_ch: int = 3, dim: int = 64, upscale: int = 4) -> None:
           ...

   def myarch_light(**kwargs):
       return myarch(dim=32, **kwargs)
   ```

2. Add one `ARCHS` row per class and factory: `type -> (module, scale kwarg)`. The type is the Python name in
   the module. Use `None` for networks without a scale (discriminators):

   ```python
   "myarch": ("myarch_arch", "upscale"), "myarch_light": ("myarch_arch", "upscale"),
   "mydisc": ("mydisc_arch", None),
   ```

3. Use it: `[network_g] type = "myarch"` (other keys go to the constructor) or `[network_d] type = "mydisc"`.
   The run `scale` is injected as `upscale`; an explicit contradicting `upscale` is an error.
4. Test: `pytest tests/test_registry.py tests/test_archs.py -k myarch`. Both pick the rows up: the row resolves,
   the scale kwarg reaches the constructor, the module imports without touching CUDA, and (on CUDA) the x2/x4
   forward has the right shape and the backward is finite. A new scale-free row must also be added to
   `NON_SR_ARCHS` in `tests/test_registry.py`.

## adding a loss

1. Subclass `neosr.losses.base.Loss`. `forward(pred, target)` returns the unweighted scalar; the trainer applies
   `loss_weight`. Set `fp32 = True` to run outside autocast with fp32 inputs, or a class `loss_weight` for a
   different default weight.

   ```python
   from torch import Tensor
   from neosr.losses.base import Loss

   class my_loss(Loss):
       def __init__(self, eps: float = 1e-6) -> None:
           super().__init__()
           self.eps = eps

       def forward(self, pred: Tensor, target: Tensor) -> Tensor:
           return ((pred - target) ** 2 + self.eps).sqrt().mean()
   ```

2. Override `evaluate(ctx)` only if the loss needs other inputs or models: the `Context` holds `sr`, `gt`, `lq`,
   `lq_up`, `it` (the update being computed) and the discriminator (`net_d`, `net_d_adv`); `ctx.log(key, value)`
   logs extra values (see `gan_loss.py`).
3. Add a row to `LOSSES` in `neosr/losses/__init__.py`: `"my_loss": ("my_loss", None)` (module, scale kwarg or
   `None`).
4. Use it with `[train.my_opt] type = "my_loss"`, `loss_weight = 0.5`, `eps = 1e-4`; it logs `l_g_my`. The builder
   moves it to the device and freezes its parameters. `tests/test_registry.py` checks the row.

## features

### supported archs

| arch | `type` | date | weight |
|---|---|---|---|
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | `esrgan` | 2021 | Medium heavy |
| [SRVGGNetCompact](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/srvgg_arch.py) | `compact` | 2021 | Ultra light |
| [SwinIR](https://github.com/JingyunLiang/SwinIR) | `swinir_small`, `swinir_medium`, `swinir_large` | 2021 | Medium heavy / Medium heavy / Heavy |
| [HAT](https://github.com/XPixelGroup/HAT) | `hat_s`, `hat_m`, `hat_l` | 2022 | Ultra heavy / Ultra heavy / Ultra heavy |
| [OmniSR](https://github.com/Francis0625/Omni-SR) | `omnisr` | 2023 | Medium heavy |
| [SRFormer](https://github.com/HVision-NKU/SRFormer) | `srformer`, `srformer_light`, `srformer_medium` | 2022 | Heavy / Medium heavy / Heavy |
| [DAT](https://github.com/zhengchen1999/dat) | `dat_s`, `dat_m`, `dat_2` | 2022 | Heavy / Heavy / Heavy |
| [DITN](https://github.com/yongliuy/DITN) | `ditn` | 2023 | Medium heavy |
| [DCTLSA](https://github.com/zengkun301/DCTLSA) | `dctlsa` | 2024 | Medium heavy |
| [SPAN](https://github.com/hongyuanyu/SPAN) | `span`, `span_fast` | 2023 | Medium / Light |
| [SPANPlus](https://github.com/umzi2/SPANPlus) | `spanplus`, `spanplus_sts`, `spanplus_s`, `spanplus_st` | 2024 | Medium / Ultra light / Medium / Light |
| [Real-CUGAN](https://github.com/bilibili/ailab) | `cugan` | 2022 | Medium heavy |
| [CRAFT](https://github.com/AVC2-UESTC/CRAFT-SR) | `craft` | 2023 | Medium |
| [SAFMN](https://github.com/sunny2109/SAFMN) | `safmn`, `safmn_l`, `light_safmnpp` | 2023 | Light / Medium / Light |
| [RGT](https://github.com/zhengchen1999/RGT) | `rgt`, `rgt_s` | 2023 | Medium heavy / Medium |
| [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary) | `atd`, `atd_light` | 2024 | Heavy / Heavy |
| [PLKSR](https://github.com/dslisleedh/PLKSR) | `plksr`, `plksr_tiny` | 2023 | Light / Light |
| [RealPLKSR](https://github.com/dslisleedh/PLKSR/issues/4) | `realplksr`, `realplksr_s`, `realplksr_l` | 2023 | Medium / Medium / Medium heavy |
| [DRCT](https://github.com/ming053l/DRCT) | `drct`, `drct_l`, `drct_xl`, `drct_s` | 2024 | Heavy / Ultra heavy / Ultra heavy / Heavy |
| [MSDAN](https://github.com/Supereeeee/MSDAN) | `msdan` | 2024 | Medium heavy |
| [HiT-SRF](https://github.com/XiangZ-0/HiT-SR) | `hit_srf`, `hit_srf_medium`, `hit_srf_large` | 2024 | Medium heavy / Heavy / Heavy |
| [HMA](https://github.com/korouuuuu/HMA) | `hma`, `hma_medium`, `hma_large` | 2024 | Medium heavy / Heavy / Heavy |
| [ASID](https://github.com/saturnian77/ASID) | `asid`, `asid_d8` | 2024 | Medium / Medium heavy |
| [MoSRV2](https://github.com/umzi2/MoSRV2) | `mosrv2` | 2024 | Medium |
| [ESC](https://github.com/dslisleedh/ESC) | `esc`, `esc_light`, `esc_fp`, `esc_large` | 2024 | Medium / Light / Light / Medium heavy |
| [CATANet](https://github.com/EquationWalker/CATANet) | `catanet` | 2024 | Medium heavy |
| [LMLT](https://github.com/jwgdmkj/LMLT) | `lmlt`, `lmlt_tiny`, `lmlt_large` | 2024 | Medium / Light / Medium |
| [PlainUSR](https://github.com/icandle/PlainUSR) | `plainusr`, `plainusr_ultra`, `plainusr_large` | 2024 | Medium / Light / Medium heavy |
| [SebiCA](https://github.com/idiosyncracies/Sebica) | `sebica`, `sebica_mini` | 2024 | Ultra light / Ultra light |
| [GRFormer](https://github.com/sisrformer/GRFormer) | `grformer`, `grformer_medium`, `grformer_large` | 2024 | Medium heavy / Heavy / Heavy |
| [KRGN](https://github.com/ZhangDY827/KRGN) | `krgn` | 2024 | Medium heavy |
| [CFSR](https://github.com/Aitical/CFSR) | `cfsr` | 2024 | Medium heavy |
| [MAN](https://github.com/icandle/MAN) | `man`, `man_tiny`, `man_light` | 2024 | Medium heavy / Medium / Medium |
| [EIMN](https://github.com/liux520/EIMN) | `eimn`, `eimn_a`, `eimn_l` | 2024 | Medium / Medium / Medium |
| [RCAN](https://github.com/yulunzhang/RCAN) | `rcan` | 2018 | Medium |
| [DeSRGAN](https://github.com/neosr-project/neosr/blob/master/neosr/archs/desrgan_arch.py) | `desrgan` | 2024 | Medium heavy |
| [DCTSR](https://github.com/neosr-project/neosr/blob/master/neosr/archs/dct_arch.py) | `dct` | 2024 | Medium |
| [MoeSR](https://github.com/umzi2/MoESR) | `moesr` | 2024 | Medium heavy |
| [SwinIR-Dense](https://github.com/neosr-project/neosr/blob/master/neosr/archs/swinirdense_arch.py) | `swinirdense` | 2024 | Ultra heavy |
| [FlexNet](https://github.com/umzi2/FlexNet) | `flexnet`, `metaflexnet` | 2024 | Medium heavy / Medium heavy |
| [NinaSR](https://github.com/Coloquinte/torchSR/blob/main/doc/NinaSR.md) | `ninasr`, `ninasr_b0`, `ninasr_b2` | 2020 | Light / Light / Medium |
| [HASN](https://github.com/nathan66666/HASN) | `hasn` | 2024 | Medium heavy |
| [AetherNet](https://github.com/phhofm/aethernet) ([docs](docs/aether.md), QAT) | `aether_small`, `aether_medium`, `aether_large`, `aether` | - | - |

> [!NOTE]
> Arch-specific parameters: [wiki](https://github.com/neosr-project/neosr/wiki/Arch%E2%80%90specific-options).
> Weights are approximate resource demands (compute + VRAM) from `architecture_categories.json` plus estimates.

Under testing: [MFG-HMoE](https://github.com/Mr-Bamboo/MFG-HMoE) `mfghmoe` (2024, Ultra heavy),
[MicroSR](https://github.com/YanHuiGUO/Submission-for-NTIRE-2025-SR-x4-) `microsr`, `microsr_light` (2025, Light /
Ultra light).

### supported discriminators

| net | `type` | date |
|---|---|---|
| U-Net w/ SN | `unet` | 2015 |
| [PatchGAN](https://github.com/NVIDIA/pix2pixHD) w/ SN | `patchgan` | 2018 |
| EA2FPN (bespoke, based on [A2-FPN](https://github.com/lironui/A2-FPN)) | `ea2fpn` | 2024 |
| [DUnet](https://github.com/umzi2/DUnet) | `dunet` | 2024 |
| [MetaGan](https://github.com/umzi2/MetaGan) | `metagan` | 2024 |
| [ResGAN](https://github.com/neosr-project/neosr/blob/master/neosr/archs/resgan_arch.py) | `resgan` | 2023 |

### other components

| kind | options |
|---|---|
| models (`model_type`) | `image` (paired SISR, optional discriminator), `otf` (`image` + on-the-fly Real-ESRGAN degradations) |
| datasets | `paired`, `otf`, `single` |
| optimizers | [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) `adam`, [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) `adamw`, [NAdam](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html) `nadam`, [Adan](https://github.com/sail-sg/Adan) `adan`, [AdamW Schedule-Free](https://github.com/facebookresearch/schedule_free) `adamw_sf`, [Adan Schedule-Free](https://github.com/neosr-project/adan_schedule_free) `adan_sf` |
| schedulers | `multisteplr`, `cosineannealing` |
| metrics | PSNR `calculate_psnr`, SSIM `calculate_ssim`, [DISTS](https://github.com/dingkeyan93/DISTS) `calculate_dists`, [TOPIQ](https://github.com/chaofengc/IQA-PyTorch) `calculate_topiq` |
| augmentations | flip `use_hflip`, rotation `use_rot`, [MixUp](https://arxiv.org/abs/1710.09412) `mixup`, [CutMix](https://arxiv.org/abs/1905.04899) `cutmix`, [ResizeMix](https://arxiv.org/abs/2012.11101) `resizemix`, [CutBlur](https://github.com/clovaai/cutblur/) `cutblur` |

## changes since the refactor

- **Log semantics**: logged values are means over the logging window, not last-micro-batch snapshots;
  `l_d_real`/`l_d_fake` are no longer divided by `accumulate`; `l_d_total` is their sum (about twice the old
  mean); new `grad_norm_g/d` and `skipped_g/d`; `gan_weight` is gone.
- **GAN term is live**: the generator's adversarial loss now carries gradient (it was computed without one), and
  D trains in train mode. `l_g_gan` values differ from old logs.
- **DISTS** applies ImageNet input normalisation in both `dists_loss` and `calculate_dists`, as in the paper;
  values shift against old logs.
- **Removed**: SAM/FSAM, ECO, DataParallel (`num_gpu > 1` needs a launcher), `warmup_iter` (`adamw_sf`/`adan_sf`
  have `warmup_steps`; the other optimizers have no warmup), the `soap_sf` and `adamw_win` optimizers, `gw`/`color`/`luma`/`resnet_perceptual` losses and
  the perceptual `patchloss`/`ipk` options, the throughput kill-switch, and the legacy model types `default`,
  `simple_otf`, `eqr_otf`, `pancam_otf` with their datasets and the `pancam`/`eqrsrgan` archs.
- **Checkpoints**: training states are format 2 (old `.state` files cannot be resumed; old `net_*.pth` load as
  pretrained weights); only the newest `logger.keep_states` states are kept.
- **Templates (stage 2)**: generator AdamW 2e-4 (schedule-free `adan_sf` with `betas = [0.98, 0, 0.99]` noted
  as the alternative), an L1 term, FDL(DINOv2) weight 0.02 (was 0.75, ~99% of the generator gradient), no batch
  augmentation by default, `train.clamp = "restore"`. On the realplksr GAN template (8k updates, held-out PSNR/SSIM/
  DISTS) this moved 27.23/0.627/0.2024 to 29.36/0.721/0.1787; see `docs/refactor/stage2/DECISIONS.md`.
- **Stage 2**: `manual_seed` no longer implies deterministic kernels (add `deterministic = true`); `fast_matmul`
  no longer enables fp16 GEMM accumulation (`fp16_accumulation = true`); a skipped update restores the buffers its
  forwards wrote and a non-finite loss in it is counted (`nonfinite_<key>`) instead of stopping the run; the fp16
  scaler grows every 200 updates (was 2000); `train.clamp` defaults to `"restore"` (`true` = the old hard clamp);
  new `train.grad_clip = <norm> | "spike"`,
  `warmup_steps` for every optimizer, `aug_multi_prob`, `logger.keep_exports`.
- **Configuration**: unknown keys warn instead of being silently ignored, removed keys are errors, and
  `match_lq_colors` became the per-loss `target = "lq"`.
- **`blur_kernel_size`, `blur_kernel_size2`** (OTF) were removed: they were never read. Kernel sizes are sampled
  as odd values from 7 to 21, as before; old configs get an unknown-key warning.

## datasets

As part of *neosr*, we have released a series of datasets. The purpose of these datasets is to distill only the best images from the academic and community datasets. A total of 14+ datasets were manually reviewed and processed, including: [Adobe-MIT-5k](https://data.csail.mit.edu/graphics/fivek/), [RAISE](http://loki.disi.unitn.it/RAISE/), [LSDIR](https://data.vision.ee.ethz.ch/yawli/), [LIU4k-v2](https://structpku.github.io/LIU4K_Dataset/LIU4K_v2.html), [KONIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html), [Nikon LL RAW](https://www.kaggle.com/datasets/razorblade/nikon-camera-dataset), [DIV8k](https://ieeexplore.ieee.org/document/9021973), [FFHQ](https://github.com/NVlabs/ffhq-dataset), [Flickr2k](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [ModernAnimation1080_v2](https://huggingface.co/datasets/Zarxrax/ModernAnimation1080_v2), [Rawsamples](https://www.rawsamples.ch/index.php/en/), [SignatureEdits](https://www.signatureedits.com/free-raw-photos/), [Hasselblad raw samples](https://www.hasselblad.com/learn/sample-images/) and [Unsplash](https://unsplash.com/).

- BHI series from [@Phhofm](https://github.com/Phhofm). Read more about the selection process [here](https://huggingface.co/blog/Phips/bhi-filtering). Full version is **recommended** for large networks, while small version is **recommended** for lightweight SISR:

|   dataset    |                         download                                |
|--------------|-----------------------------------------------------------------|
| BHI full     | [Release page](https://huggingface.co/datasets/Phips/BHI)       |
| BHI small    | [Release page](https://huggingface.co/datasets/Phips/BHI_small) |
| BHI mini     | [Release page](https://huggingface.co/datasets/Phips/BHI_mini)  |
| BHI100 (val) | [Release page](https://huggingface.co/datasets/Phips/BHI100)    |

- `Nomos-v2`: contains 6000 images, multipurpose.
- `nomos_uni`: contains 2989 images, multipurpose. Meant to be used on lightweight networks (<800k parameters).

| dataset download | sha256 |
|---|---|
| [**nomosv2**](https://drive.google.com/file/d/1vqKWGtpkYHKv8uqK_xgYZbYuPHJS2l8j/view?usp=drive_link) (3GB) | [sha256](https://drive.google.com/file/d/12eNzPqHd2N1rTWMDh_rAv3urNypJexQT/view?usp=drive_link) |
| [**nomosv2.lmdb**](https://drive.google.com/file/d/1Rzdjt3w0qXle7vHa8FeFltmyKTMIwPR5/view?usp=drive_link) (3GB) | [sha256](https://drive.google.com/file/d/1IrDjI37psiCc-Khn3_KSyov-xP4txZYe/view?usp=drive_link) |
| [nomosv2_lq_4x](https://drive.google.com/file/d/1YiCywSFwRuwaYmnZ0TgoWDvcDQifAsZo/view?usp=drive_link) (187MB) | [sha256](https://drive.google.com/file/d/1iOOte6h-AE1iD-i5wl_gVx1uJzNTS4Cq/view?usp=drive_link) |
| [nomosv2_lq_4x.lmdb](https://drive.google.com/file/d/1SE3wN2CmftBy9r9qOnizb33xtySezWnM/view?usp=drive_link) (187MB) | [sha256](https://drive.google.com/file/d/1bpuuiGFNBrDuZiRSP5hpVgFQx44MImay/view?usp=drive_link) |
| [nomos_uni](https://drive.google.com/file/d/1LVS7i9J3mP9f2Qav2Z9i9vq3y2xsKxA_/view?usp=sharing) (1.3GB) | [sha256](https://drive.google.com/file/d/1cdzVSpXQQhcbRVuFPbNtb6mZx_BoLwyW/view?usp=sharing) |
| [nomos_uni_lq_4x](https://drive.google.com/file/d/1uvMl8dG8-LXjCOEoO9Aiq5Q9rd_BIUw9/view?usp=sharing) | [sha256](https://drive.google.com/file/d/1MTJBcfaMYdfWhsZCWEEOwbKSdmN5dVwl/view?usp=drive_link) |

### community datasets

Datasets made by the upscaling community. More info can be found in author's repository.

- `digital_art_v3`: Digital art dataset from [@umzi2](https://github.com/umzi2).
- `ModernAnimation_v3`: High-quality modern anime dataset from Zarxrax.
- `ClassicAnimation`: High-quality analog (cel) anime dataset from Zarxrax.
- `DF2k-BHI`: a curated version of the classic DF2k dataset, made by [@Phhofm](https://github.com/Phhofm).
- `4xNomosRealWeb Dataset`: realistically degraded LQ's for Nomos-v2 dataset (from [@Phhofm](https://github.com/Phhofm)).
- `FaceUp`: Curated version of [FFHQ](https://github.com/NVlabs/ffhq-dataset).
- `SSDIR`: Curated version of [LSDIR](https://data.vision.ee.ethz.ch/yawli/).
- `ArtFaces`: Curated version of [MetFaces](https://github.com/NVlabs/metfaces-dataset).
- `Nature Dataset`: Curated version of [iNaturalist](https://github.com/visipedia/inat_comp/tree/master/2017).

| dataset | download |
|---|---|
| [@Phhofm](https://github.com/Phhofm) DF2k-BHI | [HuggingFace](https://huggingface.co/datasets/Phips/DF2K_BHI) |
| [@Phhofm](https://github.com/Phhofm) 4xNomosRealWeb | [Release page](https://github.com/Phhofm/models/releases/tag/4xnomosrealweb_dataset) |
| [@Phhofm](https://github.com/Phhofm) FaceUp | [GDrive (4GB)](https://drive.google.com/file/d/1WFY0siR_ERVSnE2p7ouiCfV3wQizpAKr/view) |
| [@Phhofm](https://github.com/Phhofm) SSDIR | [Gdrive (4.5GB)](https://drive.google.com/file/d/1FA8Q-T3xZ6_KA7SHYgoa6idIS7xpdrl4/view) |
| [@Phhofm](https://github.com/Phhofm) ArtFaces | [Release page](https://github.com/Phhofm/models/releases/tag/dataset_artfaces) |
| [@Phhofm](https://github.com/Phhofm) Nature Dataset | [Release page](https://github.com/Phhofm/models/releases/tag/nature_dataset) |
| [@umzi2](https://github.com/umzi2) Digital Art (v3) | [Release page](https://huggingface.co/datasets/umzi/DigitalArtV3) |
| @Zarxrax ClassicAnimation | [Release page](https://huggingface.co/datasets/Zarxrax/ClassicAnimation) |
| @Zarxrax ModernAnimation_v3 | [Release page](https://huggingface.co/datasets/Zarxrax/ModernAnimation1080_v3) |

## resources

- [Training Guide](https://github.com/Sirosky/Upscale-Hub/wiki/%F0%9F%93%88-Training-a-Model-in-NeoSR) from [@Sirosky](https://github.com/Sirosky)
- Philip's [youtube channel](https://www.youtube.com/@philip_hofmann/videos)
- Dataset processing tools: [WTP Dataset Destroyer](https://github.com/umzi2/wtp_dataset_destroyer) and [SR dataset creator](https://github.com/stinkybread/super_resolution_dataset_creator)
- [OpenModelDB](https://openmodeldb.info/)
- [chaiNNer](https://chainner.app/)

## license and acknowledgements

Released under the [Apache license](license.txt).
All licenses listed on [license/readme](license/readme.md).
This code was originally based on [BasicSR](https://github.com/XPixelGroup/BasicSR).

Thanks to [victorca25/traiNNer](https://github.com/victorca25/traiNNer), [styler00dollar/Colab-traiNNer](https://github.com/styler00dollar/Colab-traiNNer/) and [timm](https://github.com/huggingface/pytorch-image-models) for providing helpful insights into some problems.

Thanks to active contributors [@Phhofm](https://github.com/Phhofm), [@Sirosky](https://github.com/Sirosky), and [@umzi2](https://github.com/umzi2) for helping with tests and bug reporting.
