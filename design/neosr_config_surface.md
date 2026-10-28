# NeoSR training-configuration surface (reference for a TOML-generating web UI)

Scope: what `train.py`, `test.py` and `convert.py` accept, read from the code in `C:\work2000\neosr-dev` (branch `master`, `pyproject` version 1.0.0, Python >=3.13,<3.14, post-refactor) plus the two wiki pages. Claims come from code I read; `(?)` marks anything inferred or not verified by running. Where wiki and code disagree, the code (and `readme.md`, which says it describes the current code) wins.

Counts: 93 generator type strings in 45 families, 6 discriminators, 15 loss types, 6 optimizers, 2 schedulers, 4 metrics, 3 dataset types, 2 model types, 4 batch augmentations.

## 0. Provenance and caveats

| topic | finding |
|---|---|
| Config parser | `neosr/utils/options.py` does not exist. Loading and validation live in `neosr/config.py` (`load_config`, `validate_train`); argv parsing in `neosr/cli.py`. TOML only (`.toml` suffix enforced); no YAML, no `force_yml`. |
| Registries | No `@ARCH_REGISTRY.register`. Each package has one lazy table: `ARCHS` (`neosr/archs/__init__.py`), `LOSSES` (`neosr/losses/__init__.py`), `DATASETS` (`neosr/data/__init__.py`), `MODELS` (`neosr/models/__init__.py`), `METRICS` (`neosr/metrics/__init__.py`), `OPTIMIZERS` / `SCHEDULERS` (`neosr/models/base.py`). Unknown type = `ConfigError` with a "did you mean" hint. |
| Wiki drift | Configuration-Walkthrough still documents removed keys (`eco*`, `sam*`, `warmup_iter`) and calls `total_iter` epoch based. In code every counter is in optimizer updates. The Upscale-Hub guide targets an older version (Compact; L1 + perceptual + color + GAN losses). |
| Removed keys (hard `ConfigError` if present) | `train.sam`, `train.sam_init`, `train.eco`, `train.eco_init`, `train.eco_iters`, `train.eco_schedule`, `train.warmup_iter` (use `warmup_steps`), `train.gw_opt`, `train.color_opt`, `train.luma_opt`, `path.ignore_resume_networks`. Gone but not listed (so they only warn or fail as unknown type): optimizers `soap_sf`, `adamw_win`, `fsam`; model types `default`, `simple_otf`, `eqr_otf`, `pancam_otf`; losses `resnet_perceptual`; `blur_kernel_size(2)` (warns). |
| First-run downloads | `dists_loss` / `calculate_dists`: `dists_weights.pth` from huggingface.co plus torchvision VGG16 weights. `fdl_loss model="dinov2"`: `torch.hub` `facebookresearch/dinov2`. `vgg_perceptual_loss`, `fdl_loss model="vgg"`: torchvision VGG19. `ea2fpn`: torchvision ResNet18. A UI should warn these need network access on first use. |

## 1. Top-level structure

```toml
name = "4x_mymodel"            # required
model_type = "image"           # required: image | otf
scale = 4                      # required
# ... scalar run keys (section 3)

[datasets.train]               # training: required
[datasets.val]  [datasets.val_1] ...   # optional validation sets
[degradations]                 # OTF only; merged into datasets.train
[network_g]   [network_d]      # network_g required; network_d optional (GAN)
[train]  [train.optim_g]  [train.optim_d]  [train.scheduler]
[train.<slot>_opt] ...         # one table per loss term
[val]  [val.metrics.<name>]
[logger]  [logger.wandb]
[path]
[dist_params]                  # DDP only
```

| section | training (`train.py`) | inference (`test.py`) |
|---|---|---|
| root scalars | `name`, `model_type`, `scale` required | same |
| `[datasets.train]` | required: `type`, `patch_size`, `batch_size` | not used |
| `[datasets.val*]` | optional, any number; each needs `name` in practice (code indexes `ds["name"]`) | not used |
| `[datasets.test_N]` | rejected ("test datasets belong in an inference config") | one per test set; `type = "single"` (LQ only, no metrics) or `"paired"` |
| `[degradations]` | OTF only | ignored |
| `[network_g]` | required | required |
| `[network_d]` | optional; needs `train.optim_d` and `train.gan_opt` | ignored |
| `[train]` | required: `optim_g` + at least one generator loss with non-zero weight | ignored |
| `[val]` | optional | optional: `tile`, `tile_overlap`, `save_img`, `suffix`, `metrics` |
| `[logger]` | `save_checkpoint_freq` required | ignored |
| `[path]` | optional | `pretrain_network_g` (plus `param_key_g`, `strict_load_g`, `results_root`) |

## 2. Entry points, CLI and config loading

| command | arguments | notes |
|---|---|---|
| `train.py` | `-opt FILE` (required), `--launcher none\|pytorch\|slurm` (default `none`), `--debug`, `--auto_resume` | install aliases: `neosr-train FILE`, `neosr-test FILE`, `neosr-convert`. `--debug`: name becomes `debug_<name>`, `val_freq=8`, `print_freq=1`, `save_checkpoint_freq=8`, no TensorBoard. |
| `test.py` | `-opt FILE` (required), `--launcher`, `--debug` | runs `model.validation()` per dataset; writes `experiments/results/<name>/<dataset>/<img>_<suffix>.png` and logs fps |
| `convert.py` | `--input` (req), `-net/--network` (req), `-s/--scale` (4), `-window`, `--output`, `-onnx`, `--strip-qat`, `-prune`, `-opset` (17), `-static C H W`, `-nocheck`, `-fp16` (no-op), `-optimize`, `-fulloptimization` (disabled) | builds `-net` on CPU, loads `--input`, writes `<out>_fp32.onnx` (+ `_fp32_optimized.onnx` with `-optimize`); `craft` and `ditn` cannot be exported; a QAT checkpoint exports INT8 QDQ (opset >= 13); `--strip-qat` writes `<input>_float.pth` |
| `create_lmdb.py` | `--input`, `--output` | builds an LMDB plus `meta_info.txt` |

Exit codes: 0 done (or first Ctrl-C after save); 1 unexpected (incl. non-finite training); 2 `ConfigError` or bad argv; 130 second Ctrl-C. `auto_resume` and `launcher` are CLI-only; neither is read from TOML.

Loader behaviour (`neosr/config.py`):

| step | rule |
|---|---|
| 1 | Required root keys: `name`, `model_type`, `scale`, `network_g`, `datasets`. |
| 2 | Every key outside the opaque tables is type-checked against `SCHEMA`. Wrong type = `ConfigError`. Unknown key = `ConfigWarning` (with suggestion) and ignored. `float` fields accept int, never bool. |
| 3 | Opaque tables (validated by constructors; unknown kwargs become `ConfigError`): `network_g`, `network_d`, `train.*_opt`, `train.optim_g`, `train.optim_d`, `train.scheduler`, `val.metrics.*`, `logger.wandb`. |
| 4 | Dataset phase names: `train`, or `val[_N]` / `test[_N]`. |
| 5 | No defaults are filled at load; readers use `.get`. Defaults below are the readers'. |
| 6 | `~` is expanded in `dataroot_*` and path keys. Windows paths need TOML single quotes (the parse error says so). Relative `path.resume_state` is relative to the repo root. |
| 7 | `[degradations]` keys are copied into `[datasets.train]`; conflicts warn and `[degradations]` wins. |

## 3. Root scalar keys

| key | type | default | allowed / notes |
|---|---|---|---|
| `name` | str | required | run folder name; `debug` in the name forces debug frequencies |
| `model_type` | str | required | `image` or `otf` (only two) |
| `scale` | int | required | training: 1, 2, 3, 4, 8. Per-arch limits in section 5 |
| `num_gpu` | int or str | `"auto"` | int > 1 needs `--launcher` (DataParallel removed) |
| `manual_seed` | int | unset (random 1024..10000, logged) | setting it enables deterministic algorithms (slower) |
| `use_amp` | bool | false | fp16 autocast + GradScaler; nets and inputs become channels-last |
| `bfloat16` | bool | false | requires `use_amp` (error otherwise); error on compute capability <= 6; warning on 7.x |
| `fast_matmul` | bool | false | TF32 and reduced matmul precision |
| `compile` | bool | false | training only, generator only; **error on Windows** (no Triton) |
| `compile_mode` | str | `"default"` | torch.compile mode |
| `find_unused_parameters` | bool | false | DDP |
| `input_size`, `in_chans` | int | unset, 3 | with `input_size` and `torchprofile` installed, logs MACs |
| `print_network` | bool | false | also accepted as `path.print_network` |

Templates ship `use_amp = true`, `bfloat16 = true`, `fast_matmul = true`, `#compile = true`, `#manual_seed = 1024`. Walkthrough wiki: AMP on Turing or newer, bf16 on Ampere or newer (may be unstable, some ops lack bf16), compile Linux-only and experimental, `manual_seed` slows training. Upscale-Hub (older) tells beginners to leave AMP, bf16, compile off. Runtime logs a hint when `use_amp` is off on Ampere or newer.

## 4. Datasets

### 4.1 Types and keys

`DATASETS` = `paired`, `otf`, `single`. `type` is required in every `[datasets.*]` table. `scale` and `phase` are injected.

| key | type | default | applies to | meaning |
|---|---|---|---|---|
| `type` | str | required | all | `paired` (train, val, test), `otf` (train only, with `model_type="otf"`), `single` (inference/val without metrics) |
| `name` | str | (phase) | val, test | needed in practice: `validate()` and `test.py` read `ds["name"]`; used in logs and output folders |
| `dataroot_gt` | str | required | paired, otf | folder (scanned recursively) or LMDB folder (both roots must end in `lmdb`; otf: `.lmdb`) |
| `dataroot_lq` | str | required | paired, single | otf rejects it (`ConfigError`) |
| `meta_info` (legacy `meta_info_file`) | str | unset | all | text file, one name per line (first space-separated token used); relative to the roots; skips folder scan |
| `patch_size` | int | required (train) | train | **LQ** patch size; GT crop = `patch_size * scale`. There is no `gt_size`. Common: 32, 48, 64, 96, 128; transformers need multiples of their window size (wiki) |
| `batch_size` | int | required (train) | train | per process |
| `accumulate` | int | 1 | train | micro-batches per update; loss divided by it; effective batch = `batch_size * accumulate` (x world size) |
| `num_worker_per_gpu` | int or `"auto"` | 4 | train | `"auto"` = CPUs / world size; val/test use 0 workers, batch 1 |
| `pin_memory` | bool | true if CUDA | train | |
| `persistent_workers` | bool | true if workers > 0 | train | needs `num_worker_per_gpu > 0` (`ConfigError`) |
| `dataset_enlarge_ratio` | int | 1 | train | repeats the dataset per epoch (epoch length only; training is update based) |
| `use_hflip`, `use_rot` | bool | true, true | paired train, otf | flip, rotation (vflip + transpose) |
| `color` | str | rgb | all | `"y"` = grayscale (1 channel; arch `in_ch` must match, `in_chans` is separate (?)); anything else = color. 4-channel inputs stay RGBA |
| `mean`, `std` | list | unset | all | optional normalisation of lq and gt |
| `augmentation` | list[str] | unset (off) | train | subset of `"none"`, `"mixup"`, `"cutmix"`, `"resizemix"`, `"cutblur"`; `["none"]` = off |
| `aug_prob` | list[float] | uniform | train | same length as `augmentation`; relative weights, normalised; 0 disables an entry; negative or all-zero errors |
| `queue_size` | int | 180 | otf | OTF pair pool; must be >= `batch_size` |
| OTF degradation keys | see 4.2 | | otf | also valid in `[degradations]` |

Folder mode: both folders scanned recursively for `.jpg .jpeg .png .webp .tiff .tif`; every GT file needs the same relative path in the LQ folder; counts must match. LQ must be exactly `scale` times smaller than GT (checked on crop; mismatch raises with the file path). LQ smaller than `patch_size` raises. Validation/test crops GT to `lq * scale`.

Augmentation behaviour (hard-coded params, only `augmentation` and `aug_prob` are configurable): one entry is drawn by weight; with probability 0.3 (fixed) 2..n are applied together, in order cutmix, mixup, resizemix, cutblur. Anything except `cutblur` alone needs `batch_size > 1`. Defaults inside the functions: mixup alpha 0.4-0.6, cutmix alpha 0.9, resizemix scope (0.2, 0.9), cutblur alpha 0.7. Wiki: CutBlur is for real-world SR and may hurt bicubic-only training.

### 4.2 OTF degradations (`[degradations]` or `[datasets.train]`, `model_type = "otf"`)

`Degrader` raises `ConfigError` unless every stage key below exists in both stages (suffix `""` and `"2"`), plus `second_blur_prob`; `jpeg_range`/`jpeg_range2` are required unless `jpeg_compress = false`. `sinc_prob`, `sinc_prob2`, `final_sinc_prob` and the kernel keys are read by the dataset without a presence check and would crash if missing (?), so treat all as required. Values below are identical in all 42 shipped `*_otf.toml` templates.

| key (stage 1 / stage 2) | type | template value | meaning |
|---|---|---|---|
| `kernel_list` / `kernel_list2` | list[str] | `["iso","aniso","generalized_iso","generalized_aniso","plateau_iso","plateau_aniso"]` | blur kernel kinds; code also knows `skewed`, `skewed_aniso` |
| `kernel_prob` / `kernel_prob2` | list[float] | `[0.45,0.25,0.12,0.03,0.12,0.03]` | same length as the list |
| `sinc_prob` / `sinc_prob2` | float | 0.1 / 0.1 | sinc filter probability |
| `blur_sigma` / `blur_sigma2` (+ `blur_sigmay`, `blur_sigmay2`; default = x) | list[2] | `[0.2,3]` / `[0.2,1.5]` | |
| `betag_range`, `betap_range` (+ `2`) | list[2] | `[0.5,4]`, `[1,2]` | generalized / plateau shape |
| `resize_prob` / `resize_prob2` | list[3] | `[0.3,0.4,0.3]` | up, down, keep |
| `resize_range` / `resize_range2` | list[2] | `[0.5,1.5]` / `[0.3,1.5]` | |
| `gaussian_noise_prob` / `...2` | float | 0.2 / 0.2 | else Poisson noise |
| `noise_range` / `noise_range2` | list[2] | `[0,2]` / `[0,2]` | |
| `poisson_scale_range` / `...2` | list[2] | `[0.05,0.25]` / `[0.05,0.1]` | |
| `gray_noise_prob` / `...2` | float | 0.1 / 0.1 | |
| `jpeg_range` / `jpeg_range2` | list[2] | `[40,95]` / `[35,95]` | quality |
| `second_blur_prob` | float | 0.4 | |
| `final_sinc_prob` | float | 0.8 | |
| `rotation` | list[2] | `[-pi, pi]` | kernel rotation, optional |
| `jpeg_compress` | bool | true | false drops JPEG stages |

Kernel size is sampled from odd 7..21 (hard-coded). GT is cropped or reflect-padded to `ceil(max(512, patch_size*scale)/scale)*scale`; `patch_size` is the LQ patch after degradation. `otf` model + `otf` dataset must go together: the `otf` trainer reads `kernel1`/`kernel2`/`sinc_kernel` that only the `otf` dataset yields, the `image` trainer reads `lq` (?). Validation sets stay `paired`.

## 5. Networks

`[network_g]` / `[network_d]`: `type` plus the constructor kwargs of that arch (unknown kwargs = `ConfigError`). The run `scale` is injected into the arch's own kwarg name (column "scale kw"); an explicit contradicting value errors, so a UI must not emit it (the shipped `train_omnisr*.toml` sets `upsampling = 4`, which breaks for scale != 4). Discriminators take no scale. 100 rows in `ARCHS`: 93 generators, 6 discriminators, 1 helper (`VGGFeatureExtractor`, not a user network).

Scale limits found in code: `cugan` rejects 1x; `aether*` supports 2,3,4,8; `swinir_large` and any transformer using `upsampler='nearest+conv'` (`atd`, `hit_srf`, `srformer`, `swinir`, `swinirdense`) assert x4. Resource notes from `tests/test_archs.py`: `hit_srf_large` needs >9.6 GiB per forward; `hit_srf_medium` train forward needs >20 GiB. `architecture_categories.json` and the readme weight column group archs as Ultra light / Light / Medium / Medium heavy / Heavy / Ultra heavy.

### 5.1 Generators (45 families; type strings are exact `[network_g] type` values)

Params are constructor defaults (`in_chans`/`num_in_ch` = 3 omitted). "Presets" show only what differs from the base class.

| family (module) | type strings | scale kw | key params (default) / presets |
|---|---|---|---|
| aether (QAT-capable) | `aether`, `aether_small`, `aether_medium`, `aether_large` | scale | embed_dim 96, depths (4,4,4,4), mlp_ratio 2.0, drop_path_rate 0.1, lk_kernel 11, sk_kernel 3, fused_init False, quantize_residual True, use_channel_attn/use_spatial_attn True. small: 96, no attn; medium: 128, (6,6,6,6); large: 180, 5x8 |
| asid | `asid`, `asid_d8` | scale | num_feat 48, res_num 3, block_num 1, window_size 8, flash_attn True, d8 False, pe False, drop 0.0. d8: res_num 8 |
| atd | `atd`, `atd_light` | upscale | img_size 96, embed_dim 210, depths 6x6, num_heads 6x6, window_size 16, category_size 256, num_tokens 128, upsampler pixelshuffle. light: 64, 48, 6x4, heads 4, category 128, tokens 64, pixelshuffledirect |
| catanet | `catanet` | upscale | dim 40, block_num 8, qk_dim 36, mlp_dim 96, heads 4 |
| cfsr | `cfsr` | upscale | embed_dim 48, depths (6,6), dw_size 9, mlp_ratio 2.0, upsampler pixelshuffledirect |
| compact | `compact` | upscale | num_feat 64, num_conv 16, act_type prelu |
| craft | `craft` | upscale | embed_dim 48, depths [2,2,2,2], heads [6]x4, window_size 16, split_size_0 4, split_size_1 16, flash_attn False (not ONNX-exportable) |
| cugan | `cugan` | scale | in_channels 3, out_channels 3, pro True (1x unsupported) |
| dat | `dat_s`, `dat_m`, `dat_2` | upscale | embed_dim 180, depth [6]x6, heads [6]x6; split_size [8,16]/[8,32]/[8,32], expansion_factor 2/4/2 |
| dct | `dct` | upscale | img_size 64, embed_dim 80, depth [20], heads [8], expansion_factor 4.0, resi_connection 3conv, upsampler pixelshuffledirect |
| dctlsa | `dctlsa` | upscale | nf 55, num_modules 6, num_head 5 |
| desrgan | `desrgan` | scale | num_feat 64, num_block 23, num_grow_ch 32 |
| ditn | `ditn` | upscale | dim 60, ITL_blocks 4, SAL_blocks 4, UFONE_blocks 1, ffn_expansion_factor 2, patch_size 8 (not ONNX-exportable) |
| drct | `drct`, `drct_l`, `drct_xl`, `drct_s` | upscale | img_size 64, embed_dim 180, depths 6x6, window_size 16, compress_ratio 3, squeeze_factor 30, gc 32, use_checkpoint False. l: 12 groups; xl: 14; s: embed 48, depths 2x4 |
| eimn | `eimn`, `eimn_l`, `eimn_a` | scale | embed_dims 64, depths 1, mlp_ratios 2.66, num_stages 16, norm BatchNorm2d. a: num_stages 14 (`eimn` and `eimn_l` are identical) |
| esc | `esc`, `esc_light`, `esc_fp`, `esc_large` | upscaling_factor | dim 64, pdim 16, kernel_size 13, n_blocks 5, window_size 32, num_heads 4, exp_ratio 1.25, attn_type sdpa, use_dysample True, realsr True. light: n_blocks 3; fp: dim 48, heads 3, is_fp, no dysample; large: 10 blocks, exp_ratio 2.0 |
| esrgan | `esrgan` | scale | num_feat 64, num_block 23, num_grow_ch 32 (supports 1, 2, 4) |
| flexnet | `flexnet`, `metaflexnet` | scale | dim 64, num_blocks (6)x6, window_size 8, hidden_rate 4, pipeline_type linear, upsampler ps, flash_attn True. meta: dim 48, blocks (4,6,6,8), pipeline meta |
| grformer | `grformer`, `grformer_medium`, `grformer_large` | upscale | img_size 64, embed_dim 60, depths 6x4, heads 3x4. medium: 180, 6x6, heads 6; large: 180, 12x6 |
| hasn | `hasn` | upscale | feature_channels 52 |
| hat | `hat_s`, `hat_m`, `hat_l` | upscale | window_size 16, depths 6x6, upsampler pixelshuffle. s: embed 144, compress 24, squeeze 24; m: embed 180, compress 3, squeeze 30; l: 12 groups |
| hit_srf | `hit_srf`, `hit_srf_medium`, `hit_srf_large` | upscale | embed 60, depths 6x4, base_win_size [8,8], pixelshuffledirect. medium: img_size 48, win [16,16], embed 180, 6x6; large: win [24,24], 12 groups (VRAM heavy) |
| hma | `hma`, `hma_medium`, `hma_large` | upscale | embed 60, depths 6x4, window_size 8, interval_size 4. medium: window 16, 180, 6x6; large: img_size 48, 12 groups |
| krgn | `krgn` | scale | n_feats 64, n_resgroups 9, act lrelu, dilation 3 |
| lmlt | `lmlt`, `lmlt_tiny`, `lmlt_large` | upscaling_factor | dim 60, n_blocks 8, ffn_scale 2.0, window_size 8. tiny: 36; large: 84 |
| man | `man`, `man_tiny`, `man_light` | scale | n_resblocks 36, n_resgroups 1, n_feats 180. tiny: 5, 48; light: 24, 60 |
| mfghmoe (under testing) | `mfghmoe` | upscale | HAT-like: embed 180, depths 6x6, window 16, compress 3, squeeze 30 |
| microsr (under testing) | `microsr`, `microsr_light` | upscale | embed 180, patch_size 32, window 16, gc 32, rgb_norm False. light: 48, depths 2x4, window 8 |
| moesr | `moesr` | scale | dim 64, n_blocks 9, n_block 4, expansion_factor 8/3, expansion_msg 1.5, upsampler pixelshuffledirect |
| mosrv2 | `mosrv2` | scale | n_block 24, dim 64, expansion_ratio 1.5, mid_dim 32, unshuffle_mod True |
| msdan | `msdan` | upscale_factor | channels 48, num_DFEB 8 |
| ninasr | `ninasr`, `ninasr_b0`, `ninasr_b2` | scale | n_resblocks 26, n_feats 32, expansion 2.0. b0: 10, 16; b2: 84, 56 |
| omnisr | `omnisr` | upsampling (factory arg) | factory(upsampling=4, window_size=8) -> res_num 5, block_num 1, num_feat 64, pe True, bias True, ffn_bias True |
| plainusr | `plainusr`, `plainusr_ultra`, `plainusr_large` | scale | n_feat 64, im_feat (64,48,32), attn_feat 16. ultra: 32,(32,16),4; large: 80,(80,64,48),16 |
| plksr | `plksr`, `plksr_tiny` | upscaling_factor | dim 64, n_blocks 28, ccm_type DCCM, kernel_size 17, split_ratio 0.25, lk_type PLK, use_ea True. tiny: 12 blocks, kernel 13, no ea |
| rcan | `rcan` | scale | n_resgroups 10, n_resblocks 20, n_feats 64, reduction 16 |
| realplksr | `realplksr`, `realplksr_s`, `realplksr_l` | upscaling_factor | dim 64, n_blocks 28, kernel_size 17, split_ratio 0.25, use_ea True, norm_groups 4, dropout 0, dysample False. s: 12 blocks, kernel 13, no ea; l: dim 96 |
| rgt | `rgt`, `rgt_s` | upscale | embed 180, depth [6]x8, heads [6]x8, split_size [8,32], c_ratio 0.5. s: depth [6]x6 |
| safmn | `safmn`, `safmn_l`, `light_safmnpp` | upscaling_factor | dim 36, n_blocks 8, ffn_scale 2.0, bcie False. l: dim 128, 16 blocks; light_safmnpp: dim 32, n_blocks 2, ffn_scale 1.5 |
| sebica | `sebica`, `sebica_mini` | sr_rate | N 16, mini False, dropout 0.0. mini: N 8 |
| span | `span`, `span_fast` | upscale | feature_channels 48, bias True, norm False, fast False. fast: 32, no bias |
| spanplus | `spanplus`, `spanplus_sts`, `spanplus_s`, `spanplus_st` | upscale | blocks [4], feature_channels 48, upsampler dys. sts: [2], 32, ps; s: [2], 32, dys; st: ps |
| srformer | `srformer`, `srformer_light`, `srformer_medium` | upscale | embed 60, depths 6x4, window 16, pixelshuffledirect. medium: img 48, window 24, 180, 6x6, pixelshuffle |
| swinir | `swinir_small`, `swinir_medium`, `swinir_large` | upscale | window_size 8, flash_attn False. small: embed 60, 6x4, pixelshuffledirect; medium: img 48, 180, 6x6, pixelshuffle; large: embed 240, 9 groups, heads 8, nearest+conv (x4 only) |
| swinirdense | `swinirdense` | upscale | img_size 32, embed 60, depths 6x4, window 8, flash_attn False |

Registered arch kwargs seen in templates: `omnisr` (`upsampling`, `window_size`), `asid_d8` (`drop`), `craft`/`flexnet`/`swinir_*` (`flash_attn`), `realplksr` (`dysample`). Every arch also accepts its other constructor kwargs; a UI should expose only `type` plus an "advanced" free-form table.

### 5.2 Discriminators (`[network_d]`; no scale kw)

| type | params (default) | notes |
|---|---|---|
| `unet` | num_in_ch 3, num_feat 64, skip_connection True | spectral norm |
| `patchgan` | num_in_ch 3, num_feat 64, num_layers 3, max_nf_mult 8, norm_type none, use_sigmoid False, use_sn True | |
| `resgan` | num_in_ch 3, num_feat 64, skip_connection True, upscale True, attention False | |
| `metagan` | in_ch 3, n_class 1, dims (48,96,192,288), blocks (3,3,9,3), downs (4,4,2,2), drop_path 0.02, end_drop 0.2 | used by all 42 train templates |
| `dunet` | in_ch 3, dim 64 | |
| `ea2fpn` | class_num 6, encoder_channels (512,256,128,64), pyramid_channels 64, segmentation_channels 64, dropout 0.0 | A2-FPN derived, torchvision ResNet18 encoder; default `class_num` 6 looks segmentation-oriented (?) |

D is always in train mode; `network_d` is built without a scale, so passing a scale-bearing generator type there raises `ConfigError`.

## 6. `[train]`

### 6.1 Scalar keys (schema)

| key | type | default | meaning |
|---|---|---|---|
| `ema` | float | 0 (off) | generator EMA decay (templates 0.999); first successful update copies G; validation and exported `net_g` use the EMA |
| `ema_power` | float | unset (constant decay) | ema_pytorch-style ramp `min(1-(1+k)^-power, ema)`; must be > 0; ignored if `ema` is 0; templates 0.75 |
| `clamp` | bool | true | clamp G output to [1/255, 1] before losses |
| `grad_clip` | bool | true | clip each network's grad norm to a fixed 1.0 (no value key) |
| `max_skip_streak` | int | 10 | consecutive non-finite updates before the run stops; >= 1 |
| `match_lq_colors` | bool | false | legacy: sets `target="lq"` on slots named `consistency` and `msswd` |
| `wavelet_guided`, `wavelet_init` | bool, int | false, 0 | legacy: maps to `gan_opt.wavelet_init`; needs `gan_opt` |
| `enable_qat`, `qat_type`, `fused_init`, `quantize_residual`, `calibration_iters`, `qat_freeze_norm_iter`, `qat_freeze_observers_iter` | bool, str, bool, bool, int, int, int | false, `"int8"`, true, net default, 0, 0.7 x total_iter, 0.8 x total_iter | QAT for `aether*` only (section 9) |
| `optim_g`, `optim_d`, `scheduler` | table | | 6.2, 6.3 |
| `<slot>_opt` | table | | losses, section 7 |

### 6.2 Optimizers (`[train.optim_g]`, `[train.optim_d]`; `type` case-insensitive; kwargs pass to the constructor)

| type | class | key params (default) | notes |
|---|---|---|---|
| `adam` | `torch.optim.Adam` | any torch kwarg: lr, betas, eps, weight_decay, amsgrad, foreach, fused ... | |
| `adamw` | `torch.optim.AdamW` | same; wiki example: lr 1e-4, weight_decay 0, betas [0.9,0.99], fused true | Aether templates: lr 2e-4, betas [0.9,0.99], wd 0 |
| `nadam` | `torch.optim.NAdam` | torch kwargs | |
| `adan` | `neosr.optimizers.adan` | lr 5e-4, betas (0.98,0.92,0.99), eps 1e-8, weight_decay 0.02, max_grad_norm 0.0, no_prox True, foreach True | |
| `adamw_sf` | schedule-free AdamW | lr 8e-4, betas (0.9,0.99), eps 1e-8, weight_decay 0.0, warmup_steps 0, r 0.0, weight_lr_power 2.0, schedule_free True (must stay True), foreach True | |
| `adan_sf` | schedule-free Adan | lr 1.6e-3, betas (0.98,0.92,0.99), eps 1e-8, weight_decay 0.02, max_grad_norm 0.0, warmup_steps 0, r 0.0, weight_lr_power 2.0, schedule_free True | `schedule_free=false` = plain Adan |

Shipped templates (all 42 train templates): `adan_sf`, G lr 1e-3 for most (8e-4: esrgan, cfsr; 5e-4: eimn, esc, flexnet, krgn, plainusr, plksr, realplksr, sebica), D lr 1e-4, betas [0.98,0.92,0.99] (swinir G third beta 0.995), weight_decay 0.01, `warmup_steps` 1600 (G) / 600 (D). With schedule-free optimizers and no EMA, validation and exports use the evaluation weights. Resume is strict about optimizer type and SF mode.

### 6.3 Schedulers (`[train.scheduler]`, optional)

| type | class | params (default) |
|---|---|---|
| `multisteplr` | `MultiStepLR` | milestones (list of **updates**), gamma 0.1 |
| `cosineannealing` | `CosineAnnealingLR` | T_max (updates), eta_min 0 |

One table only: it is instantiated for both optim_g and optim_d. It steps once per attempted update (skipped non-finite updates count). No shipped template sets one (the SF optimizers are designed to need none; docstring). Wiki examples: MultiStepLR milestones [60000,120000], gamma 0.5; CosineAnnealing T_max 350000, eta_min 4e-5.

## 7. Losses (`[train.<slot>_opt]`)

Every non-empty table ending in `_opt` becomes one term; the slot name is free-form and sets the log key `l_g_<slot>` (`pixel` -> `l_g_pix`, `perceptual` -> `l_g_percep`; log keys must be unique and not reserved). Reserved builder keys: `type` (required), `loss_weight` (alias `perceptual_weight`, not both; finite number; 0 drops the term), `target` (`"gt"` default or `"lq"` = antialiased bicubic upscale of lq, meant as a colour reference). Everything else goes to the loss constructor. At least one generator term with non-zero weight is required. Slot names that matter: `consistency` and `msswd` (legacy `match_lq_colors`), `gan` (legacy `wavelet_guided`). Two slots may use the same type. Loss weights are constants.

| `type` | usual slot | params (default) | phase | default weight | notes |
|---|---|---|---|---|---|
| `L1Loss` | `pixel` | reduction mean | G | 1.0 | `none/mean/sum` |
| `MSELoss` | `pixel` | reduction mean | G | 1.0 | |
| `HuberLoss` | `pixel` | reduction mean, delta 1.0 | G | 1.0 | |
| `chc_loss` | `pixel` | reduction mean, criterion `huber` (or `l1`), loss_lambda 0, clip_min 0.003921, clip_max 0.996078 | G | 1.0 | clipped pseudo-Huber + cosine |
| `vgg_perceptual_loss` | `perceptual` | layer_weights {conv1_2 0.1, conv2_2 0.1, conv3_4 1, conv4_4 1, conv5_4 1}, vgg_type vgg19 (only value), use_input_norm True, range_norm False, criterion `chc` (`l1`,`l2`,`huber`,`chc`) | G | 1.0 | layer names from VGG19 list (`conv1_1`...`pool5`) |
| `gan_loss` | `gan` | gan_type `bce` (`mse`, `huber`), real_label_val 1.0, fake_label_val 0.0, wavelet_init None | G term + D term | **0.1** (G term only; D term unweighted) | adds the discriminator objective; requires `network_d` + `optim_d` |
| `mssim_loss` | `mssim` | window_size 11, in_channels 3, sigma 1.5, K1 0.01, K2 0.03, L 1, padding None | G | 1.0 | 5-scale MS-SSIM, fp32 |
| `ldl_loss` | `ldl` | criterion `l1` (`l2`,`huber`,`chc`), ksize 7 | G | 1.0 | |
| `ff_loss` | `ff` | alpha 1.0, patch_factor 1, ave_spectrum True, log_matrix False, batch_matrix False | G | 1.0 | focal frequency, internal x50, fp32 |
| `dists_loss` | `dists` | as_loss True, load_weights True | G | 1.0 | fp32; downloads weights |
| `consistency_loss` | `consistency` | criterion `chc` (`l1`), blur True, cosim True, saturation 1.0, brightness 1.0 | G | 1.0 | Oklab chroma + CIE L*; `target="lq"` for colour fidelity; cosine term only added when it is < 1e-3 (value-gated, not iteration-gated) |
| `kl_loss` | `kl` | none | G | 1.0 | internal x0.03 |
| `msswd_loss` | `msswd` | num_scale 3, num_proj 24, patch_size 11, stride 1, c 3 | G | 1.0 | internal x0.1; takes `target="lq"` |
| `ncc_loss` | `ncc` | none | G | 1.0 | |
| `fdl_loss` | `fdl` | model `dinov2` (`vgg`,`resnet`,`effnet`), patch_size 4, stride 1, num_proj 24, phase_weight 1.0, seed 0, fp32_backbone True, dino_layers (0..7), dino_weights (1,.5,.5,1,.5,.5,1,.1), dino_norm False, vgg_weights (.5,.5,1,1,1) | G | 1.0 | `fp32_backbone=false` = ~2x faster but gradients drift |

Typical combinations seen in the repo:

| source | combination |
|---|---|
| 41 of 42 `train_*.toml` | `mssim_loss` 1.0 + `consistency_loss` 1.0 + `ldl_loss` 1.0 + `fdl_loss` (dinov2) 0.75 + `gan_loss` (bce) 0.3 with `metagan`; catanet uses fdl 0.5 |
| commented alternatives in templates | `msswd_loss` 1.0; `vgg_perceptual_loss` 0.5 (criterion huber); `dists_loss` 0.5; `ff_loss` 0.35; `ncc_loss` 1.0; `kl_loss` 1.0 |
| `options/aether/*.toml` (L1 pretrain, no GAN) | `L1Loss` 1.0, `reduction = "mean"` only |
| Upscale-Hub example log (older) | L1 pixel + VGG perceptual + color + GAN |

Log keys (window means): `l_g_<slot>`, `l_g_total`, `l_d_total`, `l_d_real`, `l_d_fake`, `out_d_real`, `out_d_fake`, `grad_norm_g/d`, `skipped_g/d`.

## 8. Model types and what the trainer reads

| `model_type` | class | extra requirements |
|---|---|---|
| `image` | `neosr.models.image.image` | paired (or val) data with `lq`/`gt` |
| `otf` | `neosr.models.otf.otf` (subclass of `image`) | `datasets.train.type = "otf"`, full degradation keys; builds `Degrader` + `PairPool(queue_size, batch_size)`; validation stays paired |

Keys read from `opt["train"]` in `image.py`: `ema`, `ema_power`, `grad_clip`, `max_skip_streak`, `clamp`, `optim_g`, `optim_d`, `scheduler`, QAT keys, and every `*_opt` (via `build_losses`, also `match_lq_colors`, `wavelet_guided`, `wavelet_init`). From `datasets.train`: `accumulate`, `augmentation`, `aug_prob`, `patch_size`, `batch_size`, `queue_size`. There is no `wavelet_guided`/`sam`/`eco`/`warmup_iter` logic left outside the legacy mapping. Top-level: `use_amp`, `bfloat16`, `compile`, `compile_mode`.

### 8.1 `[val]`

| key | type | default | meaning |
|---|---|---|---|
| `val_freq` | int | unset (only final validation) | updates between validations; also validates after the last update |
| `save_img` | bool | true | writes `experiments/<name>/visualization/<img>/<img>_<update>.png` |
| `save_lq` | bool | true | also writes `<img>_lq.png` once |
| `tile` | int | -1 (<= 0 = whole image) | LQ tile size, stitched on CPU; wiki: set it if large images OOM |
| `tile_overlap` | int | 16 | context kept around a tile; rule `tile > 2 * tile_overlap >= 0` |
| `pbar` | bool | true | progress bar |
| `suffix` | str | run name | inference output file suffix |
| `metrics` | table | none | `[val.metrics.<name>]` |

Validation runs on rank 0 with EMA (or evaluation) weights; `single` datasets skip metrics.

### 8.2 `[val.metrics.<name>]` (the `<name>` is free; `type` selects)

| `type` | params (default) | better |
|---|---|---|
| `calculate_psnr` | crop_border 4, input_order HWC, test_y_channel False | higher |
| `calculate_ssim` | crop_border 4, input_order HWC, test_y_channel False (needs >= 11x11 px after crop) | higher |
| `calculate_dists` | none (device injected; do not set `device`) | lower |
| `calculate_topiq` | none | higher |

`better = "higher" \| "lower"` (default `higher`) is consumed by best-value tracking, not the metric: DISTS needs `better = "lower"`. Logged as `metrics/<dataset>/<name>` in TensorBoard.

### 8.3 `[logger]`, `[path]`, `[dist_params]`

| key | type | default | meaning |
|---|---|---|---|
| `logger.total_iter` | int | 1,000,000 | total **updates** (templates 1,000,000; wiki example 500,000) |
| `logger.print_freq` | int | 100 | updates between log lines |
| `logger.save_checkpoint_freq` | int | **required** (no default) | updates between `.state` + `net_*_<u>.pth` saves |
| `logger.keep_states` | int | 3 | newest training states kept; 0 keeps all |
| `logger.use_tb_logger` | bool | false | TensorBoard in `experiments/tb_logger/<name>` |
| `logger.save_tb_img` | bool | false | val images to TensorBoard (slow, big) |
| `logger.wandb.project` / `.resume_id` | str | none | only these two are read; needs `use_tb_logger = true`; a misspelt `project` silently disables W&B |
| `path.pretrain_network_g` / `_d` | str | none | weights; ignored while resuming; `.state` files are not accepted |
| `path.param_key_g` / `_d` | str | tries `params_ema`, `params-ema`, `params`, bare dict | |
| `path.strict_load_g` / `_d` | bool | true | false skips shape mismatches (different scale or slightly different arch) |
| `path.resume_state` | str | none | explicit state wins over `--auto_resume`; comment out pretrains when set (wiki) |
| `path.experiments_root`, `results_root` | str | `<repo>/experiments`, `<repo>/experiments/results` | |
| `path.print_network` | bool | false | |
| `dist_params.backend` / `.port` | str / int | nccl / 29500 (slurm only) | DDP is "best effort" per readme |

A fresh run renames an existing run folder to `<name>_archived_<time>`; a resume never does. Ctrl-C once: stops at the next update, saves state + exports + `*_latest`, exit 0, no final validation. Outputs under `experiments/<name>/`: `models/net_g_<u>.pth`, `net_d_<u>.pth`, `net_*_latest.pth`, `training_states/<u>.state`, `visualization/`, a timestamped copy of the TOML and `train_<name>_<time>.log`.

## 9. Iteration-scheduled behaviour (the only time-dependent knobs)

| mechanism | keys | clock | detail |
|---|---|---|---|
| Wavelet-guided GAN | `gan_opt.wavelet_init` (legacy `train.wavelet_guided` + `train.wavelet_init`) | update number being computed | identity D input before N, then SWT high-frequency view for both D and the G adversarial term; templates 80000 |
| Optimizer LR warmup | `optim_*.warmup_steps` (adamw_sf, adan_sf only) | optimizer steps (skipped updates do not step) | linear ramp, 0 = off; templates G 1600, D 600; replaces removed `warmup_iter` |
| LR scheduler | `[train.scheduler]` | attempted updates | one table applied to G and D |
| EMA ramp | `ema_power` | successful G updates | decay rises to `ema` |
| QAT phases (aether only) | `calibration_iters` (0), `qat_freeze_norm_iter` (0.7 x total_iter), `qat_freeze_observers_iter` (0.8 x total_iter), -1 = never | update number | observe-only, then fake-quant, then norm freeze, then observer freeze; norm freeze must be <= observer freeze and > calibration |
| Frequencies | `val_freq`, `save_checkpoint_freq`, `print_freq`, `total_iter` | updates | |
| Loss weights | none | | constant; no warmup/ramp/switch of any loss. `consistency_loss` cosine term is value-gated |
| Removed | ECO, SAM, `warmup_iter`, `eco_schedule` | | hard error |

### QAT (aether only, `docs/aether.md`)

`train.enable_qat = true` needs a network with `prepare_qat` (only `aether*`; others `ConfigError`). `qat_type` only `"int8"`; `fused_init=false` rejected; `network_g.fused_init=true` from scratch needs a pretrain; `quantize_residual` must not contradict `network_g`; AMP/compile accepted with a warning (fp32 eager is the reference). Shipped aether templates use `use_amp = false`, `adamw` lr 2e-4, EMA 0.999 + `ema_power` 0.75, `calibration_iters = 500`.

## 10. Cross-key rules and interdependencies (candidates for UI validation)

| rule | source |
|---|---|
| `network_d` <=> `train.optim_d` <=> `train.gan_opt` (all or none); `wavelet_guided` needs `gan_opt` | `validate_train`, `build_losses` |
| `bfloat16` needs `use_amp` | `validate_train` |
| `compile` errors on Windows; training only | `setup_runtime` |
| `num_gpu > 1` needs `--launcher` | `setup_runtime` |
| `model_type = otf` <=> `datasets.train.type = otf` + complete degradation keys; `otf` forbids `dataroot_lq` | `otf` trainer/dataset |
| `val.tile > 2 * val.tile_overlap >= 0` | `validate_train` |
| `datasets.train.patch_size`, `batch_size` required and >= 1; `accumulate >= 1`; `persistent_workers` needs workers > 0 | `validate_train` |
| `max_skip_streak >= 1`, `ema_power > 0` | `validate_train` |
| `logger.save_checkpoint_freq` required; `logger.wandb` needs `use_tb_logger` | `train.py` |
| `augmentation` and `aug_prob` same length; non-cutblur augmentation needs `batch_size > 1`; `queue_size >= batch_size` (otf) | augment / `PairPool` |
| arch scale kwarg must be unset or equal `scale`; scale-bearing arch not allowed as `network_d` | `inject_scale` |
| `train.optim_g` required; at least one non-zero generator loss | `validate_train`, `build_losses` |
| `[datasets.test_*]` not allowed in training; `train`/`val*` not meaningful in inference | `validate_train`, `build_loaders` |
| `adamw_sf` must have `schedule_free = true` | optimizer ctor |
| Resume is strict: same nets, optimizer type + SF mode, trainable params, scheduler list; other diffs only warn | `base.resume` |
| Metric/Dataset: val dataset needs `name`; `better = "lower"` for DISTS | `validate()`, `update_best` |

## 11. Template inventory (`options/`)

| group | count | content |
|---|---|---|
| `train_<arch>.toml` | 42 | paired data, GAN-from-scratch recipe in section 7; `train_esrgan.toml` is the base, others differ by network type, lr, patch_size (32 for window-attention transformers: atd, dat, dct, drct, grformer, hat, hitsrf, hma, mfghmoe, microsr, rgt, srformer, swinir; 64 otherwise), `grad_clip = false` (compact), `clamp = false` (cugan, esrgan, ninasr), `match_lq_colors = true` instead of wavelet (atd), no wavelet (dctlsa) |
| `train_<arch>_otf.toml` | 42 | same plus `[degradations]` (4.2) and `type = "otf"` |
| `test_<arch>.toml` | 43 | `[datasets.test_1] type = "single"`, `[network_g]`, `[path] pretrain_network_g`; `test_aether` loads a fused or QAT checkpoint |
| `options/aether/*.toml` | 6 | L1-pretrain recipes (aether small/medium/large, span), QAT variants, scale 2 |

Common to train templates: `batch_size = 8`, `augmentation = ["none","mixup","cutmix","resizemix","cutblur"]` with `aug_prob = [0.5,0.1,0.1,0.1,0.5]`, `[datasets.val]` paired, `val_freq = 1000`, `tile`/metrics commented out, `[logger] total_iter = 1000000, save_checkpoint_freq = 1000, use_tb_logger = true`. Pretrain lines commented in `[path]`.

## 12. Workflow guidance from the two wiki pages

Sources: Configuration-Walkthrough (neosr wiki) and "Training a Model in NeoSR" (Upscale-Hub). Neither page covers loss weights, GAN switching, or conversion in detail (both defer to other wiki pages not fetched).

| topic | content |
|---|---|
| Process (Upscale-Hub) | 1 build image-pair or OTF dataset; 2 make a small validation set; 3 pick an arch (guide uses Compact); 4 install neosr; 5 edit a template; 6 optional TensorBoard; 7 `neosr-train options/<file>.toml`; 8 monitor logs, validation scores and images; 9 use the `_g` model for inference. |
| Order of config (Walkthrough) | header options, datasets, validation, network, train + optimizers, logging; `--auto_resume` to continue. |
| Dataset prep | Full paths, Windows paths in single quotes, no special characters. `dataroot_*` may be LMDB (`create_lmdb.py` also writes `meta_info`). `meta_info` is optional but catches filename mismatches. Remove near-duplicates, avoid featureless images, include some intentional blur, keep it under about 5-7k images, mix sources. HR is 1x/2x/4x LR (3x on some archs), pairs closely aligned. Pair types: real pairs, synthetic HR (upscaled LR), synthetic LR (degraded HR); often combined. OTF suits real-life sources more than anime and can over-degrade. |
| Validation set | 8-10 images, varied; PSNR/SSIM high, DISTS low are only loose guides, inspect `experiments/<model>/visualization`; checkpoints with worse metrics can look better. |
| Beginner arch | Compact (SVRGGNet): fast, stable, low VRAM; use the normal Compact pretrain of the same scale, not UltraCompact. |
| Recommended values | `name = "4x_mymodel"` (scale prefix), `scale` 1/2/4, `patch_size` multiple of 8 or 16, batch 4/8/16, `use_hflip`/`use_rot` true, ema 0.999 (esp. with SF optimizers), `val_freq`/`save_checkpoint_freq` 1000 (author uses 2500 save, 5000 val), `tile = 200` if large validation images OOM. |
| Iteration counts | Compact often done under 40K iterations, 5-10K with a very good pretrain; examples use `total_iter` 500000; stop when validation scores stagnate or decline (Ctrl-C, resume with `--auto_resume` or `resume_state`). |
| Pretrain vs fine-tune | `pretrain_network_g/_d` load weights; `resume_state` resumes a `.state`; use one or the other. `strict_load_g = false` for a different scale or slightly different arch. A pretrain speeds training but the dataset matters more; same-arch pretrains also allow interpolation. No separate fine-tune recipe is described. |
| Losses | Example log shows pixel (L1), perceptual, color and GAN terms; non-GAN losses should trend toward 0; tuning loss values is called low-return. No schedule for adding GAN is given. In this repo the templates train with GAN from the start (wavelet view from 80000), while `options/aether` shows a GAN-less L1 pretrain. |
| Pitfalls | Misaligned pairs; over-compressed or over-blurred synthetic LR; synthetic HR inherits upscaler flaws; real pairs alone overfit; OTF too aggressive; AMP/bf16 can destabilise; large validation images without `tile` OOM; transformer `patch_size` vs window size; some archs unstable at small batch; reliance on metrics alone. |
| Convert / release | Final weights in `experiments/<name>/models`: `net_g_*.pth` is the inference model (EMA if enabled), `net_d_*.pth` only for resuming; `.state` for resume only. ONNX/NCNN conversion is on a separate wiki page (not fetched); the repo's `convert.py` covers ONNX. |

## 13. UX-relevant findings

1. GAN is a three-key coupling (`network_d`, `train.optim_d`, `train.gan_opt`) and the GAN term has its own default weight 0.1 (templates 0.3). A "GAN on/off" switch should add or remove all three, and optionally `wavelet_init`.
2. Everything is counted in optimizer updates (including `total_iter`, `val_freq`, scheduler milestones, `wavelet_init`, QAT thresholds as 0.7/0.8 x `total_iter`); `accumulate` stretches an update over several micro-batches. Only warmup (SF optimizers' `warmup_steps`) and `wavelet_init` are iteration-switched; no loss weight changes over time.
3. Losses are free-form `[train.<slot>_opt]` tables, so a UI can offer a list of (type, weight, type-specific params) rows; slot names `pixel`, `perceptual`, `consistency`, `msswd`, `gan` carry legacy meaning. `target = "lq"` is only sensible for colour losses (`consistency`, `msswd`).
4. OTF is a bundle: `model_type = "otf"` + `datasets.train.type = "otf"` + a full `[degradations]` table (the shipped one is identical across archs) + no `dataroot_lq`; validation set stays `paired`.
5. Platform and hardware gates: `compile` errors on Windows; `bfloat16` needs `use_amp` and Ampere-class GPUs; `num_gpu > 1` needs `torchrun`. Never emit an arch's scale kwarg; per-arch scale limits (x4-only variants, cugan no 1x) and VRAM-heavy transformers (hat/drct/atd/hit_srf) should be flagged.
6. Required-by-code but easy to miss: `logger.save_checkpoint_freq`, `datasets.train.patch_size`/`batch_size`, a `name` on val datasets, `better = "lower"` for DISTS.
7. `[train.scheduler]` is a single table for both optimizers and is absent from all templates because the default optimizers are schedule-free.

Open items `(?)`: whether `ea2fpn` is meant to run with its default `class_num`; minimum patch size for `mssim_loss` (5 scales, 11x11 window) and `fdl_loss`; behaviour of `color = "y"` with each arch; whether missing OTF `sinc_prob*` or `kernel_*` keys fail with a clear error (the dataset reads them unchecked).
