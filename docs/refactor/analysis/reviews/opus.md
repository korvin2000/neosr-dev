# My review (Opus, main thread) — verified items marked [V]

## Critical (training dynamics / crashes)
1. [V] image.py:609-616 GAN generator loss computed under torch.inference_mode -> l_g_gan has no grad; G never gets adversarial signal.
   Evidence: logs experiments/train_esrgan/*.log: l_d_real ~1e-10, out_d_real +40, out_d_fake -25, l_g_gan 7-10 (D wins totally).
   Repro: tiny script -> G grad norm 0.0.
2. [V] image.py:609/654 net_d.eval() during D forward for both G and D steps -> spectral_norm (unet/dunet/ea2fpn/patchgan) never runs
   power iteration (stale sigma), BatchNorm uses never-updated running stats, Dropout disabled. Explains exploding D logits.
3. image.py:501-505 + 733-738: n_accumulated incremented in BOTH optimize_parameters and closure -> accumulation broken
   (accum=2 steps every iter); grad-clip condition evaluated on wrong counter; SAM calls closure again -> 3rd increment.
4. base.py:551 dist reduce: `losses = torch.distributed.reduce(...)` returns None -> crash under DDP (+ stray print).
5. data/__init__.py:95-118 build_dataloader: `dataloader_args` undefined when dist=True -> UnboundLocalError (DDP unusable);
   unreachable dead block after raise (references psutil).
6. train.py:577-584 "speed kill": if it/s/accumulate < 0.5 -> saves and sys.exit(0). Big models silently stop training.
7. train.py:225-238 resume failure is swallowed -> starts from scratch and make_exp_dirs renames the experiment folder.
8. prefetch_dataloader.py CUDAPrefetcher: H2D copies on side stream without record_stream -> allocator can reuse memory still
   used by the main stream (silent data corruption race); pin_memory default False makes non_blocking pointless.
9. augmentations.py:39-42 mixup mixes LQ with shuffled *GT* (GT leakage into input); apply_augment always round-trips LQ
   through up/down resampling even when "none" is chosen (silent blur of every LQ batch).
10. vgg_perceptual_loss.py: patch()/PatchesKernel3D run under torch.no_grad -> PatchLoss/IPK contribute zero gradient (dead feature);
    kernels re-instantiated each call.
11. ncc_loss.py:20 `.view(C,-1)` mixes batch/channel and crashes on channels_last (non-contiguous) tensors (AMP => channels_last nets).
12. ssim_loss.py msssim: cs**w with negative cs.mean() -> NaN (known MS-SSIM instability); needs clamp.
13. consistency_loss.py:162 L* low branch is Y^2*903.3 instead of Y*903.3; `if cosim < 1e-3` is a per-iter GPU sync.
14. ldl_loss.py:103 stores self.output/self.gt on the module (keeps last graph alive -> memory).
15. otf.py: `random_add_poisson_noise_pt(scale_range=self.opt['poisson_scale_range'])` reads top-level opt keys while all other
    keys come from opt['datasets']['train'] (KeyError / inconsistent); jpeg_compress branch inconsistent between orders.

## Global design problems
- net_opt() (arch_util.py:13) and rng() (utils/rng.py) re-parse sys.argv + TOML at import time and per DropPath construction;
  parse_options has side effects (init_dist, set_random_seed, cudnn flags, env vars). Importing neosr.archs without -opt raises.
  With manual_seed every worker process builds the same numpy Generator (same seed) -> identical OTF/augment randomness across
  dataloader workers. (Checked: DropPath reseed did NOT clone weights in swinir_small/dat_s because _init_weights runs after.)
- torch.set_default_device("cuda") globally (train.py:320, test.py) — hides device bugs, forces CUDA generator in sampler,
  every tiny tensor goes to GPU.
- Two parallel model hierarchies: default.py (1200 lines, legacy BasicSR copy, prints loss dicts every print_freq!) and
  base.py/image.py; plus simple_otf/eqr_otf (subclass default), pancam_otf (subclass otf); image.py.helaman stale copy.
- Per-iteration GPU syncs: reduce_loss_dict .item() on every loss every iter, torch.isnan(l_g_total) every iter.
- empty_cache()+gc.collect() on every save/load/resume; empty_cache per val image and per tile.
- sys.exit() used as error handling inside library code (~50 sites); `except: pass` swallowing.
- GradScaler enabled for bf16 (useless); separate scaler per net ok; SAM+GradScaler mismatch.
- Loss construction: 13 copy-pasted `if train_opt.get('x_opt')` blocks; loss computation: 11 `if self.cri_x` blocks.
- `l_g_total.detach()` / `l_d_real.detach()` no-op statements; `patch_size % 4` check builds msg but never raises.
- train.py: CUDA version check raises inside try/except Exception: pass (dead); tb_logger.close() twice; torch version check
  parses minor only; prints in library code.
- build_dataloader: persistent_workers False + spawn start method -> workers respawned (re-import whole package) every epoch.
- EMA: AveragedModel deep-copies a possibly torch.compile'd/DDP-wrapped module; EMA state not saved in training state
  (resume restarts EMA from loaded weights); net_d_ema saved as net_d (resume continues D from its EMA).

## Added later
- metrics/calculate.py:197,234 calculate_dists/topiq build the full VGG/ResNet metric net (and load weights from disk) per
  image per validation -> very slow validation. Cache the metric module.
- dists_loss: alpha/beta are trainable nn.Parameters inside a loss (2950 params) -> grads accumulate forever, never zeroed.
  Generic fix: build_loss(...).requires_grad_(False) (losses are never optimized).
- aether: neosr/archs/aether_arch.py == common/v4/aether_arch.py; neither registers aether_small/medium/large, yet options/aether/*
  use them -> those configs cannot run. train.py QAT block calls net_g.calibrate(loader, iters=...) but signature is
  calibrate(calib_loader) -> TypeError; whole QAT path dead. aether_core.py duplicate of common/v5 (differs).
- Only model_type "image" (93 cfgs) and "otf" (42 cfgs) are used by options/. default/simple_otf/eqr_otf/pancam_otf unused.
