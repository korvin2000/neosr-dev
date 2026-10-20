# Consolidated findings to verify (sources: Opus main-thread review = "O", Codex gpt-6-astra review = "C" with its IDs)
Full source texts: review/my_review.md (O) and review/codex_report.md (C). Repo: /home/korvin/apps/neosr-dev (paths relative).
Competitor (for reference only): /home/korvin/apps/traiNNer-redux-dev

## G1 GAN / discriminator / training step (neosr/models/image.py closure + optimize_parameters)
- G1.1 [O,C-A1] image.py:609-616 G-side GAN loss computed under torch.inference_mode -> l_g_gan has no grad -> G gets no adversarial signal. Logs: experiments/train_esrgan/*.log D logits +-40, l_d ~1e-10.
- G1.2 [O,C-A2] image.py:609/654 net_d.eval() during both G and D forwards -> spectral norm power iteration never runs, BN running stats unused/stale, dropout off.
- G1.3 [C-A11] wavelet_guided: after wavelet_init D trains on HF wavelet bands (no_grad) while G queries D with RGB -> domain mismatch; both call sites no_grad.
- G1.4 [C-A9] FSAM: separate base optimizer not in self.optimizers (not checkpointed); SAM second closure re-runs D backward and increments counters; SAM + GradScaler mismatch for D.
- G1.5 [C-A10] ECO overwrites self.gt in place; SAM second pass blends already-blended target; extra train-mode forward updates BN stats twice.
- G1.6 [C-A27] Only G-loss NaN checked (after backward, per-iter sync); inf / D NaN / non-finite grads not caught; AMP-skipped steps still advance EMA/schedulers.
- G1.7 [O,C-C5] GradScaler enabled for bf16 (useless); `l_g_total.detach()` / `l_d_real.detach()` no-op statements; `patch_size % 4` check builds msg but never raises (image.py:331).
- G1.8 [O] D loss: logged l_d_total=(real+fake)/2 while backward on real+fake separately; logged values already divided by accum.

## G2 Accumulation / counters / scheduler / logging / train loop
- G2.1 [O,C-A3] n_accumulated incremented in both optimize_parameters (image.py:735) and closure (image.py:502) -> accumulate=2 steps every batch; grad-clip condition on wrong counter.
- G2.2 [C-A4] train.py:600/454 saves accumulated (update) iter, resumes it as micro-batch iter, divides again -> resume with accumulation repeats/shifts schedule; logger divides an already adjusted count.
- G2.3 [O,C-A8] train.py:577-584 throughput kill-switch: < 0.5 it/s (per accumulate) -> save + sys.exit(0).
- G2.4 [O,C-C1] per-iteration GPU syncs: reduce_loss_dict .item() for every loss every iter (base.py:559), torch.isnan(l_g_total), consistency_loss `if cosim < 1e-3`, ff_loss/diffjpeg tensor conditions.
- G2.5 [C-D4] conflicting total_iter locations/defaults (train.total_iter vs logger.total_iter), warmup without scheduler lacks initial_lr (base.py:253), custom experiments_root overwritten by train.py:329.
- G2.6 [O] train.py misc: CUDA version check raises inside try/except: pass (dead); torch version check parses minor only; tb_logger.close() twice; scheduler.step() timing vs accumulation (base.py:268 uses n_accumulated==0 after double increment).

## G3 Checkpoint / resume / EMA / compile / DDP wrappers
- G3.1 [O,C-A5] save() writes EMA weights as net_g (and net_d_ema as net_d); resume loads EMA as live weights, pairs them with live optimizer moments; EMA n_averaged, GradScaler, RNG, accumulation state not saved.
- G3.2 [O,C-D1] train.py:225-238 resume load failure swallowed -> silently starts from scratch and make_exp_dirs archives experiment folder; save retry loop reports success-on-3rd-try as failure; non-atomic writes.
- G3.3 [C-A6] schedule-free: weights copied to CPU before optimizer.eval() in save_network (base.py:343 vs 353); validation sets is_train=False before checking sf_optim_g and is_train so SF eval swap never happens (image.py:851,889).
- G3.4 [C-A19] torch.compile applied before load_network; get_bare_model does not unwrap _orig_mod -> pretrained loading / checkpoint keys broken with compile; AveragedModel deep-copies compiled/DDP-wrapped module.
- G3.5 [C-A18] DDP: rank0-only validation through DDP wrapper while other ranks continue -> collective mismatch / hang.
- G3.6 [O,C-A29] QAT: prepare after optimizer construction; _init_ema replaces AveragedModel by plain net -> update_parameters missing; train.py calls net_g.calibrate(loader, iters=) but signature calibrate(calib_loader).
- G3.7 [C-D2] load_network overrides requested param_key by guessed priority, bare except: pass; failed load sys.exit(0) (success code).

## G4 Optimizers (neosr/optimizers/*)
- G4.1 [C-A7] adan_sf.py:260/315 warmup lr computed but _multi_tensor_adan gets group["lr"] -> configured warmup has no effect (user config uses adan_sf with warmup_steps).
- G4.2 [C-A25] soap_sf.py state keyed by (Parameter,index) then mapping cleared; SF eval misses z; decay applied to grad but update uses grad_projected.
- G4.3 [C-A33] adamw_win divides by lr (crash when scheduler reaches 0); adamw_sf ignores schedule_free=False; beta1=0 -> div by zero on eval conversion.
- G4.4 [O] get_optimizer: optim_type alias handling duplicated in base.py and default.py; SAM base_optimizer mapping duplicated; sys.exit for unsupported.

## G5 Loss numerics (neosr/losses/*)
- G5.1 [C-A12] ldl_loss.py:92 var**0.2 at zero -> non-finite gradients when residual is zero.
- G5.2 [O,C-A13] ncc_loss.py:20 .view(C,-1) mixes batch/channel and fails on channels_last; constant inputs divide by zero.
- G5.3 [O,C-A14] ssim_loss.py msssim: negative cs/ssim mean ** fractional weight -> NaN.
- G5.4 [O,C-A15] vgg_perceptual_loss: patch()/PatchesKernel3D under no_grad -> PatchLoss/IPK give zero gradient; kernels re-instantiated per call; unprotected norms; vgg_arch loading: load_state_dict on constructor fn / non-vgg19 variants get vgg19 weights (vgg_arch.py:128).
- G5.5 [O,C-A16] consistency_loss.py L* low branch uses Y^2*903.3 instead of Y*903.3; msswd_loss keeps only last scale (msswd_loss.py:177); resnet_perceptual features trainable/train-mode, loss_weight swallowed.
- G5.6 [O,C-C5] dists_loss alpha/beta trainable params inside loss -> grads accumulate forever; generic fix build_loss(...).requires_grad_(False).
- G5.7 [O] ldl_loss stores self.output/self.gt on module (keeps last graph alive); consistency_loss non-buffer tensors relying on default device.
- G5.8 [O] vgg_perceptual_loss & every arch call net_opt() at import (parses argv/TOML) just to get scale.

## G6 Data pipeline (neosr/data/*, neosr/models/otf.py, utils/rng.py)
- G6.1 [O,C-A17] data/__init__.py build_dataloader: dataloader_args undefined when dist=True (UnboundLocalError); unreachable block after raise; base.py:551 torch.distributed.reduce returns None + print.
- G6.2 [O,C-A22] CUDAPrefetcher: side-stream H2D copies without record_stream on consumer stream -> allocator reuse race; pin_memory default False makes non_blocking useless.
- G6.3 [O,C-A24] utils/rng.py + arch_util.net_opt: re-parse CLI at import, re-seed global RNG, may init_dist; module-level numpy Generator identical across workers with manual_seed (worker_init_fn does not reseed it).
- G6.4 [O,C-A30] augmentations.py mixup mixes LQ with shuffled GT; "none" still round-trips LQ through up/down resampling; multi-aug ignores weights; aug_prob None -> len(None).
- G6.5 [O,C-A20] otf.py reads poisson_scale_range / jpeg_range from root opt while others from datasets.train; one JPEG branch ignores jpeg_compress=False.
- G6.6 [C-A21] otf queue rounding uses configured batch; zero-size queue; legacy simple_otf passes HR gt_size into LR crop API.
- G6.7 [C-A23] paired_dataset meta_info key mismatch (checks "meta_info", reads "meta_info_file"); retry loop dead/broken (indexes string); otf_dataset divides str path; random.randint(0,len) off-by-one.
- G6.8 [O,C-C2] torch.set_default_device("cuda") global (train.py:320, test.py) -> sampler uses CUDA generator + randperm on GPU then tolist; persistent_workers False with spawn start method -> workers re-spawned (re-import package) every epoch; pin_memory False.
- G6.9 [C-C4] otf queue shuffles/copies whole queue each batch; wavelet filter modules rebuilt each call; Poisson unique loops sync.

## G7 Validation / metrics / test / convert
- G7.1 [C-A28] tile_val: reflect pad fails when pad >= dim; output channels assumed = input channels; no overlap -> seams; empty val set -> _idx undefined; nonpositive SSIM handling.
- G7.2 [O,C-C3] calculate_dists/topiq build metric network + load weights per image; tile val shuttles GPU->CPU->GPU with empty_cache per tile and per image.
- G7.3 [C-D5] convert.py export device/flags issues; broad excepts.
- G7.4 [O] nondist_validation toggles self.is_train globally; validation uses net_g_ema.train() afterwards (AveragedModel train mode) ; save_tb_img path.

## G8 Structure / legacy / imports / config
- G8.1 [O,C-A31] neosr/models/default.py legacy hierarchy (unused by any options/*.toml: only "image"(93) and "otf"(42) are used) has many crashes; simple_otf/eqr_otf subclass it; image.py.helaman stale copy; otfgray_dataset.py empty.
- G8.2 [C-A32] pancam_otf / pancamsingle_dataset bugs (undefined linear_blur, wrong channel slicing).
- G8.3 [C-D3] registries import every module eagerly (an unrelated broken arch prevents startup).
- G8.4 [O] aether_arch.py == common/v4/aether_arch.py, aether_core.py ~ common/v5; no aether_* registered yet options/aether/*.toml use type "aether_small" -> those configs cannot run.
- G8.5 [O] ~50 sys.exit() calls used as error handling inside library code; bare except: pass; prints in library code.
- G8.6 [C-A26] arch_util DySample pin_memory on CUDA tensor; ea2fpn hits it in forward.
