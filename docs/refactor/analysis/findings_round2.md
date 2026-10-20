# Round-2 findings: unique to the Fable review (review/fable_report.md). Repo /home/korvin/apps/neosr-dev.

## F Fable-unique findings
- F.1 [Fable A13] image.py:98-108 AveragedModel(..., use_buffers=False default) -> BN running stats / spectral-norm u,v not averaged; EMA model validated/saved with init BN stats for BN archs (rcan, dat, hasn, sebica, msdan, dctlsa) -> garbage EMA outputs when ema>0. Fix use_buffers=True.
- F.2 [Fable A16] neosr/utils/img_util.py:102-103 tensor2img normalizes the unclamped tensor_ instead of n_tensor -> uint8 wrap-around artifacts / wrong metrics for out-of-range outputs.
- F.3 [Fable A10] metagan_arch.py:103-104: DropPath.__init__ calls net_opt() -> parse_options -> set_random_seed; with manual_seed set, every GatedCNNBlock after the first is initialised from the same RNG state (identical inits). (Note: Opus checked swinir_small/dat_s and found no identical Linear weights there because they re-init after construction; metagan may differ.)
- F.4 [Fable A29] train.py:539-548 `break` on current_iter > total_iters leaves only the inner loop; outer epoch loop continues, re-creating iterators/prefetchers for remaining epochs.
- F.5 [Fable A30] train.py:346 fast_matmul sets torch.backends.cuda.matmul.allow_fp16_accumulation=True -> overflow risk with fp16 AMP.
- F.6 [Fable C8] fdl_loss.py:317-320 custom_fwd(cast_inputs=float32) forces DINOv2 ViT-B/14 backbone to fp32 for sr and gt every iteration -> likely dominant iteration cost; backbone could run under autocast bf16 while FFT/sort stay fp32.
- F.7 [Fable C9] base.py:156-161 nets in channels_last only when use_amp; inputs stay NCHW; losses moved to channels_last -> layout conversions per conv.
- F.8 [Fable C11] base.py:164 torch.compile(mode="max-autotune", dynamic=True) -> very long compiles, recompiles on validation sizes, AveragedModel deep-copies compiled module.
- F.9 [Fable D8] dead code: transforms.py single_random_crop / paired_random_crop_ow; arch_util pixel_unshuffle_width, normal_init/constant_init; img_util tensor2img_fast; losses/resnet_perceptual_loss.py + archs/resnet_arch.py unused & broken (percep_loss unbound); losses/loss_util.py duplicate of ldl.
- F.10 [Fable A18] transforms.py:122,127 / paired dataset: imfrombytes(None) raises TypeError (not AttributeError being caught); sys.exit inside dataloader workers; one unreadable image kills the run.
- F.11 [Fable C2] adan*.py max_grad_norm path does .item() / host syncs each step; ff_loss.py asserts with .item().
- F.12 [Fable A6 detail] resume continues D from smoothed net_d_ema whose spectral-norm u/v buffers were never averaged.
- F.13 [Fable D7] archs/__init__.py build_network rescans and imports every *_arch.py on every call.
- F.14 [Fable] optimizers/soap_sf.py:17 also calls parse_options at import (toml_opt).
- F.15 [Fable A21] dists_loss downloads weights at construction from huggingface without checksum; offline runs fail.
- F.16 [Fable] LDL paper formulation (and competitor) uses EMA generator output for the artifact map; current ldl_loss uses the live output (design question, not necessarily a bug).
