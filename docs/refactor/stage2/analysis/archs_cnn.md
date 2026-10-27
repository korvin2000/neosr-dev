# Stage 2 analysis — lane `archs_cnn`: convolutional / lightweight generators, discriminators, shared arch utilities

Scope: `neosr/archs/{esrgan,compact,span,spanplus,realplksr,plksr,mosrv2,moesr,esc,aether,flexnet,cugan,rcan,krgn,safmn,
eimn,man,msdan,ninasr,sebica,microsr,cfsr,hasn,plainusr,ea2fpn,desrgan}_arch.py` (convolutional parts),
`{unet,dunet,patchgan,metagan,resgan}_arch.py`, `vgg_arch.py`, `arch_util.py`, `archs/__init__.py`; competitor
`traiNNer/archs/`. Branch `refactor/stabilize` (== master), torch 2.9.1, CPU-only probes (`CUDA_VISIBLE_DEVICES=`).
All probe scripts and logs: `/tmp/claude-1000/-home-korvin-apps-neosr-dev/c1b40b02-f454-4498-9d89-be0afa4e2973/scratchpad/archs_cnn/`
(`census.py/json`, `online_reparam.py/log`, `plainusr_fuse.py/log`, `dysample_probe2.py/log`, `ln_probe.py/log`,
`aether_probe.py/log`, `explain.py/log`, `layout_fake.py`, `d_probe*.py/log`, `span_fast_dbg*.py`).

CPU timings were taken while other lanes ran CPU jobs on the same machine: ratios are meaningful, absolute ms are not.
Every speed claim must be confirmed on the 4090 by the profiling lane before it is used to rank work.

## 1. Priority summary

| id | P | finding | kind | verified | numerics change |
|---|---|---|---|---|---|
| A-01 | P1 | Online re-parameterisation: train linear multi-branch blocks (SPAN/SPAN+ `Conv3XC`, FlexNet `OmniShift`, PlainUSR `MBConv`, Aether `ReparamLargeKernelConv`) through their fused kernel | speed + memory | measured (CPU) | no (same function, fp rounding only) |
| A-02 | P1 | DySample grid in bf16: 0.13 px (LQ 64) to 0.9-1.9 px (LQ 256-1024) sampling error under the templates' bf16 autocast; same code path makes dynamic ONNX export wrong (F07) | correctness / export | reproduced | bf16 only (more exact); fp32 bit-identical |
| A-03 | P1 | `plainusr_ultra` training graph drops `blockm` and `block2` (grad None); eval runs them untrained (train/eval max diff 0.43) | correctness | reproduced | yes (fix) |
| A-04 | P1 | metagan (D of all 84 templates) gives the generator an adversarial input gradient 800x (init) to 250x (after 150 D steps) larger than the SN U-Net; GroupNorm after the SN Down convs is the cause | GAN stability / learning | measured (synthetic D probe) | yes, A/B only |
| A-05 | P2 | Parameters that never get gradients break DDP (`find_unused_parameters=False`): msdan (other-scale upsampler), ea2fpn (ResNet fc, unused DySample), plainusr_ultra | reliability | reproduced (grad None) | no |
| A-06 | P2 | `span_fast` init: SPAB chain underflows to exact 0 by block 3; blocks 3-5 + `conv_2` get zero/subnormal gradients | learning | reproduced | yes (A/B) |
| A-07 | P2 | realplksr family starts with output std 2.6-3.3 (target [0,1]) and G grad norm 29-49 (plksr 0.39): post-branch GroupNorm with gamma 1 in 28 residual blocks | learning / stability | measured | yes (A/B) |
| A-08 | P2 | Hand-written channels-first LayerNorm (safmn, man, cfsr, eimn, moesr) under bf16: 1.7e-3 .. 1.1e-1 relative error vs 1e-7 for `F.layer_norm`; 3.5x slower fwd+bwd on CPU | numerics + speed | measured | bf16 only (more exact) |
| A-09 | P2 | Aether float training: 68 idle `QuantStub`/`DeQuantStub` calls give 10 dynamo graphs / 9 breaks; with Identity: 1 graph | speed (compile) | reproduced | no |
| A-10 | P2 | F12: MeanShift params trainable (KRGN default; RCAN only with `norm=true`) and drift ~1e-3 per AdamW step | correctness | reproduced | tiny |
| A-11 | P3 | PlainUSR deploy broken (`switch_to_deploy` raises); ASR SE is saturated at sigmoid(409.6)=1.0, its 3 params per MBConv never learn | export / dead params | reproduced | ASR fix: yes |
| A-12 | P3 | channels_last breakers on CUDA (FakeTensor-dispatch verified): GroupNorm, mid-network PixelShuffle, reflect pad, instance norm, `repeat_interleave`; ESC LayerNorm `.contiguous()` in training | speed | measured (fake CUDA) / source | no |
| A-13 | P3 | Aether full-half `prepare_qat()` primes with fp32 input | QAT | reproduced | no |
| A-14 | P3 | PLK `convert()` deploy helpers allocate identity on CPU/fp32 (dead code: never called) | export | source-verified | no |
| A-15 | P3 | MetaGAN `Attention` passes `dropout_p` in eval (non-default helper) | correctness | source-verified | no |
| A-16 | P3 | SPAN/SPAN+/FlexNet eval-time fusion recompute (Codex perf rank 4): real but small; float ONNX export embeds the fuse subgraph (`do_constant_folding=False`) | speed / export | source-verified | no |
| A-17 | P3 | Init/learning nits: ninasr output starts at mean 0.95 (Rescale ±1), cfsr edge-prior branches merged in `__init__` then re-initialised (never trained), dead CA units (hasn hidden=1, rcan hidden=4) | learning | measured | yes |
| A-18 | P3 | dunet D: DySample (no SN, bf16 grid) inside the D; did not leave chance level in 150 D steps on the probe | GAN | measured | n/a |

Ranking rationale (owner priorities: speed, reliability, memory, training stability/efficiency): A-01 is the largest
equivalence-preserving speed and memory win in the lane and touches four popular families; A-02 is a silent
precision bug in every bf16 template run of the DySample families plus a wrong-pixels export bug; A-03 silently
trains a different network than it evaluates; A-04 is the strongest stability lever but needs the A/B (S.1 in PLAN
follow-ups); A-05 is a hard crash under DDP. Everything else is local and small.

## 2. Codex items (analyze.md)

| item | verdict | evidence |
|---|---|---|
| F07 DySample dynamic ONNX | **confirmed**, plus a larger eager bf16 issue | `dysample_probe2.log`: exported at 8x8 with `convert.py` settings (legacy exporter, dynamic axes, opset 17): max ORT error 1.85 at 12x10, 2.35 at 2x33x17. Fixed variant (normaliser from the traced grid, fp32 grid): 4.8e-7 / 1.9e-6, fp32 eager bit-identical (max diff 0.0). See A-02 |
| F12 RCAN/KRGN MeanShift trainable | **partial** | KRGN: confirmed in the default config. RCAN: `MeanShift` exists only with `norm=True` (`rcan_arch.py:164`); default `norm=False` has none. Probe: all 4 MeanShift tensors are in the optimizer and move 3-5e-3 in 5 AdamW steps. See A-10 |
| PLK deploy helpers (plksr:90,184,255) | **confirmed, low value** | `torch.eye(...)` on the default device/fp32 (`plksr_arch.py:90`, `248`), new `nn.Conv2d` built on CPU/fp32 then only `.to(device)` (`255`). No caller exists (`grep` over repo: `convert()` never called; `with_idt=False` default) → dead code. See A-14 |
| MetaGAN helper dropout | **confirmed, non-default** | `metagan_arch.py:55` passes `dropout_p=self.dropout_p` regardless of `self.training`; every construction uses `attn_drop=0.0`, so default MetaGAN is unaffected. See A-15 |
| PlainUSR deploy | **confirmed** | `Block.switch_to_deploy` → `AttributeError: 'MBConv' object has no attribute 'conv'` (`plainusr_arch.py:264`); `MBConv.switch_to_deploy` → `ValueError ... does not have a parametrization on expand_conv` (it calls `remove_parametrizations` on modules it already deleted, `:113-121`). A verified differentiable fused kernel replaces it (A-01, A-11) |
| Aether full-half QAT | **confirmed** | `aether_small().half().prepare_qat()` → `RuntimeError: Input type (float) and bias type (c10::Half) should be the same` from the fp32 priming input at `aether_arch.py:619`. See A-13 |
| SPAN/SPAN+/FlexNet eval-time recomputation (perf rank 4) | **partial** | Real (each eval forward re-fuses: ~10 tiny kernels per `Conv3XC`, 20 per SPAN), but small for validation. The large, measured win is the other direction: use the fused kernel in **training** (A-01). See A-16 |
| DySample bf16 grid (analyze.md §7 note) | **confirmed and quantified** | A-02 table |

## 3. Findings

### A-01 (P1, speed + memory) Online re-parameterisation of linear multi-branch blocks

**Where.** `span_arch.py:146-150` and `spanplus_arch.py:84-88` (`Conv3XC.forward`), `flexnet_arch.py:112-155`
(`OmniShift`), `plainusr_arch.py:56-66` (`MBConv.forward`), `aether_arch.py:127-134` (`ReparamLargeKernelConv.forward`).

**Mechanism.** Each of these blocks is a purely linear function of its input (1x1→3x3→1x1 + 1x1 skip; identity + dw1x1 +
dw3x3 + dw5x5 with scalar weights; 1x1 expand → bias-padded 3x3 → per-channel ASR scale + identity → 1x1 reduce +
identity; dw-lk + dw-sk). The existing deploy/eval fusion is exact *including borders* (SPAN pads x before the 1x1
conv; PlainUSR pads with the expand bias), so `F.conv2d(x, *fused_kernel())` is the same function of `(x, θ)` as the
training branch. Its autograd gradient w.r.t. every branch parameter is therefore the same (chain rule through the
differentiable fuse); weight decay acts on the same θ; DDP sees the same parameters. Training through the fused kernel
removes 3-4 convs and their saved activations per block. MACs per pixel: `Conv3XC(48, gain 2)` 94,464 → 20,736 (4.6x);
`MBConv(64)` 163,840 → 36,864 (4.4x); OmniShift is bandwidth-bound (3 dw-conv outputs + 3 scaled copies → 1 output).

**Measured** (`online_reparam.log`, `plainusr_fuse.log`; train mode, B=8, LQ 64x64, x4, L1, fwd+bwd, fp32 CPU):

| arch | fwd+bwd | saved activations (autograd) | max abs fwd diff | max rel grad diff |
|---|---|---|---|---|
| span | 11.83 s → 6.71 s (**1.76x**) | 936 → 336 MiB (**-64%**) | 2.8e-7 | 3.7e-6 |
| spanplus | 13.43 s → 9.41 s (1.43x) | 1080 → 480 MiB (-56%) | 8.9e-8 | 4.0e-6 |
| spanplus_s | 5.97 s → 4.37 s (1.36x) | 546 → 268 MiB (-51%) | 1.3e-7 | 4.0e-6 |
| span_fast | 5.77 s → 4.26 s (1.35x) | 523 → 193 MiB (-63%) | 3.3e-7 | 4.4e-6 |
| flexnet | 45.7 s → 33.9 s (1.35x) | 9.52 → 7.79 GiB (-18%) | 1.5e-7 | 3.0e-5 |
| plainusr | 6.23 s → 4.69 s (1.33x) | — | 1.3e-7 | 4.6e-4 (dead ASR grads ~0) |
| plainusr_large | 12.27 s → 6.35 s (**1.93x**) | — | 2.4e-7 | 1.1e-5 |

MBConv fused vs training forward in fp64 on a 17x23 input (borders included): max |d| = 1.1e-14.

**Fix.**
* `Conv3XC.forward` (both files): `return F.conv2d(x, *self.fused_kernel(), stride=self.stride, padding=1)` in
  every mode (delete the `if self.training` branch). `fuse_conv3xc` already runs in fp32 under autocast.
* `OmniShift.forward`: `return F.conv2d(x, self.fused_kernel(), padding=2, groups=self.conv5x5.groups)` always.
* `MBConv`: add the differentiable `fused_kernel()` below and use `F.conv2d(x, *self.fused_kernel(), padding=1)`;
  keep `switch_to_deploy` as "copy `fused_kernel()` into a plain 3x3 conv" (fixes A-11 deploy):
  ```python
  def fused_kernel(self):
      k0, b0 = self.expand_conv.weight[:, :, 0, 0], self.expand_conv.bias
      k1, b1 = self.fea_conv.weight, self.fea_conv.bias
      k2, b2 = self.reduce_conv.weight[:, :, 0, 0], self.reduce_conv.bias
      a = self.se.se(self.se.tensor).reshape(-1)
      w = torch.einsum("omuv,mi->oiuv", k1, k0) * a[:, None, None, None]
      b = (b1 + torch.einsum("omuv,m->o", k1, b0)) * a
      w, b = w + F.pad(k0[:, :, None, None], (1, 1, 1, 1)), b + b0
      w, b = torch.einsum("om,miuv->oiuv", k2, w), k2 @ b + b2
      eye = torch.eye(w.shape[0], dtype=w.dtype, device=w.device)[:, :, None, None]
      return w + F.pad(eye, (1, 1, 1, 1)), b
  ```
  (wrap in `torch.autocast(..., enabled=False)` like `fuse_conv3xc`).
* `ReparamLargeKernelConv.forward` (non-fused): `F.conv2d(x, *self._fuse_kernel(), self.stride, self.padding,
  groups=self.groups)` (source-verified algebra; not timed).
* Keep `eval_conv` / `conv5x5_reparam` and their `_save_to_state_dict` refresh: state_dict keys and export files
  are unchanged.

**Behaviour change.** None beyond fp rounding (fewer roundings under bf16: the fused kernel is computed in fp32 and
one bf16 conv replaces 3-4). Checkpoints/configs unchanged. Train and eval now run the same code path (removes a
train/eval divergence class). **Test:** for each arch, train-mode output and all parameter gradients equal the
branch path at rtol 1e-5 (fp32, CPU) on odd sizes (e.g. 17x23) — this is what the probes do; GPU wall-clock and peak
memory A/B on the 4090 (profiling lane); a 2k-update smoke run per family to confirm loss curves overlap.

### A-02 (P1, correctness / export) DySample: bf16 sampling grid and non-dynamic normaliser

**Where.** `arch_util.py:103-131`. Used by spanplus (default `upsampler="dys"`), flexnet (always), esc (realsr
default), realplksr/mosrv2/moesr (option), dunet D, ea2fpn. All 84 templates set `use_amp = true`,
`bfloat16 = true`.

**Evidence** (`dysample_probe2.log`; offsets rounded to bf16 as a bf16 conv emits them, grid arithmetic in bf16 as the
code does under autocast; trained-like offsets):

| LQ size | grid error max / mean (LQ px) | sampled-feature rel. error | bf16 feature rounding alone |
|---|---|---|---|
| 32 | 0.063 / 0.018 | 2.1e-2 | 1.0e-3 |
| 64 (template patch) | **0.126 / 0.028** | **3.1e-2** | 1.0e-3 |
| 128 | 0.250 / 0.125 | 1.4e-1 | 1.0e-3 |
| 256 | 0.924 / 0.367 | 2.4e-1 | 1.0e-3 |
| 1024 | 1.928 / 0.754 | 3.5e-1 | 1.0e-3 |

`coords + offset` is formed in bf16 at magnitudes up to H, so a learned sub-pixel offset is quantised to 1/8-1/4 px at
the training patch and to whole pixels at inference sizes; the normaliser `torch.tensor([W, H], dtype=x.dtype)` is
also inexact in bf16 for W > 256. Training (bf16) and validation (fp32, `image.py` validates without autocast) thus
see different upsamplers. The same `torch.tensor([W, H])` is a trace constant for the legacy ONNX exporter (F07,
reproduced above) and a per-forward host→device copy in eager CUDA.

**Fix** (verified in `dysample_probe2.py::Fixed`):
```python
offset = offset.float().view(B, 2, -1, H, W)                       # grid in fp32
coords_h = torch.arange(H, device=x.device, dtype=torch.float32) + 0.5
coords_w = torch.arange(W, device=x.device, dtype=torch.float32) + 0.5
coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij")).transpose(1, 2)[None, :, None]
normalizer = (torch.stack((coords_w[-1], coords_h[-1])) + 0.5).view(1, 2, 1, 1, 1)  # [W, H], traced
coords = 2 * (coords + offset) / normalizer - 1
...  # pixel_shuffle / permute unchanged; grid_sample(x, coords): autocast promotes x to fp32
```
fp32 eager output is bit-identical (max |d| 0.0); ONNX dynamic export correct at 3 shapes. Cost under autocast: the
`grid_sample` input is promoted to fp32 (one LQ-resolution copy) and its output is fp32 until `end_conv` casts it.

**Behaviour change.** fp32 runs: none. bf16 runs: the upsampler becomes exact (training numerics change; low-risk
A/B on spanplus/flexnet/realplksr+dysample). Checkpoints unchanged. **Test:** replace
`tests/test_archs.py::test_dysample_grid_is_fp32_then_cast` (it locks in the historical bf16 rounding — obsolete
after the fix) with: (a) bf16-module grid equals the fp32 grid within 1e-6 at W=512; (b) ONNX export at 8x8 matches
ORT at 12x10 and 2x33x17 (skip without onnxruntime).

### A-03 (P1, correctness) `plainusr_ultra` trains without `blockm` and `block2`

**Where.** `plainusr_arch.py:451-477`. For `n_feat <= 32` the training branch computes
`x1 = self.block2(torch.cat([self.blockm(x1), x2], 1))` and then concatenates the block-1 output `x` with `pic`
(`:476`), so `x1` is discarded. `fast_forward` (eval) applies both blocks in place.
**Evidence:** census: every `blockm.*` and `block2.*` parameter has `grad None`; train vs eval forward of the same
weights differs by up to 0.43. Validation and exports therefore run two randomly initialised blocks.
**Fix:** in the `n_feat <= 32` branch set `x = self.block2(torch.cat([self.blockm(x1), x2], 1))` before
`torch.cat([x, pic], 1)`. **Behaviour change:** yes (the intended network now trains); existing `plainusr_ultra`
checkpoints contain untrained `blockm`/`block2` and must be retrained or fine-tuned. **Test:** train/eval output
equality at init and no `grad None` for every plainusr variant (add to the census-style test, A-05).

### A-04 (P1 for stability, A/B only) metagan's adversarial gradient dwarfs the other G terms

**Where.** `metagan_arch.py:115-119` (`Down` = SN conv + `GroupNorm(4)`), `:151` (`downs=(4,4,2,2)`: a 64x
reduction, 4x4 logits for a 256 GT patch, 2x2 for 128), `:171-177` (head with `Dropout(0.2)`); `gan_loss.py`
(`BCEWithLogitsLoss`, mean over logits). Templates: 84/84 use `network_d = metagan`, `gan_opt.loss_weight = 0.3`.

**Probe** (`d_probe.py`, CPU): real = multi-scale smooth images 128x128, fake = x4 bicubic down/up (an early-SR-like
blur), D trained with BCE, AdamW lr 1e-4 (template D lr), batch 4. `g_adv = ||∂ BCE(D(fake), real) / ∂fake||` is what
reaches G per unit `loss_weight`. Reference: L1 (mean) on the same tensor has gradient norm 1/sqrt(N) = 2.3e-3.

| D | logits / image | g_adv at init | g_adv at 150 D steps | D loss at 150 | |logit| max at 150 |
|---|---|---|---|---|---|
| metagan | 2x2 | **1.8e-1** (80x L1) | **1.6** (700x L1) | 0.000 | 16.1 (still growing) |
| unet (SN) | 128x128 | 2.2e-4 (0.1x L1) | 6.5e-3 (2.8x L1) | 0.437 | 9.3 |
| patchgan (SN) | 14x14 | 1.4e-3 | 5.7e-1 | 0.017 | 7.6 |
| dunet | 128x128 | 3.2e-4 | 6.5e-4 | 1.382 (chance) | 0.1 |
| metagan + SN fc1/fc2 | 2x2 | 1.5e-1 | 4.3e-1 @80 | 0.001 | 13.3 |
| metagan + SN fc1/fc2 + layer-scale 1e-6 | 2x2 | 1.5e-1 | 7.2e-1 @80 | 0.005 | 12.1 |
| metagan, GroupNorm in `Down` → Identity | 2x2 | **6.8e-3** | 6.0e-3 @50 | 1.386 | 0.1 |

Reading: metagan separates this pair within 25-50 updates and its logits then grow without bound (BCE overconfidence);
at `loss_weight = 0.3` its G-side gradient is ~25x (init) to ~200x (trained D) an L1 term of weight 1, whereas the SN
U-Net stays within 0.03-1x. SN on the gated blocks or a layer-scale init (competitor MetaGan2 style) does **not**
change this; removing the `GroupNorm` after the SN `Down` convs lowers the init gradient 27x (and makes the D slow to
learn). The SN on `Down` is effectively cancelled by the following `GroupNorm`, which is scale invariant
(`GN(Wx/σ + b) = GN(Wx + σb)`), so metagan has no Lipschitz control before its head. This is a synthetic probe, not a
training run: it shows the order of magnitude, not the final quality effect.

**Recommendation (no default change without the S.1 acceptance A/B; no auto-damping).**
1. Add the cheap diagnostic first: every `print_freq` updates, log `||∂(w_i L_i)/∂sr||` per G term (one
   `torch.autograd.grad` per term on the current batch, device scalars, no extra sync beyond the log read). It shows
   per run whether the adversarial term dominates — tuning by evidence, not by automatic normalisation.
2. A/B (>= 20k updates, x4, LQ 64, the template loss stack): `metagan @ 0.3` (current) vs `unet @ 0.1`
   (Real-ESRGAN recipe) vs `metagan @ 0.03`. Accept the variant with the better LPIPS/DISTS at equal PSNR drop and no
   D-logit runaway (`out_d_real - out_d_fake` stays bounded — already logged by `gan_d_loss`).
3. Architectural candidates for a stable metagan (each needs the same A/B): SN-conv without the following GroupNorm
   in `Down` plus a normalisation that is not per-sample-scale-invariant; or R1 on real images (deferred S.1).
   The competitor's MetaGan2 changes (SN fc1/fc2, gamma 1e-6, no dropout) did not move g_adv in the probe.
**Behaviour change:** objective change → A/B required. Config/checkpoint: switching D type loses D pretrain only.

### A-05 (P2, reliability) Parameters without gradients crash DDP

**Where / evidence (census).** `msdan_arch.py:367-378`: the upsampler builds the x2 and x3 branches for every scale
(`BSConv_3`, `BSConv1_3` unused at x2/x4; `BSConv1_2`, `BSConv_2` unused at x3) → grad None.
`ea2fpn_arch.py:206` keeps `models.resnet18(...)` including its unused `fc` (513k params) and every
`Conv3x3GNMish` builds a `DySample` even when `upsample=False` (`:125`) → grad None. `plainusr_ultra` (A-03).
`base.wrap` builds DDP with `find_unused_parameters=False` (default), so the second update of such a run raises
"Expected to have finished reduction in the prior iteration". msdan has two templates.
**Fix (keys unchanged):** after construction, `requires_grad_(False)` on the branches the configured scale never
uses (msdan) and on `base_model.fc` and non-upsampling `dysample` modules (ea2fpn). Building only the needed modules
would change state_dict keys and break old checkpoints. **Behaviour change:** none for single-GPU numerics; resuming a
format-2 state saved before the fix fails the strict trainable-name check (`phase.py:141-146`) → start from
`pretrain_network_g`. **Test:** a CPU census test over `ARCHS` (train mode, one backward, assert no trainable
parameter has `grad None`), with an explicit allowlist for intentional cases; it also guards A-03.

### A-06 (P2, learning) `span_fast` init kills blocks 3-5

**Where.** `span_arch.py:172-180` (SPAB) + `:257-258`: `span_fast` uses `bias=False` and
`kaiming_normal_(mode='fan_out', nonlinearity='linear')`; each SPAB returns `(out3 + x) * (sigmoid(out3) - 0.5)`,
which is ~quadratic in small activations. **Evidence** (`span_fast_dbg2.py`): block output abs-max
`refine 0.67 → b1 0.18 → b2 0.002 → b3 0 → b4 0 → b5 0 → conv_2 0` (exact zeros, fp32 underflow; fp16 underflows
earlier); census: `conv_2.*` gradients exactly 0 over 3 Adam steps, blocks 3-5 subnormal. Adam cannot move a
parameter whose gradient is << eps (1e-8). `span` (default PyTorch init, with bias) stabilises at ~0.015 per block.
**Fix:** drop the custom `_init_weights` for `fast` (PyTorch default init, as `span`) or scale the 9x9/refine init so
the block chain keeps O(0.1) activations. **Behaviour change:** init only; A/B (2-5k updates, compare PSNR curve).
Checkpoints unaffected.

### A-07 (P2, learning/stability) realplksr starts far from the target

**Where.** `realplksr_arch.py:82-97`: each of 28 blocks ends with `GroupNorm(4, 64)` (gamma 1, `:86`) on the residual
branch, so every block adds a unit-variance term to the stream. **Evidence (census, x4, 32x32, L1):**

| arch | out mean / std at init | G grad norm at init |
|---|---|---|
| realplksr | 0.47 / **2.57** | **28.9** |
| realplksr_s | 0.45 / 1.22 | 20.6 |
| realplksr_l | -0.14 / **3.32** | **49.3** |
| realplksr + dysample | 0.87 / 1.14 | 51.7 |
| plksr (no GN) | 0.46 / 0.33 | 0.39 |

The first hundreds of updates are spent shrinking the output; with the default fixed clip at 1.0 (trainer lane T-02)
every early step is clipped 30-50x. **Fix candidates (A/B):** init the block GroupNorm weight small (`1e-2`, or 0 =
"zero-gamma", keys unchanged, pretrains unaffected); or the competitor's pre-LayerNorm variant
(`traiNNer/archs/realplksr_arch.py`, `layer_norm=True` default) — new keys, a feature port for the next stage.
**Behaviour change:** init only (first option). Test: init output std < 0.5 and the 5k-update A/B.

### A-08 (P2, numerics + speed) Hand-written channels-first LayerNorm

**Where.** `safmn_arch.py:24-29`, `man_arch.py:29-34`, `cfsr_arch.py:24-29`, `eimn_arch.py:35-40`
(`data_format="channels_first"` branch), `moesr_arch.py:105-109` (always). Under CUDA bf16 autocast `mean`/`sub`/`div`
run in bf16 (only `pow` is fp32-listed), so the mean is rounded before the subtraction.
**Evidence** (`ln_probe.log`, normalised-output error vs fp64, bf16 input with channel-mean/std ratio m):
m=0: 1.7e-3, m=1: 2.4e-3, m=4: 6.9e-3, m=16: 2.7e-2, m=64: 1.1e-1; `F.layer_norm` ~1e-7 in every case. fp32 equality
manual vs fused: max 9.5e-7. CPU fwd+bwd [8,64,64,64]: 303 ms → 86 ms (3.5x); fused output stays channels_last.
**Fix:** `F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)`
(same formula: biased variance, eps inside the sqrt; free view under channels_last). **Behaviour change:** fp32 none;
bf16 more exact (light A/B). Keys unchanged. Lower priority, same pattern: `mosrv2_arch.py:129-140` RMSNorm (5
elementwise ops; `norm` is fp32-listed so precision is fine) and aether `DeploymentNorm` (bf16 batch mean/var).

### A-09 (P2, compile speed) Aether idle quant stubs break the graph

**Where.** `aether_arch.py:178,190,313,323-324,329,473,514,531-532,543,557`. **Evidence** (`aether_probe.log`,
`dynamo.explain`, float training): `aether`/`aether_small`: 68 stubs, graphs=10, breaks=9 ("Attempted to inline
function marked as skipped: QuantStub.forward"); stubs replaced by `nn.Identity`: graphs=1, breaks=0. All other
explained archs (compact, span, spanplus, realplksr(+dys), esrgan, mosrv2, moesr, flexnet, plksr) compile as one
graph in train and eval (`explain.log`).
**Fix:** guard each stub call with the existing `is_quantized` flag (`if self.is_quantized: x = self.quant_norm(x)`),
or construct `nn.Identity()` placeholders and install `tq.QuantStub()` at the top of `prepare_qat()`. Unprepared
stubs hold no state, so keys do not change. **Behaviour change:** none. **Test:** `dynamo.explain(aether_small)`
graph_count == 1 in float training; existing QAT tests unchanged.

### A-10 (P2) F12 MeanShift is trainable

**Where.** `krgn_arch.py:7-16` (used unconditionally), `rcan_arch.py:13-21` (only `norm=True`). `self.requires_grad =
False` sets a module attribute. **Evidence:** all 4 tensors in `trainable()`; 5 AdamW(lr 1e-3, wd 0.01) steps move
`sub_mean.weight` by 4-5e-3, `add_mean.bias` by 3e-3; the "fixed" colour transform drifts O(1) over 1k updates.
**Fix:** `self.requires_grad_(False)` in both `MeanShift.__init__`. Keys unchanged (old learned values still load).
**Compat:** a format-2 state saved before the fix fails the strict trainable-name check on resume (4 names fewer) —
document; pretrain loading is unaffected. Numerics: negligible.

### A-11 (P3) PlainUSR deploy and ASR

Deploy errors: see Codex table. Replace `MBConv.switch_to_deploy` by "build `nn.Conv2d(C, C, 3, 1, 1)` and copy
`fused_kernel()`" (A-01) and fold `PConv` into the preceding conv (the existing v2/v3 merge math at `:284-298` is
correct once the MBConv conv exists); then `convert.py --prune` can export the plain net. ASR (`plainusr_arch.py:20-44`):
`se[1]` and `se[3]` are filled with 1 ("to make sure the initial [0.5,...]"), giving `sigmoid(32 * SiLU(12.8)) =
sigmoid(409.6) = 1.0` exactly; census: `se.tensor`, `se.se.1.weight`, `se.se.3.weight` gradients are exactly 0 in every
MBConv of all three variants — they never learn and ASR is a constant 1. Upstream (icandle/PlainUSR) has the same
code. A real fix (e.g. small random `se` weights so the initial attention is ~0.5) changes the network → A/B; low value.

### A-12 (P3, speed) channels_last breakers on CUDA

`model.mf = channels_last` whenever `use_amp` (`image.py:46`). FakeTensor dispatch with a `cuda` device
(`layout_fake.py`, real CUDA meta/composite code, no GPU used) gives for a channels_last input:
GroupNorm → NCHW, PixelShuffle → NCHW, reflect pad → NCHW, instance_norm → NCHW, `repeat_interleave` → NCHW,
`reshape(B*g, -1, H, W)` → NCHW; conv, constant pad, cat, nearest interpolate, pixel_unshuffle, batch_norm,
avg/max pool, `F.layer_norm`/`F.rms_norm` on the NHWC view → channels_last. Affected hot paths: realplksr (28
GroupNorms per forward, 2 layout copies per block fwd and bwd), metagan D (6 `Down` GroupNorms), moesr (`MSG`
PixelUnshuffle/PixelShuffle in the residual path of 9 blocks), ESC `LayerNorm` training branch (`esc_arch.py:75-86`
calls `.contiguous()` after both permutes → 2 copies per LN; the eval branch omits them). Estimated cost for realplksr
at B8/LQ64/bf16: ~112 copies of 4 MiB per step (~1 ms) — small. **Fix:** drop the two `.contiguous()` in ESC's
training LN (zero-copy under channels_last; same values); let the profiling lane measure realplksr/moesr in
`contiguous_format` vs channels_last before adding any per-arch layout opt-out.

### A-13 (P3) Aether full-half QAT priming

`aether_arch.py:619`: `self(torch.rand(1, C, 16, 16, device=self.mean.device, dtype=self.conv_first.weight.dtype))`.
Trainer path (fp32 module + autocast) unaffected.

### A-14 (P3) PLK deploy helpers

`plksr_arch.py:90`: `torch.eye(n, dtype=k.dtype, device=k.device)`; `:248` same; `:255`: build the new conv with
`device=k.device, dtype=k.dtype`. Or delete `convert()`/`is_convert` (no caller, `with_idt=False` default).

### A-15 (P3) MetaGAN Attention dropout in eval

`metagan_arch.py:55`: `dropout_p=self.dropout_p if self.training else 0.0` (FlexNet already does this,
`flexnet_arch.py:251`). Default MetaGAN constructs it with 0.0.

### A-16 (P3) Eval-time fusion recompute and float export

With A-01 the training path fuses every step anyway; the eval cost stays ~10 small kernels per `Conv3XC` per forward
(~200 launches for SPAN): negligible for validation, visible only for many-tile inference. A safe cache would key on
the source parameters' `(data_ptr, _version)` (optimizer steps, `load_state_dict`, EMA `lerp_` all bump `_version`).
Not worth it now. Export: `convert.py:287` uses `do_constant_folding=qat`, so float ONNX of SPAN/SPAN+/FlexNet carries
the fuse subgraph (flip/permute/conv on weights) into the file; ORT folds it at load, other runtimes may not. Export
the deploy graph (`eval_conv`) or enable constant folding for float exports (convert lane).

### A-17 (P3) Init / learning nits

* ninasr (`ninasr_arch.py:58-66,81,101`): learnable scalar Rescale initialised to -1 / +1 (upstream: frozen
  per-channel RGB mean), so the init output mean is 0.95 vs a ~0.45 target; `±0.5` would start at mid-grey
  (init-only change; pretrains unaffected).
* cfsr (`cfsr_arch.py:319-321`): `merge_all()` in `__init__` folds and deletes the Sobel/Laplacian edge-prior branches,
  then `self.apply(self._init_weights)` re-initialises the merged `pos` conv: the edge prior is never trained. Keep
  (changing it changes keys); document.
* Dead channel-attention units (census): `hasn` CA has `52 // 30 = 1` hidden unit (dead in blocks 2-3 for seed 0),
  rcan CA 4 hidden units (1 of 200 RCABs dead). Architecture choice; no action.
* `rcan_arch.py:181` passes `res_scale=scale` (the upscale factor) but `RCAB.forward` ignores it (`:103`): harmless,
  remove the argument to avoid a future "fix" that scales residuals by 4.

### A-18 (P3) dunet D

`dunet_arch.py:13-18`: `Up` uses `DySample` whose offset convs are not spectrally normalised and whose grid has the
A-02 bf16 issue; in the probe dunet stayed at chance (D loss 1.382) for 150 updates while unet separated by 25-50. Not
recommended as a default; A-02 applies.

## 4. Gradient census (default config, x4, CPU, 2x3x32x32 (D: 128), 3 Adam steps)

No generator has accidental trainable constants other than MeanShift (A-10). Frozen parameters are only the deploy
copies (`eval_conv`, `conv5x5_reparam`) — intentional. Grad-None / always-zero: plainusr_ultra (A-03), msdan (A-05),
ea2fpn (A-05), span_fast `conv_2` (A-06), plainusr ASR (A-11), dead CA units (A-17). Selected init statistics:

| arch | params | G grad norm @init | out mean / std | arch | params | G grad norm | out mean / std |
|---|---|---|---|---|---|---|---|
| esrgan | 16.7M | 3.94 | 0.40 / 0.28 | compact | 0.62M | 0.008 | 0.50 / 0.29 |
| span | 2.24M | 0.49 | -0.02 / 0.09 | spanplus | 2.23M | 0.98 | 0.02 / 0.07 |
| realplksr | 7.39M | 28.9 | 0.47 / 2.57 | plksr | 7.39M | 0.39 | 0.46 / 0.33 |
| mosrv2 | 4.18M | 0.033 | 0.50 / 0.20 | moesr | 16.5M | 3.02 | -0.03 / 0.54 |
| esc | 0.97M | 14.3 | 0.00 / 0.37 | aether | 1.38M | 0.005 | 0.50 / 0.01 |
| flexnet | 3.05M | 1.29 | 0.03 / 0.19 | cugan | 1.41M | 0.38 | 0.51 / 0.32 |
| rcan | 15.6M | 2.64 | -0.01 / 0.20 | safmn_l | 5.60M | 7.97 | -0.16 / 1.28 |
| ninasr | 1.03M | 1.35 | 0.96 / 0.37 | sebica | 0.04M | 1.27 | 0.56 / 1.51 |
| metagan (D) | 6.02M | 15.0 | 0.53 / 0.53 | unet (D) | 4.38M | 2.66 | 0.01 / 0.03 |

## 5. Discriminators: placement and setup notes

* D is always in train mode (`image.py:58`): SN power iteration runs in the G adversarial pass and in both D passes
  (3 updates of u/v per iteration) — expected; EMA does not average D. SN old-style hooks clone u/v, so two forwards
  before one backward are safe in eager (compile is G-only, `base.py:142`).
* BN in D: only `patchgan(norm_type="batch")`. Real and fake run as separate batches (`gan_loss.py:63-64`), so batch
  statistics leak "real/fake" to the D; under DDP buffers are not broadcast (running stats unused in train mode).
  Recommend rejecting `norm_type="batch"` for D or documenting it; default `none` is fine.
* unet (Real-ESRGAN U-Net SN) is identical to the competitor's `UNetDiscriminatorSN`; skips are correct for even
  patch sizes (all templates). Init logits are ~0 (std 0.03): the cleanest BCE start of the four.
* resgan: one global logit (`squeeze()` gives shape [1] at batch 1; works with `full_like`).
* Most stable configuration for SR-GAN with BCE in this probe and in published practice: SN U-Net, BCE, G weight
  ~0.1, D lr ~= G lr/1-10 (Real-ESRGAN). Whether metagan is better for quality must come from the A/B in A-04.

## 6. Competitor (traiNNer-redux) notes for this lane

* DySample (`traiNNer/archs/arch_util.py:374-449`): identical code, same bf16 grid and ONNX constant (adds
  `pin_memory=True` to the per-forward tensor). Nothing to port; A-02 is new.
* SPAN `Conv3XC`: branch training + cached `eval_conv` refreshed on `train(False)` (the stale-cache design neosr already
  removed). No online re-parameterisation anywhere: A-01 is new.
* RealPLKSR: optional pre-`LayerNorm` instead of the post-branch GroupNorm (`layer_norm=True` default) — the
  stability-relevant difference (A-07); new keys, so a later feature port behind a flag.
* MetaGan2: SN on gated-block Linear layers, layer-scale 1e-6, no dropout/drop-path, downs (4,2,2,2). The probe shows
  these alone do not reduce the adversarial gradient magnitude (A-04).
* dunet uses `torch.nn.utils.parametrizations.spectral_norm` (new API; compile-friendlier, different state_dict keys
  `parametrizations.weight.original*`: needs the D key remap deferred as A4 in PLAN).
* UNetDiscriminatorSN and PatchGAN: same as neosr.
* None of the competitor's aggressive automations (dynamic loss scheduling, adaptive clipping) are recommended here.

## 7. Suggested order of work

1. A-02 (DySample fp32 grid + traced normaliser; replace the obsolete bf16-rounding test) and A-03, A-05, A-10, A-13,
   A-14, A-15 as one small "arch correctness" lane (no objective change except A-03; tests: census test + ONNX test).
2. A-01 online re-parameterisation (span, spanplus, flexnet, plainusr, aether) + A-11 deploy via the fused kernel;
   equivalence tests; GPU timing by the profiling lane; 2k-update smoke runs.
3. A-09 aether stub guards; A-08 `F.layer_norm` swap; ESC `.contiguous()` removal (A-12).
4. Experiments (A/B, owner sign-off before defaults move): A-04 (D choice / weight, per-term gradient diagnostic),
   A-06 span_fast init, A-07 realplksr GN gamma init.
