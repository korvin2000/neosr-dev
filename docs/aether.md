# AetherNet in neosr

AetherNet is a single-image SR network by Philip Hofmann (MIT, https://github.com/phhofm/aethernet). It is
built to be deployed. Large-kernel reparameterized convs and a foldable norm collapse into a plain conv
graph, and quantization-aware training (QAT) produces an INT8 QDQ ONNX.
neosr keeps one implementation, `neosr/archs/aether_arch.py` (design `refactor/design/T5_aether_qat.md`).

## Variants

| type | embed_dim | depths | channel / spatial attention |
|---|---|---|---|
| `aether_small` | 96 | 4,4,4,4 | no / no |
| `aether_medium` | 128 | 6,6,6,6 | yes / yes |
| `aether_large` | 180 | 8,8,8,8,8 | yes / yes |
| `aether` | free (`embed_dim`, `depths`, `use_*_attn`, ...) | | |

The run `scale` reaches the constructor (2, 3, 4 or 8). Other constructor kwargs (`mlp_ratio`, `lk_kernel`,
`sk_kernel`, `drop_path_rate`, `quantize_residual`, `fused_init`) go in `[network_g]`.

## Building blocks

| block | training form | deploy form |
|---|---|---|
| `ReparamLargeKernelConv` | depthwise 11x11 + 3x3 branches | one 11x11 conv (`fuse_reparam()`) |
| `DeploymentNorm` | per-channel norm, EMA running statistics (like BN) | scale/shift (`fold_norms()`) |
| `GatedConvFFN` | 1x1-conv GLU with SiLU gate and temperature | same |
| multi-stage fusion | 1x1 conv per stage, concat, `QuantFusion`, global residual | same |

`fuse_model()` does both folds in place. `[network_g] fused_init = true` builds the deploy topology directly,
for loading a `fuse_model()` checkpoint. `DeploymentNorm.fused_flag` is saved with the weights, so a file
loads into the behaviour it was saved with. Files written before the flag existed keep the constructor's
`fused_init`.

## QAT training

`[train] enable_qat = true` works from scratch (the shipped `options/aether/*.toml`) or from a float,
deployed or QAT `pretrain_network_g`. Only a QAT file is detected from its keys (its fake-quant modules). A
deployed float file (`fuse_model()` or `convert.py --strip-qat`) needs `[network_g] fused_init = true`, else the
strict load fails; this is distinct from `[train] fused_init` below. The trainer calls
`prepare_qat()` on the bare generator before EMA, optimizers, DDP and compile exist. It fuses the reparam
convs, keeps the norms live and inserts fake quant. Weights are symmetric per-channel int8 (scale
`max|w|/127`), activations symmetric per-tensor int8. `conv_first`, `conv_before_upsample.0`, `conv_last`
and the attention sigmoids stay float.

| key | default | meaning |
|---|---|---|
| `qat_type` | `"int8"` | the only policy (TensorRT / ONNX Runtime QDQ) |
| `fused_init` | `true` | reparam convs fused before QAT; `false` is rejected |
| `quantize_residual` | network default `true` | quantize the block residual add; must not contradict `[network_g]` |
| `calibration_iters` | `0` | updates `1..N` only observe (fake quant off) |
| `qat_freeze_norm_iter` | `0.7 * logger.total_iter` | DeploymentNorm statistics frozen from this update; `-1` never |
| `qat_freeze_observers_iter` | `0.8 * logger.total_iter` | activation/weight ranges frozen from this update; `-1` never |

The phase is a pure function of the update number, so resume and DDP need no extra state. Validation never
moves the ranges, and the exported `net_g_<update>.pth` holds the EMA with its quantization state. AMP
(bf16/fp16) and `compile` are accepted with a warning; fp32 eager is the reference. Under DDP the observer
and norm statistics come from rank 0 (buffer broadcast).
Recipe: from-scratch QAT trains the fused large kernel, not the two reparam branches. A float pretrain
followed by a QAT fine-tune keeps the branch benefit.

## Deployment

| goal | how |
|---|---|
| INT8 QDQ ONNX | `python convert.py --input net_g_N.pth -net aether_small -s 2` on a QAT file: ranges frozen, fake quant on, norms folded, opset >= 13; parity against the fake-quant model with ONNX Runtime unoptimized (max <= 5e-3, mean <= 1e-4) and optimized (mean <= 2e-3) |
| float ONNX | the same command on a float file (exported unfused) |
| evaluate a QAT file | `test.py` with `pretrain_network_g` = the QAT file: fake-quant inference, frozen ranges |
| deployed float `.pth` | `net.eval(); net.fuse_model(); torch.save({"params": net.state_dict()}, ...)`, then load with `fused_init = true` (`options/test_aether.toml`) |
| float `.pth` from a QAT file | `python convert.py --input net_g_N.pth -net aether_small -s 2 --strip-qat`: drops the QAT state, fuses and folds, checks parity, writes `net_g_N_float.pth` (`--output` overrides); `--onnx` also writes its float ONNX (`net_g_N_float_fp32.onnx`). Load the `.pth` with `[network_g] fused_init = true` |

## Limits and open points

- `torch.ao.quantization` (eager QAT) is deprecated and slated for removal in torch 2.10 (torchao replaces it).
- medium/large: the attention convs get weight Q/DQ, but their inputs stay float. x4 and x8: the convs after
  a PixelShuffle have no input Q/DQ.
- TensorRT engine builds, native CPU INT8 (`torch.ao.quantization.convert`), multi-GPU DDP and `compile` with QAT are unverified.

## History

Upstream shipped five drafts (`common/v1`..`v5`) plus `neosr/archs/aether_core.py`. Commit `3514432` shows that
file is byte-identical to `common/v3`. All copies and the standalone `aether_scripts` (PTQ calibration,
eager INT8 conversion) were removed. `convert.py` covers export.
