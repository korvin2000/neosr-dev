# Stage 2 analysis: window-attention transformer architectures

Lane scope: `neosr/archs/{swinir,swinirdense,hat,drct,dat,rgt,atd,srformer,craft,hitsrf}_arch.py`,
`neosr/archs/arch_util.py`, `neosr/archs/__init__.py` (ARCHS rows), `tests/test_archs*.py`; competitor
`traiNNer/archs/` (whose SwinIR/ATD/RGT/CRAFT/SRFormer/DAT/HAT come from `spandrel` /
`spandrel_extra_arches`; read from `~/miniconda3/envs/redux-dev/.../site-packages/spandrel`). Checkout
`refactor/stabilize` (= master), torch 2.9.1+cu128, Python 3.13. All probes ran on CPU
(`CUDA_VISIBLE_DEVICES=`; the GPU belongs to the profiling lane) on a shared machine, so CPU wall-clock numbers are
relative indications, not speed claims. Probe scripts and raw outputs: `$SCRATCH/archs_window/` (session
scratchpad, not committed): `p1_grad_census.py`, `p2_flash_parity.py`, `sdpa_patch.py`, `p3_sdpa_parity.py`,
`p3b_bcast_parity.py`, `p5_srformer_ckpt.py`, `p6_hitsrf_bias.py`, `p7_counts.py`, `p8_tiny.py`,
`p9_maskops.py`, `p10_saved.py`, `p11_module_parity.py`, `p12_onnx.py`, `p12b_onnx64.py`, `p13_craft_pad.py`, their
`*_out*.txt`, and `bench_window_sdpa_cuda.py` (a CUDA benchmark prepared for the profiling lane, **not run here**).

Evidence levels: **reproduced** (a probe shows the failure), **measured** (number from a probe),
**source-verified** (read, not executed), **hypothesis** (plausible mechanism; needs CUDA profiling or an A/B).

What matters most is the default path. Every lane template (`options/train_{swinir,hat,drct,dat,rgt,atd,srformer,
craft,hitsrf}*.toml`) sets only `type` (`swinir_small`, `hat_m`, `drct`, `dat_m`, `rgt`, `atd`, `srformer_light`,
`craft`, `hit_srf`), `use_amp=true`, `bfloat16=true`, `fast_matmul=true`, `batch_size=8` and `patch_size=32`
(LQ; `craft` 64). Because the factories use `img_size` 48-96, **the training patch never equals `img_size`**: every
SwinIR-style block takes its "size mismatch" branch on every training step (this drives W-04/W-05).

## 1. Summary and priorities

The families are numerically correct on their default (manual) attention path: a full gradient census on every
default row finds **no parameter with `grad is None`** (p1). The remaining value is in four places: (1) the manual
`softmax(QK^T*s + B [+M]) V` keeps ~6 bytes per attention element for backward and can be replaced exactly by
SDPA with an additive float mask; (2) HiT-SRF builds quadratic position-bias intermediates; (3) host syncs and
per-block mask rebuilding on the hot path; (4) a few real bugs (SRFormer checkpointing, flash switches, CRAFT's
always-extra padding and dead positional bias, tiny-input crashes, an RGT train/eval mismatch at the template
patch).

| # | Pri | Finding | Kind | Evidence | Numerics |
|---|---|---|---|---|---|
| W-01 | P1 | Exact SDPA (mem-efficient kernel) path for window attention with bias + shift mask, all 9 families. Manual path stores 6 B/attention element for backward: ~0.42 GiB swinir_small, ~3.5 GiB hat_m, ~1.55 GiB drct, ~5.1 GiB atd per step | speed+memory | parity measured; memory measured per element; speed hypothesis | rounding only |
| W-02 | P1 | F04: `flash_attn=True` changes SwinIR/SwinIR-dense (no bias, no shift mask; 24/48 bias tables grad None). CRAFT's flash branch is exact (its bias is constant, see W-08) but leaves 448 params grad None (DDP error) | correctness | reproduced | fix = W-01 |
| W-03 | P1 | F09: HiT-SRF position bias is O(N^2); exact O(N*base_N) pooled form (avg_pool2d + small index) | memory+speed | reproduced, fix measured exact | rounding only |
| W-04 | P1 | Blocking host->device copies in forward: ATD 73 per forward, SRFormer 24, RGT 24 (each a `cudaStreamSynchronize`) | speed | counted (p7) | none |
| W-05 | P2 | Shift masks rebuilt in every block at the template patch (SwinIR 24/fwd, DRCT 30, SRFormer 24, DAT 9, RGT 12), half of them all-zero masks for unshifted blocks; 49 aten ops per build = 38 % of swinir_small forward dispatches | speed | counted (p7, p9) | none |
| W-06 | P2 | CRAFT always pads one extra window (64 -> 80 px at its template patch, +56 % tokens); CPU fwd+bwd 30.8 s -> 23.4 s with aligned padding | speed | measured | border context (A/B) |
| W-07 | P2 | F05: SRFormer `use_checkpoint=True` raises TypeError; fixed call is exact under `backward(inputs=params)` | reliability | reproduced, fix verified | none |
| W-08 | P2 | CRAFT DynamicPosBias is dead: `pos_dim = 1`, LayerNorm(1) makes it a per-head constant; 416 params get exactly zero grad, CRAFT has no positional bias (also upstream/spandrel) | learning | reproduced | variant only |
| W-09 | P2 | RGT recursion depth differs between training at LQ 32 (`_time=1`) and evaluation (`_time>=2`) | learning | source-verified; quality impact hypothesis | yes (A/B) |
| W-10 | P2 | Tiny inputs crash: SRFormer <= 8 px, ATD-light < 8 px, CRAFT < 8 px, RGT any side < 16 px in eval (also 12x12, 17x9) | reliability | reproduced | none for working sizes |
| W-11 | P2 | Checkpointing coverage: ATD (heaviest default) has none; HAT/DRCT checkpoint whole groups (coarse peak); SwinIR/SwinIR-dense checkpoint also in eval | memory | source-verified | none |
| W-12 | P3 | Template patch 32 vs large windows: HiT-SRF pads to 48/64 px in 2 of 6 blocks per layer (2.25x/4x tokens), srformer_medium pads 32 -> 48 | tuning | source-verified | config only |
| W-13 | P3 | DAT/RGT run softmax in bf16 on bf16 logits+bias under autocast (`softmax(..., dtype=attn.dtype)`) | numerics | source-verified | W-01 improves |
| W-14 | P3 | F08 (partial): converter's 20x20 sample exports SwinIR/HAT dynamically correctly; export at `img_size` or a window multiple freezes branches and fails | export | reproduced both ways | none |
| W-15 | P3 | `self.mean = self.mean.type_as(x)` mutates a plain attribute in forward (SRFormer, ATD, CRAFT, RGT) | compile/hygiene | source-verified | none |
| W-16 | P3 | Persistent `attn_mask` buffers sized by `img_size` are never used at patch 32 but live on GPU twice (net + EMA) and in every checkpoint (drct 48 MiB, drct_xl 112 MiB per copy) | memory | source-verified | none (keep keys) |

Priority reasoning: W-01 is the largest single lever for the owner's priorities (memory and, very likely, speed of
every window transformer) and is exact, so it is P1 despite needing a CUDA confirmation. W-03 and W-04 are exact,
small and remove pathological costs. W-02, W-07 and W-10 are bugs that make advertised switches or inputs fail.
W-06/W-08/W-09 change what is trained and need an A/B before they become defaults. Nothing here proposes damping,
loss normalisation or adaptive clipping: all stability comes from exact math and correct gradients.

## 2. Codex items: verdicts

| Codex item | Verdict | Evidence |
|---|---|---|
| F04 SwinIR / SwinIR-dense flash path omits bias and shift mask | **confirmed** | p1: `flash_attn=True` leaves all 24 (swinir_small) / 48 (swinirdense) `relative_position_bias_table` grads None; p2: whole-net eval output diff 4.2e-3 / 6.1e-3 (swinir_small 32/40 px) and 4.9e-2 (swinirdense) already at init, where the tables are only N(0, 0.02) |
| F04 CRAFT flash branch omits learned dynamic position bias | **partial (refuted in substance)** | CRAFT's DPB output is constant over positions (p2: spread 0.0), so softmax is invariant and the flash branch is numerically equal to manual (p2: 1.3e-6 / 1.5e-6 whole-net). The remaining defect is 448 params with grad None under flash (p1), which breaks DDP with `find_unused_parameters=false` (base.py:153 default) |
| F05 SRFormer checkpointing crash | **confirmed** | p5: `TypeError: PSA_Block.forward() missing 1 required positional argument: 'x_size'`; fixed call `checkpoint(blk, x, x_size, use_reentrant=False)` gives grads equal to the plain path within 4.6e-13 under `backward(inputs=params)` (the trainer's call, phase.py:86) |
| F08 arch side: Python shape conversions / pad branches freeze dynamic export | **partial** | p12: with convert.py's default 20x20 sample, dynamic ONNX of `swinir_small` and `hat_s` matches PyTorch within 1.9e-6 at 20x20, 24x24, 32x24, 64x64, 37x29. p12b: exporting at 64x64 (= `img_size`, window-aligned) freezes the buffer branch (swinir_arch.py:367) and the `if ph or pw` pad branch (:1044) and ORT fails at 32x24 and 37x29 ("Reshape ... requested shape {1,4,8,3,8,60}"). So the defect is real but only for aligned/`img_size` samples |
| F09 HiT-SRF quadratic position bias | **confirmed**, fix verified | p6: exact pooled replacement (max bias diff 5.4e-7, grad rel 3.0e-6); default `hit_srf` window 64: dense index 128 MiB saved for backward per block -> 2 MiB; medium window 128: 2 GiB index + 4 GiB coords + 3 GiB gathered bias per block -> 32 MiB |
| Tiny-input rows (SRFormer 1x1, ATD-light) | **confirmed and widened** | p8: SRFormer fails at 1x1, 5x7, 7x5 (reflect pad >= size); ATD-light at 1x1, 5x7, 7x5 (one flip-cat insufficient); additionally CRAFT (same flip-cat) and RGT (`math domain error` for any side < 16 in eval, incl. 12x12 and 17x9). SwinIR/SwinIR-dense/HAT/DRCT/DAT/HiT pass all sizes |
| Mask caching guidance ("bounded, never learned outputs") | **agree, refined** | the cheapest exact win is not a cache but (a) no mask at all for unshifted blocks and (b) one mask per forward instead of one per block (W-05); a 1-entry memo is enough and bounded; validation-size masks must not persist |

## 3. Gradient census on the default config (p1)

`build_network({"type": name}, scale=4).train()`, batch 2, LQ 32x32 (craft 64x64), `y.mean().backward(inputs=params)`:

| Row | grad None | notes |
|---|---|---|
| swinir_small, swinir_medium, swinirdense, hat_s, hat_m, drct, drct_s, dat_s, dat_m, rgt, rgt_s, atd, atd_light, srformer_light, srformer_medium, hit_srf | 0 | a few exact-zero grads in single blocks are DropPath drops (both samples dropped) or dead ReLUs in HAT's 6-channel CAB squeeze at init: random, not structural |
| craft (manual) | 0 | **416 params exactly zero**: every `attns.*.pos.*` (W-08) |
| swinir_small `flash_attn=True` | **24** | all `relative_position_bias_table` (F04) |
| swinirdense `flash_attn=True` | **48** | all `relative_position_bias_table` (F04) |
| craft `flash_attn=True` | **448** | all `pos.*` (W-02) |

`tests/test_archs_a2.py::test_forward_backward` already asserts `grad is not None` for the default rows (CUDA only);
`test_flash_attn_trains_qkv` checks only QKV grads, which is why F04 survived.

## 4. Findings

### W-01 (P1) Exact SDPA path for window attention with additive bias and shift mask

**Where.** Manual cores: swinir_arch.py:181-208, swinirdense_arch.py:148-176, hat_arch.py:185-210 (WA) and
:487-501 (OCAB), drct_arch.py:199-225, atd_arch.py:179-202 (WA) and :344-356 (AC_MSA), srformer_arch.py:242-269,
dat_arch.py:263-291, rgt_arch.py:227-255, craft_arch.py:190-218.

**Mechanism (measured).** Under bf16 autocast the manual path computes `attn = bf16(q@k^T)`, adds the fp32 bias
table gather (type promotion to fp32), adds the mask, runs `softmax` in fp32 and casts the probabilities to bf16
for `@v`. Autograd keeps the fp32 softmax output (softmax backward) **and** its bf16 copy (matmul backward). p10 for a
shifted HAT_m block at the template geometry (32 windows x 6 heads x 256 x 256 = 12.6 M elements): saved
`(32,6,256,256) fp32 48 MiB` + `(192,256,256) bf16 24 MiB` = **6 B per attention element**, 80 % of everything the
block saves. Scaled to the template step (b8, LQ 32):

| Default row | attention calls / fwd | elements / call | saved attention probabilities / step |
|---|---|---|---|
| swinir_small (N=64, 6 heads, 24 blocks) | 24 | 3.1 M | ~0.42 GiB |
| swinir_medium (36 blocks) | 36 | 3.1 M | ~0.63 GiB |
| hat_m / hat_s (36 HAB N=256 + 6 OCAB 256x576) | 42 | 12.6 M / 28.3 M | ~3.5 GiB |
| drct (30 blocks, 22 heads per RDG) | 30 | 9.2 M avg | ~1.55 GiB |
| atd (36 WA + 36 AC_MSA groups of 256) | 72 | 12.6 M | ~5.1 GiB |
| craft (16 blocks x 2 branches, N=64) | 32 | 6.3 M | ~1.1 GiB |
| srformer_light (24 PSA, 256x64) | 24 | 3.1 M | ~0.42 GiB |
| dat_m (18 spatial blocks; bf16 softmax, 2 B/elem) | 36 | 6.3 M | ~0.42 GiB |

The CUDA memory-efficient kernel stores only q, k, v, out and a log-sum-exp per row (O(N*d)), so these bytes
disappear; it also fuses the two N^2 passes (add bias/mask, softmax) that dominate the memory traffic of these small
head dims. Flash attention cannot take a float mask; the mem-efficient kernel can, including a gradient for the mask
(the learned bias). On CPU the SDPA path falls back to the math kernel when the mask requires grad, so CPU cannot
show the saving (p3: fp32 saved bytes 850 -> 931 MiB, i.e. not representative); the speed and peak-memory effect on
the 4090 is a **hypothesis** to be measured with `bench_window_sdpa_cuda.py`.

**Kernel eligibility caveat.** For fp16/bf16 the mem-efficient kernel needs the head dim to be a multiple of 8
(`minimum_gemm_alignment` in PyTorch's sdp_utils, to be confirmed by `can_use_efficient_attention(..., debug=True)`
in the benchmark). Default head dims: swinir_small 10, swinir_medium/large 30, hat_m/l 30, hat_s 24, drct
30/53/122/46/77, atd 35, dat 30, srformer_light 10, craft 8. Without padding SDPA would silently take the math
backend, which upcasts bf16 to fp32 (`allow_fp16_bf16_reduction_math_sdp` off) and is **slower** than the manual
path. So the helper zero-pads the head dim to a multiple of 8 with an explicit `scale` (q.k is unchanged, padded
output columns are zero and sliced off): overhead 4-14 % FLOPs on the two attention matmuls only (60 % for d=10).

**Fix (exact).** One helper in `arch_util.py`, used by all call sites; the learned bias is computed exactly as
today and passed as the float mask; the shift mask is folded into it once per call and kept at
`[1, nW*nH, N, M]` (stride-0 over the batch), never repeated `B` times:

```python
def window_sdpa(q, k, v, bias, mask=None, *, scale, dropout_p=0.0):
    """softmax(q k^T * scale + bias [+ mask]) v. q: [B*nW, nH, N, d]; k, v: [B*nW, nH, M, d];
    bias: [nH, N, M] (learned, keeps its grad); mask: [nW, N, M] or None (window index is the fast batch index)."""
    b_, nh, n, d = q.shape
    nw = 1 if mask is None else mask.shape[0]
    pad = -d % 8  # mem-efficient kernel: head_dim % 8 == 0 for 16-bit
    q, k, v = (F.pad(t, (0, pad)) if pad else t.contiguous() for t in (q, k, v))
    q, k, v = (t.view(b_ // nw, nw * nh, t.shape[2], d + pad) for t in (q, k, v))
    ab = bias.unsqueeze(0) if mask is None else bias.unsqueeze(0) + mask.unsqueeze(1).to(bias.dtype)
    ab = ab.reshape(1, nw * nh, n, -1).to(q.dtype)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=ab, dropout_p=dropout_p, scale=scale)
    return out.view(b_, nh, n, d + pad)[..., :d]
```

Call-site specifics: HAT OCAB passes `mask=None` and `M = 576`; SRFormer PSA passes its `[nH, N, N/4]` aligned
bias; DAT/RGT/CRAFT pass `pos[rpi].view(N, N, -1).permute(2, 0, 1)`; ATD AC_MSA has no bias: use
`SDPA((q * logit_scale).flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), scale=1.0)` (flash-eligible after the
kernel's own head-dim padding; `logit_scale` keeps its gradient through the multiplication). Keep `-100` as the mask
value (exact equivalence with trained checkpoints); dropout as `dropout_p if self.training else 0.0`.

**Parity (measured, fp32 CPU, train mode, bias tables set to N(0,1) so the bias matters).** Whole networks
(p3/p3b): swinir_small 32 px max |dy| 1.3e-6, worst relative grad diff 8.9e-7; swinir_small 40 px (shift mask built
in forward) 1.2e-6 / 1.0e-6; swinirdense 3.0e-7 / 2.3e-6; swinir_medium 3.6e-7 / 1.9e-6; hat_s 3.9e-7 / 1.9e-6; all
bias tables receive grads. Modules (p11): SRFormer PSA with shift mask rel 2.7e-7 (dx 1.6e-7, table 3.8e-7);
DAT Spatial_Attention with mask 3.0e-7 (dpos 3.6e-7); ATD AC_MSA rel 2.0e-5 (logits ~60 after `logit_scale=10`;
fp32 rounding). Broadcast and repeated-mask variants are bit-identical (0.0).

**Numerics / compatibility.** State dict unchanged (no new parameters or buffers). Under bf16 the bias is rounded
to bf16 inside SDPA while the logits are accumulated in fp32 (the manual path rounds the logits to bf16 instead), so
results differ at rounding level: short A/B (e.g. 5-10 k iterations of swinir_small and hat_s, loss and val PSNR
curves) before making it the only path. With `manual_seed` (deterministic mode, runtime.py:94 `warn_only=True`) the
mem-efficient backward may warn about non-determinism.

**Tests.** For each family: one test that builds the network twice (manual reference implemented in the test, or the
old forward kept as a private `_attn_reference`), sets bias tables to N(0,1), compares outputs and **all** parameter
grads (`rtol 1e-4`, `atol 1e-6` fp32 CPU) at an aligned and a non-aligned size; on CUDA additionally bf16 within
`2e-2` and `torch.backends.cuda.can_use_efficient_attention` true for the default head dims.

### W-02 (P1) F04: the `flash_attn` switches

**SwinIR / SwinIR-dense** (swinir_arch.py:163-179, swinirdense_arch.py:135-146): the SDPA branch has no bias and no
shift mask: a different model (p1, p2). **Fix:** make `flash_attn=True` call the W-01 helper (or make it a no-op
alias once W-01 is the default path); keep the kwarg so existing configs load. Extend `test_flash_attn_trains_qkv`
to assert grads on `relative_position_bias_table` and output parity with the manual path.

**CRAFT** (craft_arch.py:183-189): numerically equivalent because its bias is constant (W-08), but the 448 dead
`pos.*` parameters get grad None and a DDP run with `find_unused_parameters=false` fails. Two exact options: pass
the constant bias as `attn_mask` (keeps params in the graph; mem-efficient kernel, d=8 is aligned), or freeze the
dead DPB (`requires_grad_(False)`, state dict unchanged) and use bias-free SDPA (flash-eligible). The second is
cheaper; it is output-exact because those params only affect a softmax-invariant constant.

### W-03 (P1) F09: HiT-SRF position bias

**Where.** hitsrf_arch.py:288-329 (`SCC.spatial_self_correlation`), called by every block every forward.

**Mechanism (reproduced).** Per call it rebuilds an `[N,N,2]` int64 coordinate tensor, an `[N,N]` int64 index,
gathers `[N*N, nH]` and copies it once more (permute+contiguous) before averaging over the `rh x rw` keys of each
base cell. For default `hit_srf` (base 8, windows 4/8/16/32/48/64) the window-64 block has N = 4096: 256 MiB
coordinates (+256 MiB during the permute copy), 128 MiB index **kept for backward**, 192+192 MiB bf16 gathered bias;
the backward then scatter-adds 100 M values into a 127x127x6 table. Per forward, the saved indices alone are ~0.7
GiB (4 x 128 + 4 x 40.5 + 4 x 8 MiB). For `hit_srf_medium` (window 128): 4 GiB coordinates, 2 GiB index, 3 GiB bias
per block, which is why tests skip it (test_archs.py:34-37).

**Exact fix (measured).** With `P = pos` viewed `[nH, 2H-1, 2W-1]`, `rh = H/bh`, `rw = W/bw`:

`bias[h, n, (i,j)] = mean_{a<rh, c<rw} P[h, nh - i*rh - a + H-1, nw - j*rw - c + W-1]
                   = avg_pool2d(P, (rh, rw), stride=1)[h, nh - i*rh + H - rh, nw - j*rw + W - rw]`

```python
# __init__ (graph-independent, non-persistent: state dict unchanged)
self.register_buffer("rpe_grid", <current rpe_biases>, persistent=False)        # [(2H-1)(2W-1), 2] float
self.register_buffer("pool_idx", pooled_index(window_size, base_win_size), persistent=False)  # [N, bh*bw] int64
# forward
pos = self.pos(self.rpe_grid)                                                     # [(2H-1)(2W-1), nH]
p = pos.t().reshape(1, self.num_heads, 2 * H - 1, 2 * W - 1)
bias = F.avg_pool2d(p, (H // bh, W // bw), stride=1).flatten(2)[0][:, self.pool_idx]  # [nH, N, bh*bw]
```

with `pooled_index` = `(nh - i*rh + H - rh) * (2W - rw) + (nw - j*rw + W - rw)` over the `(H, W, bh, bw)` grid
(p6 has the 8-line function). p6 against the verbatim current code: windows 4/8/16/48/64 max |dbias| <= 5.4e-7, max
relative grad diff of the DPB MLP <= 3.0e-6. Window 64: saved index 128 MiB -> 2 MiB, CPU fwd+bwd of the bias
1034 ms -> 394 ms (contended machine, indicative); window 128 (medium): pooled 32 MiB instead of 2+4+3 GiB. Works
because window sizes are fixed per block at construction. Note the base window is clamped
(`min(window, base)`, hitsrf_arch.py:247-249), so `rh = 1` reduces exactly to the standard bias.

**Numerics/compat.** Summation order only; no state-dict change. **Test:** the p6 comparison as a unit test on
windows (4, 8, 16, 48) plus a `hit_srf_medium` forward/backward at 32x32 on CUDA that currently cannot run.

### W-04 (P1) Blocking host->device copies inside forward

`tensor.to(cuda)` from pageable CPU memory with `non_blocking=False` is a `cudaMemcpyAsync` followed by a
`cudaStreamSynchronize`: the CPU waits for the whole queue, then the GPU idles until new kernels arrive, and the
CPU can never run ahead of the GPU again in that forward. Counted per training forward at the template size (p7):

| Arch | Source | Syncs / fwd | Exact fix |
|---|---|---|---|
| atd | `index_reverse`: `torch.arange(n).to(index.device)` + a Python loop of `b` `index_put_` (atd_arch.py:14-19), once per layer | 36 (+288 small kernels) | `torch.empty_like(index).scatter_(-1, index, torch.arange(n, device=index.device).expand_as(index))` (p11: bit-exact) |
| atd | `torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01)).to(qkv.device))` (atd_arch.py:347-349) | 36 | `max=math.log(100.0)` (p11: identical fp32 value) |
| atd | `self.calculate_mask([h, w]).to(x.device)` (atd_arch.py:1126, 1173) | 1 | build on `x.device` (as HAT does) |
| srformer_light | `self.calculate_mask(x_size).to(x.device)` for **every** block (srformer_arch.py:383, 406, 470) | 24 | build on device, once per forward (W-05) |
| rgt | `calculate_mask(_H, _W)` on CPU, then `mask_tmp[0].to(x.device)`, `mask_tmp[1].to(x.device)` (rgt_arch.py:341-342, 465-467) | 24 | build on device, once per forward |

`atd_light` has 49 per forward. spandrel's ATD/RGT/SwinIR/HAT/DAT (competitor) have the same CPU-built masks and
the same `index_reverse`; nothing to port there. **Impact:** hypothesis 1-5 % of step time for ATD/RGT (more for
the launch-bound light variants), to be confirmed by counting `cudaStreamSynchronize` in a torch.profiler trace
(profiling lane). **Numerics:** none. **Test:** the existing forward/backward tests plus a CPU test that
`index_reverse` equals the scatter form on random permutations.

### W-05 (P2) Shift masks rebuilt per block, including all-zero masks for unshifted blocks

**Where.** swinir_arch.py:366-373, swinirdense_arch.py:313-317, drct_arch.py:375-379, srformer_arch.py:464-471:
the size-mismatch branch calls `calculate_mask` for **every** block, also when `shift_size == 0`; with shift 0 the
slices degenerate and the result is an all-zero `[nW, N, N]` mask (p9: verified for SwinIR and SRFormer) that is then
added to the full attention tensor. DAT (dat_arch.py:525-528) and RGT (rgt_arch.py:461-466) rebuild for every
shifted block. Per training forward (p7): swinir_small 24 builds (12 all-zero), drct 30 (18 all-zero), srformer_light
24 (12 all-zero, each also a sync, W-04), dat_m 9, rgt 12. One build is 49 aten dispatches (~16 CUDA kernels):
**1176 of swinir_small's 3080 forward dispatches (38 %)** and 1470 of drct's 4002 (37 %) are mask construction; the
light models are launch-bound at batch 8 / LQ 32, so this is directly visible in step time (hypothesis; profiler).

**Fix (exact, minimal).**
1. In the mismatch branch pass `mask=None` when `self.shift_size == 0` (as the aligned branch already does via the
   `None` buffer).
2. One mask per forward: a module-level helper `shift_mask(h, w, ws, shift, device, dtype)` in `arch_util.py` with a
   **one-entry** memo keyed by its arguments (the mask is a graph-free constant), bypassed when
   `torch.compiler.is_compiling()` (memo lookups would graph-break; inductor folds the build anyway). All blocks of a
   model share the entry; training at a fixed patch hits it every step; a validation size replaces it and the next
   training step replaces it back, so at most one mask is retained (worst case one validation-size mask until the
   next step, e.g. 64 MiB for SwinIR at 512 px LQ). HAT/ATD already hoist one mask per forward; the alternative
   for the SwinIR family is to hoist the same way (pass the mask down `BasicLayer -> block`), which avoids any memo
   but touches four call chains.
3. Keep the persistent `attn_mask` buffers for state-dict compatibility (W-16) and keep using them when
   `x_size == input_resolution`.

Do not cache learned tensors (bias, DPB outputs) and do not key on unbounded size sets. **Test:** output equality
before/after at aligned, non-aligned and `img_size` inputs; a counter test that a forward builds at most one mask.

### W-06 (P2) CRAFT always pads one extra window

craft_arch.py:936-942: `h_pad = (h_old // ws + 1) * ws - h_old` is 16 (not 0) for aligned inputs, so the template
patch 64 runs the whole body at 80x80 (6400 instead of 4096 tokens, +56 %). Inherited from upstream (spandrel CRAFT.py
:910 is identical). p13 (CPU, b8, fwd+bwd): 30.75 s -> 23.43 s (-24 %) with `h_pad = (-h_old) % ws`.
**Numerics:** the right/bottom border windows no longer see one window of mirrored context; train/inference stay
consistent with each other but differ from existing CRAFT checkpoints at the borders. Needs an A/B (val PSNR) and a
note for old checkpoints; for strict backward-compatible inference keep the old formula behind a kwarg
(`pad_extra_window=True` default until the A/B). **Test:** shape tests at aligned/non-aligned sizes; token count.

### W-07 (P2) F05: SRFormer checkpointing

srformer_arch.py:644 `checkpoint.checkpoint(blk, x, use_reentrant=False)` drops `x_size`. Fix:
`if self.use_checkpoint and self.training: x = checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)`
(also adds the `self.training` guard used by HAT/DRCT). p5: grads match the plain path within 4.6e-13 with
`backward(inputs=params)`. Non-reentrant is required for `backward(inputs=...)` and preserves the DropPath RNG.
**Test:** add `srformer_light` to `test_checkpoint_grad_parity` (test_archs_a2.py:118) and run it on CPU too.

### W-08 (P2) CRAFT's dynamic position bias is dead

craft_arch.py:153 builds `DynamicPosBias(self.dim // 4, ...)` with `self.dim = 24` (embed 48 split over two
branches), so `pos_dim = 24 // 4 // 4 = 1`. `LayerNorm(1)` maps every input to its bias, so the MLP output is a
per-head constant (p2: spread 0.0 over all 217 positions), softmax ignores it, and all 416 `pos.*` parameters get
exactly zero gradient (p1). CRAFT therefore trains without any positional bias, in neosr, upstream and spandrel.
DAT/RGT (`pos_dim` 5) and HiT-SRF (3) are fine.
**Options.** (a) Now, exact: skip the dead computation (no MLP, gather and N^2 bias add per attention) and freeze
the params (state dict unchanged; outputs identical up to rounding). (b) Later, a learning improvement: a variant
with `pos_dim = max(dim // 4, 4)` or a plain relative-position table: new parameter shapes, so a new ARCHS row or
kwarg; old checkpoints still load with `strict_load_g=false` (load_weights drops the mismatched `pos.*` tensors).
Needs an A/B; expected to help, since every other family in this lane benefits from a positional bias.

### W-09 (P2) RGT recursion depth: training at LQ 32 differs from evaluation

rgt_arch.py:573-579: training uses `_time = int(log4(H // 4))`, evaluation `max(int(log4(H // 16)), 2)`. At the
template patch 32 (also 48) training applies the shared `reduction1` conv once, evaluation twice (p8 table: H=32
train 1 / eval 2; H=64 2 / 2). The comment "testing _time must equal or larger than training _time (2)" assumes the
paper's 64 px patches. So every RGT trained from the template is validated and deployed with a k/v reduction depth
it never saw. Quality impact is a **hypothesis** (the RGM is meant to generalise across depth, but here it was
trained at depth 1 only). Options: template `patch_size = 64` for `train_rgt*.toml` (paper setting, 4x compute per
sample), or an explicit `train_time` kwarg (default 2, current behaviour) used as the evaluation floor. Measure first:
validate one template-trained checkpoint with eval `_time` 1 vs 2.

### W-10 (P2) Tiny / small inputs

p8 (eval, x4): SRFormer reflect pad needs pad < size (srformer_arch.py:1171) and fails at 1x1, 5x7, 7x5 (any side
<= 8 with window 16); ATD (atd_arch.py:1165-1166) and CRAFT (:941-942) mirror only once and fail when the input is
smaller than half a window; RGT fails with `math domain error` whenever a side is < 16 in eval, also for 12x12 and
17x9 (rgt_arch.py:576 `math.log(H // 16, 4)` and the view at :624). Validation images, tiles and convert.py samples can
hit these. **Fix (exact for currently working sizes):** SRFormer: the SwinIR pattern
(`mode = "reflect" if ph < h and pw < w else "replicate"`, swinir_arch.py:1045); ATD/CRAFT: repeat the flip-cat until
large enough (`while x.shape[2] < h: x = torch.cat([x, x.flip(2)], 2)`), identical to today when one step suffices;
RGT: pad the input to >= 16 per side (replicate) in `forward` and crop. **Test:** extend `test_tiny_input_and_half_masks`
to all ten families on CPU.

### W-11 (P2) Gradient-checkpointing coverage

| Family | Current | Recommendation |
|---|---|---|
| swinir, swinirdense | per block, non-reentrant (swinir_arch.py:540-541) but also in eval | add `and self.training` |
| hat | per RHAG (hat_arch.py:1126-1127): peak = one group of 6 HAB + OCAB | per HAB + OCAB (competitor traiNNer/archs/hat_arch.py:611-640, `use_reentrant=False`): ~6x lower recompute peak, same extra compute |
| drct | per RDG (drct_arch.py:910-911) | per swin block inside RDG (competitor drct_arch.py:356-395) |
| dat, rgt, hit_srf | per block | fine |
| srformer | broken | W-07 |
| atd | **none** (heaviest default, ~5.1 GiB attention probs alone) | per `ATDTransformerLayer` (`checkpoint(layer, x, td, x_size, params, use_reentrant=False)`; tuple outputs are supported) |
| craft | none | not needed (0.75 M params) |

All checkpoint calls must stay non-reentrant: the trainer's `backward(inputs=params)` is not supported by the
reentrant variant. With W-01, attention probabilities no longer dominate, so checkpointing becomes an opt-in for
hat_l/drct_xl/atd at larger patches rather than a necessity. **Test:** extend `test_checkpoint_grad_parity` to
atd_light and per-block HAT/DRCT.

### W-12 (P3) Template patch 32 vs large windows (tuning)

`hit_srf` windows are `8 x [0.5, 1, 2, 4, 6, 8]`; at LQ 32 the 48/64 windows pad the 32x32 features to 48x48/64x64
(hitsrf_arch.py:470-485): 2.25x and 4x tokens in 2 of 6 blocks per layer, filled with reflections; the upstream
HiT-SR trains at LQ 64. `srformer_medium` (window 24) pads 32 -> 48 (2.25x). Recommend `patch_size = 64` in
`train_hitsrf*.toml` (with W-03 the memory allows it) and documenting a multiple of the largest window for
srformer_medium. Config-only; changes what is trained (A/B against the current template).

### W-13 (P3) DAT/RGT softmax precision under bf16

dat_arch.py:288 and rgt_arch.py:252 use `softmax(attn, dim=-1, dtype=attn.dtype)`; under autocast `attn` is bf16
(bf16 matmul + bf16 DPB output + `mask.to(attn.dtype)`), and an explicit dtype makes autocast keep the softmax in
bf16, unlike the other families (fp32 via promotion). W-01 removes the issue (fp32 accumulation inside the kernel).
If W-01 is not adopted, dropping the `dtype=` argument restores the fp32 softmax (numerics change, A/B).

### W-14 (P3) F08 export, arch side

Confirmed mechanism, narrower than reported (section 2). Exact, cheap arch-side hardening: in the SwinIR family use
the `calculate_mask` path when `torch.jit.is_tracing()` (never the frozen buffer branch), always call `F.pad` (zero
pad is a no-op) instead of `if ph or pw`, and write `window_reverse` as `windows.view(-1, h // ws, w // ws, ws, ws, c)`
instead of `int(windows.shape[0] / (h * w / ws / ws))` (swinir_arch.py:70, atd_arch.py:121, hitsrf_arch.py:168).
Converter side (not this lane): reject window-aligned or `img_size` samples for dynamic export and verify at two or
more shapes.

### W-15 (P3) Plain-attribute `mean` mutated in forward

srformer_arch.py:1177, atd_arch.py:1170, craft_arch.py:944, rgt_arch.py:942: `self.mean = self.mean.type_as(x)`.
It is not moved by `.to()`, the first forward does an H2D copy, and the attribute mutation is a side effect for
`torch.compile` guards. Use `register_buffer("mean", ..., persistent=False)` and a local `mean = self.mean.to(x)` as
SwinIR/HAT/DRCT already do (state dict unchanged: the attribute was never in it). ATD's mean is only used with
`norm=True`.

### W-16 (P3) Unused persistent mask buffers

SwinIR-style blocks register `attn_mask` for `img_size` (swinir_arch.py:306-311, drct_arch.py:309-313,
srformer_arch.py:366-370, dat_arch.py:371-379, rgt_arch.py:322-332). At the template patch they are never used, but
they are on the GPU for net_g and the EMA copy and in every `net_g` file: drct 12 x 4 MiB = 48 MiB per copy, drct_xl
112 MiB, swinir_small/srformer_light 12 MiB. They also make a strict load fail when `img_size` differs between
checkpoint and config. Keep them (old `net_*.pth` must load); optional later: a `_register_load_state_dict_pre_hook`
that drops `*.attn_mask*` keys plus `persistent=False`, which keeps old files loading and new files smaller.

## 5. Competitor comparison (same families)

- traiNNer's SwinIR, ATD, RGT, CRAFT, DAT come from spandrel and SRFormer from spandrel_extra_arches: same equations,
  but masks are built on CPU and copied (`calculate_mask(x_size).to(x.device)` in SwinIR.py:351, HAT.py:1104,
  DAT.py:519-521, RGT rgt.py:462-466, ATD atd_arch.py:1170) and `index_reverse` / `torch.tensor(...).to()` in ATD are
  unchanged. neosr's on-device masks (SwinIR, HAT, DRCT, DAT) are already better; nothing to port.
- traiNNer's own `hat_arch.py` / `drct_arch.py`: per-block non-reentrant checkpointing (worth porting, W-11), and a
  `calculate_mask(..., dtype)` that neosr already covers with `.to(attn.dtype)`.
- traiNNer's `hit_srf_arch.py:281-326` has the same dense bias (W-03); spandrel CRAFT has the same extra-window pad
  (W-06) and the dead DPB (W-08). No SDPA in any of these families. No damping-style mechanism is involved here.

## 6. Suggested order of work

1. W-07, W-04, W-05(1), W-10, W-15: exact, small, no A/B. Tests: grad parity, tiny sizes, scatter equality.
2. W-03 (HiT-SRF pooled bias): exact; unlocks hit_srf_medium/large; unit test against the dense reference.
3. W-01 + W-02: helper + call sites, family by family, each with a whole-net output+grad parity test; profiling lane
   runs `bench_window_sdpa_cuda.py` and a full-step trace (time, peak memory, `can_use_efficient_attention`), then a
   short bf16 A/B before removing the manual path.
4. W-05(2), W-11, W-16: memo/hoist, finer checkpointing.
5. A/B items: W-06 (CRAFT padding), W-08(b) (CRAFT positional bias variant), W-09 (RGT depth), W-12 (patch 64 for
   HiT-SRF), W-13 if W-01 is not adopted.
