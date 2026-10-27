# Stage 2 analysis: mixed, channel-attention and MoE architectures

Lane scope: `neosr/archs/{hma,mfghmoe,asid,ditn,grformer,catanet,dct,dctlsa,lmlt,omnisr,esc,microsr,cfsr,man,safmn,flexnet}_arch.py`,
`neosr/archs/arch_util.py`, the `ARCHS` table, `tests/test_archs*.py`, and `traiNNer/archs/` in the competitor.
Code at HEAD `f360edd` (refactor/stabilize). Every probe ran on CPU (`CUDA_VISIBLE_DEVICES=`); scripts and raw outputs are in
`/tmp/claude-1000/-home-korvin-apps-neosr-dev/c1b40b02-f454-4498-9d89-be0afa4e2973/scratchpad/archs_mixed/` (names are given below).
All shipped templates (`options/train_<arch>{,_otf}.toml`) use the ARCHS defaults with `use_amp = true`, `bfloat16 = true`,
batch 8, LR patch 64 (32 for hma, mfghmoe, grformer, dct, microsr), so the **default constructor path is what matters**.

Verification labels: **reproduced** (probe shows the defect), **measured** (numbers from a probe), **source-verified**
(read, not run), **hypothesis** (GPU effect not measurable here: the GPU is reserved for the profiling lane).

## 1. Summary and priority order

| # | Pri | Finding | Arch(es) | Kind | Changes a trained checkpoint's output? |
|---|---|---|---|---|---|
| M-01 | P1 | 64 MiB CPU-resident position index per attention module: host copies + 2 pageable H2D uploads + syncs per module per step, 19 s build | esc (all variants; template default) | speed | no (bitwise identical bias) |
| M-02 | P1 | MoE router and the whole multi-level-feature branch get no gradient; dense 16-expert stack (1.4 GiB transient at template batch); 10 host syncs; bf16 flips 4.7 % of routes | mfghmoe | learning, memory, speed | no (straight-through gate keeps the forward) |
| M-03 | P1 | Default (flash) attention runs under `no_grad`, ignores temperature, grid variant's scale depends on input resolution, 5-D SDPA never fused | asid, asid_d8 | learning | yes (needs the legacy remap below) |
| M-04 | P1 | No padding: validation crashes on any non-aligned image (67x53; MFGHMoE also at the template's own `tile = 200`) | hma, mfghmoe, microsr | reliability | no |
| M-05 | P1 | Temperature dead, attention computed and discarded, 3-D SDPA (math only); neosr-trained weights differ from spandrel/chaiNNer DITN | ditn | learning, export | yes (legacy remap) |
| M-06 | P2 | 30 all-zero shift masks built on CPU and uploaded (30 syncs) every forward | microsr, microsr_light | speed | no |
| M-07 | P2 | `use_checkpoint` silently ignored; per-layer checkpoint cuts saved activations 444->16 MiB (hma), 1603->65 MiB (mfghmoe) | hma, mfghmoe | memory | no |
| M-08 | P2 | Manual window attention keeps fp32 softmax probabilities = 22-37 % of saved activations | mfghmoe, grformer, hma (+ dct, microsr, dctlsa, lmlt small) | memory | no (parity test) |
| M-09 | P2 | DynamicPosBias with `pos_dim = 1` is a per-head constant: dead parameters, wasted (at val: ~0.8 GiB) work; no relative-position information at all | hma (default), dct (default) | speed, memory (+arch quality opt-in) | no for the skip |
| M-10 | P2 | Shift masks built on CPU per forward (H2D sync): hma 1, mfghmoe 1, dct 10 at template patch | hma, mfghmoe, dct | speed | no |
| M-11 | P2 | `if not self.initted` on a device bool: 8 (eval) / 16 (train) host syncs per forward, 14 graph breaks | catanet | speed, compile | no (bitwise) |
| M-12 | P2 | `ctx.saved_variables` in LayerNorm2d backward: 43 dynamo graph breaks | asid | compile | no (bitwise) |
| M-13 | P2 | DySample builds `torch.tensor([W, H], device=...)` each forward: one H2D sync; also the F07 trace constant | esc (default `use_dysample`), any DySample user | speed, export | no (bitwise) |
| M-14 | P2 | SDPA called with 3-D/5-D tensors: CUDA never dispatches a fused kernel (math backend) | flexnet (default flash), ditn, asid | speed, memory | no |
| M-15 | P2 | Default MicroSR runs 2x2 windows without shift (patch_size 32 hack, commit 75557f4) instead of the advertised 16x16/8x8 shifted windows | microsr, microsr_light | arch quality | n/a (shape-bound; owner decision) |
| M-16 | P2 | Arch tests drop `None` gradients and never check host syncs or awkward sizes, so F02/F03/F11 and M-01/M-04 pass CI | tests/test_archs*.py | tooling | - |
| M-17 | P3 | Tiny-input limits: reflect padding fails for sides <= 32 (lmlt), <= 17 (esc, grformer), ASID < 9 (ESA pool) | lmlt, esc, grformer, asid, catanet, safmn | reliability | no |
| M-18 | P3 | Hygiene: OmniSR `AdaConvGuidedFilter` NameError (dead), ASID `Dropsample` legacy-constructor bug (unreachable), ASID `block_num > 1` crashes, MFGHMoE NaN print/zeroing | omnisr, asid, mfghmoe | hygiene | no |
| M-19 | P3 | Optional MoE load-balance auxiliary term and exact sparse expert dispatch | mfghmoe | learning/speed opt-in | numerics change |

Recommended implementation order (independent files, can run in parallel lanes): M-01, M-02+M-04+M-07+M-10 (mfghmoe/hma
together), M-03+M-12+M-14 (asid), M-05+M-14 (ditn), M-06 (microsr), M-11, M-13, M-14 (flexnet), M-09, M-16 (tests with
every fix), then M-08 (needs GPU parity/benchmark), M-15 (owner decision), M-17..M-19.

## 2. Codex item verdicts

| Item | Verdict | Evidence (this lane) |
|---|---|---|
| F02 ASID attention under no_grad, temperature ignored | **confirmed, worse than stated** | Full default net: 30 parameter tensors with `grad None` (BlockC qkv, qkv_dwconv, temperature **and canorm1/2**) in all 3 IDSG groups, ~16 % of the 312 k parameters; `x.grad None` through the unit. Flash vs manual differ by 0.040 max (2.5x the output std). Extra: the 5-D SDPA call never reaches a fused kernel (MATH backend), and the grid variant's implicit scale is `1/sqrt((H/8)(W/8))`, i.e. resolution-dependent (M-03). |
| F03 MFHGMoE router gets no gradients | **confirmed** | Codex unit probe reproduced; full default net: 33 tensors `grad None`: router (7), all 6 `mfg` conv stacks (24), `aggr_conv` (2) - the whole multi-level feature branch is dead weight. |
| F06 HMA/MFHGMoE ignore use_checkpoint | **confirmed** | Source: `self.use_checkpoint` stored at hma_arch.py:826 / mfghmoe_arch.py:534, never read. Measured potential of the HAT-style fix in M-07. |
| F10 MFHGMoE dense experts | **confirmed** | Codex arithmetic matches (16 experts x 1024 ch; 21.5 GMAC at 64x64). Measured CPU peak at template batch 8x32x32: 1585 MiB old vs 217 MiB with the masked-accumulate dispatch. The fp32 `one_hot` also promotes the bf16 product to fp32. MoE upsampler = 17.3 % of fwd+bwd FLOPs (FlopCounterMode). |
| F11 DITN discarded attention | **confirmed** | `ISA.temperature.grad is None` in all 4 ITLs after an optimizer step; the discarded `attn` costs a c x c softmax per window; SDPA input is 3-D (math backend). New: neosr-trained DITN != spandrel DITN (M-05). |
| LMLT tiny inputs | **confirmed, broader** | Fails for any side <= 32 (reflect pad to a multiple of 64), not only 7x5/1x1; 40x40 and 67x53 work. |
| ASID minimum dimensions | **confirmed** | Whole-network minimum is 9 px per side (window 8 pads 8->8, ESA stride-2 conv gives 3x3, `max_pool2d(7)` fails); 9..16 pad to 16 and work. |
| HMA/MFHGMoE window alignment | **confirmed, broader** | hma fails at 67x53, 30x30, 17x13; mfghmoe fails at every probed size not divisible by 16 incl. 40x40 and **200x200 = the `#tile = 200` hint in train_mfghmoe.toml**; microsr fails at any odd side. |
| OmniSR helper NameError | **confirmed (harmless)** | ruff F821 at omnisr_arch.py:449-450; class never instantiated anywhere. |
| FlexNet attention under no_grad | **refuted for HEAD** | Fixed in 09ffccc (A1); `test_flexnet_sdpa_matches_manual` covers it; census: 0 missing grads. New issue instead: the SDPA input is 3-D, so the default flash path is the math kernel (M-14). |

## 3. Census results (all CPU, default constructors, x4)

Gradient census (`census.py`, `census2.py`; B=2, 64x64 or 32x32; "persistent" = still exactly zero after one Adam step):

| arch | grad None | exactly-zero (persistent) | notes |
|---|---|---|---|
| asid / asid_d8 | 30 / 80 | 0 | F02 |
| ditn | 4 (temperature) | 0 (48 at step 0 = zero-init `SDA.scale`) | F11 |
| mfghmoe | 33 | 26 at step 2 = unselected experts (expected for top-1) | F03 |
| hma | 0 | **104** (all grid-attention DynamicPosBias) | M-09 |
| dct | 0 | **270** (all DynamicPosBias) | M-09 |
| esc*, man*, microsr, grformer* | 0 | 0 (step-0 zeros are zero-init layer scales; grformer_large zeros = DropPath with B=2) | ok |
| catanet, dctlsa, omnisr, cfsr, safmn*, flexnet, metaflexnet, lmlt* | 0 | 0 | ok |

Host-sync census per forward (`sync_census.py`: Tensor.__bool__/item/tolist/nonzero/bool-mask indexing; `h2d_census.py`:
CPU tensors copied to the net's device, run on the meta device; plus source review for `torch.tensor(list, device=)` and
CPU index tensors):

| arch | train | eval | sites |
|---|---|---|---|
| esc (default) | >= 10 + 10 in backward | >= 10 | rpe index H2D x2 per WindowAttention (M-01), DySample normalizer (M-13) |
| microsr / microsr_light | 30 / 20 | 30 / 20 | microsr_arch.py:352 (M-06) |
| catanet | 16 | 8 | catanet_arch.py:221, 255 (M-11) |
| mfghmoe | 11 | 11 | mfghmoe_arch.py:855, 861, 864, 876 (x2 groups), 963, 971, mask at 1300 |
| dct | 10 (+1 first call) | 10 | dct_arch.py:461-462 at the template patch 32 (`img_size=64`) (M-10) |
| hma | 1 (+1 first call) | 1 | hma_arch.py:1345 (M-10) |
| all others in scope | 0 | 0 | - |

Each listed copy is a pageable host->device copy; PyTorch's non-`non_blocking` copy ends in `cudaStreamSynchronize`
(`memcpy_and_sync`), so each one drains the GPU queue (source-verified; wall-clock effect = hypothesis for the profiling lane).

Dynamo graph breaks, train mode (`explain_probe.py`): asid 43, catanet 14, mfghmoe 6, every other arch in scope 0.
After the proposed fixes (prototypes): asid 0, catanet 0, mfghmoe 0.

bf16 autocast vs fp32 (`amp_probe.py`, CPU autocast, B=2): every arch finite, gradient cosine >= 0.9996, output relative error
<= 1.8e-2 **except mfghmoe: 0.316** (hard routing flips, M-02).

Arbitrary sizes, eval (`size_probe.py`, sizes 1x1, 7x5, 12x12, 17x13, 30x30, 40x40, 67x53): dct, cfsr, man, flexnet, metaflexnet
pass all; failures as listed in sec. 2 and M-04/M-17.

## 4. Findings

### M-01 (P1, speed) ESC keeps a 64 MiB position index on the CPU and re-uploads it twice per attention per step

Where: `esc_arch.py:237` (`self.rpe_idxs = self.create_table_idxs(...)`, a plain attribute), `:241-258` (Python loop),
`:301`, `:310`, `:266` (`relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]`).

Evidence (`esc_probe.py`, measured on host): `rpe_idxs` is `(4_194_304, 2) int64` = 64 MiB per WindowAttention (esc 5 modules,
esc_light 3, esc_large 10, esc_fp 48 MiB x5); it is not a buffer, so `net.to(device)` leaves it on CPU (checked on meta).
Advanced indexing a CUDA tensor with CPU indices moves the indices to the device every call; the two strided columns are
first made contiguous on the host: **22 ms host time per call** + 64 MiB pageable upload + sync. `IndexBackward0` saves the
CPU indices (checked), so the backward re-uploads them. Default `esc`: ~10 such copies per training step = ~640 MiB H2D and
>= 220 ms of host work during which the GPU starves (GPU magnitude = hypothesis). Building the net takes **19.1 s**
(3.2 s per module, 4.2 M-iteration Python loop); the EMA deep copy duplicates the CPU tensors (another 320 MiB host RAM).

Fix (exact): replace the attribute with a non-persistent `(N, N)` buffer and gather per head:
```python
ws = window_size[0]; c = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
rel = c[:, None, :] - c[:, :, None] + ws - 1          # k - q, same convention as create_table_idxs
self.register_buffer("rpe_idx", rel[0] * (2 * ws - 1) + rel[1], persistent=False)   # (N, N) long, 8 MiB
...
bias = self.relative_position_bias[:, self.rpe_idx].unsqueeze(0)   # (1, heads, N, N)
```
`torch.equal(old_bias, new_bias)` is True (probe). Adapt `to_mobile()` (same expression, then `del self.rpe_idx`). The
flex path is untouched. state_dict unchanged (the attribute was never saved; the buffer is non-persistent), so the A3
manifest test stays green. Expected: removes all per-step uploads/syncs of ESC attention and ~19 s startup; GPU memory +8 MiB
per module (or 0 with the `unfold` formulation: the bias is a sliding window over the `(2ws-1)^2` table).

Test: bias equality old vs new for ws in {8, 32}, heads in {3, 4}; forward equality of a full `esc_light` before/after with
the same weights; `torch.cuda.set_sync_debug_mode("error")` around a CUDA forward+backward (CUDA-only test); build time < 2 s.
Numerics: none. Compat: none.

### M-02 (P1, learning + memory + speed) MFGHMoE: router and multi-level features never learn; dense expert stack

Where: `mfghmoe_arch.py:828-885` (DualRouting), `:939-984` (HMoE.forward), `:1296-1327` (x_t branch), `:1208` (MoEUpsample).

Evidence: (`moe_probe.py`, `moe_full.py`, `moe_peak.py`, `moe_amp.py`)
- Gradients: the gate returns a constant one-hot (topk indices + index_put of 1). Full default network: 33 tensors with
  `grad None`: router, `mfg.{1,3,5,7,9,11}` conv stacks and `aggr_conv`. The x_t branch costs forward compute on every
  step (6 x two 3x3 convs 180<->64 + aggr) but its parameters stay at init forever; routing is a frozen random function.
- Memory: dense 16-expert list + `torch.stack` + fp32 one-hot product. CPU peak at template batch 8, LR 32x32: **1585 MiB**
  (analytic CUDA bf16: list 256 MiB + stack 256 MiB + fp32 product 512 MiB; validation at LR 256x256: ~12 GiB fp32).
- Syncs: 10 per forward (`group_mask.any()`, boolean-mask getitem/setitem per group, two `isnan().any()`), 6 graph breaks;
  `print("NaN in gate scores")` and silent NaN->0 replacement hide the first non-finite op from the trainer's finite policy.
- AMP: training is bf16 (models/image.py:221), validation fp32 (`inference_mode`, image.py:342). With near-tied untrained
  logits, **4.7 % of tokens pick a different expert in bf16 than in fp32** -> output rel-err 0.316 (other archs <= 0.018).
  Computing the gate in fp32 cuts flips to 0.7 % (the rest comes from bf16 x_t).

Fix (prototype `moe_fix.py`, verified):
1. Vectorised, sync-free routing: group softmax -> `topk(1)`; `x + group_embedding(g)`; both expert gates on all tokens and
   `torch.where(g == 0, ...)`; global id `g*G + e`; selected probability `p = p_group[g] * p_expert[e]`. Run it under
   `torch.autocast(..., enabled=False)` with `x_t.float()`.
2. Value-preserving straight-through weight `w = 1 + p - p.detach()`: the forward is exactly the old one-hot (old checkpoints
   compute the same output: full net max diff 4.8e-7), while d loss/d p equals the Switch-Transformer router gradient
   (`<g, E_sel(x)>`); the router, group_embedding, x_t and therefore `mfg`/`aggr_conv` learn (census: 0 missing grads).
3. Memory-lean exact dispatch: `out = sum_e expert_e(x) * (idx == e)` with constant masks (autograd saves only the masks),
   then `out * w` (saves `out` once). Routing bitwise identical; expert grads equal to 1.8e-11; CPU peak 1585 -> 217 MiB,
   CPU time 5.6 s -> 2.7 s at the template batch. 0 syncs, 0 graph breaks.
4. Delete the NaN print/zeroing (the trainer already skips non-finite updates).

Numerics: training changes (router and 26 more parameters now learn; fp32 routing) -> **A/B before default** (PSNR/SSIM curve
and expert-usage histogram/entropy at 20k-50k iters; watch for collapse, see M-19). state_dict: unchanged. Old checkpoints:
identical forward except the fp32 gate (routing of near-tie tokens can differ from a bf16-trained run; same as the current
train/val mismatch). Test: routing equals the old one-hot on random inputs; forward equality old/new; strict gradient census
(no `None`); `set_sync_debug_mode("error")` on CUDA; peak-memory regression (`torch.cuda.max_memory_allocated`) at 8x32x32.

### M-03 (P1, learning) ASID default attention: no_grad, ignored temperature, resolution-dependent scale

Where: `asid_arch.py:331-367` (Channel_Attention), `:370-427` (Channel_Attention_grid), default `flash_attn=True` (`:735`).

Evidence (`asid_probe.py`, census): `torch.no_grad()` around SDPA -> output detached; 30 tensors without grad in the
default network (canorm1/2 too, because nothing upstream of SDPA gets a gradient). SDPA uses its default `1/sqrt(E)`; the
manual path uses the learned temperature. In the grid variant `E = (H/8)(W/8)`, so the effective temperature is 1/8 at the
64 training patch but 1/32 at a 256x256 input (`|flash - manual(t=1/sqrt(E))| = 0`, `|flash - manual(t=1/8)| = 3.6e-3` at 256):
a train/inference mismatch on the default path. The tensors are 5-D, so CUDA uses the MATH backend: the no_grad hack never
bought speed. Under DDP with `find_unused_parameters=False` the 30 unused parameters raise the reducer error on step 2
(source-verified; DDP best-effort per DECISIONS 8c).

Fix: one equation for both paths, computed by SDPA on 4-D tensors, canonical (paper/manual) semantics:
```python
q = F.normalize(q, dim=-1) * self.temperature; k = F.normalize(k, dim=-1)
s = q.shape                                   # (b, n, heads, d, E)
out = F.scaled_dot_product_attention(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), scale=1.0).view(s)
```
(`flash_attn` then only selects SDPA vs explicit matmul; probe: `|manual - SDPA(q*t, scale=1)| = 3.4e-8`.)
Legacy remap so existing default-trained checkpoints keep their output: their temperatures were never updated and are
exactly 1.0 (no grad -> Adam/AdamW skip; EMA lerp keeps 1.0 exactly). In both attention classes:
```python
def _load_from_state_dict(self, sd, prefix, *a, **k):
    t = sd.get(prefix + "temperature")
    if self.flash_attn and t is not None and bool((t == 1).all()):
        sd[prefix + "temperature"] = t / self.ps      # 1/sqrt(ps*ps): the old implicit scale at the 64 patch
    super()._load_from_state_dict(sd, prefix, *a, **k)
```
This reproduces the old local attention exactly and the old grid attention exactly at `patch_size = 8 * window_size`
(the template's 64); at other inference sizes the result now equals the training-size behaviour (the old one was
resolution-dependent). Do **not** use the shorter "SDPA(q*t, k, v) with default scale" variant: it keeps the
resolution-dependent grid scale. Numerics: training changes (A/B); compat: state_dict keys unchanged; old flash_attn=False
checkpoints untouched (flag False, t learned != 1). Test: manual/SDPA parity incl. gradients for all params; strict gradient
census of `asid`/`asid_d8`; remap test (old forward == new forward after load at 64x64); a test that the grid attention's
effective temperature does not depend on the input size (compare against the manual path at 64 and 128).

### M-04 (P1, reliability) HMA, MFGHMoE, MicroSR crash at non-aligned validation sizes

Where: `hma_arch.py:1361-1374`, `mfghmoe_arch.py:1330-1339`, `microsr_arch.py:751-762` (no `check_image_size`).
Evidence (`size_probe.py`): `RuntimeError: shape '[1, 8, 8, 6, 8, 1]' is invalid` for hma at 67x53; mfghmoe fails at 40x40
and at 200x200 (the template's `#tile = 200` hint); microsr fails at any odd side. Validation and `infer_tiled` (tile windows
are snapped to the image, not to the lattice) feed arbitrary sizes, so a run can die at the first validation hours in.
Fix (pattern already in hat_arch.py:1136-1162): pad to `m = lcm(window_size, interval_size)` (hma), `window_size` (mfghmoe),
`window_size` of the blocks (microsr; 2 by default), `mode = "reflect" if ph < h and pw < w else "replicate"`, crop the output
to `h*scale, w*scale`. No numerics change for aligned inputs (pad is a no-op; test it); state_dict unchanged.
Test: add hma/mfghmoe/microsr to a size sweep (67x53, 17x13, 40x40, 200x200) asserting the output shape.

### M-05 (P1, learning/export) DITN: dead temperature, discarded attention, spandrel mismatch

Where: `ditn_arch.py:102-129`. Evidence (`ditn_probe.py`): temperature `grad None` in all ITLs; the code computes
`softmax(q k^T * t)` and throws it away, then `SDPA(q, k, v)` (scale 1/8) on 3-D tensors (math backend). spandrel's
DITN_Real (used by chaiNNer and the competitor) computes `softmax(t * q k^T) v`: on the same weights neosr and the canonical
forward differ by 0.0137 max already at init, so every neosr-trained DITN silently changes when exported/loaded elsewhere.
Fix: canonical semantics in one fused call: `q, k, v = (t.transpose(-1, -2).unsqueeze(1) ...)`,
`q = F.normalize(q, -1) * self.temperature`, `out = F.scaled_dot_product_attention(q, k, v, scale=1.0)`; delete the dead
`attn` lines; add the same `_load_from_state_dict` remap (`t == 1` exactly -> `1 / patch_size`). Probe: canonical vs fixed
max diff 0.0; neosr vs fixed-with-remap 4.2e-7; fixed: 0 missing grads. Re-saved old checkpoints become spandrel-correct.
Numerics: training changes (A/B); compat: keys unchanged. Test: parity with an inline copy of the spandrel ISA; remap test;
strict gradient census.

### M-06 (P2, speed) MicroSR builds 30 all-zero shift masks on the CPU every forward

Where: `microsr_arch.py:350-353`. Default `patch_size=32`, `img_size=64` -> `input_resolution=(2,2)` never equals `x_size`,
so every block calls `calculate_mask(x_size).to(x.device)` even with `shift_size == 0` (the mask is then all zeros).
Measured (host only): 30 masks per forward, 8.5 ms at LR 64x64, 239 ms at 256x256 (validation), 30 H2D syncs; with
`patch_size=1` (16x16 windows) 322 ms / 120 MiB at 64x64 and 1.76 s / 1.9 GiB at 256x256 per forward when off-size.
Fix (exact): `mask = None if self.shift_size == 0 else (self.attn_mask if self.input_resolution == x_size else
self.calculate_mask(x_size, x.device))`, with `calculate_mask` building `img_mask` on `device` (hat_arch.py:1069 pattern).
Adding a zero mask is the identity, so outputs are bitwise unchanged. Test: forward equality + sync debug mode on CUDA.

### M-07 (P2, memory) HMA/MFGHMoE: implement `use_checkpoint`

Where: `hma_arch.py:893-902` / `:1340-1359`, `mfghmoe_arch.py:579-588` / `:1295-1327`.
Measured (`ckpt_probe.py`, saved-tensor bytes between forward and backward, B=2, LR 32x32, fp32): hma 444.2 -> 15.7 MiB,
hma_medium 2216 -> 27 MiB, mfghmoe 1602.6 -> 65.4 MiB (+ one layer's activations during backward, ~1/4 resp. ~1/6 of the
original); gradient norm identical to all printed digits; time +25 % (hma) on CPU. Fix: as hat_arch.py:1122-1126:
`x = checkpoint(layer, x, x_size, params, use_reentrant=False) if self.use_checkpoint and self.training else layer(...)`
in `forward_features` (per RHTB/RHAG; params dict is fine with non-reentrant). Numerics none; opt-in flag; template comment
`#use_checkpoint = true` for the heavy variants. Test: gradient parity with/without checkpoint (as test_archs_a2
`test_checkpoint_grad_parity`) through the real trainer backward.

### M-08 (P2, memory) Manual window attention stores fp32 softmax probabilities

Measured (`softmax_share.py`, autograd graph walk over `SoftmaxBackward0._saved_result`, B=2, fp32): softmax outputs are
37.1 % of all saved activations in mfghmoe (594 of 1603 MiB), 25.6 % in grformer, 21.6 % in hma, 5.9 % dctlsa, 7.4 % lmlt,
<= 2.8 % asid/dct/microsr_light. CUDA autocast runs softmax in fp32, so under bf16 the share is larger (the rest halves).
Fix: SDPA with an additive float bias (relative-position bias [+ shift mask]) for HAB/OCAB (mfghmoe), FAB/GAB window
attention and AffineTransform (hma), GRFormer's cosine attention (`q = normalize(q) * logit_scale.exp().clamp`, `scale=1`,
`attn_mask=bias`). The memory-efficient kernel supports a bias that requires grad. Exact up to kernel rounding; needs a
manual/SDPA output+gradient parity test per class and a GPU benchmark (the profiling lane): the fused kernel with a
materialised (b*nW, heads, N, N) mask may cost speed; broadcast the bias where possible. Shares the work with Codex F04
(SwinIR/HAT family, other lane). Hypothesis for the CUDA gain: ~30-50 % less activation memory for mfghmoe.

### M-09 (P2, speed/memory; P3 arch opt-in) DynamicPosBias collapses to `pos_dim = 1`

Where: `hma_arch.py:508` + `:146`, `dct_arch.py:187` + `:105` (`DynamicPosBias(dim // 4)` and again `pos_dim = dim // 4`,
the DAT pattern). Evidence: pos_dim histogram hma {1: 8}, dct {1: 20} (hma_medium/large: 5). `LayerNorm(1)` outputs its bias,
so the MLP emits one constant per head; adding a per-row constant before softmax is a no-op: 104 (hma) / 270 (dct)
parameter tensors with persistent exact-zero gradients; with random MLP weights the outputs with/without the bias differ by
3e-7. The default hma grid attention and **all** default dct window attentions therefore have no relative-position
information. Cost of the dead work in hma's GAB at a 256x256 LR validation input (N = 4096 grid tokens): int64 coords
256 MiB + index 128 MiB + gathered bias 192 MiB + contiguous copy 192 MiB per AffineTransform, 8 per forward (analytic).
Fix (exact): in `AffineTransform.forward` / `Window_Attention.forward`, skip the bias when `self.pos.pos_dim == 1`
(keep the parameters for state_dict compatibility). Opportunity (changes the architecture, A/B, new checkpoints only):
an opt-in kwarg `pos_dim` (e.g. `max(dim // 4, 8)`) restores positional bias for small models.

### M-10 (P2, speed) HMA/MFGHMoE/DCT shift masks built on CPU every forward

Where: `hma_arch.py:1301-1331, 1345`, `mfghmoe_arch.py:1258-1285, 1300`, `dct_arch.py:337-414, 459-462` (DCT recomputes
both masks per shifted block whenever the input differs from `img_size=64`, i.e. always at the template patch 32).
Fix (exact): `calculate_mask(x_size, device)` building on `device` (as hat_arch.py:1069), computed once per forward and
passed to blocks (dct: compute per (H, W) once in `forward_features`, pass down). Measured counts in sec. 3.

### M-11 (P2, speed/compile) CATANet `initted` host syncs

Where: `catanet_arch.py:221` and `:255`. 8 syncs per eval forward, 16 per train forward, 14 graph breaks.
Fix (prototype `catanet_probe.py`, bitwise identical outputs and EMA means over 4 train/eval rounds, 0 graph breaks):
always compute `init_means` (cheap mean), `x_means = torch.where(self.initted, self.means, init_means).detach()`; update
`self.means.copy_(torch.where(self.initted, self.means.mul(d).add_(x_means, alpha=1 - d), x_means)); self.initted.fill_(True)`.
The k-means `no_grad` blocks are upstream design (not a bug). Numerics none; state_dict unchanged.

### M-12 (P2, compile) ASID LayerNorm2d custom backward uses `ctx.saved_variables`

Where: `asid_arch.py:24`. 43 dynamo graph breaks (`Length mismatch when unpacking ... UNPACK_SEQUENCE`). Fix: `ctx.saved_tensors`.
Probe: graphs 44 -> 1, breaks 43 -> 0, gradients bitwise equal. Numerics none.

### M-13 (P2, speed/export) DySample normalizer is a host tensor every forward

Where: `arch_util.py:114` (`torch.tensor([W, H], dtype=x.dtype, device=x.device)`): one pageable H2D copy + sync per
forward (esc default `use_dysample=True`; flexnet/spanplus/realplksr/moesr/mosrv2 when selected) and the F07 trace constant.
Fix (bitwise identical over fp32/bf16/fp16 and sizes up to 1000x513, `dysample_probe.py`):
`normalizer = torch.cat((coords_w[-1:], coords_h[-1:])).add(0.5).to(x.dtype).view(1, 2, 1, 1, 1)` (values come from the
`arange` already built, so tracing keeps them shape-derived). Coordinate with the lane that owns F07.

### M-14 (P2, speed) SDPA calls that can never use a fused kernel

`torch._fused_sdp_choice` on CPU: 3-D -> MATH, 4-D -> FLASH, 5-D -> MATH; libtorch_cuda contains the CUDA check
"All fused kernels requires query, key and value to be 4 dimensional". Affected: flexnet/metaflexnet `LMLTVIT`
(flexnet_arch.py:245, default `flash_attn=True`, q (B', 64, C) 3-D), ditn (3-D), asid (5-D). Fix: `unsqueeze(1)` /
`flatten(0, 1)` to `(batch, heads, L, E)`; head dims 64/48/64 are flash-eligible in bf16. Exact math. GPU speed/memory gain =
hypothesis (measure in the profiling lane). Not applicable: catanet IASA (q/k head dim 9 != v 10, not 8-aligned), lmlt
(head dim 15/9/21).

### M-15 (P2, arch quality; owner decision) MicroSR default attention is 2x2 windows without shift

Where: `microsr_arch.py:614` (`patch_size=32`, changed from 1 in upstream commit 75557f4 "fix ... microsr inference").
`patches_resolution = 64 // 32 = 2` makes every SwinTransformerBlock fall back to `window_size = 2, shift_size = 0`
(all 30 blocks; light: 20). The advertised design (16x16 / 8x8 shifted windows, `relative_position_bias_table (961, 6)`)
is not what trains; tables are `(9, 6)`. Changing the default breaks every existing microsr checkpoint (shape mismatch),
so: keep the default, add padding (M-04) and an explicit variant/kwarg for the real window design (A/B; new checkpoints),
and note it in the README. Speed note: the 16x16 design costs far more attention compute; M-06 then matters more.

### M-16 (P2, tooling) Tests that would have caught F02/F03/F11, M-01, M-04

`tests/test_archs.py:190` filters `p.grad is not None`; `test_arch` is CUDA-only, so CPU CI runs no arch numerics.
Add (CPU, fast, parametrised over ARCHS rows of this lane, small inputs):
1. strict gradient census: every `requires_grad` parameter has a grad after one backward, allowlist empty for this lane
   after fixes (structural exceptions documented in the test);
2. sync census: meta-device run counting CPU->device copies + `Tensor.__bool__`/`item` interception (the probe scripts),
   and on CUDA `torch.cuda.set_sync_debug_mode("error")` around forward+backward;
3. size sweep (17x13, 67x53, tile-like 200x200) asserting shape for archs that claim arbitrary sizes;
4. per-arch state_dict manifest (pattern of test_archs_a3) for every changed file.

### M-17 (P3, reliability) Tiny-input limits

lmlt fails for sides <= 32, esc/grformer <= 17 (reflect pad to the 32/16 window), asid < 9 (ESA), catanet/dctlsa/omnisr/
safmn at 7x5/1x1. Fix: the HAT rule (`reflect` if `pad < size` else `replicate`) in `lmlt_arch.py:345-351`,
`esc_arch.py:260-263`, grformer's pad; document ASID's 9 px minimum. Numerics unchanged for supported sizes.

### M-18 (P3, hygiene)

- omnisr_arch.py:416-470 `AdaConvGuidedFilter` (NameError `channel`, never used): delete.
- asid_arch.py:154 `torch.FloatTensor((B,1,1,1), device=device)` builds a 4-element tensor (and fails on CUDA); unreachable
  (dropout fixed 0.0). Use `x.new_empty((B,1,1,1)).bernoulli_(1 - p)` or delete Dropsample.
- asid `block_num > 1` crashes (`nn.Sequential` of blocks that return tuples, asid_arch.py:714-719): loop explicitly or
  reject `block_num != 1` in the ctor.
- mfghmoe: `self.text_locals = []`, unused `capacity_factor`/`epsilon`; NaN print (removed by M-02).

### M-19 (P3, optional) MoE balance term and sparse dispatch

After M-02 the router learns; top-1 routing without balancing can collapse (graceful for an upsampler: it degrades to one
conv). Decide from the A/B's expert-usage entropy. If needed: the arch stores `self.aux = E * sum_e f_e * mean_p_e`
(`f_e` from `bincount`, no sync; `p_e` = full hierarchical probabilities), a `moe_balance` loss row reads it through one new
`Context` field (e.g. `ctx.aux`, filled generically by the trainer from the bare net_g), no per-loss branching.
Exact sparse dispatch (unfold 3x3 patches, sort tokens by expert, per-expert GEMM) cuts the upsampler's 17.3 % of FLOPs by
~16x but needs one host sync (`counts.tolist()`) and dynamic shapes (`torch.compiler.disable` region); not worth it before
profiling shows the upsampler matters.

## 5. Do not "fix"

- dctlsa SwinBlock without attention residual (`shortcut` unused, dctlsa_arch.py:181): identical in spandrel/upstream.
- catanet `torch.no_grad()` blocks: k-means clustering and argmax indexing are upstream design.
- hma grid attention is global over `(H/4)(W/4)` tokens (quadratic); validation needs tiling: upstream design.
- lmlt manual attention: head dims 15/9/21 are not fused-kernel eligible; SDPA would not help.
- Zero gradients at step 0 in ditn/esc/man/microsr/grformer: zero-initialised layer scales or DropPath (vanish after one step).
- traiNNer-redux has no ASID/HMA/MFGHMoE/CATANet/DCT/MicroSR/CFSR/GRFormer; its DITN/DCTLSA/OmniSR/SAFMN are spandrel
  wrappers (DITN = the canonical temperature equation, used as the parity reference in M-05); its FlexNet adds layer scales
  (an architecture change, not a fix). Nothing else to port for this lane; none of its damping mechanisms are proposed.

## 6. Probe index (scratch dir)

census.py / census2.py (gradient census), sync_census.py, h2d_census.py (meta device), size_probe.py, amp_probe.py,
explain_probe.py (dynamo), asid_probe.py, ditn_probe.py, moe_fix.py + moe_probe.py + moe_full.py + moe_peak.py + moe_amp.py,
ckpt_probe.py, softmax_share.py, esc_probe.py, dysample_probe.py, catanet_probe.py; outputs `*.txt` next to them.
