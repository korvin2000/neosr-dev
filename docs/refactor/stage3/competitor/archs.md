# Stage 3 competitor harvest: lane `archs`

T = neosr (`.worktrees/refactor`, `refactor/stabilize` at e0019ee). C = traiNNer-redux (`/home/korvin/apps/traiNNer-redux-dev/traiNNer`).
SP = spandrel 0.4.1 in the competitor env (`~/miniconda3/envs/redux-dev`). BENCH = `C/../docs/source/benchmarks.md`. Its numbers are
inference only: RTX 4090, 1x3x480x640 input, fp16/bf16, with and without channels_last. PSNR/SSIM there come from Urban100 Y using the
published L1 models (DF2K or DIV2K). This is the only cross-arch table in either repo.

Evidence key: **paper** (numbers from the publication) · **bench** (BENCH table) · **code** (read in C/T) · **logs** (the owner's
own traiNNer runs in `C/../experiments`; mixed losses, pretrains and resumes, so not an A/B) · **reasoning**. No probe was run in
this lane. Every finding below is settled by code reading, the BENCH table or a paper.

Prior work this builds on: `analysis/competitor/H4.md` (helpers and the shared archs), `stage2/analysis/archs_{cnn,mixed,window}.md`.
This lane does not repeat those findings, apart from the realplksr/A-07 link.

## 1. Verdict

Most of C's 90 arch files do not give neosr a better network.

- **About 40% are spandrel wrappers.** These include grl, swin2sr, nafnet, seemore, mosr, rrdbnet and srvgg.
- **About 35% are community or bespoke designs with no controlled numbers.** These come from Philip Hofmann (paragon*, elysium,
  hyperion, detailrefiner, nerve, heart, munet, gfisrv2), Kim2091 (dis, temporal*, tfdat), umzi (gaterv3, lawfft, metagan3,
  rtmosr) and the-database (spanf/f3, fdat variants, escrealm, autoencoder).
- **Most of the rest are paper archs.** On BENCH they are dominated by archs neosr already ships.

The real gaps in neosr are narrow:

1. There is no **1x restoration U-Net** for denoise, deblur, JPEG or compression. Neosr archs only handle scale 1 through
   unshuffle tricks.
2. Neosr's window transformers keep **vanilla per-token LayerNorm**. A 2025 paper shows this LayerNorm drives feature
   magnitudes to about 1e6, and that fp16 inference collapses as a result.
3. The **template arch (realplksr)** has a slow, init-unstable post-GroupNorm. C fixed it with an opt-in pre-LN variant.
4. There is no **normalisation-free R3GAN discriminator** to pair with the rpgan + R1/R2 support that stage 2 already added.

Ranked recommendation (5 to port, 3 optional):

| # | Item | Kind | Prio | Cost | Approach | Evidence |
|---|---|---|---|---|---|---|
| 1 | RealPLKSR `layer_norm` (pre-LN instead of post-GroupNorm), opt-in | variant | **P1** | S | port+fix | bench + T census A-07 + code |
| 2 | i-LN (holistic LN + input-std rescale) as `norm="iln"` for HAT/DRCT/SwinIR/SRFormer(/DAT) | arch_util idea | **P1** | M | rewrite from paper (C's iLN is not faithful) | paper |
| 3 | NAFNet (1x restoration baseline) | new family | **P1** | S-M | port+fix | paper |
| 4 | R3GAN scalar discriminator (`RpGANDiscriminatorScalar`, the owner's own code) | D | **P2** | S | port+fix | paper (generation), logs |
| 5 | ICNR init for every conv that feeds a PixelShuffle | arch_util idea | **P2** | S | rewrite (15 lines) | paper (Aitken 2017) + reasoning |
| 6 | SCUNet (1x blind real-world denoise), after NAFNet | new family | P3 | M | port | paper |
| 7 | Real-time presets: ArtCNN, RTMoSR, compact `ultracompact`/`superultracompact`, `esrgan_lite`, `dat_light` | presets/small | P3 | S each | port (RTMoSR via T's online reparam) | bench |
| 8 | Video SR (TSPANv2 first) | new pipeline | P3 | L | rewrite | code only |

Skip everything else (section 3). Two ideas are worth remembering but not porting now: RIB position bias, which makes window
attention FlashAttention-compatible, and head-dim-aligned presets (section 6).

## 2. Recommended items

### 2.1 (P1, S) RealPLKSR pre-LayerNorm variant

**C** `realplksr_arch.py:15-27,73-122`. `use_layer_norm` adds a channel LayerNorm *before* the block (`x = layer_norm(x)` at
`:115`) and replaces the post-branch `GroupNorm(4)` with Identity. C defaults to `layer_norm=True` at `:143,202`, which breaks
the official weights. H4 already said not to copy that default.

**Value.**

- **Speed (bench).** At 4x, fp16, realplksr pixelshuffle runs 8.99 fps (LN) vs 6.08 fps (GN), and 10.93 vs 6.41 fps with
  channels_last (1.48x/1.71x). The 1x row is the same (9.04 vs 6.10).
  - Mechanism: GroupNorm is a slow kernel in fp16, and it breaks channels_last (T A-12: 28 GN per forward, 2 layout copies per
    block in fwd and bwd).
- **Stability (T census A-07).** The post-GN with gamma 1 adds a unit-variance term to the residual stream in all 28 blocks.
  - At init, realplksr outputs std 2.57 (realplksr_l: 3.32) and has a G grad norm of 29-49, against 0.39 for plksr without GN.
  - Every early step is clipped 30-50x. A pre-LN residual block leaves the stream untouched at init.
- **Quality (bench).** C's published L1 DF2K model with `layer_norm=True` scores 26.94/0.8140 on Urban100 x4. PLKSR (DF2K)
  scores 26.85. There is no published GroupNorm number to compare against.

**C implementation quality.** The block logic is faithful and small. The LayerNorm is hand-written: mean and var run in the
input dtype, and `(x-u).pow(2)` overflows in fp16 once |x-u| > 256.

**Port recipe.**

- Add `layer_norm: bool = False` to `T/archs/realplksr_arch.py` `PLKBlock`/`realplksr`.
- Implement the norm as `F.layer_norm` on the NHWC permute (`x.permute(0,2,3,1)`). This is zero-copy under channels_last, and
  autocast runs it in fp32 (T A-08 already chose this form).
- The new keys `layer_norm.weight/bias` appear only when the flag is on. The default keeps GN, so every old `net_*.pth` loads
  unchanged.

**Test.** Check the key set for both settings, that init output std is < 0.5, and that an fp16 forward stays finite with large
activations.

**Next steps.**

- Measure ms/update in T, including with `compile=true`, which may fuse GN and shrink the gap.
- Run a 5k/8k template A/B against A-07's GN-gamma-init option.
- If LN wins on both, add `layer_norm=true` to the realplksr templates. The arch default stays False.

### 2.2 (P1, M) i-LN for the window transformers

**Paper.** Lee et al., arXiv 2504.06629 (v3, Feb 2026), "Analyzing the training dynamics of image restoration transformers: a
revisit to layer normalization".

**Mechanism.** Per-token LN in image-restoration transformers lets feature magnitudes grow to about 1e6 and collapses
channel-wise entropy. i-LN has two parts:

- **LN\*.** Mean and variance are computed per sample over all tokens *and* channels, with the usual per-channel affine.
- **Input-adaptive rescale.** Each sublayer becomes `x + sqrt(σ²+ε) · f(LN*(x))`.

**Reported results (paper).**

- HAT, lightweight setting, x4 Urban100: 26.55 → 27.17. x2: 32.16 → 33.32.
- DRCT1 x4: 26.63 → 26.79. SRFormer1 x4: 26.43 → 26.49.
- HAT, full public config, x4: 27.97 → 28.04. x2: 34.45 → 34.60. The gain shrinks as the model grows.
- Feature magnitudes stay near 1.2 instead of about 1e6.
- Weights and features in fp16: LN collapses to 7.46 dB, while i-LN keeps 27.58.
- Weight-only int4: 25.02 → 26.83.
- Gradient clipping does *not* fix the LN dynamics (Table 8).

This is the owner's principle in paper form: stability from a correct normalisation, not from damping. It also matters for neosr's
fp16 ONNX export path.

**C implementation.** `C/archs/arch_util.py:21-71` (`iLN`), used by `hat_iln_arch.py` and `dat_iln_arch.py`, and `heart` builds on
it. **It is not the paper's i-LN.**

- C adds an EMA "spike guard": `scale = min(std, 2·std_ema)`, with the EMA updated in training at α = 1e-3. That is
  automatic damping of exactly the kind the owner rejects.
- `std_ema` is a **non-persistent** buffer (`:33`). After a checkpoint reload it is 0, so eval uses the raw std. In the
  training process, validation uses the clamped std instead.
- The same weights therefore produce different outputs in-process and after reload. The difference shows whenever an input's
  std exceeds 2x the EMA, for example full images or high-contrast content against 64 px training patches.
- The rest is right: holistic statistics, and the rescale applied to both attention and FFN branches (`hat_iln_arch.py`
  `HAB.forward`).

**Port recipe (rewrite from the paper, about 20 lines).**

- Add `class iLN` to `T/archs/arch_util.py`. It returns `(γ·(x-μ)/σ + β, σ)`, computes σ over `(L, C)` per sample in fp32, and
  has no buffers.
- For plain LN, return `(F.layer_norm(...), 1)` so the block code is shared.
- Add a `norm: Literal["ln","iln"] = "ln"` kwarg to the HAT, DRCT, SwinIR and SRFormer blocks: `x = x + s1*attn(...)`,
  `x = x + s2*mlp(...)`. These are the four backbones the paper validates. DAT, ATD and HMA come later and need their own
  check.
- State-dict keys and shapes are unchanged (the LN weight/bias are `[C]`). Old weights *load* into an i-LN net but compute a
  different function. The flag is therefore per-run config and is never auto-detected.

**Risks.**

- Statistics over the whole image make the output **tile-dependent**. Tiled validation or inference can show seams, so test
  tile vs full image.
- Under `train.clamp="restore"` nothing interacts.

**A/B.** Run HAT-S or SwinIR-M with LN vs i-LN on the owner's data (5k L1, then the GAN template). Then run fp16 eval of both.

### 2.3 (P1, S-M) NAFNet: the missing 1x restoration baseline

**Paper.** Chen et al., ECCV 2022, "Simple baselines for image restoration". SIDD denoise 40.30 dB at about 65 GMACs, GoPro
deblur 33.69 dB, beating Restormer/MPRNet at a fraction of their compute.

**Mechanism.** A pure-conv U-Net with SimpleGate (x1·x2) in place of activations, simplified channel attention and LayerNorm2d.

**Fit with neosr.** It is AMP-, compile- and channels_last-friendly (no attention, no exotic ops), and it pads to `2^levels`.
T supports scale 1 in paired training (`paired_dataset.py`), but no T arch is a real restoration network.

**C/SP implementation.** `C/archs/nafnet_arch.py` wraps `SP/NAFNet/__arch/NAFNet_arch.py`, a faithful copy of the official MIT code.
Its LayerNorm is a custom `autograd.Function`:

- It uses the deprecated `ctx.saved_variables`.
- Mean and var are computed in the input dtype, which is fp16 under autocast.
- It is opaque to compile (the same class of problem as T M-12).

**Port recipe.**

- Write about 150 lines in `T/archs/nafnet_arch.py`.
- Replace LayerNorm2d with `F.layer_norm` on the NHWC permute.
- Add `nafnet` (width 32) and `nafnet_l` (width 64) rows. C's `nafnet32`/`nafnet64` defaults are enc `[2,2,4,8]`, mid 12,
  dec `[2,2,2,2]`.
- Give the factory a `scale` kwarg that raises unless it is 1, so the ARCHS row can pass the run scale.
- Keep spandrel-compatible key names, so official NAFNet `.pth` files load. That is a free pretrain source.

**Optional P3 variant: CGNet (CascadedGaze, TMLR 2024, `C/archs/cgnet_arch.py`).** It is NAFNet plus a global-context extractor
in the encoder.

- The paper reports +0.09 dB on SIDD and +0.06 dB on GoPro over NAFNet.
- C's port is faithful but has the same custom LayerNorm.
- That gain is not worth a second family. If wanted later, add it as a `gce=True` block option.

### 2.4 (P2, S) R3GAN scalar discriminator

**Paper.** Huang et al., NeurIPS 2024, "The GAN is dead; long live the GAN!". RpGAN + R1 + R2 with a modern ConvNeXt-style D: no
normalisation and no spectral norm. The last conv of each residual block is zero-init, and downsampling uses an anti-aliased FIR.
The paper's stability claim rests on the *combination* of loss, penalties and this D.

**Fit with neosr.** Stage 2 already has `gan_type="rpgan"` and lazy R1/R2 (`T/losses/gan_loss.py:26-74`), but no D designed
for them. metagan's GroupNorm cancels the SN, which is T A-04, the dominant adversarial gradient.

**C implementation.** `C/archs/rpgan_r3gan_scalar_discriminator_arch.py`, written by the owner in January 2026. It is a clean
reimplementation of the reference `Networks.py`.

**Fixes for the port.**

- **Drop the optional StyleGAN dependency.** Remove the `torch_utils.ops` `bias_act`/`upfirdn2d` imports (`:52-58,93-98`); the
  pure-torch fallback is exact.
- **`stage_dtype=float32` does nothing under autocast.** `F.conv2d` is still autocast to bf16/fp16. Either wrap the forward in
  `torch.autocast(..., enabled=False)` (fp32 D, the reference behaviour, safer for R1/R2 double backward) or drop the knob.
- **Remove the duplicated adaptive pool.** It appears at both `:301-305` and `:483-484`. For SR patches the final map is not
  4x4 (gt 128 gives 8x8), so pooling stays, but once.
- The output is a scalar per sample, which T's losses already handle (resgan precedent).

**Evidence.** The paper's results are on FFHQ, ImageNet and CIFAR generation, not SR. The owner's ESRGAN runs with this D
(`experiments/4x_ESRGAN`) peak around 29.47 PSNR and 0.173 DISTS, but those are resumed, multi-loss runs with no metagan
baseline.

**Caveat (reasoning).** A scalar D judges global realism only. U-Net and patch Ds give per-pixel texture signal, which SR usually
benefits from.

**A/B.** Run the realplksr template with metagan + bce, metagan + rpgan + R1/R2, and R3GAN-D + rpgan + R1/R2, and compare
DISTS@8k. Only an A/B decides whether it becomes a template default.

### 2.5 (P2, S) ICNR initialisation of PixelShuffle heads

**Source.** Aitken et al. 2017, "Checkerboard artifact free sub-pixel convolution". C has it in `heart_arch.py:32-60` and
`nerve_arch.py`, and the code is correct. The `weight[i::sub] = base` layout matches PixelShuffle's `c·r²+k` channel order.

**Mechanism.** Every sub-pixel phase starts from the same kernel, so the initial upsampler is nearest-neighbour instead of a
random checkerboard. It is init-only: no keys change and pretrains are unaffected.

**Fit.** It complements the A-07 init work. It is a correctness-style init, not damping.

**Port recipe.**

- Add one helper `icnr_(conv, r)` in `T/archs/arch_util.py`.
- Apply it opt-in from the archs' upsampler builders, or via one `init_icnr` call in `build_network` that walks
  `Conv2d → PixelShuffle` pairs.

**A/B.** Measure early PSNR (500/1000 updates) and the checkerboard at 1k with the A/B harness. Expected effect is small and
early-only.

### 2.6 (P3) Optional additions, in order

- **SCUNet (Zhang et al., MIR 2023; `C/archs/scunet_aaf6aa_arch.py`, aaf6aa fork with SR heads; SP has the 1x original).**
  - It is the reference blind real-world denoiser and pairs with its own degradation pipeline. That pipeline belongs to the
    data lane.
  - Cost: 9.7M params at 1x (bench 27 fps fp16) and 15.2M at 4x. It uses window attention, so it needs T's `window_sdpa` and
    padding.
  - Only worth it after NAFNet, if blind denoising becomes a product.
- **Real-time class.** T already has compact, span/span_fast, spanplus*, sebica, plainusr and microsr, so these add little:
  - **ArtCNN** (`C/archs/artcnn_arch.py`, 60 useful lines, EDSR-baseline style, ReLU, no norm). bench x4 r8f48 132→198 fps
    with channels_last (1.50-1.65x CL gain). No quality numbers anywhere.
  - **RTMoSR** (umzi, `rtmosr_arch.py`, MambaOut-style gated CNN, 2 blocks, Conv3XC/RepConv). bench x4 264-326 fps, the
    fastest non-compact class; channels_last is slower (0.81-0.86x). No quality numbers.
    - It needs T's online reparam (A-01). C uses the old eval-cache pattern.
    - Fix before any use: bench lists 2.96M params at `dim=32, n_blocks=2`; reconcile that with the code.
  - **Presets** (no new code): `ultracompact` (64f/8conv) and `superultracompact` (24f/8conv) on T's configurable compact,
    `esrgan_lite` (32f/12 blocks), and `dat_light` (the official DAT-light; bench 26.64 x4).
- **LKFMixer** (Tang & Hu, arXiv 2508.11391; `C/archs/lkfmixer_arch.py` with umzi's ONNX fixes). A pure-CNN 31x1/1x31 partial
  large kernel.
  - Paper, x4 Urban100: L 26.85 at 927K/49.5G and 194 ms (512², 3090), vs SwinIR-light 26.47 at 1001 ms. LKFMixer-T: 26.23 at 207K.
  - Neosr's plksr_tiny, omnisr, esc_light and hit_srf already cover this class, so add it only if a pure-conv light model is
    wanted.
- **Video SR.** C has TemporalSPAN, TSPANv2 (`temporal_span_v2_arch.py`: 5-frame channel stack, SPAN blocks, checkpointing),
  TSCUNet and TFDAT, plus `paired_video_dataset.py` and `single_video_dataset.py`.
  - In T this needs a new clip dataset, the model step and validation: L cost, owned by the data and trainer lanes.
  - TSPANv2 is the only sensible first target. It is real-time class and its blocks are SPAN, which T already has with
    online reparam; C's `params_updated once per eval session` cache at `:211-213` must not be copied.
  - Do it only if video is a goal.

## 3. Families evaluated and not recommended

| Family (C file) | What / evidence | Why not |
|---|---|---|
| GRL (`grl_arch.py`, SP) | CVPR 2023 restoration transformer. bench grl_b x4 28.53, 13.2 GB, 0.17 fps | Dominated by HAT-L (28.60, 10.4 GB, 0.29 fps) and DRCT-L (28.70), both in T. Heavy anchor/stripe attention |
| Swin2SR (SP) | ECCVW 2022 SwinV2 variant. bench m 27.51 vs SwinIR-M 27.45, slower | +0.06 dB. Its JPEG-SR value is a data question, not an arch one |
| HiT-SIR / HiT-SNG | Siblings of HiT-SRF (ECCV 2024). bench DIV2K 26.71/26.75 vs SRF 26.80 | T has the best of the three (hit_srf + medium/large) |
| EMT | 2024 light transformer. bench 26.98 DF2K, 925K, 3.6 fps | OmniSR in T: 26.95 at 7.8 fps |
| ELAN / ELAN-light | ECCV 2022. bench light 26.54 DIV2K | Superseded by omnisr, hit_srf, esc_light |
| SeemoRe | ICML 2024. SeemoRe-T 26.23 at 232K (LKFMixer paper table) | Same class as lmlt_tiny, plksr_tiny, safmn. Spandrel wrapper |
| DWT (dilated window transformer) | bench 27.81 at 0.15 fps (6.4 s/img) | DAT-2 in T: 27.86 at 0.28 fps. bench lists dwt_s with SwinIR-M's exact param count (table error) |
| SRFormerV2 (`srformerv2_arch.py`, 1640 lines) | TPAMI-extension SRFormer, window 36, embed 240 | No numbers gathered. Heavy, and 36x36 windows without a flash path are expensive. T has srformer + HAT-L/DRCT-L/ATD. Revisit with RIB (section 6) |
| FDAT / TFDAT (stinkybread / Kim2091) | "Simplified DAT": unshifted window + channel attention, AIM. Popular in the community | No controlled numbers. Overlaps dat_* and the many T light transformers |
| DAT-iLN / HAT-iLN / HEART (C) | i-LN variants | Use 2.2 instead. C's iLN damps and is train/reload-inconsistent. HEART is a 2026 community mix of HAT-iLN + RIB, without numbers |
| ESCRealM | ESC real-world preset with UniUpsampleV3, flex attention via triton | T has esc/esc_light/esc_fp/esc_large. Only the upsampler differs |
| SPANF / SPANF3 / SpanPP / DIS / NERVE / GateRV3 / LAWFFT | the-database / Philip / Kim / umzi variants of tiny conv nets | No numbers. Overlap span_fast, compact, sebica. GateRV3 multiplies a param grad by 10 via `register_hook(lambda ...)`, which is hidden LR scaling and does not pickle. SPANF3 bounds attention with tanh "for stability under aggressive losses", which is damping by design |
| Paragon SR/SR2/Diffusion, ElysiumSR, HyperionSR, DetailRefinerNet, MUNet, GFISRv2 (Philip Hofmann) | Bespoke designs described in VRAM tiers | No benchmarks, and their quality claims come from the authors' own descriptions. The owner's gfisrv2a/b runs (29.15-29.33 @2-3k, batch 24) are not comparable to anything |
| DocRes (CVPR 2024) | Document restoration with task prompts | Niche. Needs document data and dewarp tasks |
| HumArch, AutoEncoder | Human-anatomy SR, latent AE | Niche / not SISR |
| MoSR (SP) | umzi, MambaOut-style | T has mosrv2 |
| HyFreqMoE_SR/3/4 (the owner's own, Jan 2026) | MoE + frequency + noise-injection SR. logs: sr4 29.79 @30k (mixed losses). 3B collapsed to 14.03 with router noise 0.07 | The owner decides. As written it conflicts with neosr rules in three ways. (1) The aux loss is injected as a gradient hook on the router logits (`hyfreqmoe_sr4_arch.py:600-614`), bypassing the loss Context; under fp16 GradScaler the injected term is not scaled. (2) A Python `_step` counter drives noise warm-up and is lost on resume and in the EMA copy. (3) The noise injection makes G stochastic. A port would need the aux term as a Context loss and the counter as state |

## 4. Discriminators

| D | Verdict |
|---|---|
| R3GAN scalar (owner) | **P2** port+fix (2.4) |
| MetaGAN2 (`metagan2_arch.py`) | Skip. SN gated blocks with layer-scale 1e-6. T's A-04 probe: it does not reduce the adversarial gradient magnitude |
| MetaGAN3 (`metagan3_arch.py:165-193`, umzi) | Skip, buggy. There is **no output head**, so the "logits" are a 256-channel map at H/32. `blocks[-1]` is unused (`range(len(blocks)-1)`). `GroupNorm(1)` follows SN convs, the same SN-cancelling pattern as A-04 |
| MUNet (Philip) | Skip. U-Net + gradient + FFT + patch branches with attention fusion. No evidence, and the FFT branch has to run in fp32 |
| VGGStyleDiscriminator | Skip. Needs a fixed 128/256 input, ESRGAN-2018 vintage |
| UNetDiscriminatorSN, PatchGAN SN, DUnet | Already in T |
| MultiscalePatchGAN | Skip (H4). It averages away the per-patch map |

## 5. Variants of archs T already has

- **realplksr.** The `layer_norm` variant is item 2.1. `realplksr_tiny`/`large` equal T's `_s`/`_l`. Keep T's GroupNorm
  default (official weights).
- **spanplus / SPANF.** T is better: online reparam (A-01, `7cedb5b`) vs C's eval-cache `Conv3XC` (stale after load/EMA,
  `.data` writes). SPANF is a SPAN with a nearest-neighbour upsampler; T has span_fast.
- **esrgan / rrdbnet.** T is equal or better. C's spandrel wrapper derives the unshuffle from the channel ratio and drops
  `out_nc` (H4). `esrgan_lite` is just a preset (2.6).
- **DySample.** T is better: fp32 grid and a traced normaliser (A-02), where C has a bf16 grid and `pin_memory` per forward.
- **HAT/DAT/FDAT "aligned" presets (C `e8eeb10c`).** These round `embed_dim` up, e.g. HAT-L 180 → 192, so head_dim becomes 32
  instead of 30 (tensor-core and fused-SDPA friendly). They are new models with no official weights. At most a P3 preset idea
  for new from-scratch runs; measure with `window_sdpa` first.
- **compact.** T's is configurable, so ultracompact and superultracompact are ARCHS presets.

## 6. arch_util and approach ideas

| Idea | Verdict |
|---|---|
| i-LN (C `arch_util.py:21`) | **P1**, rewrite from the paper (2.2) |
| ICNR (C heart/nerve) | **P2** (2.5) |
| RIB: rank-factorised implicit position bias (arXiv 2603.06738, used by C `heart_arch.py:63-170`) | P3 research idea. Position features are concatenated onto Q/K, so the bias becomes a dot product and window attention runs on FlashAttention. Shifted windows use padding instead of masks. The paper reports 2.1x faster training and 2.9x faster inference than PFT, and 35.63 dB Urban100 x2 when scaling windows to 96. This is the enabler for large-window transformers in neosr, but it is a new arch (no weight compatibility), so it belongs in a future arch-design stage, not a port |
| `pad_to_multiple`, `use_checkpoint`, mask dtype | Already done in stage 2 (H4 #3/#11/#13, `b253acf`) |
| `arch_info.py` central capability table (1437 lines) | Skip. H4 #19: per-class attributes instead |
| UniUpsampleV3 / SampleMods3 / LDA_AQU / MetaUpsample | Skip (H4: uint8 buffer overflow, caller-dict mutation). "transpose+conv" is checkerboard-prone and has no evidence |
| MagicKernelSharp2021 resampler (paragonsr, munet) | Skip. A fixed-kernel upsample with no shown benefit over bicubic/nearest + conv |
| Spandrel dependency / `store_hyperparameters` | Skip. T's explicit ARCHS table and local helpers are the design rule |

## 7. What neosr already does better (archs)

- **Building nets.** An explicit `ARCHS` lookup table, with the scale kwarg named per arch. C eagerly imports all `*_arch.py`
  files into 3 registries (SPANDREL → ARCH → TESTARCH), at startup cost, with silent name shadowing and a hard
  spandrel/spandrel_extra_arches/timm dependency.
- **Reparameterisation.** Branches train through their fused kernel (A-01) and eval kernels are cached safely. C keeps the
  eval-cache pattern with stale-after-load risk in SPAN, SPANF3, TSPANv2 and RTMoSR.
- **Window attention.** An exact SDPA helper (`window_sdpa`) with bias + shift mask, fixed masks, padding and checkpointing for
  HAT, DRCT, SwinIR, SRFormer and others. C's HAT and DAT use manual attention.
- **FlexNet.** SDPA is fixed (H4 #1); C deleted SDPA.
- **Correctness fixes.** C's own `hit_srf_arch.py:281-326` keeps the dense-bias bug that T fixed (W-03). C has no ASID or
  MFGHMoE at all. C's masks are built on the CPU and copied every forward (archs_window section 5); T builds them on the device.
- **DySample.** The fp32 grid and the fp16 ONNX DySample handling.
- **Export.** Dynamic ONNX for asid, omnisr, ditn, dctlsa, grformer, microsr and mfghmoe (`4ada015`), with export
  capability re-derived per arch.
- **Defaults.** RealPLKSR keeps the official-weight default, where C's `layer_norm=True` default breaks it. `use_reentrant=False`
  and `meshgrid(indexing=)` are used throughout, where C regressed both in dat and hit_srf (H4).
- **Coverage.** T has several archs C lacks: aether, asid, catanet, cfsr, dct, ea2fpn, grformer, hasn, hma, krgn, mfghmoe,
  microsr, msdan, ninasr, plainusr, resgan, swinirdense, desrgan.

## 8. Suggested order

1. **2.1 realplksr `layer_norm` flag.** Measure speed in T (eager + compile), then a template A/B against A-07 GN-gamma init.
2. **2.3 NAFNet.** This is an add-only change: one file, two ARCHS rows, loading of official keys, and a 1x smoke run. It does
   not touch any existing run.
3. **2.2 i-LN.** Add the helper, then wire it into HAT and SwinIR first. A/B it (L1 5k plus fp16 eval), check tiles vs full
   image, then extend to DRCT and SRFormer.
4. **2.4 R3GAN D.** Bundle it with the pending rpgan/R1/R2 A/B, since stage 2 left those opt-in.
5. **2.5 ICNR.** Run a cheap early-PSNR A/B alongside item 1.
6. Treat the P3 items as owner choices: SCUNet if blind denoise, TSPANv2 if video, presets on request.

## Sources

- i-LN: Lee et al., arXiv 2504.06629, https://arxiv.org/html/2504.06629
- RIB: Lee et al., arXiv 2603.06738, https://arxiv.org/abs/2603.06738
- LKFMixer: Tang & Hu, arXiv 2508.11391, https://arxiv.org/html/2508.11391v1
- CascadedGaze: Ghasemabadi et al., TMLR 2024, arXiv 2401.15235, https://arxiv.org/pdf/2401.15235
- NAFNet: Chen et al., ECCV 2022, arXiv 2204.04676. R3GAN: Huang et al., NeurIPS 2024. ICNR: Aitken et al., arXiv 1707.02937
  (cited from prior knowledge; not re-fetched in this lane)
- C benchmark table: `/home/korvin/apps/traiNNer-redux-dev/docs/source/benchmarks.md` (4x rows lines 10-99, 1x rows 280-372)
