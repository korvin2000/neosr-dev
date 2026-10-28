# Stage 3 lane: losses and metrics (traiNNer-redux vs neosr)

Scope: `C = /home/korvin/apps/traiNNer-redux-dev/traiNNer` (`losses/`, `metrics/`, `archs/{lpips,topiq}_arch.py`,
`models/sr_model.py` and the `options/` configs) against `N = neosr/{losses,metrics}` on `refactor/stabilize` (e0019ee).
Starting points: `analysis/competitor/H1.md` (ranked list), `stage2/analysis/{losses,metrics,stability}.md`.

Evidence tags: **[code]** read this session; **[probe]** a capped CPU probe run this session
(`scratchpad/stage3_lm/p1..p3`, the owner's 67-image held-out split, GT 256 / LQ 64);
**[paper]** paper text fetched and read this session; **[paper, memory]** from memory, not re-fetched;
**[C-config]** traiNNer's own templates/configs (no ablations exist in its repo); **[reasoning]**.

## 0. Bottom line

- **Nothing in traiNNer's loss/metric zoo is a must-have (no P1).** Stage 2 already took everything with a clear
  mechanism: MS-SSIM guard, DISTS normalisation, consistency L* fix and fp32, VGG loader, NCC fix, rpgan + lazy R1/R2,
  GAN start/ramp. What is left is either cheap and modest, or expensive and speculative.
- **Worth doing (P2, both S, metrics side):** (1) the PIPAL-trained TOPIQ-FR weights as an option of the existing
  `calculate_topiq` (same network, one weight file); (2) an LPIPS metric, as an independent check next to DISTS
  (needed whenever `dists_loss` is in the objective) and for comparability with published numbers.
- **Nice (P3):** `hinge` GAN type; a generic per-loss `start_iter`/`ramp_iters`/`end_iter` on the `Loss` base (the GAN
  one, generalised); D feature matching (M); MS-SSIM reflect padding (opt-in); TOPIQ-NR for GT-less test sets;
  PSNR (log-MSE) loss for restoration tasks; AESOP and gradient-variance as ideas only.
- **Everything else: skip.** Most of traiNNer's extra losses are dead (ConvNeXt/DINO perceptual have no gradient),
  buggy (A-DISTS, MS-SSIM-L1, LumaLoss, feature matching wiring), redundant with what neosr has (HFEN, Sobel,
  Laplacian, Cosim, ColorLoss, HSLuv), niche (anime, JPEG-block TV, CLIP contrastive, segmentation BCE-Dice), or
  unsuited to aligned SISR (contextual, FLIP). Their configs carry no ablations, and 11 of its losses get weight²
  (sec. 6), so traiNNer's recommended weights are not evidence either.
- **Two neosr-internal follow-ups found on the way (not ports):** LDL at weight 1.0 is inert (p3: its pixel gradient
  is 1.4 % of L1's at 28 dB, 0.03 % at 42 dB), so H1 #5 (EMA-refined LDL, +1 G forward per step) should **not** be
  ported; instead A/B dropping LDL from the templates (stage-2 L-07, still open). MS-SSIM's zero padding is a small,
  measured bias (p1), not a bug.

## 1. H1's ranked list, re-verdicted against today's neosr

| H1 # | item | neosr now ([code]) | verdict |
|---|---|---|---|
| 1 | MS-SSIM NaN guard | done: per-(N,C) mean then `clamp_min(1e-6)`, mul chain (no `prod` sync), separable window (`N/losses/ssim_loss.py:135-152`) | closed. Optional border handling: sec. 5, P3 |
| 2 | consistency: L*, buffers, cosine | L* low branch fixed, fp32, fixed separable blur buffer, buffers. Cosine still gated `torch.where(cosim < 1e-3, ...)`, luma cosine over H (`consistency_loss.py:198-203`) | closed except the cosine; stage-2 L-14 measured it at ~1e-5 (always on, negligible). Leave it or drop it; not a port |
| 3 | DISTS normalisation + loading | done: ImageNet mean/std, `map_location="cpu"`, alpha/beta frozen in ctor and by the builder, loud download failure; streaming metric twin | closed; neosr better than C |
| 4 | VGG extractor | done: local-weights branch fixed, explicit vgg19-only `ValueError`, `ReLU.inplace=False` instead of clones, no CUDA buffers | closed |
| 5 | LDL EMA-refined map | not ported. neosr: guarded `var**0.2`, box-filter moments in fp64 (better than C's unfold) | **skip** (changed from H1 "med"): LDL itself is inert at template weight (p3, sec. 5; stage-2 L-07). The EMA mask can only zero more pixels, i.e. make it smaller, at +1 no-grad G forward per step. First A/B LDL off vs a re-weighted LDL; revisit only if LDL shows an effect |
| 6 | hinge / softplus / R1 | lazy R1/R2 done (fp32 eager D, SDPA math, per sample). softplus is BCE-with-logits (`softplus(-x)`, `softplus(x)` are exactly BCE with labels 1/0), so neosr's `bce` already is it | hinge only: **P3** |
| 7 | chc honours `reduction` | done | closed |
| 8 | loss-weight scheduling | GAN only (`start_iter`, `ramp_iters`, `Loss.active`) | generalise to the `Loss` base: **P3** (sec. 3) |
| 9 | gradient-variance loss | not ported | downgraded to **idea / P3** (sec. 3: the paper's evidence is weak) |
| 10 | relativistic pairing (R3GAN) | done (`gan_type = "rpgan"`), true pairwise RpGAN | closed; neosr better than C's `R3GANLoss` |
| 11 | D feature matching | not ported | **P3, M** (sec. 3) |

## 2. Metrics

### 2.1 Inventory

| metric | traiNNer ([code]) | neosr ([code]) |
|---|---|---|
| PSNR, SSIM (RGB or Y, crop) | numpy + `_pt` twins; the `_pt` ones skip uint8 quantisation and add `mse + 1e-8` (stage-2 metrics.md §6) | numpy reference + exact on-device twins on quantised tensors, fp64 statistics, one sync per event |
| DISTS | builds `DISTSLoss` **per image per call** (`C/metrics/dists.py:18`) | built once, parked on CPU between events, streaming per VGG stage (-28 % peak at 1080p) |
| TOPIQ-FR (CFANet R50, KADID weights) | builds `CFANet()` and loads the checkpoint **per image** (`C/metrics/topiq.py:18`); needs `timm` | same network without `timm`, built once |
| TOPIQ-NR (CFANet R50, KonIQ weights) | `calculate_topiq_nr`, per-image build | missing |
| LPIPS (alex / vgg / squeeze, v0.1) | `C/archs/lpips_arch.py` (port of richzhang's code, weights shipped, `load_state_dict(strict=False)` silently tolerates a mismatch), per-image build | missing |
| validation loop | OOM -> retry at tile 256; any other exception is logged and `continue`d, so a failed image silently drops out of the mean and the best-model tracking (`C/models/sr_model.py:1241-1260`) | errors are explicit; OOM guidance (stage 2) |

traiNNer's GAN templates validate with TOPIQ-FR only (PSNR/SSIM/LPIPS/DISTS commented out) [C-config].

### 2.2 What correlates with perceived SR quality ([paper])

TOPIQ paper (Chen et al., arXiv 2308.03060, Table II, PIPAL = GAN/SR/denoising outputs with 2AFC labels), SRCC:
PSNR 0.249, SSIM 0.361, MS-SSIM 0.369, LPIPS-VGG 0.595, DISTS 0.655, AHIQ 0.813, TOPIQ-FR trained on PIPAL 0.813
(PLCC 0.830). PIPAL is intra-dataset there (trained on its train split), so 0.81 is an upper bound, not a guarantee on
the owner's content; the paper does not report KADID-trained TOPIQ on PIPAL. Ding et al. (IJCV 2021, arXiv 2005.01338,
Fig. 8c), networks *optimised* for each metric and ranked by humans for SR: DISTS 2.50 > LPIPS 1.88 > MS-SSIM 1.20 >
MAE 1.02 > NLPD > MAD > FSIM, VIF, VSI, GMSD, CW-SSIM (negative); DISTS is statistically better than LPIPS for SR.

Consequences for neosr:
1. PSNR/SSIM are near-useless selectors for GAN runs (SRCC 0.25/0.36 on PIPAL). Stage 2 already reported DISTS for
   GAN A/Bs; that was the right call.
2. **TOPIQ-FR PIPAL weights, P2, S.** neosr's `topiq` is the same CFANet-R50 FR network; only the checkpoint
   differs (`cfanet_fr_pipal_res50-69bbe5ba.pth`, URL in pyiqa's `topiq_arch.py:28`, local checkout
   `/home/korvin/apps/IQA-PyTorch`; check the Hugging Face mirror name before switching URLs). Recipe: a `dataset =
   "kadid" | "pipal"` kwarg on `calculate_topiq(_pt)`, the net cache keyed `topiq_<dataset>`, default unchanged
   (`kadid`), strict load as now. PIPAL contains SR/GAN distortions; KADID only synthetic ones, so PIPAL weights are
   the better-matched judge for GAN SR. Risk: PIPAL's 2020-era GAN artefacts may not cover modern ones; validate by
   rank agreement first (sec. 7).
3. **LPIPS metric, P2, S.** Not a better selector than DISTS (lower on both PIPAL and Ding's SR ranking), but (a) it
   is an *independent* perceptual check (AlexNet, different training data) when `dists_loss` is part of the
   objective, where the DISTS metric is no longer an unbiased judge; (b) LPIPS-Alex is the number SR papers report;
   (c) it is the cheapest learned metric (AlexNet trunk, ~2.5 M parameters). Rewrite from the reference (richzhang
   LPIPS v0.1, BSD-2) in ~60 lines: torchvision AlexNet features (weights already in the torch hub cache), lin
   layers from the official v0.1 file, `strict=True`, input `/255 -> [-1, 1]`, ScalingLayer, unit-normalised
   features, spatial mean. Plug into `METRICS` + `_metric_builders` with the existing build-once/park-on-CPU and
   `inference_mode` path; `better = "lower"` in the lookup. Do not port traiNNer's per-image construction or its
   `strict=False` load. Default net: alex (the standard); vgg as an option.
4. **TOPIQ-NR (KonIQ), P3, S-M.** Only for GT-less evaluation (`test.py` on real inputs). KonIQ is in-the-wild
   camera quality, not SR artefacts, and NR metrics reward over-sharpening/hallucinated texture [reasoning]; never
   a model-selection criterion when GT exists. Needs the `use_ref=False` CFANet variant (different `weight_pool` /
   `dim_reduce` widths) and a metric path that takes one image.
5. Skip as metrics: A-DISTS (bare `except` fallbacks, NaN on 1-element windows, H1), FLIP (rendering metric),
   MS-SSIM metric (adds little over SSIM/DISTS).

## 3. Loss candidates (beyond H1, or re-verdicted)

| candidate | value for SISR/restoration + evidence | traiNNer impl ([code]) | approach | cost | prio |
|---|---|---|---|---|---|
| Generic per-loss schedule (`start_iter`, `ramp_iters`, `end_iter`) | "L1 first, add perceptual/GAN later" and late disable in one run. No quality evidence: stage-2 GAN start 1000 + ramp 1000 gave no gain (28.69/0.1900 vs 28.81/0.1896); traiNNer configs schedule weights but carry no ablation [C-config] | `iterative_loss_wrapper.py` 290 lines, linear/cosine/step/exp-decay, `target_weight`, `hasattr(loss, "loss_module")` branches in the loop | rewrite: move `gan_loss`'s `start_iter`/`ramp_iters` into `Loss` (builder pops the keys for any term; `active()` and a host-float factor in `compute()`), add `end_iter`. Linear only | S | P3 |
| Hinge GAN | standard alternative to BCE; no SR evidence it beats BCE/rpgan [reasoning] | `gan_loss.py` fine | port idea: one branch next to `rpgan` (D `relu(1-r)+relu(1+f)`, G `-f.mean()`) | S | P3 |
| D feature matching | pix2pixHD-style stabiliser [paper, memory]; not standard in SISR; D arch interface change | broken wiring (only `munet_arch_old` exposes features), silent layer skips, resizes mismatched maps, CPU zero tensor fallback | idea only: D archs get an explicit features output; a `Loss` reads them from Context; reuse the G-phase D forward (for `rpgan` the real side is already computed under no_grad) | M | P3 |
| PSNR loss (mean of per-image log MSE) | the objective NAFNet/HINet train restoration with [paper, memory]; per-image normalised gradient (geometric mean of MSE). No SR ablation vs L1 | `basic_loss.py:186-215`, device hack, otherwise right | port+fix (5 lines, fp32, eps) | S | P3 (restoration users) |
| AESOP (auto-encoded supervision) | replaces L1 in perceptual SR by a distance after a frozen AE whose decoder is a pretrained fidelity SR net, so the pixel term stops fighting the GAN [paper, memory: CVPR 2025]; claims better LPIPS/DISTS at similar PSNR | `aesop_loss.py` + `autoencoder_arch.py` (RRDB encoder, decoder = any SR arch), the AE needs its own training run | idea only; needs an AE-training model type and per-scale weights | L | P3 |
| Gradient-variance loss | [paper] (arXiv 2202.00997): baselines far below published numbers (EDSR x4 Set5 30.47 vs 32.46 in the EDSR paper, x3 below x4); EDSR+GV loses PSNR at x3/x4 on 4 of 8 rows; L1 31.00 -> L1+GV 31.12 dB (one run), while an SSIM loss alone gives 31.77. Gradient-based objectives (GMSD) rank near the bottom as SR objectives in Ding 2021 [paper] | simple; `loss_weight` applied outside; zero-padded Sobel; drops the unfold remainder | idea only (rewrite S if ever A/B'd: one `(2,1,3,3)` Sobel buffer, reflect pad, `unfold`) | S | P3 / idea |
| HSLuv loss | colour term; traiNNer's GAN templates use it at 1.0 [C-config], no ablation. Since its Feb-2026 fixes it is numerically sane: [probe p2] finite gradients, hue-term max/median ~40x, no blow-up near grey | `basic_loss.py:324-430`, `utils/hsluv.py` (batch-global `max` fill couples images) | skip: neosr's consistency (Oklab chroma + L*) covers the role | - | skip |
| CHC / Charbonnier / Cosim | equal: neosr `chc_loss` (fp32, reduction honoured, `clip_min=0, clip_max=inf` = Charbonnier, `loss_lambda` = cosine term) | C's CHC weight² (sec. 6), cosine eps 1e-20 | skip (optional `charbonnier` alias: usability only) | - | skip |
| MS-SSIM-L1 (Zhao 2017) | the *principle* (MS-SSIM + L1 beats either alone for restoration) [paper, memory] is already the neosr template (L1 1.0 + MS-SSIM 1.0) | grouped-conv bug mixes scales across RGB (H1), device fixed at init | skip | - | skip |
| Focal frequency | identical code on both sides | same | skip (neosr doc: `ave_spectrum=True` is not upstream's default; stage-2 L-10) | - | skip |
| ConvNeXt / DINO perceptual | would be new backbones; no SR evidence | **still dead**: `_extract_feats` is `@torch.no_grad()` and used for the prediction (`convnext_perceptual_loss.py:118,164`, `dino_perceptual_loss.py:90,222`), so these terms never had a gradient. ParagonSR2 configs use ConvNeXt at 0.5 as their "core structural loss" [C-config] | skip; neosr's FDL(dinov2) is gradient-correct | - | skip |
| HFEN, Sobel edge, Laplacian pyramid, checkerboard, TV, adaptive block TV | high-frequency / smoothness terms overlapping FF, wavelet D view, FDL and MS-SSIM; TV erodes texture; block TV is JPEG-specific; no SR evidence | HFEN rebuilds its LoG per forward; Laplacian has no prefilter; checkerboard is O(scale^4); most get weight² | skip | - | skip |
| Contextual loss | for misaligned pairs (Mechrez 2018) [paper, memory]; aligned SISR gains nothing over L1/perceptual | random 100x100 sampling (unseeded), a 10^4 x 10^4 affinity per layer per image (~400 MB fp32) | skip (idea only if a misaligned real-pair dataset appears) | - | skip |
| FLIP | rendering-oriented error (67 ppd viewing model); no SR evidence | rebuilds filters per call, `device="cuda"` zeros, input clamp | skip | - | skip |
| A-DISTS | IQA metric variant; unverified as an SR loss | bare `except` -> global statistics, NaN on 1-element windows (H1) | skip | - | skip |
| Line distiller, perceptual anime (Danbooru R50) | anime-domain only; weights of unclear provenance (line distiller) | ok-ish | skip (idea if the owner trains anime models) | - | skip |
| Structure-texture, CLIP contrastive, BCE-Dice, ModelAsLoss, Bicubic/Average/Color/Luma, ColorLoss | homegrown or off-task; LumaLoss keeps the Y^2 L* bug | - | skip | - | skip |
| DynamicLossScheduler | rejected by the owner (non-stationary objective, syncs) | 905 lines | skip | - | skip |

## 4. What neosr already does better

1. Weighting contract: the weight is applied once, in `Loss.compute`. traiNNer multiplies `loss_weight` in the loop
   *and* inside CHC, Huber, consistency, HFEN, TV, adaptive-block TV, Laplacian, checkerboard, contrastive, ConvNeXt
   and DINO ([code] `grep "self.loss_weight"`, `sr_model.py:845-866`): weight², e.g. its consistency 0.25 is 0.0625.
   Dict-valued losses also run `if loss_val > 0` on a tensor (a host sync per sub-term).
2. Losses read inputs from Context; traiNNer's G step is an `isinstance`/`hasattr(loss, "loss_module")` chain per
   loss type (`sr_model.py:690-825`).
3. MS-SSIM: RGB with the luminance term, per-(N,C) guard, separable window, no `prod` sync. traiNNer's default is Y
   only, no luminance, per-pixel ReLU, `[0,1]` input clamp and `1 - clamp(score, 0, 1)`.
4. Consistency: correct L*, fp32, fixed blur buffer, `brightness = 1.0` (traiNNer's 0.9 darkens the target; its luma
   cosine over a size-1 channel is identically 0).
5. DISTS: normalised, frozen, built once, streaming metric; traiNNer builds it per validation image.
6. FDL(dinov2): gradient-correct, cached position encoding, optional checkpointing; traiNNer's DINO/ConvNeXt
   perceptual losses have no gradient.
7. LDL: fp64 box-filter moments (no k² unfold copy), guarded `var**0.2`; traiNNer has the unguarded power (NaN grad
   at zero variance).
8. NCC: per-(N,C) correlation with an eps; traiNNer's `reshape(C, -1)` mixes batch and channel, no eps.
9. GAN: true RpGAN, lazy R1/R2 in fp32 eager D (fails loudly without double backward), one concatenated D forward,
   `start_iter`/`ramp_iters` inside the loss; traiNNer's `R3GANLoss` is relativistic-average hinge with a silent
   no-op penalty fallback and `.item()` syncs.
10. Metrics: exact quantised on-device PSNR/SSIM, nets built once and parked on CPU, one sync per event; traiNNer
    rebuilds DISTS/LPIPS/TOPIQ per image and silently drops failed images from the mean.

## 5. Probes (CPU, capped 4 GB, one at a time; 8 held-out pairs, GT 256)

**p1, MS-SSIM border handling** (`p1_mssim_pad.py`). neosr pads the Gaussian window with zeros ("same"); traiNNer
filters "valid" when the map is at least 11 px; reflect is the third option.

| pred | variant | 1-MS-SSIM | grad share in the 8-px border (area 12.1 %) | cos(grad, zero) |
|---|---|---|---|---|
| bicubic (28.4 dB) | zero (now) | 0.0634 | 3.2 % | 1 |
| | reflect | 0.0701 | 11.6 % | 0.960 |
| | valid | 0.0724 | 1.3 % | 0.995 |
| 0.5 gt + 0.5 bicubic | zero / reflect / valid | 0.0139 / 0.0154 / 0.0158 | 3.1 / 11.6 / 1.2 % | 1 / 0.958 / 0.994 |

Zero padding lowers the value ~10 % and gives the outer 8-px ring a quarter of its fair gradient density; "valid"
(traiNNer) is no better (it starves the ring further). Coarse scales are nearly saturated at this patch size (cs at
32/16 px: 0.998/1.000), so MS-SSIM here is effectively a 3-scale SSIM. Verdict: if touched, reflect padding as an
opt-in (objective change, A/B); P3. Not worth a default flip on this evidence.

**p2, traiNNer HSLuv** (`p2_hsluv.py`, its conversion imported from `C/utils/hsluv.py`). Finite gradients for
bicubic, noisy and near-grey predictions; |g| max/median: hue 44x, saturation 8x, lightness 1.4x (L1: 1x). It is a
working colour loss now, but nothing neosr's consistency loss lacks.

**p3, pixel-gradient size of the template's cheap terms at weight 1.0** (`p3_ldl_share.py`, norms of d(term)/d(sr)):

| pred (PSNR) | L1 | MS-SSIM | consistency | LDL | LDL / L1 |
|---|---|---|---|---|---|
| bicubic (28.4 dB) | 8.0e-4 | 5.2e-3 | 5.9e-4 | 1.1e-5 | 1.4 % |
| 0.5 gt + 0.5 bic (34.5 dB) | 8.0e-4 | 2.3e-3 | 5.9e-4 | 2.1e-6 | 0.26 % |
| 0.8 gt + 0.2 bic (42.4 dB) | 8.0e-4 | 8.2e-4 | 5.9e-4 | 2.3e-7 | 0.03 % |

LDL's map scales as residual^2.4, so at weight 1.0 it is inert next to L1 (GAN-level residuals raise it a few times,
still a few percent). The EMA refinement could only shrink it further. Side fact: MS-SSIM carries 6.5x L1's pixel
gradient early and ~1x late, so the template's pixel anchor is MS-SSIM-dominated early (consistent with Zhao's
0.84 : 0.16 mix, no action).

## 6. Fit with the owner's rules

- LPIPS, TOPIQ-PIPAL, TOPIQ-NR: one `METRICS` row and one builder each; no registry; nets cached as today.
- Generic schedule and hinge: inside `Loss` / `gan_loss`; the trainer stays if-free (`Loss.active` already exists).
- Feature matching: needs a D features output; the loss reads it from Context. Breaks no checkpoint (no new params).
- AESOP: a frozen AE inside a `Loss`; the AE pretraining is a separate model type (the large part).
- Old `net_*.pth` are untouched by every item here (losses and metrics hold no generator state).

## 7. Suggested order and checks

1. TOPIQ-PIPAL option + LPIPS (S each). Before choosing a selection metric, score the stage-2 A/B checkpoints (P/Q
   sets) with DISTS, LPIPS, TOPIQ-KADID and TOPIQ-PIPAL and compare their rankings with a visual check of a few arms:
   the metric that agrees with the owner's eye picks models. Keep selection off any metric whose network is also a
   training loss.
2. Template A/B (neosr-internal, from this lane): LDL off vs on (and, if wanted, a re-weighted LDL); drop it if tied.
3. P3 items only on request: generic schedule, hinge, PSNR loss, MS-SSIM reflect padding, feature matching.

Not verified here: TOPIQ-PIPAL and LPIPS rankings on the owner's content (needs step 1); AESOP and NAFNet-PSNR-loss
claims (memory, not re-fetched); the Hugging Face mirror filename of the PIPAL weights.
