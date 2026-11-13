# HR data, sampling, crop/batch and degradation

Study section, group G1 (bundle `experiments/study/bundles/g1_data_degradation.json`: 266 claims, 22 conflicts,
16 gaps). Nothing here is a decision; the owner chooses. Conventions:

- Community anchors are `ALIAS:LINE` of the knowledge base (`/home/korvin/apps/Trainings/tools/aliases.tsv`); the
  evidence label of the bundle is kept: **official** (wiki / author card), **release** (a released config or
  dataset card), **discord** (a cited message, usually one), **code** (inspected upstream code), **paper**,
  **generic** (LLM boilerplate, numbers not empirical), **hypothesis**, **design** (a synthesis doc's own choice).
- Scale tags: redux = traiNNer-redux keys, basicsr = Real-ESRGAN/BasicSR keys, neosr = NeoSR keys. Degradation
  probabilities and ranges are quoted in the source trainer's keys; they port to neosr-dev only after the key
  semantics are checked (part 4), and loss weights never port as values.
- neosr-dev claims carry `path:line` (repository root `/home/korvin/apps/neosr-dev`).
- **AU** (arm unit) = one 0 -> 20k arm of the A/B template = ~37 min on the RTX 4090 (batch 8, LQ 64, paired,
  95-110 ms/update). An 8k screen is ~0.4 AU. A D10 decision needs the 12k-20k window; a default additionally
  needs a second seed and a paired 20k -> 100k run (~2.3 h per arm at template speed).

## 1. What the community converges on

How thin it is, stated first: for this group the knowledge base has **two released degradation configs of one
author** (a 1x "sharpen" and a 1x "restore" model, both redux), a handful of dataset cards, three official wiki
rules, and single Discord observations. There is **no controlled ablation** of blur dose, severity mix, crop size,
sampling quota, kernel, USM or noise anywhere in the corpus (gaps: PH:707, DX:137, PG:452, DR:105, PG:101,
PG:127). Every numeric severity mix and every quota is a self-declared hypothesis or a synthesis doc's design.

### 1.1 Official wiki rules and released configs / dataset cards

- **Crop keys are LR-side in NeoSR (`patch_size`) and redux (`lq_size`); legacy BasicSR `gt_size` is HR-side.**
  LR 64 = HR 128 at x2, 256 at x4 (official, HB:287-295, DX:395-399; neosr default history patch 48, batch 6 -> 8:
  discord, PH:1008). Redux default `lq_size` 96, `batch_size_per_gpu` 8, crop a multiple of 8 (official, RX:168,
  RX:173, RX:175); the upscale-hub neosr guide asks a multiple of 16 (official, UH3:153).
- **Fidelity stages keep the batch and `lq_size` of the 2x scratch run, taken from the official paper configs; the
  degraded/GAN stage "benefits from larger `lq_size`, and batch size can be reduced to allow it"** (official,
  redux, TG:12, TG:22, TG:31). TG gives no crop number and no size of the benefit; "LR 64" for fidelity stages is
  the example configs of a synthesis (HB:446, HB:538), not TG.
- **Larger batch helps training from scratch; reach it with accumulation** (batch 8 x `accum_iter` 4 = 32 on one
  4090; official, redux, TG:15-16).
- **Store 512 px tiles and crop randomly inside them** (tiling: official/tooling, RX:129-137, 512 is the
  example; random crop inside: discord, DX:237-241: a crop equal to the stored tile removes the spatial
  variation). 512 px caps the LR crop at 128 for x4 (PF:82). UH3:22 is about removing duplicates (czkawka), not tiling.
- **Data set size caps of the wikis: about 5,000 images for light archs, 5,000-7,000 in general** (official,
  UH3:127, RX:121). They count images, not tiles.
- **Keep natural depth of field in HR** (official, UH3:48; repeated HB:97, FG:122). The same page records the
  dissent: its author says this "diverges from musl's perspective", whose example HR is "sharp all the way
  through" (UH3:33, UH3:48); an official-vs-official disagreement, not a consensus.
- **Untuned OTF over-degrades and loses detail** (official, RX:105, UH3:111). **NeoSR must not mix paired and
  OTF in one `model_type`** (official only as reported by HB:236 and HB:646, citing the NeoSR Configuration
  Walkthrough wiki, which is not copied into the KB). "One degradation mode per run" is a synthesis doc's wording
  (design, DR:86), not a wiki rule; UH3:105 even suggests mixing the three ways of *making* paired LR.
- **The only released recipe for a detail ("sharpen") model, 1x, redux keys** (release, PH:722-763, values
  verified at source): `blur_prob` 0.85, `blur_sigma` [0.2, 2.2], `blur_kernel_size` 7, kernels
  iso/aniso/generalized_iso/plateau_iso at 0.6/0.2/0.1/0.1; second pass sigma [0.1, 1.2], `blur_kernel_size2` 9; `sinc_prob` 0.05,
  `final_sinc_prob` 0.5; `gaussian_noise_prob` 0.3, `noise_range` [0, 3], `poisson_scale_range` [0, 0.5],
  `gray_noise_prob` 0.2; `jpeg_prob` 0.6, `jpeg_range` [75, 100] / [80, 100]; `resize_prob` [0.25, 0.5, 0.25],
  `resize_range` [0.25, 1.7] / [0.3, 1.6]; `high_order_degradation` True, `lq_usm` True, `lq_size` 96,
  `queue_size` 192. The second-pass blur probability is not in the snapshot. **Key semantics (code, two local
  traiNNer-redux checkouts, 2024-12 `d51c1954` and 2025-12 `29006500`; the release's own redux version is
  unknown):** `blur_kernel_size` 7 / `blur_kernel_size2` 9 are read but never used, the size is drawn from
  `kernel_range` (default 5..17; `traiNNer/data/realesrgan_dataset.py:81,103,150`), which the snapshot does not
  show; an absent `blur_prob2` defaults to **0** (`traiNNer/utils/redux_options.py:579-584`), so if the config
  left it out the second blur never ran; `blur_prob`, `blur_prob2`, `jpeg_prob`, `jpeg_prob2` are one draw per
  batch (`traiNNer/models/realesrgan_model.py:172,218,227,298`), and an absent `jpeg_prob2` defaults to 1.
- **Its sibling cleanup ("restore") release, 1x, redux keys** (release, PH:617-658): `blur_prob` 0.0, noise
  selector 0.6 with Gaussian [0, 18] / Poisson [0, 2.0], gray 0.35, JPEG 0.5 at [50, 100] / [55, 100],
  `final_sinc_prob` 0.8, `lq_usm` True. Reading of the two (design, PG:121, PH digest): the task is separated
  mainly in the degradation. The loss stacks differ as well (redux weights, not portable: DISTS 0.45 vs 0.25,
  perceptual 0.07 vs 0.02, HSLuv 0.6 vs 1.0, LDL 0.35 vs 0.5; PH:709-716 vs PH:604-611), so "not in the loss
  stack" overstates it. Both are milder than the Real-ESRGAN defaults (noise [1, 30], JPEG [30, 95], blur sigma
  [0.2, 3]; official via generic restatement, SKG:24-25, not re-verified upstream).
- **Explicit variant coverage instead of always-on degradation**: RealWebPhoto v4 merged 3 upload chains x
  {none, blur, noise, both} = 12 LR variants into one LR set (release, PH:308-309; discord, RP:52-54); offline
  lens blur radius [2, 3] (v4: PH:306); the ~4 components and the sharp-and-blurred duplicate of every HR are
  documented for v2 (PH:295); "never always-on" is the playbook's design principle drawn from these notes
  (PH:328). The first version had every LR recompressed, no clean baseline (release, PH:302); v4 has a
  downscale-only variant (PH:273).
- **Strict HR filtering yields** (reports, not targets): 56K+ images -> 4,297 tiles of 512 px with ~10 IQA metrics
  (release, PH:111-115); ~50,000 sources -> ~3,000 tiles (discord, DX:205). Metric names and thresholds are given
  nowhere. The BHI card: dropping the worst tiles helped more than keeping only the best; criteria blockiness,
  quality, complexity (release, FG:65, FG:134; card not opened by the KB).
- **A 2x live-action SPAN release trained for compression/rescale/halo repair but explicitly not for denoising**,
  to keep grain and micro-texture (release, DX:155-157; numbers at DX:515: 48 channels, batch 20, GT 128, 490k
  iterations).
- **Siax v1 card** (official, SI:5-25): ESRGAN x4, HR crop 256 (= LR 64), batch 2, flip/rot only, 175k
  iterations, DIV2K train+valid, 2,700 images of which 900 unique (SI:12-13); the author's detached line
  "mitchell/bicubic/box randomized" (SI:25) has no weights and its link to this model is uncertain.

### 1.2 Cited community reports (mostly one message each)

- **Blur is a two-sided lever; probability and strength are separate knobs**: no blur -> the model cannot deblur,
  excessive blur erases HF and favours smoothing (discord, DX:139-143; a 2021 workflow, no numbers).
- **When a model over-restores or over-smooths mild inputs, reduce the degradation tail** (release/discord,
  PH:137, PH:285, RP:58); "before adding losses" is a synthesis reading (design/hypothesis, PG:138, FG:413), the
  sources only report reducing strength between versions.
- **More context helped a blur task**: HR crop 128 -> 256 gave a "substantial improvement" on an RGT lens-blur
  model; HR 512 fit only batch 1 and is not a demonstrated further win (discord, DX:403-407). One `LQ192/batch3`
  comparison improved lines but worsened an eyelash, confounded (discord, RG:104).
- **A model trained on TOPIQ-filtered, sharper data reconstructed detail better than clean-data scratch training,
  then was detuned at LR 1e-5 on cleaner HR** (discord, one interim report, DX:279); the same circle doubts strict
  multi-IQA filtering (it may select a sharpened, haloed look; an "artifact-free" filter rejected nearly
  everything; discord, DX:197-201, RG:99). NR-IQA rewards sharp edges and rejects natural bokeh (discord, PF:83).
- **Sharpened targets were overdone by their own author**, who wanted to fine-tune back on unsharpened data
  (release/discord, DX:229-233); the one paired 12-hour comparison preferred "no synthetic noise and no USM
  targets", two changes at once (release, RG:141, FG:161).
- **A broken LR set makes every run look wrong**: one case was fixed by regenerating the LR (discord, PH:1010).
- **Batch below 8 alone caused no NaN; suspect the D learning rate** (discord, PH:1009).
- **Mixup + CutMix are used by one prolific author in NeoSR** (discord, RP:209); UltraSharp abandoned Mixup for
  stability (release, DX:511).
- **Offline pairs give inspectable, reproducible LR; OTF gives wider sampling and is not inherently harsher; save
  the sampled LR either way** (discord, DX:171-181).
- **Real-ESRGAN's clip-then-round to 8 bit changes weak-noise statistics; reproduce it** (discord, RP:96; code,
  RG:90).
- **Mitchell as LR kernel: negative training reports** (consistently bad; "ringing yet too blurry"; discord,
  DX:91, DX:95) against use in working pipelines (SI:25, RP:102, PH:1161).

### 1.3 Inspected code facts (Real-ESRGAN family; unpinned live repo, FG:525)

- `gaussian_noise_prob` selects Gaussian versus Poisson; 0 selects Poisson, it does not disable noise (code,
  FG:165). Noise is therefore on every sample in those configs.
- The LQ is synthesised from the sharpened `gt_usm` independently of the three loss-target USM flags (code,
  FG:166); final resize + sinc and JPEG run in two orders; the pair queue raises within-batch diversity because
  some resize decisions are shared across a batch (code, FG:167, FG:169, RG:63).
- Checked here against two local traiNNer-redux checkouts (2024-12, 2025-12; competitor code read for key
  semantics only): the selector semantics hold (`realesrgan_model.py:201-216`, noise on every sample), and
  redux turns upstream's always-sharpened LQ source into a switch, `lq_usm` (default False) with a random USM
  radius from `lq_usm_radius_range` (default 1..25; `realesrgan_model.py:152-164`). The releases' `lq_usm` True
  therefore restores the upstream Real-ESRGAN behaviour, at a radius range the snapshot does not show; that is a
  likely reading of the "unexplained" flag, not the author's stated reason.

### 1.4 Hypotheses and designs (not evidence; listed because the strategy docs lean on them)

- Severity mixes: 25/60/15 (FG:173), 0.40/0.45/0.15 (PF:139-145), 20-35/25-35/25-35/5-15 (HB:250-253),
  30/30/30/10 (DR:92), 15/40/10/25/10 by bucket (PG:127). Each source calls its own mix a hypothesis.
- Smooth/defocus/near-clean crop quota >= 25 % (design, PG:101); texture-only sampling teaches texture on
  walls, skin and bokeh (hypothesis, PF:53, FG:126, RG:48) versus "oversample HF patches" (generic/hypothesis,
  SKG:69, SX:57-60).
- "A crop must cover one texture period; try LR 96 for fur/fabric before adding losses" (generic, SKT:17, SKR:23;
  hypothesis, FG:260-265).
- HR rules: reject AI-generated, beauty-filtered, haloed, denoised, upscaled sources; never "fix" HR afterwards;
  split by source group before tiling; dedup; cap tiles per parent (hypothesis/discord mix: SX:40-43, RG:44,
  FG:121-132, PF:52, PF:74, PH:57, DR:72-77). The direction is unanimous, the measurements are absent.
- "The target sets the rewarded sharpness: blur only the LR, never soften the HR" (hypothesis, FG:74-84, PF:104).
- 4,000-6,000 unique 512 px tiles as a set size (design, DR:12, DR:69); 2-5k matched images per failing texture
  class (generic, SKT:15, SKG:62).
- "Measure the real inputs (noise sigma in flat patches, edge-spread blur width, 8-px blockiness, resize
  history) and build the degradation distribution from them" (generic, SKG:33; hypothesis, RG:38; HB:277
  distribution check). No source reports doing it with numbers.
- Gaps that matter for a people-close-up domain (FOCUS:296-297): no portrait/skin/hair HR data guidance beyond
  FFHQ mentions and a wish list (SKT:55, SKG:54, SX:27; person-disjoint holdouts only as advice, RG:146), and no
  method to keep film grain while removing sensor noise (one release recipe, DX:153; one code note, RP:96).

## 2. Open conflicts

| question | positions (anchors, evidence) | what would settle it | can our setup settle it now? |
|---|---|---|---|
| Blur dose for a detail model | high probability, mild strength: `blur_prob` 0.85 (redux: per batch), sigma [0.2, 2.2] at 1x, second blur possibly off (redux default `blur_prob2` 0, part 1.1) (release, PH:741-745) / very mild: p 5-15 %, sigma 0.2-0.6 (hypothesis, SX:90-91) / clean-heavy mixes (hypothesis, FG:173, PF:139-145) / too much erases HF, none cannot deblur (discord, DX:139) / release values plus a 15 % blur-free share (design, PG:127) | probability x strength grid on cached LQ, scored per bucket (clean / mild / strong / bokeh negatives) and on real inputs (X1, PG:452) | **No.** The data is clean; needs degraded LQ (parked OTF or a bake tool), bucketed validation sets and real blurred inputs |
| Overall degradation strength | mild (release, PH:726, PH:730) / very mild band (hypothesis, SX:69-76) / web-harsh JPEG ~Q40 twice (discord, RP:64) and Real-ESRGAN defaults (SKG:24) / over-degrading gives the "AI-restored" look (generic + discord, SKG:35, RP:58, PH:137) | per-stratum A/B with a near-identity clean bucket; depends on the owner's real inputs | **No** (same reason; also needs the owner's real LQ samples) |
| Synthetic noise and waxy skin | light noise prevents wax (hypothesis, SX:95-97) / noise severity causes wax (hypothesis, RG:220, FG:413) / omit denoising to keep grain (release, DX:155) / reviewers preferred no noise, confounded (release, RG:141) | 2x2 noise on/off x target USM on/off on cached LQ, skin bucket, flat-region residual (X11, PG:462) | **No.** Needs noisy LQ; the people/skin content of the local set would suit it once the LQ exists |
| USM-sharpened targets for L1/perceptual | use them (paper, Real-ESRGAN x4plus split, SKT:24, SKB:81-83, RG:56) / they bake in halos, keep an unsharpened baseline (discord, DX:229-231; hypothesis, PF:217) / off, arms U0-U3 (design, PG:99, FG:213-216) | same checkpoint, target USM on/off per loss, halo/edge-overshoot score | **Partly.** The target-side arm needs no degradation, so the clean data can host it, but neosr-dev has no per-loss target routing (part 4) |
| `lq_usm` (LQ built from sharpened GT) | True in both 1x releases, unexplained (release, PH:634, PH:739) / LQ from unsharpened GT (design, DR:101; discord, DX:231) | on/off arm with everything else cached | **No.** Feature absent; OTF parked |
| Mitchell as LR kernel | use or include (official line SI:25; hypothesis SX:80, PF:48; discord RP:102) / trained badly (discord, DX:91, DX:95) / accept only if gains transfer to unseen kernels and real inputs (hypothesis, FG:113) | same HR, arms cubic / Mitchell / mixture / cubic + mild LR-only blur, cross-evaluated on every kernel's held-out set and on real inputs (FG:104-113, PF:114) | **Partly.** The kernels exist (`aa_mitchell`), the HR is fixed; what is missing is the LQ generation for train and dev per kernel (bake tool or `clean_prob = 1`, both in the parked track) and a real-input set |
| HR filtering: detail teacher or halo teacher | strict multi-IQA may select a polished, haloed look (discord, DX:197-201, RG:99) / TOPIQ-filtered data taught more detail (discord, DX:279) / drop the worst, do not keep only the best (release, FG:134) | same pool: defect-only vs strict vs TOPIQ-filtered, then a low-LR detune (X10, PG:461), halo and grain-retention scores | **Partly.** Subsets of the owner's 10k tiles can be selected through `meta_info`; needs a tile scorer and an unfiltered held-out set; the pool is one homogeneous domain |
| Crop size for micro-texture | larger context (official TG:31; discord DX:403; generic SKT:17, SKR:23; hypothesis FG:263, RG:46) / smaller patches for micro-texture (hypothesis, compiler inference, RP:190-192) / one confounded counter-example (discord, RG:104) | LR 64 / 96 / 128 with batch as a separate factor, per-class scoring (X9, PG:460; HB:1116 E2) | **Partly.** The stored tiles are 64 px LQ: crops above 64 need re-tiled originals; the "smaller patches" side (LR 48 or 32 inside the 64 px tile, which also brings back random crop positions) runs on the current tiles, against the generic "64 is the floor" (SKR:23) |
| Texture-biased sampling vs smooth quota | oversample HF (generic, AN:65, SKG:69; hypothesis, SX:57) / keep smooth and defocus negatives (hypothesis, PF:53, FG:126, RG:48) / bounded bias + quota >= 25 % (design, PG:101); related official split on HR content: keep depth of field (UH3:48) vs musl's "sharp all the way through" and low-information images "won't do much" (UH3:24, UH3:33) | quota 0 / 25 / 40 % with a smooth/bokeh validation bucket | **Partly.** On this set a tile is a crop, so tile-level selection equals crop sampling; needs a tile scorer and a smooth-region bucket in dev; integer weights already work by repeating `meta_info` lines (code-read, part 4) |
| Pre-baked pairs vs OTF (and mixing them) | pre-bake everything (hypothesis, SX:107-113) / OTF on GPU with a pair queue (paper, SKG:27; release configs use OTF, PH:732) / never mix paired and OTF in NeoSR (official as reported, HB:236) / redux mixes paired + OTF per batch (discord, NP:55, NP:593) | same HR, same degradation code, offline frozen variants vs OTF (HB:1118 E4) | **No** (parked OTF, no bake tool) |
| Mixup / CutMix for photo models | used (discord, RP:209) / washes out micro-texture (uncited, RP:211), abandoned (release, DX:511), situational (generic, SKG:70) | one augmentation per arm under D10 | **Yes** (config only). Local stage-2 result exists but is weak (part 3) |
| Batch in a GAN stage | never below ~8 effective (hypothesis/generic, SKR:25, DR:118-119) / no NaN from batch < 8 alone (discord, PH:1009); Siax used batch 2 (official, SI:10) / larger batch helps scratch (official, TG:15) | batch 4 / 8 / 16 and true batch vs accumulation at matched exposure (HB:1117 E3) | **Yes** (config only; cost in part 6) |
| Data set size | <= 5,000-7,000 images (official, UH3:127, RX:121); 4,000-6,000 tiles (design, DR:69) / ~107k tiles for one pretrain (discord, RG:102); "small is better" not universal (HB:208) | subsets at equal updates on one pool | **Partly.** Subsets via `meta_info`; the pool is one domain, so only "fewer tiles of the same kind" is tested |
| Colour mismatch in paired data | never auto-correct, offer preserve / match / report (hypothesis, NP:246) / treat drift as a symptom (hypothesis, HB:1062) | none needed until real pairs are used | not applicable to the synthetic local pairs |

## 3. What our own evidence says

None of this group's local findings is a D10 verdict. They are measurements, one stage-2-gate decision (one seed,
8k, 67 held-out tiles, before the R5 split) and probes.

| local finding | numbers | source | relation to the community | limit |
|---|---|---|---|---|
| The data all local evidence rests on | 10,607 x4 pairs, GT 256 / LQ 64; the LQ matches an antialiased cubic or lanczos2 downscale at 52.7 dB (area 43.7 dB, plain bicubic 31.2 dB); no blur, noise, JPEG or sharpening | `docs/refactor/stage3/r3gan/RESULTS.md:10-17`, `docs/refactor/stage3/otf/DESIGN.md:28` | It is the "clean / downscale only" stratum at a 100 % share (HB:250). By the community's own reports such a model is not an artifact remover (DX:147-151) and cannot deblur (DX:139): an inference, not a local measurement | people close-ups only; says nothing about any degradation advice |
| "Two bicubics differ" | 52.7 dB vs 31.2 dB between the antialiased and the non-antialiased cubic against the owner's LQ | `docs/refactor/stage3/otf/DESIGN.md:28` | **Agrees** with PF:108-112 / FG:95 (kernel, antialias and phase are part of the degradation model) | a probe on one data set |
| Split by source group, leakage | groups had to be recovered after tiling (10,144 groups); a fuzzy check still found 164 of 600 held-out tiles (27 %) with a train twin; verdicts use 219 dev tiles in 211 groups | `docs/refactor/stage3/r3gan/RESULTS.md:10-17` | **Agrees** with "split by source before tiling, dedup across splits" (PF:52, PF:74, FG:123, DR:73, NP:329): the failure the community warns about occurred here | independence is only overlap-verified; rescaled/rotated copies were not searched |
| Crop = tile | `patch_size` 64 equals the stored 64 px LQ tile, so there is no random crop; flips and rot90 are the only spatial variation | `neosr/data/paired_dataset.py:113-123`, `options/train_realplksr.toml:15`, `docs/refactor/stage3/r3gan/neosr_fit.md:404-406` | **Departs** from "store 512 px tiles, crop inside" (RX:129-137, DX:237-241); matches the "LR 64 floor" (design DR:114-115, from generic SKR:23) | larger-crop advice untestable on this set; only smaller crops (LR < 64, random position) are possible |
| Batch augmentation off | touched 73 % of batches; off: DISTS 0.1862 vs 0.1879 at 8k, equal PSNR | `docs/refactor/stage2/DECISIONS.md:45-46`, `docs/refactor/stage2/CLOSURE.md:173` | **Agrees** with "flip/rot only" (SI:14, RP:217-219, SKG:70) and DX:511; **disagrees** with RP:209 | low confidence: one seed, one no-aug arm, old gate; the augmented arm ran with the hidden `aug_multi_prob` 0.3 and a resizemix that aliases without `aug_antialias` (`neosr/data/augmentations.py:117-118`), so the loss may belong to that defect, not to augmentation as such |
| Batch size cost | b8 104.6 ms, 3.19 GiB (76 img/s); b16 161.7 ms, 5.88 GiB (99 img/s); b32 323 ms, 11.2 GiB (99 img/s) | `next_tasks.md:30`, `experiments/s9_ab/consult/brief.md:26-27` | Neutral: throughput saturates at batch 16; the community's quality claims about batch (TG:15, SKR:25) were never A/B'd here | the owner's redux run switched 8 -> 16 together with the losses (`docs/refactor/stage3/owner_recipe.md:48-49`): confounded |
| Pre-tiling pays in the loader | 73 ms to decode a 4.6 MiB 2K PNG vs ~7 ms for a 512 px tile | `next_tasks.md:186-188`, `docs/refactor/stage2/CLOSURE.md:176` | **Agrees** with the tiling practice (RX:137, UH3:22) | the owner's set is already tiled |
| Degradation pipelines | no A/B. Probes: the default OTF chain has no antialiased resampler, applies blur1, noise twice and JPEG twice to every sample, has no clean sample | `docs/refactor/stage3/otf/DESIGN.md:22-30`, `docs/refactor/STATUS.md:91-96` | The default chain is exactly what the community warns against (always-on degradation, PH:328; no clean bucket, PF:50, RG:38), but that is a code reading, not a result | **local evidence can neither confirm nor refute any degradation advice** |

Common limits: clean x4 data, one domain, one generator (realplksr), every arm a fine-tune from one 20k L1
pretrain, nothing past 20k updates. The owner's own recipe used the same crop (LQ 64 / GT 256, flips + rot90;
`docs/refactor/stage3/owner_recipe.md:46`), so it adds no evidence on crop or degradation either.

## 4. neosr-dev support

| community practice | status | exact config keys | evidence path:line | gap or caveat |
|---|---|---|---|---|
| Offline paired LQ (pre-baked variants train as paired) | present | `[datasets.train] type = "paired"`, `dataroot_gt`, `dataroot_lq`, `meta_info` | `neosr/data/paired_dataset.py:57-90` | one GT root and one LQ root (single strings, `neosr/config.py:113-114`): several LQ variants per HR need duplicated or linked HR folders |
| LR-side crop key | present | `patch_size` (LQ pixels) | `neosr/data/transforms.py:57`, `train.py:137` | same side as redux `lq_size`; old `gt_size` configs are 4x off at x4. Uniform crop position only |
| Exact scale ratio of a pair | present (at load) | none | `neosr/data/transforms.py:59-61` | raises per sampled pair; no offline audit of alignment, sub-pixel shift, kernel or decode health (HB:149, NP:246, RG:39) |
| Flip / rot90 | present, default on | `use_hflip`, `use_rot` | `neosr/data/paired_dataset.py:115-123`, `neosr/data/otf_dataset.py:201-203` | in OTF they act on the GT before degradation (HR-first, as DX:131 asks) |
| Mixup, CutMix, ResizeMix, CutBlur | present, opt-in | `augmentation`, `aug_prob`, `aug_multi_prob` (0.3), `aug_antialias` (false) | `neosr/data/augmentations.py:33-180`, `neosr/data/augmentations.py:186-249`, `neosr/models/image.py:343-353` | off in the template (`options/train_realplksr.toml:18-19`); ResizeMix aliases unless `aug_antialias` |
| Effective batch by accumulation | present | `batch_size`, `accumulate` | `neosr/models/image.py:361-364`, `train.py:132-141` | an update is one optimizer step (`micro` counts micro-batches); the log prints updates per epoch and total updates. LQ pixels or images seen per update are not logged (DX:411-417) |
| Texture-biased / quota / region-aware crop sampling | missing | none | `neosr/data/transforms.py:67-69`, `neosr/data/data_sampler.py:45-58` | uniform crop, uniform shuffle; no realised-composition log |
| Per-source caps, weighted sources, several roots | missing | none | `neosr/config.py:113-114` | `meta_info` selects a subset list (`neosr/data/paired_dataset.py:74-79`); repeated lines are kept (`neosr/data/data_util.py:50-53`, `:259-264`; the duplicate-id check runs only for val/test, `paired_dataset.py:87-88`), so integer weights by repetition work: code-read, not tested; no per-parent cap, no composition log |
| HR curation: IQA/defect scoring, dedup, group split, multi-scale tiling | missing | none (tiler only: `--crop`, `--step`, `--bits8`) | `scripts/extract_subimages.py:1-13`, `scripts/extract_subimages.py:78-88` | the R5 split and leak check were done ad hoc in `experiments/`, not as a shipped tool; the tiler writes no manifest, though its tile names `<stem>_s001.png` keep the parent (`scripts/extract_subimages.py:7`); the owner's tiles carry content-hash names, no parent (`docs/refactor/stage3/r3gan/RESULTS.md:10`); no multi-scale option |
| Real-ESRGAN two-stage OTF chain | present (realism work parked) | `model_type = "otf"`, `[datasets.train] type = "otf"`, `[degradations]`, `queue_size` | `neosr/models/otf.py:14-41`, `neosr/data/otf_degradation.py:147-174` | default chain: blur1 always, noise twice, JPEG twice, no clean sample (`docs/refactor/stage3/otf/DESIGN.md:22-30`) |
| Blur probability separate from strength | present, opt-in | `blur_prob` (first blur; absent = always), `second_blur_prob` (required), `blur_sigma{,2}`, `kernel_list{,2}`, `kernel_prob{,2}`, `betag_range{,2}`, `betap_range{,2}`, `sinc_prob{,2}`, `final_sinc_prob` | `neosr/data/otf_dataset.py:157`, `neosr/data/otf_dataset.py:291-293`, `neosr/data/otf_degradation.py:160-161` | `blur_prob` is drawn per sample (dataset), `second_blur_prob` once per batch (Degrader); redux draws both blur probabilities per batch (part 1.1): long-run shares port, within-batch composition does not. Kernel size is drawn from 7..21, hard-coded (`neosr/data/otf_dataset.py:173-174`); redux draws it from a configurable `kernel_range` (default 5..17) and ignores `blur_kernel_size`, so the snapshot's "7" and "9" (PH:742, PH:749) are not kernel sizes and the release's effective range is unknown; no lens-disc or motion kernels (PH:295) |
| Noise: selector vs gate | present | `gaussian_noise_prob{,2}` = Gaussian-vs-Poisson selector; `noise_prob{,2}` (default 1.0) = per-sample gate; `noise_range{,2}`, `poisson_scale_range{,2}`, `gray_noise_prob{,2}` | `neosr/data/otf_degradation.py:245-269`, `neosr/data/otf_degradation.py:51-58` | same selector trap as FG:165, plus a real off switch: a noise-free recipe is `noise_prob = 0`, not `gaussian_noise_prob = 0`. The selector is one draw per batch (`neosr/data/otf_degradation.py:248`) |
| JPEG probability and range, two passes | present | `jpeg_prob{,2}` (default 1.0), `jpeg_range{,2}`, `jpeg_compress` | `neosr/data/otf_degradation.py:271-279` | per-sample gate here, one draw per batch in redux (release `jpeg_prob` 0.6 = 60 % of batches); DiffJPEG only: no chroma-subsampling choice, no 1-3 cycle chains, no WebP or real codecs (RP:72-76, PH:298-300) |
| Clean (downscale-only) bucket | present, opt-in | `clean_prob` (0), `clean_mode` (`["aa_bicubic"]`), `clean_mode_prob` | `neosr/data/otf_degradation.py:134-145` | `clean_prob = 1` is a GT-only resize degrader; no named buckets with shares, gates are independent draws (PF:127-129 asks for coherent families) |
| Resize-kernel mixture including Mitchell | present, opt-in | `resize_mode{,2}`, `resize_mode_prob{,2}`, `final_resize_mode`, `final_resize_mode_prob`; modes `area`, `bilinear`, `bicubic`, `aa_box`, `aa_bilinear`, `aa_bicubic` (Keys 0, 1/2), `aa_mitchell` (1/3, 1/3), `aa_lanczos2`, `aa_lanczos3` | `neosr/data/otf_degradation.py:38-48`, `neosr/data/otf_degradation.py:211-243`, `neosr/data/resample.py:46-53` | default list is the legacy non-antialiased trio; mode and resize factor are one draw per batch, mixed over time by the pair pool (`neosr/data/otf_degradation.py:359-362`) |
| 8-bit clip and round of the LQ | present | none | `neosr/data/otf_degradation.py:70-71`, `neosr/data/otf_degradation.py:140-145`, `neosr/data/otf_degradation.py:252-266` | noise calls use `clip=True, rounds=False`, the chain output is quantised once; believed equal to upstream, not re-verified against a pinned Real-ESRGAN revision (no parity test against the reference implementation) |
| Pair queue | present | `queue_size` (180) | `neosr/models/otf.py:26-31`, `neosr/data/otf_degradation.py:374-380` | rounded down to a multiple of the batch, so the "batch must divide it" rule (DR:107) is not needed; release configs use 192 (PH:732) |
| Degrade a larger region, then crop | present | `tight_canvas` (opt-in) | `neosr/data/otf_dataset.py:31-42`, `neosr/models/otf.py:40` | default canvas >= 512 px: on 256 px tiles 75 % of the degraded GT is mirror padding (`docs/refactor/stage3/otf/DESIGN.md:31`); one crop offset per batch |
| Sharpening in the pipeline | partial, different semantics | `sharpen_prob` (0), `sharpen_sigma`, `sharpen_amount` | `neosr/data/otf_degradation.py:59-65`, `neosr/data/otf_degradation.py:176-193` | an unsharp mask on the LQ after the final resize (sharpening as a degradation). There is **no** GT-scale `lq_usm` and **no** USM of the loss targets, so the FG:166 trap does not exist and the community's "unsharpened baseline" (DX:231) is the only mode |
| Paired + OTF mixing | missing | none | `neosr/data/otf_dataset.py:107-109` | the otf dataset rejects `dataroot_lq`; consistent with the NeoSR wiki's "do not mix" rule as reported by HB:236 |
| Dump of realised LQ / contact sheet / sampled-parameter log | missing | none | no match for debug/preview/dump in `neosr/models/otf.py`, `neosr/data/otf_degradation.py` | redux has `high_order_degradations_debug` (DR:107); the community asks for it before any long run (PF:151, DX:179-181) |
| Per-stratum validation | partial | several `[datasets.val*]` sections; `analyze.py --dataset` | `train.py:109-121`, `scripts/ab/analyze.py:337` | the mechanism exists; bucketed sets (clean / mild / severe, smooth negatives, texture classes) and fixed OTF validation sets do not (`docs/refactor/stage3/otf/DESIGN.md:32`, gap G5) |
| Real-input (unpaired) validation | partial | `[datasets.val*] type = "single"` | `neosr/data/single_dataset.py:20`, `neosr/models/image.py:576` | outputs are saved for the eye, no metric runs on a `single` set; no NR-IQA metric exists (TOPIQ-NR is a redux-only item in the bundle). The community asks for real deployment inputs as validation (PH:1011, PF:75); the metric side belongs to the validation group |
| Per-term gradient shares after a data change | present | `python scripts/term_grads.py -opt <toml>` | `scripts/term_grads.py:1-12` | builds the train loader as `train.py` does, so it measures any data config; the tool for "magnitude matching" questions |
| Crop grown in a later stage | not verified | edit `patch_size` / `batch_size` and resume | - | no test or doc states that a resumed state accepts a changed crop or batch (pair pool, compiled shapes) |

## 5. Framework work this implies (strategy-independent)

None of these changes a default; all are tools or opt-in keys. The owner's order (stable core, then Studio, then
new features) decides whether any of them starts. "Prereq" names the part-6 questions that cannot run without it.

| rank | feature | why (anchors) | size | risk | verify | prereq for |
|---|---|---|---|---|---|---|
| 1 | **Pair audit / data preflight tool** (`scripts/`): decode health, exact scale, bit depth, channel count, per-pair PSNR of the LQ against each `resample.py` kernel of the GT, sub-pixel shift estimate, report file; plus a contact sheet / HR-LR overlay of what the train loader actually yields after crop, flip and batch augmentation | a broken LR set wastes days (PH:1010); 1 px misalignment biases toward blur (HB:107-113, PF:86, RG:39); `audit_pairs.py` idea (HB:149, DR:77); NP:246; inspect decoded arrays through the actual loader (FG:129, HB:276, DX:133) | S-M | low; read-only | fixtures with a 1 LQ-px shift and a wrong kernel are flagged; the owner's set reports ~52.7 dB for the antialiased cubic | Q7, Q12 (trust in generated sets) |
| 2 | **Leak / duplicate detector and group-safe splitter** as a shipped tool (exact hash, block hash, NCC, parent manifest) | split by source before tiling (PF:52, PF:74, FG:123), dedup across splits (DR:73, FG:124), leakage detective (NP:329, NP:510); local 27 % twin rate | M | thresholds are data-specific; must report, not silently drop | reproduces the R5 exclusion list (`docs/refactor/stage3/r3gan/RESULTS.md:10-17`) | every question that builds a new held-out set (Q4-Q7, Q12, Q13) |
| 3 | **Tile scorer writing `meta_info` selections and strata** (SR headroom, Laplacian variance, blockiness, noise sigma, flat and clipped share, optional NR-IQA columns) | filter tiles, not images (PF:59); drop the worst rather than keep the best (FG:134); IQA only for triage, audit rejects (DX:199, FG:127, DR:74); already sketched as C3 (`docs/refactor/stage3/competitor/data_degradation.md:186-216`); run on the owner's real LQ samples, the same noise / blockiness / blur columns give the measured input profile the sources ask for before choosing any degradation (generic SKG:33, HB:277, RG:38) | M | NR scores reward sharpening (PF:83): it must output scores, never decide | unit tests per score on synthetic tiles; deterministic CSV | Q5, Q6, Q13; part 7 Q2 |
| 4 | **Exposure accounting in the run log and `done.json`** (`done.json` is the A/B harness's file, `scripts/ab/README.md:7`): effective batch, LQ pixels and images per update, cumulative images seen | compare exposure, not iterations (DX:411-417, FG:248, HB:299-307, PF:271); re-derive units when crop or accumulation changes (DR:120) | S | none | values equal `batch x accumulate x patch^2`; analyze.py prints them | Q1, Q4 (matched-budget arms) |
| 5 | **Weighted / stratified tile sampler with a realised-composition log** (weights or strata per `meta_info` entry; default uniform, index stream bitwise unchanged) | smooth quota and region quotas (RG:48, PG:101), per-parent caps (DR:72, FG:125), logging of what was sampled (RG:48) | M | resume and DDP determinism; must not touch the default sampler path | parity test: no weights = today's permutation; realised shares within tolerance of the weights | Q5 (quota arms beyond plain subset lists) |
| 6 | **One HR, several LQ variants** (a paired dataset drawing one of N LQ roots per sample, seeded, with per-root weights) | 12-variant sets (PH:308, RP:52), "one random variant per HR per load" (DR:86, DR:88), requested upstream (NP:164, NP:590) | M | variant leakage into validation; needs rank 2's manifest | deterministic draw per (seed, epoch, index); shares logged | Q8-Q11 if run offline instead of OTF |
| 7 | **Bucket tags in the A/B analysis** (per-image stratum from a sidecar file; per-stratum paired deltas under the D10 rules) | every source that proposes a mix asks for per-stratum monitoring (FG:157, FG:175, PF:151, HB:250-253, RG:38) | S | small strata widen the CIs; tau must be re-derived per stratum | strata of the current dev set sum to the pooled verdict | Q5, Q6, Q8-Q10 |
| 8 | **Resume with a changed crop / batch: test and document** | larger `lq_size` in the GAN stage (TG:31, DR:115), larger patches late (discord, PH:1007), crop schedules (hypothesis, SX:171-179) | S | none if it already works; otherwise a small fix | resume test with `patch_size` 64 -> 96 and `batch_size` 8 -> 4 | Q4 (staged variant) |
| 9 | **Per-loss target routing (opt-in USM of the GT per loss)** | the x4plus split (SKT:24, SKB:81-83) versus the unsharpened baseline (DX:231); needed only to run the U0-U3 arms (FG:213-216) | S-M | an objective change: opt-in only; touches the loss context; the thin community evidence (one author's regret, DX:229; one confounded comparison, RG:141) leans against using it | off = bitwise today's losses; on = target equals a reference USM | Q10 (target-USM factor) |
| 10 | **Tiler manifest and multi-scale option** in `scripts/extract_subimages.py` (parent id, scale, offset per tile; optional GT scale set such as {1, 0.75, 0.5, 0.25}) | keep parent identity with every tile and cap repeats (DX:215, FG:125); multi-scale GT in the releases (PH:294, PH:321; ~107k multi-scale tiles, RG:102); group-safe split before tiling (PF:74) | S | multi-scale copies change texture scale and sampling weight (DX:211-215): opt-in, never the default | manifest round-trip test; default output bitwise unchanged | Q4 (re-tiling), feeds rank 2 and Q3 / Q13; needs the originals (part 7 Q3) |

Reported, not proposed: the **OTF realism track is parked by the owner** (`docs/refactor/STATUS.md:91-96`). What
the community practices of this group would need from it, should the owner reopen it:
`scripts/degrade.py` (O4: bake the Degrader's output to disk with a manifest of the effective keys and frozen
seeds; it doubles as the preview / contact-sheet tool and generates the fixed validation sets,
`docs/refactor/stage3/otf/DESIGN.md:165-177`, step list `:220-231`; a per-sample trace of the drawn parameters
is explicitly not planned, "may record its own draws later", `DESIGN.md:148`); the O6 cost check (OTF
ms/update is unmeasured); named buckets with shares instead of independent gates; a configurable kernel-size
range (`neosr/data/otf_dataset.py:173-174`); lens-disc and motion kernels; JPEG chroma subsampling, multi-cycle
chains and real codecs; a GT-scale `lq_usm` arm; a realised-diversity log for the per-batch draws; a
pinned-revision parity test of the chain against upstream Real-ESRGAN. Questions Q7-Q11 below are blocked on
the first two items at least. Not decided here: whether a resize-only GT -> LQ generator over the existing
`resample.py` kernels (no degradation chain; enough for Q7 and Q12) counts as part of the parked track is the
owner's call (part 7, question 1).

## 6. Research questions for the strategy phase

Costs are single-seed 20k decisions in AU (37 min), excluding the second seed and the long run a default needs.
The existing reference pool (A0 + B0 + B0s2) and the null RMS are valid **only on the current train and dev
sets**: any question that changes the training LQ, the tile selection or the held-out set needs its own reference
arm and, for a new held-out set, its own same-config null pair (+2 AU). OTF step cost is unmeasured, so OTF arms
are given as ">= n AU".

| id | question | single-factor arms | prerequisites (data, feature) | rough GPU cost | blocked by | priority for the owner's goal |
|---|---|---|---|---|---|---|
| G1-Q1 | Does batch size or accumulation change GAN-stage quality at fixed crop? (HB:1117 E3; TG:15; SKR:25; PH:1009) | batch 8 (reference) / 16 / 8 x accumulate 2; optionally batch 4 | none (config only); rank 4 for the exposure report | b16: ~1.5 AU at matched updates (161.7 ms/update), ~0.75 AU at matched images; 8 x 2: ~2 AU | nothing; the matched-budget rule is an owner call | medium: cheap, removes a confound from the owner's own recipe (8 -> 16) |
| G1-Q2 | Does any single batch augmentation help or hurt texture under the D10 gate? (RP:209 vs DX:511, SKG:70; local stage-2 result) | none (reference) / mixup / cutmix / cutblur / resizemix with `aug_antialias`, one per arm, `aug_multi_prob` 0 | none | 1 AU per arm (4 AU) | nothing | low: weak community support, and a local low-confidence result already points to "off" |
| G1-Q3 | Do fewer tiles of the same pool train as well at equal updates? (UH3:127, RX:121 vs RG:102) | full 10,007 / random 5,000 / random 2,500 tiles (`meta_info`) | subset lists drawn by source group (rank 2 logic) | 1 AU per arm (2 AU) | nothing | low: one homogeneous domain; informs how much data a new domain needs |
| G1-Q4 | Does a larger LR crop help oriented texture (hair, fabric) in the GAN stage, separately from batch? Maps **X9** (crop half), HB:1116 E2 | 2x2: LR 64 / 96 x batch 8 / 4 (64 x b8 and 96 x b4 are near pixel-matched: 32,768 vs 36,864 LQ px per update); LR 128 only if 96 wins | originals re-tiled to >= 384 px GT (512 px for LR 128) with a group-safe split; a hair/fabric bucket; rank 4, rank 8 for a staged variant | 96 x b8 ~2.3 AU, 96 x b4 ~1.1 AU, 64 x b4 ~0.6 AU (scaled by pixels, unmeasured; the measured b8 -> b16 step cost 1.55x for 2x pixels, so pixel scaling likely overstates) + new reference and null pair | **data** (tiles are 64 px LQ) | high if the originals exist: the only conflict where official guidance (TG:31) and the owner's fixed crop disagree. Runnable now without new data: the "smaller patches" side, LR 48 x b8 vs 64 x b8 (RP:190-192 vs SKR:23), ~0.6-0.8 AU (unmeasured), which also tests random crop position; G and D fit at GT 192 not checked |
| G1-Q5 | Does a quota of smooth / defocus tiles remove false texture without costing texture? Maps **X9** (quota half) | smooth-tile share: as-is / 25 % / 40 % (tile-level, because a tile is a crop here); texture-only as a negative control | rank 3 scorer, rank 5 sampler (or duplicated list lines), a smooth/bokeh bucket in dev, rank 7 | 1 AU per arm (3 AU) + bucket null | features (ranks 3, 5, 7) | medium: targets the "texture on skin and bokeh" failure that matters for portraits; all positions are hypotheses |
| G1-Q6 | Does HR tile filtering teach detail or a sharpened look, and does a low-LR detune undo it? Maps **X10** | unfiltered (reference) / defect-only (drop worst 20-30 %) / top-score only; then a detune stage on the unfiltered set as a second step | rank 3 scorer, an unfiltered held-out set, a halo / edge-overshoot score, rank 7 | 1 AU per arm (2 AU) + detune arms (~0.5 AU each) | features (rank 3, halo score) | medium: the owner steers by IQA metrics, so a filter that selects what the judges reward is a direct Goodhart risk (DX:197, DR:74) |
| G1-Q7 | Is the LR kernel a sharpness lever or only a hidden deblur task? (Mitchell conflict; FG:104-113, PF:114, RG:55) | same GT, train LQ = `aa_bicubic` (close to today's) / `aa_mitchell` / mixture / `aa_bicubic` + mild LR-only blur; every model scored on every arm's dev LQ and on real inputs | LQ generation per kernel for train and dev (bake tool or a resize-only generator; OTF with `clean_prob = 1` is a poor substitute on this set: on 256 px tiles the >= 512 px canvas makes 75 % of the degraded GT mirror padding and 96.9 % of patches hold a seam, `docs/refactor/stage3/otf/DESIGN.md:31`, still 12 % padding with `tight_canvas`, `neosr/data/otf_dataset.py:31-42`; dev LQ must be baked anyway), rank 1 audit, a real-input set | 1 AU per paired arm (4 AU) plus the 4 x 4 cross-evaluation (validation only) | **parked OTF** (generation; owner call on a resize-only generator), no real-input set | medium: cheap once LQ sets exist and it stays on clean data; no community measurement exists |
| G1-Q8 | Which blur dose makes the model re-draw detail before bokeh breaks? Maps **X1** (first step) | `blur_prob` {0.5, 0.85, 1.0} x sigma max {1.5, 2.2, 3.0} as a 3x3 grid, everything else fixed and cached (neosr-dev's `blur_prob` is per sample, the release's per batch: the grid tests shares, not the release's batch composition) | degraded train LQ (OTF or baked), bucketed dev sets (clean / mild / strong blur / bokeh negatives), real blurred inputs, rank 7 | >= 9 AU at 20k, or >= 3.6 AU as an 8k screen, + reference and nulls per bucket | **parked OTF**; clean data; no 2x/4x ranges in any source (DR:105) | high for the stated goal (re-drawn detail on soft inputs); it is also the largest open gap of the corpus |
| G1-Q9 | How large must the clean share be so that good inputs are not redrawn? Maps **X1** (shares step) | `clean_prob` 0 / 0.15 / 0.30 / 0.40 at one fixed degraded recipe (the published mixes differ in several factors at once and are not single-factor arms) | as Q8, plus a near-identity clean bucket | >= 3 AU | **parked OTF** | high once Q8 is open: every source demands a clean bucket, none measured its size |
| G1-Q10 | Do synthetic noise, target USM and LQ-side sharpening each raise wax or halos? Maps **X11** | 2x2: noise on/off (`noise_prob` 0 / 1 at range [0, 3]) x target USM on/off; `lq_usm` as a third factor later | noisy LQ (OTF or baked), rank 9 (target USM), a GT-scale `lq_usm` (absent), skin and hair buckets, halo and flat-region residual scores | >= 3 AU for the 2x2 | **parked OTF**, missing features | medium: the local content (skin, hair) fits; the target-USM factor alone could run on clean data after rank 9 (1 AU) |
| G1-Q11 | Do frozen offline variants and OTF sampling train differently on the same HR and degradation code? (HB:1118 E4; SX:107 vs SKG:27) | paired on N baked variants / OTF with the same keys | bake tool, rank 6 | >= 1 AU + 1 AU | **parked OTF** | low: a tooling question; matters only after Q8-Q10 choose a recipe |
| G1-Q12 | Native x2 versus x4 followed by a downscale for a x2 deliverable (hypothesis, PF:90; nobody measured it; FG:283 is a related 1x point: use a 1x model rather than resizing for a 4x one) | native x2 model / existing x4 model + antialiased downscale, same GT | a x2 LQ set from the same GT (kernel = the one matched in part 3), rank 1 audit, a x2 pretrain | ~1 AU pretrain + ~1 AU arm (x2 at LQ 64 sees GT 128, cheaper per update; unmeasured) | LQ generation only | medium: the owner trains x2 as well; stays on clean data |
| G1-Q13 | Are the current verdicts sensitive to residual train / held-out leakage? (local open question; NP:329) | re-analysis of finished arms with a stricter exclusion list (no training) | rank 2 tool (rescaled / rotated / flipped twins) | 0 AU | nothing | medium: protects every other verdict of the study |

PG experiments in this group: **X1** -> Q8 + Q9; **X9** -> Q4 + Q5; **X10** -> Q6; **X11** -> Q10. X12's first arm
(region-aware sampling) needs the same sampler as Q5 but belongs to the face/identity group; X13 (per-term
gradient norms across trainers) is answered on the neosr-dev side by `scripts/term_grads.py` and belongs to the
loss group. Run `term_grads.py` on any arm that changes crop, batch or the LQ distribution: the per-term shares
measured at LQ 64 on clean data (FDL 81 %, GAN 16.5 % at P0) need not hold there.

## 7. Questions for the owner

1. **Scope of the strategy phase:** stay on clean x4 (then Q1-Q6, Q12, Q13 are the whole of this group, Q12
   only with a x2 LQ set), or reopen at least the offline bake tool so that the kernel, blur, clean-share and
   noise questions (Q7-Q11) can run? Reopening anything of the OTF track is the owner's call, including whether
   a resize-only GT -> LQ generator (no degradation chain; it would unblock Q7 and Q12) counts as part of it.
2. **Real inputs:** are 50+ real LQ samples of the intended inputs available (the open Q1 of
   `docs/refactor/stage3/otf/DESIGN.md:272-281`), and which matters more: staying faithful on clean inputs or
   repairing damaged ones? Without them no degradation question has a real-input endpoint.
3. **Originals:** do the untiled source photos of the 10,607 pairs still exist? They decide whether crop size
   (Q4), multi-scale GT and a source-group split made before tiling are possible at all.
4. **Data tooling in the repository:** should pair audit, leak detection, tile scoring and the weighted sampler
   (part 5, ranks 1-3, 5, 10) become shipped `scripts/` and core features, or stay ad hoc in `experiments/`? And
   where do they sit relative to Studio in the priority order?
5. **Matched-budget rule for crop and batch arms:** matched updates (the D10 windows are defined in updates),
   matched images seen, or matched LQ pixels? Both of the latter need a window mapping the owner accepts.
6. **Buckets and classes:** which held-out strata count for the goal (smooth skin / bokeh negatives, hair,
   fabric, eyes), who labels them, and is "close-ups of people" the target domain or will other content be added?
7. **Sharpened targets:** is an opt-in target-USM path wanted only to test the conflict, given that neosr-dev's
   present behaviour (no USM anywhere on the GT) already equals the community's recommended baseline?
8. **Batch augmentation:** re-test under D10 (Q2) or leave the stage-2 decision as it stands?

## Verification

Adversarial pass, 2026-11-13, by reading only (no GPU, no runs). Counts: **~110 KB anchor spans opened** (every
release-config value of part 1.1, every anchor behind part-6 priorities "high" and "medium", the official wiki
rules, the 1.2 reports, the 1.4 hypothesis labels, and the conflict-table positions); **52 neosr-dev code and doc
claims** checked at `path:line` (parts 3-5, including every config key and default named in part 4); the redux
key semantics of the two release configs checked against two local traiNNer-redux checkouts (2024-12 `d51c1954`,
2025-12 `29006500`; competitor code read, nothing taken over). All 101 relevance-3 bundle claims, the 22 conflicts
and the 16 gaps were compared with the section for omissions.

Confirmed as written (sample): both release configs value by value (PH:617-658, PH:722-763); Siax card; the
crop-key crosswalk; TG:15 and TG:31; DX:139, DX:403, DX:279, DX:197-205, DX:229-233; FG:165-167 labels [I];
every severity mix is self-labelled a hypothesis (FG:173, PF:131-151, HB:246-255, DR:92, PG:127); local numbers
(10,607 pairs, 52.7 / 43.7 / 31.2 dB, 164 of 600 = 27 %, 219 / 211, 73 % / 0.1862 vs 0.1879, b8 / b16 / b32
cost, 73 ms vs ~7 ms); every part-4 key, default and line (paired and OTF datasets, Degrader gates and defaults,
`queue_size` 180 rounded down to a batch multiple, per-batch noise selector, `tight_canvas`, resample kernels,
no USM anywhere in `neosr/`, no LQ dump in the OTF path); AU arithmetic (20k x 110 ms = 37 min; 80k x 104.6 ms =
2.3 h; 32,768 vs 36,864 px). Rules: no averaged conflict, no proposal changes a default, no redux loss weight is
given as a neosr value, OTF work is reported, not planned, local evidence is not claimed for degradation or
long runs.

Corrections (before -> after, source):

1. "Fidelity stages use LR 64 (TG:22)" -> fidelity stages keep the 2x scratch run's batch and `lq_size` from the
   official paper configs; LR 64 comes from a synthesis's example configs (TG:12, TG:22; HB:446, HB:538).
2. "Store 512 px tiles ... (official/tooling, RX:137, UH3:22)" -> tiling RX:129-137; random crop inside is
   discord (DX:237-241); UH3:22 is about duplicate removal.
3. "One degradation mode per run; NeoSR must not mix paired and OTF (official, RX:105, UH3:111, HB:236)" ->
   over-degrading OTF: official RX:105, UH3:111; "do not mix": official only as reported by HB:236 / HB:646
   (NeoSR wiki not in the KB); "one mode per run" is DR:86's design. Same fix in part 2 and part 4.
4. Sharpen release "kernel size 7 ... size 9" and part 4 "the release's fixed sizes 7 and 9 cannot be
   reproduced" -> in redux `blur_kernel_size{,2}` are read but unused, the size comes from `kernel_range`
   (default 5..17), so the release's effective range is unknown (`traiNNer/data/realesrgan_dataset.py:81,103,150`).
5. Added to the release reading: redux draws `blur_prob`, `blur_prob2`, `jpeg_prob`, `jpeg_prob2` once per batch,
   neosr-dev draws `blur_prob`, `noise_prob`, `jpeg_prob` per sample; an absent `blur_prob2` defaults to 0 in
   redux, so the release's second blur may never have run (`realesrgan_model.py:172,218,227,298`,
   `redux_options.py:579-584`). Carried into parts 2, 4 and Q8.
6. "the task is separated in the degradation, not in the loss stack (PG:121)" -> mainly in the degradation; the
   loss stacks also differ (DISTS 0.45 vs 0.25, perceptual 0.07 vs 0.02, ...; PH:604-611, PH:709-716).
7. "v1/v2 had every LR recompressed (PH:273, PH:302)" -> the first version (PH:302); lens-blur components and
   sharp-and-blurred duplicates are documented for v2 (PH:295), the v4 radius at PH:306.
8. SPAN release numbers anchored at DX:515, not DX:155-157; Siax data set "DIV2K 900" -> DIV2K train+valid, 2,700
   images, 900 unique (SI:12-13).
9. "reduce the degradation tail before adding losses (PH:137, PH:285, RP:58)" -> the sources report reducing
   strength; "before adding losses" is a synthesis reading (PG:138, FG:413).
10. Depth-of-field rule (UH3:48) -> the page itself records musl's dissent ("sharp all the way through", UH3:33);
    added to the 1.1 bullet and the sampling conflict row.
11. Crop-size conflict "can our setup settle it now? No" and part 3 "crop-size advice untestable" -> partly: the
    smaller-patch side (LR 48/32 inside the 64 px tile, with random position) runs now; added as a Q4 option.
12. Part 4 "weights only by editing the list (repeated lines: not verified)" -> repeated `meta_info` lines are
    kept for training (`neosr/data/data_util.py:50-53`, `:259-264`; duplicate-id check val/test only,
    `paired_dataset.py:87-88`): code-read, not tested.
13. Part 4 "the tiler keeps no parent manifest" -> no manifest file, but tile names keep the parent stem
    (`scripts/extract_subimages.py:7`); the owner's tiles have content-hash names (`RESULTS.md:10`).
14. O4 described as "the contact-sheet / realised-parameter dump" (DESIGN.md:220-231) -> previews and fixed sets
    yes (DESIGN.md:165-177), a per-sample parameter trace is explicitly not planned (DESIGN.md:148).
15. Q7 "OTF with `clean_prob = 1`" as an LQ source -> qualified: on 256 px tiles the OTF canvas puts mirror
    seams in 96.9 % of patches (DESIGN.md:31), 12 % padding even with `tight_canvas`; dev LQ must be baked anyway.
16. Anchor fixes: Q12 FG:283 is a 1x point, not native-x2-vs-x4 (PF:90 only); rank 8 PF:271 (exposure logging)
    moved to rank 4, replaced by PH:1007; part 3 SI:14 is the card itself, not "via SX digest";
    `neosr_fit.md:404-406` moved from the leakage row to the crop = tile row; "NeoSR wiki" (UH3) -> upscale-hub
    neosr guide; "the bundle ran with" -> "the augmented arm ran with".

Additions: 1.3 note on redux `lq_usm` (a switch for upstream's sharpened LQ source, random radius 1..25; a likely
reading of the "unexplained" flag); 1.4 "measure the real inputs" (SKG:33, RG:38, HB:277) and two omitted gaps
for a people domain (portrait/skin/hair HR guidance, grain vs sensor noise; FOCUS:296-297); part 4 row for
real-input (`single`) validation without any metric; part 5 rank 1 loader-output contact sheet, rank 3 real-input
profile, new rank 10 (tiler manifest and multi-scale option); the owner question on a resize-only LQ generator
(parts 5 and 7).

Unverified / open:

- The GaterV3 releases' own redux version, their `kernel_range`, `blur_prob2`, `lq_usm_radius_range` and whether
  the snapshot omitted keys (corrections 4-5 rest on two later or contemporaneous checkouts, not the release).
- Upstream Real-ESRGAN behaviour (FG:165-167, SKG:24-25, the 8-bit rounding parity) was not re-read at a pinned
  revision; neosr-dev's noise call matches redux's (`clip=True, rounds=False`, quantise once).
- Discord messages behind the syntheses (DX, PF, FG, RG, RP, PH tables) were not opened; their labels are taken
  from the KB documents. The NeoSR wiki page behind HB:236 is not in the KB.
- Whether a `single` val set is accepted in a training config end to end, whether resume accepts a changed
  crop or batch (part 4, last row), and the fit of G and D at GT 192 were not run.
- All GPU costs for crop changes are pixel-scaled estimates; OTF step cost is unmeasured.
