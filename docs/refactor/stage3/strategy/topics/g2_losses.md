# Losses and perceptual backbones

Study section G2 (bundle `experiments/study/bundles/g2_losses.json`: 128 claims, 17 conflicts, 14 gaps, 24 capability
rows, 5 local findings). Nothing here is a decision: the owner chooses.

Conventions used below.
- Evidence labels: **[official]** trainer wiki or model card; **[release]** a config that produced a released model;
  **[discord]** a message-cited community report; **[paper]**; **[generic]** LLM boilerplate or unattributed example;
  **[H]** hypothesis; **[D]** a design or interpretation made by a KB synthesis (PG, DR), never evidence.
- Scale tags: every weight carries its trainer scale (**redux**, **neosr**, **basicsr**). Weights on different scales
  are never compared, averaged or converted; a foreign weight enters neosr-dev only as a magnitude-matching question.
- KB anchors are `ALIAS:LINE` (`/home/korvin/apps/Trainings/tools/aliases.tsv`); neosr-dev anchors are `path:line`.
- "Arm" in cost terms = one seed 0 -> 20k from P0 at the template step: ~37 min (~0.6 GPU-h); with a DISTS loss +37 %
  (~0.85 GPU-h).

## 1. What the community converges on

How thin the evidence is, stated first: **no controlled ablation of any loss weight with numbers exists in the
corpus** (GAN, DISTS, LDL, FF, gradient variance, CX, perceptual). The nearest things are one participant's DF2K/Urban100
GAN ablation reported without numbers (a low perceptual weight improved some metrics but gave severe visual
artifacts; PF:209 [C]) and Zarxrax's per-term on/off tests on Compact/anime (DX:337). Every number is a single release
point, one person's habit, a doc example or a proposal (gap, anchors PH:707-716, PH:814-823, SX:145-148, SKL:26). No
source gives a tuned NeoSR-scale weight set for a complete GAN-stage stack (DR:503, HB:631). No source compares
perceptual backbones on photo texture (PH:418, RG:111, NL:79). All three release stacks come from one section of one
playbook (PH:551-908, an LLM-written extract of Phhofm's attached configs; the originals are not in the KB, so the
values cannot be re-verified there: PH card), on the redux scale, for 1x and 2x models trained with degradations; none
is x4 and none matches our clean x4 data.

### 1.1 Release configs (the best evidence available; one source, redux scale)

- **1x sharpen/deblur detail stack (GaterV3)**: MS-SSIM 0.4, VGG perceptual 0.07 (charbonnier), HSLuv 0.6
  (charbonnier), cosim 1.0, vanilla GAN 0.14, DISTS 0.45, LDL 0.35, FF 0.15; **no L1/Charbonnier pixel term**
  [release, redux] (PH:707-716). Per the PH card, the only release GAN recipe in the corpus that re-draws detail on
  blurred input [D]. No batch size in the snapshot; LQ crop `lq_size` 96 (PH:740; the restore snapshot also 96,
  PH:635).
- **1x restore/cleanup stack (GaterV3)**: MS-SSIM 0.3, perceptual 0.02, HSLuv 1.0, cosim 1.0, vanilla GAN 0.15,
  DISTS 0.25, LDL 0.5, FF 0.2, discriminator rvgg; no pixel term [release, redux] (PH:555, PH:604-611).
- **2x RealPLKSR stack**: Charbonnier 0.1 (eps 0.01), MS-SSIM 0.3, VGG perceptual 0.1 on conv2_2 0.3 / conv3_4 0.5 /
  conv4_4 0.2 (charbonnier), FF 0.3 (log_matrix true, ave_spectrum false), LDL 0.8, gradient-variance 0.25
  (patch 24), vanilla GAN 0.05 with labels 0.9 / 0.1, cosim 0.3 (cosim_lambda 3) [release, redux] (PH:814-823). The
  only release stack with a pixel term and soft labels; the snapshot lacks D, degradation and crop keys (PH:765; the
  uncited prose at PH:443 names a DUnet D). Its eps 0.01 matters: redux computes `sqrt(d^2 + eps)` (local redux
  checkout `traiNNer/losses/basic_loss.py:27-28`; version match with the release not verified), i.e. a near-quadratic
  zone for |d| below ~0.1, unlike a near-L1 Charbonnier with a tiny eps.
- **No complete x4 release stack exists** (PG:211). The nearest thing is a default block posted on Discord:
  MS-SSIM 0.5, perceptual 0.01 (charbonnier), HSLuv 1.0, cosim 1.0, vanilla GAN 0.1, with DISTS (0.25 in the header,
  0.3 in the inline comment), LDL 1.0 and FF 0.2 shipped disabled at weight 0 [discord, redux] (PH:1032-1067,
  DR:150-155). The block itself names no scale factor (a generic redux template block, RG:112). Switching all three
  optional terms on for x4 is PG's choice [D] (PG:211), not a release.
- What the three stacks share, as a reading of the numbers [D]: many small terms; a structure anchor (MS-SSIM
  0.3-0.4) plus a colour anchor (HSLuv + cosim) instead of a strong pixel term; a small VGG term (0.02-0.1); DISTS as
  the largest perceptual term of the detail model (0.45 vs 0.25 in the cleanup model) (PH:714, PH:609, PG:230);
  LDL lowest in the detail model (0.35) and highest where the GAN is weakest (0.8), which PG reads as "artifact brake,
  not texture source" (PH:715, PH:610, PH:820, PG:231); FF 0.15-0.3 in all three (PH:716, PH:611, PH:819).
- **NeoSR fidelity-stage practice**: L1 0.6 + MS-SSIM 0.4 [config snapshot of a 2x DRCT-XL pretrain, neosr]
  (PH:894-895, PH:393, DR:411-418). Carrying
  that split into the GAN stage is DR's choice [D] (DR:151); the NeoSR GAN-stage candidate block leaves the VGG and
  consistency weights as "template value" and puts GAN at 0.1 [D] (DR:476-494).

### 1.2 Official documents

- **Redux curriculum**: stages 1-2 use only "MS-SSIM L1"; stage 3 enables MS-SSIM L1, perceptual, HSLuv and GAN
  together, with no weights [official, redux] (TG:11, TG:21, TG:29).
- **NeoSR losses wiki** lists wavelet_guided, pixel (L1/MSE/Huber/chc), mssim, ncc, fdl, vgg perceptual, dists, gan,
  ldl, ff, gw, kl, consistency (+ match_lq_colors), msswd with example keys; the weights (mostly 1.0, gan 0.3,
  dists 0.5) are syntax examples, not recommendations [official, neosr] (NL:1-253, DR:503). It has no HSLuv,
  contextual or R3GAN loss and none named Charbonnier (its `chc` is a "Clipped Huber with Cosine Similarity" pixel
  type, NL:31).
- **NeoSR prerequisites**: wavelet_guided works better for fine-tuning, train at least ~40k before enabling
  (NL:16; doc example wavelet_init 80000, NL:11-12); ff_loss can be unstable without a pretrain (NL:182); kl_loss
  only with a pretrain (NL:210) [official, neosr].
- **FDL** (model dinov2 | vgg | resnet | effnet): num_proj defaults to 24 for speed (official implementation 256),
  "raise it at the end of a finetune"; no quality evidence in the wiki [official, neosr] (NL:62-79). FDL layer
  weights: avoid raising a layer above 1.0 (NL:80-90).
- **Real-ESRGAN x4plus**: USM-sharpened GT for L1 and perceptual, original GT for the adversarial real samples, LQ
  built from sharpened GT [official, basicsr] (RG:56, FG:207). VGG layer weights conv1_2 0.1, conv2_2 0.1, conv3_4 1,
  conv4_4 1, conv5_4 1 at weight 1.0 with L1 1.0 and GAN 0.1 [paper/generic, basicsr] (SKL:15, SKL:52).
- **Siax model card**: "relies heavily on Contextual Loss in combination with PatchGAN"; no weights, layers or LR
  [official] (SI:17, SI:23). This is the only verified CX fact in the corpus.

### 1.3 Message-cited community reports

- **Weights do not transfer across trainers** (reductions, feature scaling, label conventions differ); copy the
  trainer's own template and compare weighted magnitudes and gradient behaviour [discord] (DR:6, HB:499, HB:797-801,
  PF:213). A perceptual or GAN weight of 1.0 is "super high" on the redux scale (PH:1005, PH:1024) and normal on the
  basicsr / NeoSR-doc scale (SKL:15, NL:101).
- **Magnitude matching**: print each term raw at loss_weight 1.0, then scale to the intended contribution; example: a
  DINOv2 term printing ~6.6 scaled to contribute ~0.035 [discord] (RP:139-141, RP:357).
- **Pixel term**: DX's action from reported tests that all kept a pixel loss: start from an interpretable
  reconstruction baseline and watch geometry, colour and hallucination if the term is reduced (DX:321-323) [discord
  synthesis]. The opposite assertion, "remove every L1/SSIM-like objective for photorealism", is in the logs without
  experimental support (DX:321, DX:600) [discord, uncontrolled]. The often-quoted "avoid turning off pixel loss
  entirely ... unless you explicitly want hallucination" (PH:491) is **not** message-cited: it sits in PH's
  LLM-written, uncited section 5.2 (PH card) [generic].
- **Negative report (Compact, anime footage)**: perceptual loss produced dots, noise and blocks that grew with
  training; gradient-variance sharpened but added artifacts; LDL and focal-frequency additions were negligible
  [discord, reported test] (DX:337-341, DX:875). Anime only; DX itself restricts the conclusion.
- **Colour**: a low-frequency consistency loss can block dehaloing; widen its low-pass instead of only lowering the
  weight [discord, release recipe of a video model] (DX:361-365). A 2022 anime run improved early colour after
  removing one perceptual configuration, then degraded for other reasons (HB:789, HB:1249): neither "always remove
  perceptual loss" nor "perceptual is the only cause" is defensible (HB:789); outcomes were "config-conditional"
  (NP:260).
- **Backbone directions without results**: ConvNeXt-Tiny against the "waterpainterly GAN look" (PH:418, PH:421)
  [uncited LLM paraphrase of Phhofm's notes, PH card: "stated direction, no weights given"]; a DINOv3 feature-loss
  script shared with no comparison (RG:111) [discord].
- NeoSR configs "often include" perceptual-type losses at weight 0, enabled by a weight above 0 (example: DISTS)
  [discord] (RP:133). In neosr-dev's template they are commented out instead, and a zero-weight G term is dropped at
  build (`neosr/losses/__init__.py:90`).

### 1.4 Paper-level statements (not practitioner results)

- Wavelet-domain loss or D input when an RGB GAN adds speckle or false hair; keep a spatial anchor and compare at
  matched sharpness [paper] (RG:57). No photo result and no schedule in the corpus beyond NL:4-16 and one piece of
  advice to switch it on while the LR still matters (DX:303) [discord].
- LDL: add once GAN artifacts appear and inspect its maps first; misregistration and valid stochastic texture look
  like artifacts to it [paper] (RG:58).
- Focal frequency only as a small auxiliary: magnitude agreement cannot locate detail and can learn JPEG peaks
  [paper] (RG:75, FG:193). FDL forgives small misalignment, hence displacement: keep exact-edge checks (RG:82, FG:194).
- DISTS tolerates texture rearrangement by design, so it cannot be the sole exact-detail criterion (FG:190,
  FG:405); a DISTS-type loss can lower the metric while defects remain [H] (PF:203).
- Contextual loss / CoBi is "the tool for misaligned real pairs and texture transfer" [generic skill text, paper
  background] (SKL:17, SKL:54). RG:89 [paper + community] proposes it for aligned pairs too, as a bounded CX-vs-VGG
  comparison when L1/VGG penalise plausible texture shifts, and warns it can reward texture in the wrong place.

### 1.5 Hypotheses and designs (listed so they are not mistaken for evidence)

- Balance by weighted per-term **gradient norms** at a shared trunk layer on fixed probes; do not force equal norms
  [H] (RG:59, RG:169, FG:201, PF:213, DX:371). Nobody in the corpus has logged these norms (PF:213, PG:261).
- "Weighted terms within about one order of magnitude after warm-up" [generic] (SKL:57). The two official logs do
  not settle it: redux 0.19 / 0.25 / 0.056 / 0.075 for mssim / perceptual / hsluv / gan (RX:239) is **within** one
  order (max/min 4.5x); NeoSR pixel 0.0024 vs perceptual 0.77, colour 0.0009 (UH3:212) is two-plus orders apart. Both
  lines are at iteration 100, before any warm-up ends, so neither tests the "after warm-up" wording.
- "LDL on by default, best quality-per-effort add-on" [generic] (SKL:29, SKT:18, SKT:36).
- "One opt-in loss per observed defect, stop when it is fixed" [H/D] (FG:181-183, DR:157, HB:789).
- SX stack: L1 1.0, VGG 0.2, CX 0.3-0.4 as "texture engine", multi-scale PatchGAN at "effective" 0.003-0.005
  [H, speculative reconstruction; scale assumed basicsr] (SX:145-148). High-pass residual loss with no sigma or
  weight (SX:155-159). HFEN 0.08, A-DISTS 0.15, ConvNeXt 0.2, LDL 0.8 with decay at 35000: unattributed examples
  [generic] (SL:1-31); DR advises against them (DR:548). Local note: SL's ConvNeXt 0.2, HFEN 0.08 and LDL decay
  0.8 / 35000 match entries of the owner's own redux run, where ConvNeXt had no gradient, HFEN and LDL were
  numerically nil and the decay keys were ignored (`docs/refactor/stage3/owner_recipe.md:28-29`, `:39`, `:41`, `:57`).
- Loss API proposal: per-term declared range, precision, reduction, raw and weighted logging [H] (NP:285-300).
- USM ablation arms U0-U3 [H] (FG:213-218); HB's five-branch loss ablation A-E [H] (HB:803-813).

## 2. Open conflicts

| question | positions (anchors, evidence) | what would settle it | can our setup settle it now? |
|---|---|---|---|
| Pixel term in the GAN stage | off only if hallucination is wanted (PH:491) [generic: uncited LLM-written PH section]; remove every L1/SSIM-like objective for photorealism (DX:321, DX:600) [discord, uncontrolled assertion]; none in both 1x releases (PH:707-716, PH:604-611) [release, redux]; Charbonnier 0.1 at 2x (PH:816) [release]; keep an interpretable reconstruction baseline (DX:323) [discord synthesis; the cited tests all kept pixel loss, DX:321]; MS-SSIM + HSLuv + cosim act as the anchor at 1x (PG:426) [D] | single-factor arms on the pixel weight with the structure and colour anchors held fixed, judged on perceptual metrics and a colour/luminance check | **Partly.** x4 clean: yes (arms on `pixel_opt`). The 1x no-pixel regime: no (no 1x degraded data; OTF parked) |
| DISTS / LDL / FF: on from the start, or one opt-in per defect | all on from the start in the releases (PH:714-716, PH:609-611, PH:819-820) [release]; one opt-in per observed defect (FG:183, DR:157, DX:561) [H/D] | attribution: each term toggled alone from one checkpoint (PG X5) | **Yes** for x4 clean (all three terms exist); the "observed defect" side needs defects our clean data may not produce |
| DISTS as a loss and as a judge | loss 0.25-0.45 (PH:413, PH:714) [release, redux]; it can hide defects (PF:203) [H] and tolerates resampled texture by design (FG:512, FG:392) [paper property]; loss yes, selection metric no (FG:405, PG:428) [D] | a DISTS arm judged only by metrics it does not train on, plus an eye check for hidden defects | **Yes**: the D9/D10 rule already reports DISTS and gates on TOPIQ + LPIPS (`docs/refactor/DECISIONS.md:198`, `:227`) |
| LDL: default-on, brake, texture source, weight | on by default (SKL:29, SKT:18) [generic]; 0.35 / 0.5 / 0.8 in releases (PH:715, PH:610, PH:820) [release]; 1.0 suggested, shipped off (PH:1063) [discord]; "regain micro-texture" (PH:414) [compiler wording]; flags valid stochastic texture (RG:58) [paper]; negligible alone on Compact/anime (DX:337) [discord]; post-hoc fix (FG:192) [H] | LDL on/off with artifact-map review on stochastic texture, at a weight where it has a measurable gradient | **Partly.** On/off: yes, but needs >= 20k on two seeds or a long run (section 3). Misregistration behaviour: no (our pairs are exact) |
| Contextual loss as the texture engine | core of Siax (SI:17) [official]; 0.3-0.4 and "moves the needle faster than GAN weight" (SX:147, RP:127) [H / extrapolation]; questioned, no CX-only ablation, Siax's full stack unknown (DX:329-333, DX:601) [discord]; can place plausible texture in the wrong spot (RG:89) [paper]; release stacks use DISTS/LDL/FF instead (PH:707-716, PH:814-823) | base + CX vs base + perceptual in one trainer under the same D and anchor | **No**: no CX loss in neosr-dev (section 4) |
| Perceptual (VGG) weight | 0.01 default, 1.0 "super high" (PH:1005, PH:1024, PH:1044) [discord, redux]; 0.02 / 0.07 / 0.1 (PH:605, PH:710, PH:818) [release, redux]; 1.0 (SKL:15, SKL:52, NL:101) [basicsr / neosr doc example]; 0.2 (SX:146) [H]; scales are incommensurable, magnitude-match (HB:499, RP:139) [discord] | not a conflict of values but of scales: a measured raw magnitude and gradient share per unit weight in each trainer (PG X13) | **Half.** The neosr-dev side can be measured today (`scripts/term_grads.py`); the redux side would mean running the competitor |
| Perceptual backbone | VGG19, the only one with release weights (PH:818) [release]; ConvNeXt-Tiny (PH:418, SL:6) [discord / generic]; DINO features (RP:357, RG:111) [discord]; one backbone at a time, none suits all content (PF:175, PF:209) [H / discord] | backbone swap at matched gradient share (PG X8) | **Partly**: VGG19, DISTS (VGG16) and FDL on DINOv2/VGG/ResNet/EffNet exist; ConvNeXt does not |
| Balancing method | raw magnitudes at weight 1 (RP:139) [discord], within one order (SKL:57) [generic]; per-term gradient norms on fixed probes (RG:59, RG:169, FG:201, DX:371) [H]; magnitudes to initialise, norms to verify (PG:435) [D] | measure both on the same checkpoints and see which predicts the A/B outcome | **Yes** (measurement only; needs raw-value output, section 5) |
| Do frequency / gradient-variance terms add useful texture pressure | FF 0.15-0.3 in every release, gradient-variance 0.25 in the 2x release (PH:716, PH:611, PH:819, PH:821) [release]; prefer FFT/SWT over GAN weight (SKT:19, SKL:32-33) [paper/generic]; FF negligible and gradient-variance sharpened with artifacts on Compact/anime (DX:337) [discord]; gradient-variance removed from NeoSR for instability (DX:385) | FF on/off; any HF-band term at matched magnitude (PG X7) | **Partly**: `ff_loss` yes; gradient-variance, HP-band, HFEN, gw no |
| USM-sharpened GT for L1/perceptual | official Real-ESRGAN does it (RG:56, FG:207, SKT:24, SKB:81) [official]; halos, keep an unsharpened baseline (DX:231-233, PF:217-219) [discord]; reviewers preferred USM off but the change was confounded with noise removal (RG:141, FG:161) [practitioner comparison, C; not a release config] | arms U0-U2 with the D target always original (FG:213-216) and an edge-overshoot measure | **No**: no per-loss USM target (section 4); U3 (sharpened LQ source) is OTF, parked |
| Is VGG perceptual the cause of colour/contrast drift | removing one perceptual config improved early colour in a 2022 anime run (HB:789, HB:837, HB:1249) [discord]; later degradation had other causes (HB:789); config-conditional (NP:260) | perceptual on/off from the same pretrain with a colour-bias metric (HB:830-837) | **Partly**: the arm is trivial; the colour-bias metric is missing (group G5) |
| When to switch on the wavelet-guided view | before LR has decayed, not at the very end (DX:303) [discord]; train at least ~40k first (NL:16) [official] | switch-point arms in a run long enough to cross the switch | **No, not yet**: no local run has passed 20k or crossed `wavelet_init = 80000` (group G3 owns the D question). One related 8k screen exists: `wavelet_init = 0` (SWT view from update 1) was a LOSS vs B0 (PSNR -0.11, LPIPS +0.0032 at 1.7x its floor, DISTS +0.0032; one seed; `docs/refactor/stage2/DECISIONS.md:91`, `:99-100`), weakly consistent with "not from scratch" (NL:16) |
| GAN weight band (cross-reference to G3) | 0.14 / 0.15 / 0.05 releases, 0.1 default (PH:713, PH:608, PH:822, PH:1054) [release/discord, redux]; start 0.1-0.2, raise to 0.3 once stable, 0.5 more unstable (PH:1021) [discord, redux]; 0.003-0.005 "effective" (SX:148) [H]; 0.001-0.01 (AN:139) [generic]; 0.3 NeoSR doc example (NL:142), also the neosr-dev template value (`options/train_realplksr.toml:102-105`, bce) | see G3; here only: nominal weights are meaningless without the D regime (`out_d_acc`, `grad_sr_ratio`) | see G3 |

## 3. What our own evidence says

Limits that apply to every row: clean antialiased x4 downscale (no degradations), GT 256 / LQ 64 tiles of people,
one generator (realplksr 7.39 M) fine-tuned from one 20k L1 pretrain, nothing past 20k updates, no run across
`wavelet_init = 80000`. Rows marked **(study)** are computations made during this study from existing run files,
not owner-accepted verdicts.

| id | local result | source | agrees / disagrees with the community | limits |
|---|---|---|---|---|
| G16 | FDL(DINOv2) weight 0.75 -> 0.02: at 0.75 FDL held ~98-100 % of the G gradient; PSNR/DISTS @8k 27.23/0.2024 (0.75), 28.17/0.1905 (0.1), 28.69/0.1830 (0.05) with the old code; with the new code 0.02 vs 0.05 = 29.36/0.1787 vs 28.58/0.1829 (AdamW), so 29.36 is not a point of the same sweep (it is also the stage-2 "net effect" of several changes, `:49-50`). Even at 0.02 FDL holds 74-81 % of the summed **squared** G-gradient norms (~2.1-2.2x the GAN term's norm) | `docs/refactor/stage2/DECISIONS.md:39-43`, `:49-50`; `docs/refactor/stage2/PROGRESS.md:73`; `readme.md:217-220` | **Agrees** with "1.0-style weights are not sane across scales, magnitude-match" (PH:1024, RP:139) and with the RP example of scaling a DINOv2 term down by ~200x (RP:141). It is the corpus-wide missing measurement (per-term gradient share) done once, locally | stage-2 gate: one seed, 8k, 67 tiles, judged partly on DISTS |
| template shares | per-term share of the summed **squared** G-gradient norms (`share = \|\|g_t\|\|^2 / sum`, `scripts/term_grads.py:58-65`, `:133`) at P0: FDL 81.2 %, GAN 16.5 %, MS-SSIM 1.4 %, L1 0.53 %, consistency 0.35 %, LDL 0.000 %; at B0's 8k weights 74.3 / 16.6 / 7.1 / 1.7 / 0.44 / 0.000 %. In norm terms at P0 (FDL \|g\| 0.36): FDL is ~2.2x GAN, ~7.6x MS-SSIM, ~12x L1, ~15x consistency, ~1,600x LDL (2.3e-4) **(study arithmetic)** | `docs/refactor/stage3/PROGRESS.md:428-430`; `experiments/s9_ab/NOTES.md:25-26`; tool `scripts/term_grads.py:33-85` | **Mostly agrees** with the gradient form of the "within one order of magnitude" heuristic (SKL:57): every term except LDL is within ~15x of FDL in norm; only LDL breaks it (by three orders). The earlier "three orders apart" reading compared squared shares. Weakly supports RG:169: FDL and GAN log near-equal weighted values (0.22 / 0.21 at 17k, A0 log) yet differ ~2x in gradient norm at 8k [D] | shares are over all G parameters on training batches, not a trunk layer on fixed probes; P0 is an L1 pretrain, so the L1 share starts near its minimum |
| G17 | adding L1 1.0 to the FDL GAN objective: 28.34/0.1879 vs 28.17/0.1905 (PSNR/DISTS @8k) | `docs/refactor/stage2/DECISIONS.md:44` | **Agrees** with "keep a reconstruction anchor" (DX:323); says nothing about the release practice of no pixel term at 1x (PH:707-716) | small deltas, one seed, 8k; confidence low |
| G37 | owner's redux run: Charbonnier 0.01 without HSLuv -> val8 PSNR 29.5 -> 17.7 dB in 1,556 iterations; restoring Charbonnier 0.1 + HSLuv 0.1 -> 28.05 dB and the best DISTS of the run (0.1107) | `docs/refactor/stage3/owner_recipe.md:21-23`, `:42-43`, `:74` | **Agrees** with DX:323 and with PH:491's condition; the clearest causal reading in the owner's logs | confounded with the GAN weight rising x6 in the same step; one run; redux scale, other arch |
| G14 | DISTS 0.15 added to the template (T_DISTS): official D9 verdict **TIE** at 20k (TOPIQ +0.0015 inside the floor, LPIPS -0.0047 better, DISTS -0.0087 reported). Window 12k-20k: LPIPS -0.0067, TOPIQ +0.0022 (better at 9/9 checkpoints). **(study)** recomputed vs the pooled reference with D10 tau: TOPIQ +0.0025, LPIPS -0.0060 -> WIN under D10 rules (re-run by the verifier with `scripts/ab/analyze.py --window 12000:20000`, same numbers). D10's tau binds only for arms started after 2026-11-12 (`docs/refactor/DECISIONS.md:222`) and the consult rejected retroactive re-gating (`synthesis.md:32`): the WIN is a reading, not a verdict. Side observation: the smallest `grad_norm_g` maximum of the compared arms (3.99 vs 6.45-34.6 for A0/B0/B0s2; analyze.py diagnostics). Cost 130 ms/update (+37 %), steady 4.70 vs 3.19 GiB | `experiments/s10/consult/synthesis.md:6-10`, `:23`; `experiments/s10/consult/brief.md:17`, `:35`; `docs/refactor/DECISIONS.md:216-217`; `experiments/s10/queue2.held.txt` | **Agrees in direction** with DISTS being the texture term of the detail releases (PH:714, PG:230) and with the owner's own ranking (`owner_recipe.md:73`). The strongest local signal for any objective change | one seed (seed 2 held); DISTS is also a judge, so only TOPIQ + LPIPS count; cost is real |
| G15 (G3) | 2x2 GAN x DISTS: without a D, DISTS loses to GAN + DISTS (window 12k-20k: LPIPS +0.0085, TOPIQ -0.0072) | `experiments/s10/consult/synthesis.md:24-25`, `:31` | **Agrees** with "GAN adds HF beyond DISTS" (PG X5 hypothesis, PH:713-714 [D]); does not confirm the reading that DISTS could replace the D (`owner_recipe.md:90-91`) | one seed; the comparison vs the pooled template is **(study)** |
| G18 | LDL at weight 1.0: G-gradient share 0.000 % (\|g\| 2.3e-4 at P0 vs FDL 0.36); removing it saves 4.8 % ms/update. Seed 7: TIE at 8k and in the 12k-20k window. Seed 2: LOSS (LPIPS +0.0029 at 8k; window 12k-20k TOPIQ -0.0066, LPIPS +0.0088) with a G-gradient spike of 248 at update 4400. Decision D1: keep LDL | `docs/refactor/stage2/DECISIONS.md:72-83`; `docs/refactor/DECISIONS.md:190`; `experiments/s10/NOTES.md:41-42`; probe `docs/refactor/stage3/competitor/losses_metrics.md:28-30`, `:167-175` | **Agrees** with "LDL negligible alone" (DX:337) on the gradient measurement; neither supports nor refutes "LDL as brake" (PG:231) or "default-on" (SKL:29): a term with zero measured share cannot be shown to brake anything, yet the seed without it diverged | seeds disagree; the LOSS sits on a D outage that the removal may or may not have caused; unexplained; confidence low |
| G13 (G3) | **(study)** OWN, the owner's recipe ported (DISTS 0.05, L1 0.1 + consistency 0.1, weak slow rpgan D; no FDL/MS-SSIM/LDL/wavelet): LOSS vs the pooled template on dev (window 12k-20k TOPIQ -0.0093, LPIPS +0.0062, PSNR -0.277) but the best val8 LPIPS/DISTS of all arms | `experiments/s10/cfg/OWN.toml`; `experiments/s10/runs/OWN/per_image.jsonl`; `docs/refactor/DECISIONS.md:224` | Illustrates the community warning that a recipe does not transfer as weights (DR:6, PF:213): a multi-change port is not attributable | not an official verdict (D10 rule 4: multi-change recipes are not screens); different arch from the owner's; one seed |
| inert terms | in the owner's redux run the ConvNeXt and DINOv3 perceptual terms had no gradient (features under no_grad), LDL and HFEN were numerically nil, the loss-decay keys were ignored | `docs/refactor/stage3/owner_recipe.md:28-29`, `:55-57`, `:80`; `docs/refactor/stage3/competitor/losses_metrics.md:113` | A caution about the community's ConvNeXt/DINO "directions" (PH:418, RG:111): configs that list such terms prove nothing about their effect unless the gradient was checked | code and log reading of one fork commit |
| wavelet view (G3 overlap) | `wavelet_init = 0` (D sees the SWT HF view from update 1): LOSS vs B0 at 8k (PSNR -0.11, LPIPS +0.0032 at 1.7x its floor, DISTS +0.0032) | `docs/refactor/stage2/DECISIONS.md:91`, `:99-100` | Weakly consistent with NL:16 ("better for fine-tuning", ~40k first); says nothing about the switch at 80k or DX:303's timing advice | one seed, 8k, marginal; stage-2 single-checkpoint gate |

Summary of the local picture: (1) the measured facts are about **scale and share**, and they back the community's
strongest non-numeric advice (do not copy weights; measure); (2) DISTS is the only added perceptual term with a local
signal, one seed; (3) LDL is unresolved; (4) no local evidence exists for VGG perceptual, FF, MS-SWD, backbone swaps,
pixel-weight sweeps under D10, colour-anchor removal, USM targets or any HF-band term; the wavelet view has one
marginal 8k screen (`wavelet_init = 0`) and no run across the 80k switch.

## 4. neosr-dev support

| community practice | status | exact config keys | evidence (path:line) | gap or caveat |
|---|---|---|---|---|
| Any number of weighted terms; weight 0 = off | present | `[train.<slot>_opt] type, loss_weight` (alias `perceptual_weight`) | `neosr/losses/__init__.py:53-111`; `neosr/losses/base.py:108-121` | weight applied once (`base.py:115`); a zero-weight G term is dropped (`__init__.py:90`); class default weight 1.0, `gan_loss` 0.1 (`neosr/losses/gan_loss.py:88`) |
| L1 / MSE / Huber pixel term | present | `type = "L1Loss" \| "MSELoss" \| "HuberLoss"`, `reduction` | `neosr/losses/basic_loss.py:53-143`; template `options/train_realplksr.toml:81-83` | template L1 1.0 (neosr scale), not the release 0 / 0.1 (redux scale) |
| Charbonnier (release 2x; owner's recipe) | present, with a trap | `type = "chc_loss", criterion = "huber", loss_lambda = 0, clip_min = 0, clip_max = 1e6` | `neosr/losses/basic_loss.py:147-213` (defaults `:170-172`); `docs/refactor/stage3/owner_recipe.md:109-115` | default `clip_min = 1/255`, `clip_max = 254/255` **clip the per-pixel loss** (zero gradient below 1/255 error): plain Charbonnier needs both overridden; no `charbonnier` type name; the eps is fixed, `sqrt(d^2 + 1e-12)` (`:204`), so the 2x release's eps 0.01 (PH:816; redux `sqrt(d^2 + eps)`) cannot be expressed: neosr's form is near-L1 |
| cosim (redux `cosimloss`) | partial | `chc_loss loss_lambda` (cosine term); `consistency_loss cosim = true` | `neosr/losses/basic_loss.py:205-206`; `neosr/losses/consistency_loss.py:199-206` | consistency's cosine term is added only below 1e-3 with fixed 0.5 weights; no `cosim_lambda`; equivalence to redux cosim is unverified |
| MS-SSIM structure anchor | present | `[train.mssim_opt] type = "mssim_loss"`, `window_size 11, sigma 1.5, K1, K2, L, in_channels` | `neosr/losses/ssim_loss.py:58-154`; `options/train_realplksr.toml:85-87` | true 5-scale MS-SSIM, zero padding (measured border bias, `docs/refactor/stage3/competitor/losses_metrics.md:148-170`); at GT 256 the coarse scales saturate; "MS-SSIM L1" (redux, with alpha) has no key: L1 + MS-SSIM as two slots |
| VGG19 perceptual with per-layer weights | present | `[train.perceptual_opt] type = "vgg_perceptual_loss", layer_weights = {conv2_2 = ..}, criterion = "chc" \| "l1" \| "l2" \| "huber"`, `use_input_norm, range_norm` | `neosr/losses/vgg_perceptual_loss.py:15-103` | features scaled x0.1 before the criterion (`:99`), so values match neither basicsr nor redux; `chc` here is unclipped (`:74`) = Charbonnier; default criterion `chc`, template comment `huber` (`options/train_realplksr.toml:111-114`); VGG19 only; the wiki's `patchloss` / `ipk` / `patch_weight` (NL:105-120) and redux `pd` / `fd` criteria (PH:1043) were not found in `neosr/` (grep) |
| DISTS as a loss | present, off | `[train.dists_opt] type = "dists_loss"`, `as_loss`, `load_weights` | `neosr/losses/dists_loss.py:37-80`; `options/train_realplksr.toml:116-118` | the DISTS metric is the same network (`neosr/metrics/calculate.py:303-312`): with `dists_opt` on, DISTS is not an independent judge (`options/train_realplksr.toml:42`, `docs/training_guide.md:57`); +37 % ms/update, +47 % steady VRAM at 0.15 (`experiments/s10/consult/brief.md:17`, `synthesis.md:9-10`); runs in fp32 outside autocast (`dists_loss.py:51`) |
| LDL | present, on | `[train.ldl_opt] type = "ldl_loss", criterion, ksize (7)` | `neosr/losses/ldl_loss.py:15-97`; `options/train_realplksr.toml:93-95` | code default criterion `l1` (`ldl_loss.py:28`; docstring and NL:159 say huber); no EMA-refined map (upstream LDL uses the EMA output; porting it was rejected as it can only shrink the term, `losses_metrics.md:28-30`, `:175-176`); measured ~inert at weight 1.0 (`docs/refactor/stage3/competitor/losses_metrics.md:28-30`); no map dump; no decay keys (SL:19-23 has them, unattributed) |
| Focal frequency | present, off | `[train.ff_opt] type = "ff_loss", alpha, patch_factor, ave_spectrum, log_matrix, batch_matrix` | `neosr/losses/ff_loss.py:9-70` (defaults `:30-31`) | default `ave_spectrum = true` differs from upstream FFL and from the 2x release (ave_spectrum false, log_matrix true, PH:819); template comment weight 0.35 (`options/train_realplksr.toml:120-122`) is neosr scale |
| Colour anchor (redux HSLuv) | partial | `[train.consistency_opt] type = "consistency_loss", criterion, blur, cosim, saturation, brightness, target` | `neosr/losses/consistency_loss.py:15-66`, `:163-208`; removed `color_opt` / `luma_opt` (`neosr/config.py:212-213`) | no HSLuv loss; consistency (Oklab chroma + CIE L*) is a stand-in, the mapping is DR's [D] (DR:153) and the port note's (`docs/refactor/stage3/owner_recipe.md:116-118`); rejected with `color = "y"` (`neosr/config.py:512-519`) |
| "Widen the consistency low-pass" (DX:363) | missing | only `blur = true \| false` | `neosr/losses/consistency_loss.py:52-55`, `:68-69` | the blur is a fixed 21-tap sigma-3 Gaussian; its width is not a key; it applies to the luma (L*) term only, the Oklab chroma term is compared per pixel unblurred (`:177-188`), so chroma near halos is constrained at full resolution |
| FDL with selectable backbone | present, on | `[train.fdl_opt] type = "fdl_loss", model = "dinov2" \| "vgg" \| "resnet" \| "effnet", patch_size, stride, num_proj (24), phase_weight, vgg_weights, dino_layers, dino_weights, fp32_backbone` | `neosr/losses/fdl_loss.py:292-413`; `options/train_realplksr.toml:97-100` | DINOv2 score has no x0.01 scale, the others do (`:413`); value grows with patch side, so weights do not transfer across patch sizes (bundle capability note); `num_proj` 24 vs official 256 (NL:79): the cost of raising it is unmeasured |
| Other perceptual backbones (ConvNeXt, LPIPS as loss) | missing | none | `docs/refactor/stage3/competitor/losses_metrics.md:113` | redux's ConvNeXt/DINO perceptual losses never had a gradient (same line); LPIPS exists as a metric only |
| Wavelet-guided GAN | present, on | `[train] wavelet_guided = true, wavelet_init = 80000` | `neosr/losses/wavelet_guided.py:8-49`; `neosr/losses/__init__.py:64-65`; `options/train_realplksr.toml:60-61` | implemented as the D input view (SWT HF bands of Y after `wavelet_init`), not as a pixel loss; `gw_opt` raises "gw_loss was removed" (`neosr/config.py:211`); the switch at 80k never A/B'd (no run reached it); one 8k screen of `wavelet_init = 0` was a marginal LOSS (`docs/refactor/stage2/DECISIONS.md:91`) |
| Soft GAN labels (2x release 0.9 / 0.1) | present | `[train.gan_opt] real_label_val, fake_label_val` | `neosr/losses/gan_loss.py:93-94`, `:108-109` | labels only for bce / mse / huber; hinge and rpgan take none; the D term is unweighted (`:88`) |
| Per-loss schedule (start, ramp, end) | present | `[train.<slot>_opt] start_iter, ramp_iters, end_iter` | `neosr/losses/__init__.py:40-44`, `:81-87`; `neosr/losses/base.py:80-121` | linear ramp to full weight only; no decay to a non-zero final weight (redux `final_weight`, PH:1027); the one local use (GAN start 1000 + ramp 1000) gave no gain at 8k (`docs/refactor/stage2/DECISIONS.md:47-48`) |
| Per-loss target | partial | `[train.<slot>_opt] target = "gt" \| "lq"` (legacy `train.match_lq_colors`) | `neosr/losses/__init__.py:27`, `:62-63`, `:77-80`; `neosr/losses/base.py:51-58` | no USM-sharpened GT target per objective (Real-ESRGAN `l1_gt_usm` / `percep_gt_usm` / `gan_gt_usm`) |
| Raw (unweighted) value per term | partial | none in the trainer; offline `scripts/term_grads.py` prints the weighted value | `neosr/losses/base.py:115-120`; `scripts/term_grads.py:44-45`, `:127-131`; `readme.md:195` | only weighted `l_g_<slot>` is logged; raw = value / (weight x ramp factor) has to be computed by hand, and the logged value is a window mean, so it is exact only outside a ramp |
| Per-term gradient norms | partial | `python scripts/term_grads.py -opt cfg.toml [--net_g] [--batches] [--iter]` | `scripts/term_grads.py:1-12`, `:33-85`; `readme.md:217-220` | offline, over all trainable G parameters, on training batches; prints norm, share of the summed **squared** norms and cosine with the total gradient (`:58-66`); no shared-layer variant, no fixed probe crops, no pairwise cosine, no JSON output. In-run: only `grad_sr_gan` / `grad_sr_ratio` for the adversarial term (`readme.md:205-216`) |
| Non-finite detection per term | present | automatic (`nonfinite_<key>`, `skipped_g/d`, `max_skip_streak`) | `neosr/models/phase.py:412-416`, `:483-486`; `readme.md:197-198` | covers values in applied updates, not per-term gradient finiteness |
| Frozen perceptual backbones | present | automatic | `neosr/losses/__init__.py:109-110` | answers NP:300's "never silently trainable" |
| MS-SWD, NCC, KL, PSNR loss | present, off | `msswd_loss`, `ncc_loss`, `kl_loss`, `psnr_loss` | `neosr/losses/msswd_loss.py:104-169`; `neosr/losses/ncc_loss.py:7-23`; `neosr/losses/kl_loss.py:9-28`; `neosr/losses/basic_loss.py:27-50` | no community evidence for any of them on photo detail (NL gap) |
| Contextual loss, HFEN / Sobel / Laplacian / HP residual, gradient-variance, A-DISTS, NR-IQA loss | missing | none | `docs/refactor/stage3/competitor/losses_metrics.md:108`, `:114-117` | CX, HFEN/Sobel/Laplacian, gradient-variance and A-DISTS were reviewed and skipped in the competitor harvest, on stated grounds (CX: for misaligned pairs; gradient-variance: weak paper baselines; A-DISTS: fragile implementation); an HP-residual term and an NR-IQA loss were not part of that harvest |

## 5. Framework work this implies (strategy-independent)

Candidates for the owner, ranked by how many community practices and research questions depend on them. None changes
a default; every objective-changing item is opt-in.

| # | feature | why (anchors) | size | risk | changes defaults? | how to verify | prerequisite for |
|---|---|---|---|---|---|---|---|
| 1 | **Raw and weighted value per term** as an offline "loss scale card": `term_grads.py` prints raw value, weight, weighted value, gradient norm and share per term; optional JSON output | the one balancing method the community actually used (RP:139-141); "log unweighted and weighted" (DX:369-373, HB:797-801, PF:213, NP:285-298); today only the weighted value exists (`neosr/losses/base.py:115-120`) | S | low: offline script, trainer untouched | no | unit test: raw x weight x ramp == weighted; reproduce the documented shares (81.2 / 16.5 / 1.4 / 0.53 / 0.35 / 0.000 % at P0) | G2-Q1, Q2, Q5, Q6, Q7 |
| 2 | **`term_grads.py` probe mode**: gradients w.r.t. `sr` or a named trunk module instead of all parameters; a fixed list of probe crops (skin, hair, flat, edge) instead of training batches; pairwise cosine between terms; a loop over saved checkpoints | the verification method nobody has run (RG:59, RG:169, FG:201, DX:371, PG:260-261; PG X13) | S-M | low: offline; probe-crop selection is a judgement call | no | shares sum to 1; cosine matrix symmetric; all-parameter mode stays bit-identical to today's output on a fixture | G2-Q1, Q6, Q7 |
| 3 | **Exact Charbonnier under a plain name** (alias of `chc_loss` with `criterion = "huber"`, `loss_lambda = 0`, no clipping) **with an `eps` key** (default 1e-12 = today's kernel), plus a doc row on the `chc_loss` default clip | every release stack uses the charbonnier criterion (PH:707-716, PH:816); the 2x release's pixel term uses eps 0.01, which neosr cannot express (`neosr/losses/basic_loss.py:204` fixes 1e-12); the owner's recipe is Charbonnier; the default `clip_min` silently zeroes gradients below 1/255 (`neosr/losses/basic_loss.py:170-172`) | S | none if exact; a non-default eps is an objective change (opt-in) | no | parity test: alias at the default eps == `chc_loss` with the override keys, bitwise | G2-Q5 |
| 4 | **LDL artifact-map dump** (offline script or opt-in validation visual) | "inspect its maps first" (RG:58, PG:231); local LDL question unresolved (`docs/refactor/DECISIONS.md:190`) | S | low | no | the dumped map equals the loss's internal map on a fixture (`neosr/losses/ldl_loss.py:71-96`) | G2-Q4 |
| 5 | **DISTS loss cost**: exact speed/memory rewrite (channels_last, compile, shared GT features) | DISTS is in both 1x releases (PH:714, PH:609) and is the only local signal; costs +37 % time, +47 % steady VRAM (`experiments/s10/consult/synthesis.md:9-10`, `:33-35`) | M | low if exact (ships behind a parity test) | no | bit or tolerance parity with `dists_loss` on fixtures; ms/update and steady VRAM before/after | G2-Q2, Q3 (cost only) |
| 6 | **Per-objective sharpened-GT target** (`target` gains a USM-sharpened GT view for pixel/perceptual slots; the D real target stays original) | official Real-ESRGAN practice vs halo reports (RG:56, FG:207, PF:217, DX:231; arms FG:213-216) | S-M | medium: an objective change; the USM definition must match the reference implementation | no (opt-in) | parity of the USM view against a reference implementation; default `target = "gt"` unchanged bitwise | G2-Q10 |
| 7 | **Config guards as warnings**: DISTS loss on while DISTS is the best-checkpoint metric; `ff_loss` / `kl_loss` without a pretrain; `wavelet_init` below ~40k | FG:405, PF:203; NL:16, NL:182, NL:210 | S | low (warnings, never an automatic change) | no | config tests | none (hygiene); overlaps group G5 |
| 8 | **High-pass band wrapper**: apply a criterion to HR - Blur(HR) (covers HP residual, HFEN-like terms and a `gw_loss` substitute) | SX:155-159, SL:26-27, NL:187 (removed locally: `neosr/config.py:211`); PG X7 | S-M | medium: gradient/edge terms can reward ringing and halos (PF:205, FG:195 [H]); the HP residual has no sigma or weight (SX:155-159); the only numbers are an unattributed HFEN example (kernel 7, sigma 1.0, weight 0.08, SL:26-31), and HFEN was numerically nil in the owner's redux run (`docs/refactor/stage3/owner_recipe.md:28-29`) | no (opt-in) | finite-gradient test under bf16; value zero for identical images; fixed-kernel determinism | G2-Q8 |
| 9 | **Consistency low-pass width as a key** | DX:361-365 (the only source that names an objective conflict rather than a weight); today only the luma term is blurred, the chroma term is per pixel (`neosr/losses/consistency_loss.py:177-188`), so a width key alone does not low-pass chroma | S | low; default must stay the 21-tap sigma-3 kernel | no | default path bitwise unchanged | G2-Q11 |
| 10 | **ConvNeXt-Tiny perceptual backbone** (as a `vgg_perceptual_loss`-style term or an FDL backbone) | PH:418, PH:421, SL:6; PG X8 | M | medium: no evidence, no weights; "texture stamping" risk (PH:421) | no (opt-in) | gradient is non-zero (the redux version had none); backbone frozen; fp32/bf16 gradient check | the ConvNeXt arm of G2-Q6 only |
| 11 | **Loss weight decay shape** (ramp to a non-zero final weight) | PH:1027 (redux schedulers), SL:19-23 (unattributed) | S | low; a pure function of the update, like the existing ramp | no | resume exactness test as for `ramp_iters` | none today |
| 12 | **HSLuv loss** | colour anchor of both 1x releases and the redux default block (PH:711, PH:606, PH:1048; the 2x release has only cosim 0.3, PH:823); owner's recipe used it | M | medium: the redux hue term had a bug (`docs/refactor/stage3/owner_recipe.md:62`); stand-in exists | no | gradient probe near grey; comparison with `consistency_loss` | only if the owner rejects the stand-in |
| 13 | **Contextual loss** | SI:17; SX:147, RP:127 (H); PG X7 | M-L | high: ~400 MB affinity per layer per image, NaN under AMP at large weights (DX:424), judged unsuited to aligned pairs (`docs/refactor/stage3/competitor/losses_metrics.md:115`) | no | fp32 distances; memory bound test | the CX arm of G2-Q8 only |
| 14 | **NR-IQA training loss on an unpaired stream** | PG X14 (warnings: never optimise toward NIQE, SKL:70 [generic]; NR metrics can favour hallucination, HB:863) | L | high; needs NR metrics (absent) and a second dataloader | no | - | G2-Q12; blocked (below) |

Not proposed: gradient-variance loss (removed from NeoSR for instability, DX:385; weak paper baselines,
`docs/refactor/stage3/competitor/losses_metrics.md:108`), A-DISTS, adaptive loss weighting (owner rule: no automatic
damping; `docs/refactor/stage3/competitor/losses_metrics.md:120`), the EMA-refined LDL map (one extra G forward per
step for a term that is already inert; `docs/refactor/stage3/competitor/losses_metrics.md:28-30`).

## 6. Research questions for the strategy phase

Mapping of the KB's PG experiments: **X5** (loss attribution) -> G2-Q2, Q3, Q4, Q5; **X7** (HF-band supervision) ->
G2-Q8; **X8** (perceptual backbones) -> G2-Q6, Q7; **X13** (per-term gradient norms) -> G2-Q1; **X14** (NR-IQA loss)
-> G2-Q12; the target half of **X11** (USM) -> G2-Q10. Outside this group: X4, X6, X15 (GAN weight ladder,
discriminator, generator levers: G3); X1, X9, X10, X12 (degradation, crops, data, faces: G1); X2, X3, X16
(metric calibration, selection, diffusion gap: G5).

Costs: "arm" = one seed 0 -> 20k (~0.6 GPU-h; ~0.85 with DISTS); a D10 decision needs two seeds; a default needs the
paired 20k -> 100k long run on top (~4.5-6 GPU-h per pair, `experiments/s10/consult/synthesis.md:70`).

| id | question | single-factor arms | prerequisites (data, feature) | rough GPU cost | blocked by | priority (reason) |
|---|---|---|---|---|---|---|
| G2-Q1 | What raw magnitude and gradient share does each available term have per unit weight in neosr-dev (template terms plus VGG, DISTS, FF, MS-SWD, FDL backbones), at P0 and along a run; do raw magnitudes or gradient shares predict the A/B outcomes we already have? (X13) | none: a measurement over existing checkpoints (P0, A0 at 8k/20k, T_DISTS) | section 5 items 1-2 (item 1 alone suffices to start) | minutes; no training | the cross-trainer half (same stack in redux) would require running the competitor | **high**: every weight question below needs a scale, and no one in the corpus has this data |
| G2-Q2 | Does adding DISTS to the template help beyond noise, and at what weight? (X5) | DISTS weight {0, 0.15 (run), one lower, one higher}, weights chosen from G2-Q1 magnitudes, not copied from redux 0.25 / 0.45 | none (seed 2 of T_DISTS is the held `experiments/s10/queue2.held.txt` arm) | 0.85 GPU-h for the held seed; ~1.7 per further weight (two seeds) | the owner's hold on tactic selection | **high**: strongest local signal and the community's largest perceptual term |
| G2-Q3 | FDL(DINOv2) and DISTS: redundant, complementary, or can DISTS replace FDL? | 2x2 FDL {0, 0.02} x DISTS {0, 0.15}: two cells exist (A0/B0, T_DISTS); missing noFDL and noFDL + DISTS (held T_DISTS_noFDL) | none | ~1.5 GPU-h one seed, ~3 two seeds | hold | **high**: FDL holds 74-81 % of the summed squared G-gradient norms (~2x the GAN term's norm) and a large share of the step time |
| G2-Q4 | LDL: inert, artifact brake or hidden stabiliser; does it flag valid texture? (X5) | LDL {0, 1.0} past 20k on two seeds; separately LDL at a weight with a non-zero gradient share (from G2-Q1) | item 4 (map dump) for the texture question | long pair ~4.5-6 GPU-h; a weight arm ~1.2 (two seeds) | misregistration behaviour cannot be tested (pairs are exact); artifact-suppression needs a regime with visible GAN artifacts | **medium**: unresolved locally, but removal saves only ~5 % time |
| G2-Q5 | How strong must the pixel anchor be in the GAN stage; L1 or Charbonnier? (X5) | pixel weight {1.0, one lower, 0} with MS-SSIM and consistency fixed; separately L1 vs Charbonnier at matched gradient share | item 3 for the Charbonnier arm; a colour/luminance drift check (G5) | ~1.2 GPU-h per extra arm (two seeds) | the 1x no-pixel regime of the releases needs 1x degraded data (OTF parked) | **high**: the sources are split exactly where the owner's goal sits (no pixel term in the 1x releases vs a reconstruction baseline, DX:321-323; the "unless you want hallucination" line, PH:491, is uncited generic text) and the owner's own run collapsed without an anchor (confounded) |
| G2-Q6 | Which feature term gives the best texture at equal fidelity: FDL-DINOv2 (current), FDL-VGG, VGG19 perceptual, DISTS? (X8) | swap the feature term, each at a matched gradient share; ConvNeXt as a later arm | G2-Q1; item 10 only for ConvNeXt | 3 arms ~1.8 GPU-h one seed, ~3.6 two seeds | ConvNeXt arm: missing feature; fur/foliage probes do not exist in a people-only data set | **high**: the corpus has no backbone comparison at all |
| G2-Q7 | VGG19 layer set: mid layers (conv2_2 / conv3_4 / conv4_4, as in the 2x release) vs the default with conv5_4 | `layer_weights` only | G2-Q1 for matching | ~1.2 GPU-h (two seeds) per arm | none | **low-medium**: only meaningful if G2-Q6 keeps VGG |
| G2-Q8 | Does HF-band supervision add micro-texture with less grain than more GAN weight? (X7) | base; + FF; + HP-band L1; + CX; each magnitude-matched (FDL is already in the base) | FF: none; HP band: item 8; CX: item 13 | ~0.6 GPU-h per arm-seed | HP and CX arms: missing features; "grain" needs a flat-region measure (G5) | **medium** (FF arm is cheap and runnable); **low** for CX |
| G2-Q9 | FF settings: neosr default (`ave_spectrum = true`) vs upstream/release (`ave_spectrum = false`, `log_matrix = true`) | one key at a time | none | ~0.6 GPU-h per arm-seed | none | **low**: only if G2-Q8 shows FF does anything |
| G2-Q10 | Do USM-sharpened L1/perceptual targets add real detail or only edge acutance? (X11, target half) | U0 original; U1 perceptual sharpened; U2 pixel + perceptual sharpened; D target original throughout | item 6; an edge-overshoot / halo measure (G5): FR metrics against unsharpened GT penalise a sharpened-target arm by construction | ~0.6 GPU-h per arm-seed | missing feature; U3 (sharpened LQ source) and the noise factor need OTF (parked) | **low-medium**: a real community conflict, but the judges are not ready |
| G2-Q11 | Is the colour anchor needed with the current objective, and does its low-pass width matter? | consistency weight {1.0, 0}; blur width as a second arm | colour-bias metric (HB:830-837; G5); item 9 for the width arm | ~1.2 GPU-h (two seeds) per arm | dehaloing cannot be tested on clean data (no halos in the LQ) | **medium**: the owner's collapse was confounded with the colour anchor |
| G2-Q12 | Does a small NR-IQA loss on real unpaired inputs help? (X14) | template; + NR loss at a matched small weight | NR metric, unpaired real-LQ set, item 14 | unknown | no real LQ data, no NR metric, OTF parked; the sources warn against NR as a target (NIQE: SKL:70 [generic]; NR favours hallucination: HB:863) | **low**: blocked three ways |
| G2-Q13 | Is FDL `num_proj` 24 leaving quality on the table (official 256; "raise at the end of a finetune", NL:79)? | `num_proj` {24, higher} as a late fine-tune fork | cost measurement first | one fork ~0.6 GPU-h plus the cost probe | meaningful only near the end of a long run (none exists) | **low** |

## 7. Questions for the owner

1. **The hold.** Do the two held arms (T_DISTS seed 2, T_DISTS without FDL; `experiments/s10/queue2.held.txt`) resume
   as they are, or wait for the whole study? They are the cheapest high-value runs in this group.
2. **Measurement before arms?** May the offline tooling (section 5, items 1-2) be built first, so that every later
   weight is chosen from a measured scale rather than from a redux number?
3. **Which matching rule defines "magnitude-matched"**: equal weighted loss contribution (the community's practice,
   RP:139) or equal gradient share (the hypothesis, RG:169)? The two give different weights; the arms need one rule.
4. **How much fidelity may the detail model give up?** The releases drop the pixel term at 1x; our evidence and the
   owner's own run favour an anchor. Is a no-pixel or weak-pixel regime in scope for x4?
5. **DISTS cost.** Is +37 % time and +47 % steady VRAM acceptable for a DISTS term if it survives two seeds, or is the
   exact cost rewrite (item 5) a precondition?
6. **Colour anchor.** Accept `consistency_loss` as the stand-in for HSLuv, or is a (fixed) HSLuv loss wanted?
7. **New loss terms.** Should any of HP-band, ConvNeXt, contextual loss be built before evidence exists (they are
   prerequisites only for single arms), or only after G2-Q1/Q6 show a gap the existing terms cannot fill?
8. **LDL.** Is a long pair (~4.5-6 GPU-h) worth spending on a term that saves ~5 % time, or does D1 ("keep") stand
   without further runs?
9. **USM targets.** Is the per-objective sharpened target wanted as an ablation switch at all, given that its
   LQ-source half is OTF (parked)?
10. **Scope by scale.** The only release loss stacks are 1x and 2x with degradations. Is the loss study limited to
    clean x4 for now, or should 1x/2x questions be queued for when degraded data exists?
11. Still open from the recipe analysis (`docs/refactor/stage3/owner_recipe.md:146-150`): was the final redux model
    judged by eye or by val8 DISTS, and would a recipe without a D be acceptable if it tied the GAN arms?

## Verification

Adversarial pass, 2026-11-13 (CPU only; no training). Opened every source below at the cited line.

**What was checked**
| area | count | how |
|---|---|---|
| KB anchors (parts 1-2, plus those parts 5-6 lean on) | ~150 anchors in ~60 spans | PH (both 1x stacks, 2x stack, the redux block, 5.2, 7.x snapshot keys, cheat sheet), PG, DR, TG, NL (all loss sections), RG, FG, SKL, SKT, SKB, SI, SX, SL, DX (+ footnotes), HB, PF, RP, NP, RX, UH3, AN, and the PH card for trust level |
| neosr-dev code claims (parts 3-5) | ~55 | `neosr/losses/*.py` (all 16 registered terms), `neosr/config.py`, `neosr/models/phase.py`, `neosr/metrics/calculate.py`, `options/train_realplksr.toml`, `scripts/term_grads.py`, `readme.md`, `docs/training_guide.md` |
| local numbers | ~35 | the cited docs, plus a re-run of `scripts/ab/analyze.py` on the s10 run files: T_DISTS, OWN, NoGAN_DISTS (window 12k-20k vs pooled), B2_noldl / B2s2 (pooled and seed-matched), val8 at 20k |

All three release stacks, the redux block, the TG curriculum, the NL keys and examples, the Real-ESRGAN facts, the Siax
card, every code path/default in part 4 and the s10 window deltas (T_DISTS +0.0025 / -0.0060, OWN -0.0093 / +0.0062 /
PSNR -0.277, B2s2 -0.0066 / +0.0088, NoGAN_DISTS vs T_DISTS) reproduce as written.

**Corrections made (before -> after; source)**
| # | before | after | source |
|---|---|---|---|
| 1 | "no ablation of any loss weight exists anywhere in the corpus" | no controlled ablation with numbers; one numberless DF2K/Urban100 perceptual-weight ablation (PF:209 [C]) and per-term tests (DX:337) exist | PF:209, DX:337 |
| 2 | release stacks: "one playbook, apparently one author" | an LLM-written extract of Phhofm's configs, originals not in the KB | PH card (TL;DR, caveats) |
| 3 | 1x sharpen snapshot: "no batch or patch size" | no batch size; `lq_size` 96 (both 1x snapshots) | PH:740, PH:635 |
| 4 | PH:491 "off only if hallucination is wanted" labelled [discord] (parts 1.3, 2, 6 Q5) | [generic]: PH's uncited LLM-written section 5.2 | PH card ("sections 4-6 ... generic advice without sources"), PH:485-491 |
| 5 | DX:323 quoted as "keep a reconstruction anchor of some kind" [discord] | DX's synthesized action; the cited tests all kept pixel loss; the opposite assertion (remove all L1/SSIM) added as uncontrolled | DX:321-325, DX:600 |
| 6 | "not universal law" (NP:260) | "config-conditional" (NP:260); HB:789's actual wording | NP:260, HB:789 |
| 7 | PH:418 ConvNeXt direction under "message-cited" [discord] | uncited LLM paraphrase, no weights | PH card |
| 8 | "In NeoSR the perceptual-type losses ship at weight 0" | RP says configs "often include" them at 0; neosr-dev comments them out and drops zero-weight G terms | RP:133, `neosr/losses/__init__.py:90`, template :111-130 |
| 9 | CX "the tool for misaligned pairs [paper] (SKL:17, SKL:54, RG:89)" | SKL generic wording; RG:89 also proposes a bounded CX-vs-VGG test on aligned pairs | SKL:17, RG:89 |
| 10 | official logs "contradict" the one-order heuristic | RX:239 is within one order (4.5x); only UH3:212 is 2+ orders apart; both at iteration 100 | RX:239, UH3:212 |
| 11 | template "trains with terms three orders apart"; FDL "holds 74-81 % of the G gradient" | shares are of summed **squared** norms; in norm FDL is ~2-15x every term except LDL (~1,600x) | `scripts/term_grads.py:58-65`, `experiments/s9_ab/NOTES.md:25-26` |
| 12 | G16 sweep 0.75 / 0.1 / 0.05 / 0.02 as one series | 0.02 = 29.36 is new code (and the stage-2 net effect); like-for-like 0.02 vs 0.05 = 29.36 vs 28.58 | `docs/refactor/stage2/DECISIONS.md:42-43`, `:49-50`; `stage2/PROGRESS.md:73` |
| 13 | G14 "(study) WIN under D10 rules" without qualification | D10 tau binds only for arms after 2026-11-12; retroactive re-gating was rejected; numbers re-run and confirmed | `docs/refactor/DECISIONS.md:222`, `experiments/s10/consult/synthesis.md:32` |
| 14 | wavelet view "never A/B'd locally" | one marginal 8k LOSS screen of `wavelet_init = 0` exists (new part-3 row; part 2 and 4 updated) | `docs/refactor/stage2/DECISIONS.md:91`, `:99-100` |
| 15 | USM comparison labelled [release] | practitioner comparison [C], not a release config | RG:141, FG:161 |
| 16 | item 8 "no sigma or weight exists anywhere" | HP residual has none; SL gives an unattributed HFEN example (kernel 7, sigma 1.0, 0.08), nil in the owner's run | SL:26-31, `owner_recipe.md:28-29` |
| 17 | item 12 HSLuv "colour anchor of all redux stacks" | both 1x releases and the default block; the 2x release has only cosim | PH:604-611, PH:707-716, PH:814-823 |
| 18 | raw value "undefined during a ramp" | computable as value / (weight x ramp factor); window means make it exact only outside a ramp | `neosr/losses/base.py:112-115` |
| 19 | inert-terms source `owner_recipe.md:29-30` | `:28-29` (plus `:55-57`) | owner_recipe.md |
| 20 | last part-4 row: all missing terms "reviewed and skipped in the harvest" | HP residual and NR-IQA loss were not part of it | `losses_metrics.md:104-120` |
| 21 | NR-IQA warnings "(SKL:70, HB:863) warn against optimising NR" | SKL:70 is about NIQE only (generic); HB:863 says NR favours hallucination | SKL:70, HB:863 |

**Additions**: Charbonnier eps gap (neosr fixes `sqrt(d^2 + 1e-12)`, the 2x release uses eps 0.01; part 4 row and item
3 now carry an `eps` key); consistency blur applies to luma only, chroma is per pixel (part 4, item 9); LDL code default
criterion `l1` vs doc `huber`; GAN-weight row gains PH:1021 [discord], AN:139 [generic] and the template's own 0.3;
GAN start/ramp local no-gain screen (part 4 schedule row); EMA-refined LDL listed as not proposed; SL's values match the
owner's inert redux entries; T_DISTS has the smallest late `grad_norm_g` maximum (one seed); DISTS loss runs fp32;
DX:303 added as the only wavelet timing advice; the x4 default block names no scale.

**Rule check**: no averaged conflict found; no redux/basicsr weight presented as a neosr value (part 6 Q2 explicitly
re-derives weights); no work planned on the OTF track (U3, Q10, Q12 stay blocked by it); no proposal changes a default
(item 3's non-default eps is marked opt-in); local limits (clean x4, <= 20k, no 80k crossing) are stated in part 3.

**Still unverified**
- The PH config values against the original config files (not in the KB) and whether those exact configs produced the
  released models.
- The redux Charbonnier form at the release's version (only the local redux checkout was read; ideas only, no code
  used).
- Stage-2 numbers (G16, G17), the owner's run numbers (G37) and the competitor probes (MS-SSIM border, LDL share): taken
  from the cited docs; their run files and logs were not re-read.
- "Best val8 LPIPS/DISTS of all arms" for OWN: checked at 20k against 7 other arms only (OWN 0.1756 / 0.1231 lowest).
- Equivalence of `consistency_loss` to HSLuv + cosim, and of `chc_loss loss_lambda` to redux cosim: open, as stated.
- Part-6 GPU-hour costs: arithmetic from the ~37 min arm and +37 % DISTS figures, not re-measured.
