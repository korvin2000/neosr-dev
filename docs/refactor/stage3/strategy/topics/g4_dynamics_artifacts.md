# Curriculum, optimizer/schedule, EMA/finetune, precision, artifacts, faces/skin/hair, objective

Input: `experiments/study/bundles/g4_dynamics_artifacts.json` (226 claims, 17 conflicts, 20 gaps, 14 capabilities,
10 local findings), KB sources spot-checked (TG:1-33, NL:4-16, PH:1000-1090, RP:168-184, DX:271-307, DX:436-442,
PG:204-292, PG:446-469) and neosr-dev code read for every status below. Nothing here is a recommendation: the owner
decides.

Reading rules used throughout:

| label | meaning | weight |
|---|---|---|
| official | trainer wiki / upstream doc (TG, NO, NL, UH*, SI, RX) | what the trainer authors say; mostly no measurements |
| release_config | a config snapshot of a released model (PH:553-900) | what was actually run; no ablation behind it |
| discord_cited | a message-cited community report (PH cheat sheet, RP, DX, RG E-items, HB) | one practitioner, usually one run, often confounded |
| paper | a literature result quoted by a synthesis (not re-read here) | sound in general, untested for this setup |
| generic | uncited LLM-written text (SK*, AN, parts of PH) | low trust; numbers are not empirical |
| hypothesis / design | a proposal or a synthesis author's own choice (DR (D)/(H), PG (D), NP H, RG designs) | not evidence |

Bundle evidence mix: 63 discord_cited, 60 hypothesis, 31 official, 28 generic, 16 paper, 15 design, 13 release_config.
**No source in the corpus ran a controlled ablation** (RG:8, FG:15, DX:10, PG:8); every number below is a setting someone
used, not a setting someone showed to be better. Scales: `redux` = traiNNer-redux YAML weights, `neosr` = NeoSR TOML,
`basicsr` = BasicSR/Real-ESRGAN. Weights never transfer across scales.

## 1. What the community converges on

**1.1 Curriculum (stages, lengths, hand-offs)**

- **Easy before hard, fidelity before GAN** [official, redux]: x2 bicubic from scratch -> x4 bicubic initialised from x2
  (`strict_load_g` false) -> x4 with degradations and all losses, initialised from the x4 pretrain (`strict_load_g` true).
  Fidelity stages use one MS-SSIM-L1 loss; LR 2e-4 from scratch, 1e-4 when fine-tuning, G = D = 1e-4 in the GAN stage;
  fidelity runs "until validation metrics peak" (x2 often >= 500k, x4 50k to 250k+), the GAN stage "until it looks good",
  with a larger `lq_size` and a smaller batch (TG:7-33). TG gives **no** loss weights, GAN weight, ramp, EMA, warm-up or
  crop numbers for the GAN stage (TG:28-31).
- **The GAN stage is one run with perceptual and GAN losses on together** [official, redux] (TG:29). No trainer doc in the
  corpus documents a ramp key (DR:166).
- **Start from a matched pretrain** [official card + release practice + synthesis]: Siax from a 4x PSNR model (SI:15,
  official card); Helaman's x4 DAT2 / ATD / DRCT-L releases from the paper pretrains DAT_2_x4, 003_ATD_SRx4_finetune,
  DRCT-L_X4 (PH:1199-1201; RP:166 calls the DRCT-L parent an "ImageNet pretrain"); his RealPLKSR releases from his own
  earlier release 4xNomos2_realplksr_dysample, and 1xDeH264 from 1xDeJPG_realplksr_otf (PH:1196-1198), so the releases
  show "matched parent", not "non-GAN parent" (the parents' objectives are not stated). A GAN parent transfers its
  defects (HB:81, HB:83, FG:241; synthesis). A NeoSR maintainer suggested at least ~30k pixel-only updates before
  perceptual/GAN phases for OmniSR from scratch [discord_cited via HB [D04], arch-specific] (HB:681); local P0 is 20k.
  Paper framing: early GAN feedback rewards texture before geometry (RG:41; SKT:14 generic).
- **Distortion pretrain = L1 + MS-SSIM only** [release_config, neosr scale]: DRCT-XL x2 pretrain L1 0.6 + MS-SSIM 0.4,
  AdamW 1e-4, betas 0.9/0.99, wd 0, MultiStep 60k/120k gamma 0.5, EMA 0.999 (PH:889-895; total length not printed,
  PH:853-855); an ESRGAN pretrain on L1 + MS-SSIM for about 300k (PH:392, weights not stated). This is the only cited
  concrete NeoSR fidelity split.
- **GAN-stage budgets from a pretrain** [release iteration counts, loss stacks not stated]: RealPLKSR 139k-265k at batch
  6-8, patch 64, 5,630-7,000 images (PH:1196-1198); DAT2 140k / ATD 180k / DRCT-L 200k at batch 2-4, patch 48-64
  (PH:1199-1201), all six on static data (OTF "No"); 1x releases 250k (PH:664, PH:559), x2 RealPLKSR 500k (PH:769);
  Siax 175k at batch 2, HR 256 / LR 64 (the local tile size), 2,700 images = 900 unique (SI:9-14, official card);
  Compact typically under 40k, some 5k-10k from a very on-point pretrain (RX:273, official beginner wiki).
- **Stage = new experiment, G as a pretrain, fresh D and optimizer; exact resume only for an unchanged config**
  [discord_cited] (HB:599, HB:700-711, PF:177, PF:188, RG:64). One major variable per branch (PF:177, HB:221).
- **GAN weight is raised late, as a separate decision** [discord_cited; messages in neosr.md and training-06.md, weight
  scale (redux or neosr) not stated]: start 0.1 (or 0.2), raise to 0.3 after validation stops drifting; 0.5 was used
  once (an OTF config) and called more unstable (PH:1004, PH:1021). The same practitioner uses loss schedulers
  (`start_iter` / `final_weight` / `final_weight_iter`) but changes many knobs at once (PH:1027, RG:100: Charbonnier
  0.8 -> 0.3 with GAN from ~10k, trainer not stated, confounded). The released stacks themselves run one constant GAN
  weight for the whole run: vanilla 0.15 / 0.14 (1x) and 0.05 with labels 0.9 / 0.1 (2x) [release_config, redux scale]
  (PH:608, PH:713, PH:822). The neosr-dev template runs bce 0.3 from update 1 (neosr scale,
  `options/train_realplksr.toml:102-105`); none of these numbers transfers across scales.
- **Wavelet-guided D is a late objective** [official, neosr]: "train at least ~40k before enabling", doc example
  `wavelet_init = 80000` (NL:12, NL:16); [discord_cited, musl `advice`] enable it while LR remains, not at the tail of a
  decayed schedule (DX:303, DX:305). Local: `wavelet_init = 0` on a 20k pretrain (below NL:16's ~40k) lost at 8k
  (section 3).
- **HF quality arrives late; short proxies under-report texture and can invert ranks** [generic SK* text citing the
  frequency principle] (SKM:39); SKR:52 adds an unsourced proxy budget of 30k-80k updates on a data subset, with rank
  inversions (bigger patches / lower LR look worse early, win late). This bears directly on 8k/20k screens: the D10
  decision window (12k-20k) sits below that (generic) bracket.
- Hypotheses and designs only: SX stage lengths 100k / 100k / 200k+ with a 50k GAN warm-up (SX:182-196, a speculative
  reconstruction; verified Siax is 175k total); a 5k GAN ramp (SKR:14); a 20k-100k perceptual-only prefix (DR:140 (H));
  "skip fidelity stages when a matched pretrain exists" (DR:55 (D)); the plateau playbook with Adam-moment reset and D
  pause (SX:206-218, no supporting observation); a small-subset / high-LR curriculum (DX:311-315, untested by its
  proposer); a stage engine (NP:63, NP:441); replay 10-30 % of the old distribution (SKR:43; FG:232 gives no number).

**1.2 Optimizer and schedule**

- **AdamW, betas [0.9, 0.99], weight decay 0** [official neosr doc example + redux guideline] (NO:27-28, TG:14); every
  release config uses AdamW for G and D (PH:672-702, PH:567-598, PH:779-808).
- **Three released schedules** [release_config, redux scale]:

  | release | G lr / D lr | wd | warm-up | EMA | clip | length | MultiStep | anchors |
  |---|---|---|---|---|---|---|---|---|
  | 1x sharpen | 1e-4 / 1e-4 | 0 | 2000 | 0.9995 | off | 250k | 125k/200k/225k/237.5k, gamma 0.5 | PH:664-702 |
  | 1x restore | 1e-4 / 1e-4 | 0 | none | 0.999 | off | 250k | same | PH:559-598 |
  | 2x RealPLKSR | 5e-5 / 1e-5 | 0.01 | none | 0.999 | on | 500k | 50k/100k/150k/200k/300k/400k, gamma 0.85 | PH:769-808 |

  All use MultiStep, none cosine (PH:588, PH:693, PH:798; the DRCT-XL pretrain too, PH:893). The snapshots lack batch
  size (1x) and D, degradation and crop keys (2x) (PH:553, PH:660, PH:765); both 1x runs are OTF high-order runs with
  `lq_usm` and `lq_size` 96 (PH:633-635, PH:738-740).
- **Schedule-free optimizers need `schedule_free = true`, EMA is recommended, warm-up goes through `warmup_steps`**
  [official, neosr] (NO:77-93). `soap_sf` is the only place an official NeoSR doc gives different G and D LRs (1e-3 /
  1e-4, NO:136-145). NO gives syntax examples (5e-4, 8e-4, 1e-3), not advice on LR by stage (NO:13, NO:82, NO:121).
- **Leave the shipped train section alone without a reason** [official beginner wikis] (UH3:156, RX:183); lower LR on
  repeatable divergence or a stable plateau, never because one checkpoint looks bad [synthesis] (HB:748).
- **EMA + cosine annealing from scratch over-smoothed** [discord_cited, one report, confounded] (RP:172): the report
  does not separate EMA from cosine, so it is weak evidence against cosine as such; "cosine not default" is DR's design
  reading (DR:186).
- **Warm-up about 2 % of the stage, ~12k in unstable phases, never a very short ramp at a high LR** [discord_cited via
  HB, NeoSR maintainer [D09]] (HB:742; HB:738 lists "commonly 3.2k-10k"); releases use none or 2000 (PH:560, PH:665).
- **RealPLKSR from a pretrain** [discord_cited, arch-specific, one 2024 message]: LR 1e-4 with warm-up, dropout only
  after ~60k, larger GT later (PH:1090; HB:742 [D10]). The local template fine-tunes realplksr at 2e-4 with no G
  warm-up (`options/train_realplksr.toml:64-68`).
- Hypothesis/design: milestones at 60 % / 85 % (DR:185 (D)); fine-tune LR 3e-5 (DR:183 (D)).

**1.3 EMA and fine-tuning**

- **EMA on, decay 0.999** in every release stack (0.9995 in one) [release_config] (PH:561, PH:666, PH:771, PH:891);
  Real-ESRGAN x4plus 0.999 [upstream basicsr config, via RG "code + community + derived"] (RG:63). Half-life at 0.999 is about 693 G steps [derived] (RG:172).
- **Score raw and EMA at the same step; neither is assumed better** [paper + discord_cited] (PF:58, PF:269, RG:63,
  DX:442-444). One case where EMA was too soft and the non-EMA weights seeded the perceptual finetune (RP:172-174).
- **Keep a step-0 baseline of every finetune; early quality may belong to the parent** [discord_cited] (DX:273, DX:287-291).
- **Detail first, then detune** [discord_cited, DX `reported_test`, ONE interim message]: a detailed (BHI
  TOPIQ-filtered-data) model fine-tuned at LR 1e-5 on the reporter's own less-oversharpened HR gave better detail than
  his clean-data scratch model with a reduced sharp edge; 50k *planned*, "seems to be working so far" (DX:279-283);
  PF:183 restates it as done. No optimizer, arch, scale or data sizes.
- **Split conservative and strong variants from one pretrain** [release practice] (DX:295-299).
- **Weight interpolation between same-lineage checkpoints as the fidelity/texture dial** [paper (ESRGAN) + [E]/[H]]
  (RG:65, PF:184, FG:250-252): one release practice, Helaman interpolating two late RealWebPhoto v4 checkpoints to
  trade sharpness against artifacts (RP:56, discord_cited), but no measured photo-GAN result and no alpha value
  anywhere in the corpus.
- Generic/hypothesis: finetune LR 3-10x lower, 50k-150k, mix 10-30 % old data (PH:523-530, SKR:42); "EMA is almost
  always cleaner" (SKR:26).

**1.4 Precision and stability**

- **bf16 over fp16; on `grad_norm = inf` enable clipping, try bf16, disable AMP to confirm, optionally fp32 early in the
  GAN stage** [discord_cited] (PH:1006, PH:1071-1072, RP:180-184). BF16 vs FP16 changes outputs measurably "across many
  SR models" (RP:182, RP:359; apparently inference outputs, training is not stated), so scores are comparable only at a
  fixed precision; no magnitude is given. neosr-dev's training validation is fp32 whatever the training precision
  (section 4).
- **AMP failures can be silent** [discord_cited, confounded anecdotes]: NaN gradients with large CX weights while
  training continued; several new losses at once then NaN under bf16 (DX:424-434). Monitor finite losses *and* gradients.
- **NaN after a batch change: suspect D LR first** [discord_cited] (PH:1009, PH:1087). Some NaNs were aggressive defaults,
  not trainer bugs (NP:129, NP:596).
- **No universal clip threshold** (DR:122, DR:551; upstream fixed 1.0, NP:125); releases run clip off at 1x and on at
  2x (PH:664-667, PH:772); one release note (UltraSharp) reported that clipping greatly helped its stability, threshold
  not given [discord_cited] (DX:424).
- **Prove the load**: step-0 validation against standalone inference plus loaded/missing/unexpected key counts
  [discord_cited] (DX:273, FG:242, HB:1069). Smoke test: 100-500 fp32 updates, one pixel loss, overfit 8-32 pairs, save
  and reload (HB:33, HB:673, DR:17).
- The fp32 + larger-patch "polish pass" is one practitioner's habit with no quantified gain (PH:1007, PH:1074).

**1.5 Artifacts** (all rows are checklists; no source measured a fix)

| symptom | community first checks, in the order most sources give | anchors (evidence) |
|---|---|---|
| process rule | one change per branch from a common checkpoint; data and degradation before loss weights | PF:177 [H], HB:221 (synthesis actions) |
| GAN grain, worms, speckle on flats | same crop on the non-GAN anchor; HR tiles for grain; fork earlier, lower GAN weight **or** D LR; LDL; SN on D; wavelet D | FG:415, PF:229, HB:1065 (synthesis checklists); PF:182 ([C] Helaman, fork + lower GAN or D LR); SKT:36 (generic) |
| halos, ringing | edge profiles of HR and LR, USM flags; disable target sharpening one source at a time; low-LR finetune on cleaner HR | PF:230, FG:166, FG:416, DX:585 (discord_cited); SKT:39 (generic) |
| waxy / plastic | degradation too hard for the real inputs; raw vs EMA; noise severity; beauty-retouched HR; add mild/clean LQ; do not just train longer | HB:1064, RG:220, RP:172 (discord_cited/hypothesis) |
| checkerboard, grid, dots | flat fields, ramps, impulses; residual by pixel phase; GAN-off checkpoint; parent model; nearest+conv or ICNR; resize/JPEG phase | RG:77, HB:1063, PF:233 (synthesis); DX:446-448 (`reported_test`, June 2024: a NeoSR-vs-redux grid failure separated from milder patterns present in several **RealPLKSR** models, causes not all resolved); RP:363 (cited); SKT:38 (generic) |
| colour drift | RGB means of LR/HR/output; range, channel order; raw vs EMA; drop one loss from the same pretrain | HB:1062, PF:234, FG:417, PH:415 |
| texture in bokeh | raise the smooth/defocus share of crops; defocused HR and clean inputs as negatives | FG:419, PF:232, RG:45 |
| quality drops late | sweep earlier full checkpoints; fork earlier; stop at the best | PF:235, PF:182, HB:1078 |
| "AI look", repeated motifs | data diversity, LDL, lower GAN, interpolate toward the anchor | SKT:44 (generic), FG:415 |
| diagnostics | residual `SR - bicubic(LR)` (SKT:31, SKT:47 generic); DeSRA diff against the L1 sibling (SKA:58 paper) | |

**1.6 Faces, skin, hair**

- **Nothing is measured**: no skin, pore, hair or fur result in the corpus; RG A02, PG X12 and the DX stage D are unrun
  designs (FOCUS:279, PG:463, RG:146). One observed datum: LQ 192 / batch 3 gave better lines and a worse eyelash,
  crop and batch not separated (RG:104).
- [paper-derived] Identity-bearing features (eyes, teeth, moles, wrinkles, hairline) are fidelity constraints (RG:47);
  region-aware *sampling* before region losses; soft masks, valid-pixel normalisation, no hard compositing (RG:61-62).
- [hypothesis] Do not force pores; smooth skin stays smooth (PF:246, DX:531). Waxy skin: check noise severity and
  beauty-retouched HR first (RG:220, RG:147). Etched hair: crop context, saturation, slanted-edge halos, raw vs EMA,
  clean vs USM targets (RG:219).
- [generic, unsourced numbers] region weights x3-10 on eyes/teeth/lips with skin x1; ArcFace loss 0.1-1 (SKT:56-57);
  fur needs crop 64 -> 96+ and mid-layer perceptual weights (SKT:37, SKT:62).
- PF:27's "little or no adversarial loss" is the row for the objective "accurate restoration, text, faces, archival
  evidence"; the owner's case is PF:28, "natural photographic upscaling": strong reconstruction pretrain, modest
  perceptual loss, carefully introduced GAN, failures wax skin and generated freckles. Both are unlabelled table rows;
  PF:27 reads as the identity-constraint side, not as a rule against GAN on portraits.

**1.7 Objective**

- [paper] Pixel losses return a central tendency of the posterior, so averaging plausible textures is blur (SKM:9-12);
  the perception-distortion trade-off is distribution-level: keep a fidelity checkpoint beside the perceptual one
  (PF:33, FG:42).
- [release_config + one posted template block, redux scale] **Neither 1x release stack nor the redux default block
  Helaman posted (PH:1005, discord_cited) has an L1/Charbonnier term**; MS-SSIM 0.3-0.5 + HSLuv 0.6-1.0 (Charbonnier
  criterion) + cosim 0.3-1.0 anchor fidelity; only the x2 stack has Charbonnier 0.1 (PH:707-716, PH:604-611, PH:816,
  PH:1038-1054). No source explains the absence. Against it: the official redux GAN stage lists "MS-SSIM L1" (TG:29),
  by its name a combined MS-SSIM + L1 loss, and Helaman's own 2025 schedule keeps Charbonnier 0.8 -> 0.3 (RG:100,
  confounded).
- [DX synthesis action over discord reports] Keep a reconstruction anchor unless an ablation justifies removal; add
  perceptual / CX / GAN one at a time (DX:321-325, DX:561). The cited evidence is thin: Zarxrax's individual-loss tests
  all kept a pixel loss (DX:321); the contrary chat claim is an assertion (DX:869-873, footnotes E044-E046).
- [synthesis thesis] Natural texture depends on the whole data distribution and objective, not on the backbone or a
  sharpness loss; a bigger loss stack sometimes did worse (DX:23, DX:516, DX:937).
- Designs: PG's four-condition definition of "diffusion-level detail" (PG:37-42); three detail labels preserved /
  reconstructed / synthesized (DX:52-56); the fidelity-creativity ladder (SKT:68). **No GAN-versus-diffusion number
  exists** (PG:44, SKA:71).

## 2. Open conflicts

| # | question | positions (anchors, evidence) | what would settle it | can our setup settle it now? |
|---|---|---|---|---|
| C1 | EMA or raw as the candidate and the stage seed | EMA almost always cleaner, GAN stage from stage-1 EMA (SKR:26, SKR:12, generic) / EMA + cosine from scratch over-smooth, seed from raw (RP:172-174, discord_cited, one case) / EMA lags a new regime (DX:444) / score both (RG:63, PF:58) | raw and EMA scored at identical updates along one run, then two finetunes seeded from each | Partly: needs raw-G scoring (missing, section 4). Then yes, on clean x4 |
| C2 | EMA decay | 0.999 (PH:561, PH:771, PH:891, RG:63) / 0.9995 (PH:666) / 0.9999 for very long runs (SKR:26, generic) | decay arms at a fixed length, raw as control | Yes for <= 20k finetunes only if `ema_power` is off or accounted for: with the template's 0.75 the decay reaches 0.999 after ~10k EMA updates and 0.9995 after ~25k (`phase.py:371-374`, derived); says nothing about 250k+ runs |
| C3 | Pixel term in the GAN stage | none in both 1x stacks and the posted redux default (PH:707-716, PH:604-611, PH:1038-1054, release_config) / Charbonnier 0.1 at x2 (PH:816) / "MS-SSIM L1" in the official GAN stage (TG:29, official) / Charbonnier 0.8 -> 0.3 in Helaman's 2025 schedule (RG:100, discord_cited, confounded) / never drop it (PH:491, generic; DX:323, DX synthesis) / L1 1.0 (SKL:52, SX:145, generic) | pixel-weight arms with the other anchors fixed, judged on fidelity guard + perceptual judges | Yes (L1 weight is one key); the redux anchors HSLuv/cosim have no exact neosr twin |
| C4 | D LR relative to G LR | equal 1e-4 (PH:676, PH:685; TG:30) / D below G, 1e-5 vs 5e-5 (PH:781, PH:790) / raise D LR (PH:477, generic) / lower D LR on NaN (PH:1009) | D-LR arms with D health logged | Yes mechanically; detail belongs to the GAN group. Local D is adan_sf, not AdamW |
| C5 | When the GAN starts and how | all losses at once, constant weight (TG:29, official) / start low, raise later (PH:1004, PH:1021) / ramp 5k (SKR:14) or 10k-50k (PH:508-509, generic) / 50k warm-up after 200k (SX:182-196, hypothesis) | start/ramp arms at the same final weight (RG:170) | Yes: `start_iter` / `ramp_iters` exist; only <= 20k from one pretrain |
| C6 | Perceptual-first prefix or one combined stage | one run (TG:29, DR:140) / perceptual first, GAN last (HB:676-677, PH:399; NP:309 hypothesis) | prefix arm vs combined at equal total updates | Yes |
| C7 | One texture loss at a time or the full stack from the start | one per branch (DR:157, FG:183, DX:561) / all release stacks carry DISTS or LDL + FF from the start (PH:609-611, PH:714-716, PH:819-820; PG:216 (D)) | attribution grid (PG X5) | Yes, but it is the losses group's question; multi-change recipes are not D10 screens |
| C8 | When to enable wavelet-guided | >= ~40k, example 80000 (NL:12, NL:16, official) / near the LDL stage while LR remains (DX:303-305) | arms with different `wavelet_init`, one crossing the switch inside the decision window | Yes: `wavelet_init` is a key. No local run has crossed 80000; `wavelet_init = 0` from P0 was screened (8k LOSS, one seed) and an acceptance run switched at 500 (D transient, section 3) |
| C9 | First fix for soft / plastic output | raise GAN or perceptual (SKT:35, AN:275, PH:954, generic) / data and degradation first (HB:1064, HB:221, RG:220, FG:57) | degradation-severity arms vs GAN-weight arms on mild inputs | **No**: degradation advice needs degraded data; OTF is parked. GAN-weight side only |
| C10 | First fix for artifacts | lower GAN first, CX second, data last (SX:221-223, hypothesis) / data first (PH:533-543, HB:1064) | same | Only the loss-side branch |
| C11 | How much texture to invent on skin | hallucinate freely (SKT:7, generic) / do not force pores (PF:246, DX:531, hypothesis) / identity marks are constraints (RG:47, paper) | skin and smooth-skin buckets with an identity check | Not yet: local data is the right content, but there are no region buckets or identity metric |
| C12 | Faces: region weights and paste-back, or sampling only | weights x3-10, ArcFace, composite restorer (SKT:56-58, generic) / sampling first, soft masks, no compositing (RG:61-62) | PG X12 | No: needs a face parser, mask-weighted losses, an identity metric (all missing) |
| C13 | fp32 + larger-patch polish pass | habit (PH:1007) / rejected as a standard step (DR:543, design) | bf16 tail vs fp32 tail from one checkpoint, equal updates | Precision half yes; the larger-patch half no (LQ tiles are 64 px) |
| C14 | What an "iteration" counts | optimizer step (HB:307) / upstream NeoSR counts micro-batches (NP:106-108) | read the code | **Settled for neosr-dev**: every frequency counts completed updates (`train.py:182-183`) |
| C15 | NaN: trainer defect or configuration | weak detection upstream (NP:123-127) / aggressive defaults (NP:129, NP:596; HB:1059) | per-term finite counters on a failing run | neosr-dev already separates the two (section 4); no local NaN case to study |
| C16 | LR, warm-up, scheduler numbers | scratch 2e-4 (TG:14) vs ~1e-4 (PH:383, PH:393) vs 5e-4 doc example (NO:26); finetune 1e-4 (TG:24) vs 3e-5 (DR:183 (D)) vs 1e-5 (DX:279); warm-up none / 2000 / ~2 % (PH:560, PH:665, HB:738); MultiStep 2 vs 4 vs 6 milestones, gamma 0.5 vs 0.85 (NO:172, PH:694-700, PH:799-807) vs cosine (NO:182, RP:172) | LR and decay arms on a run long enough for the decay to act | LR level: yes at 20k. Decay: only on a long run (>= 100k) |
| C17 | Fidelity budget and degradation-adaptation stage | >= 500k x2 (TG:13) vs 100k-400k (PH:388, generic) or 100k-500k+ (HB:690, planning example) vs >= ~30k pixel-only for OmniSR (HB:681, maintainer [D04]); adaptation stage before GAN (HB:675, HB:681) vs only on failure (DR:141) | pretrain-length arms; adaptation needs degraded data | Length: yes (E5). Adaptation: no (parked OTF) |
| C18 | Batch size | Siax batch 2 (SI:10) and releases at batch 2-8 (PH:1196-1201) vs "larger batch benefits training from scratch", example effective 32 (TG:15, official, fidelity pretrain) vs effective >= 8 in the GAN stage (SKR:25, generic; DR:18 (D)) | batch arms at matched images seen | Yes (64 px tiles, 24 GB) |

## 3. What our own evidence says

All local evidence is clean antialiased x4 on 64 -> 256 px tiles of people, one generator (realplksr 7.39 M), fine-tunes
of <= 20k updates from one 20k L1 pretrain, no run past `wavelet_init = 80000` (bundle `local_dataset`). It cannot speak
to degradations, large crops, x2/1x or long runs.

| id | local result | source | vs community | status / limits |
|---|---|---|---|---|
| G21 | Pretrain-first is assumed, not measured: every arm starts from P0 (20k L1-only, EMA). 2k of R3GAN from P0 cost 1.17 dB PSNR for +0.013 TOPIQ, -0.012 LPIPS | `docs/refactor/stage3/r3gan/RESULTS.md:80`, `:87-89`; `docs/refactor/DECISIONS.md:195` | consistent with fidelity-first (TG:7-33) but no "from scratch" arm exists (E5 deferred) | observation, no verdict |
| stage-2 d12 | GAN from update 1000 + 1000-update ramp: 28.69 dB / DISTS 0.1900 vs 28.81 / 0.1896 from update 1: no gain; GAN 0.1 vs 0.3: "mixed" | `docs/refactor/stage2/DECISIONS.md:47-48`; `docs/training_guide.md:27-29` | does not support a ramp (SKR:14, PH:1004), does not refute it | one seed, 8k, 67 tiles, arms differ in more than the start: "settles nothing" |
| G23 | No template decays the LR (AdamW 2e-4 constant to 1M). The owner's redux run was also effectively constant 2e-4. Every A/B arm fine-tunes P0 at 2e-4, although the project's own guide says to lower G to 1e-4 when fine-tuning a finished model | `docs/refactor/stage3/CONSENSUS.md:19`; `docs/refactor/stage3/owner_recipe.md:47`, `:58`; `docs/training_guide.md:45` | differs from every release config (MultiStep, PH:694-700, PH:799-807), from TG:24 (1e-4 for finetunes) and TG:30 (G and D 1e-4 in the GAN stage) | untested locally (E2 deferred to last) |
| G22 | G optimizer: AdamW 2e-4 beats SF-Adan at 8k (29.36 / DISTS 0.1787 vs 29.24 / 0.1890); tie on L1 alone | `docs/refactor/stage2/DECISIONS.md:20-25` | agrees with "AdamW, betas 0.9/0.99, wd 0" (NO:27-28, TG:14) and "schedule-free only as an arm" (DR:187) | stage-2 gate: one seed, 8k, low confidence |
| G24 | Fixed clip 1.0 vs off vs "spike": TIE wherever tested. Clip activity depends on the regime: stage 2 recorded that max-norm 1.0 "clips almost every update" (AdamW clip 29.87 vs off 29.85, L1); in the session-9 finetunes from P0 it is near idle (clipped_g ~2 %) | `docs/refactor/stage2/DECISIONS.md:26-28`, `:66-71`; `docs/refactor/DECISIONS.md:194` | agrees with "no universal threshold" (DR:551); cannot test "clip on in the GAN stage" where the clip is active | open where the clip acts (R3GAN first 2k: G clipped 95-100 %) |
| G25 | Restoring clamp vs hard clamp: +4.2 dB at 500 updates, tie at 5k | `docs/refactor/stage2/DECISIONS.md:31-32` | no community counterpart: a correctness fix | accepted default; one seed |
| G26 | EMA without warm-up: EMA 6.55 dB vs live 23.51 dB at update 1000; templates ship `ema = 0.999`, `ema_power = 0.75`. Derived from the code: decay(k) = min(1 - (1+k)^-0.75, 0.999) reaches 0.999 only after ~10k EMA updates (0.994 at 1k), so in every 20k arm (EMA re-seeded at update 1) the first half runs at a shorter horizon than nominal | `docs/refactor/DECISIONS.md:40-42`; `neosr/models/phase.py:371-387` | a concrete case of "EMA lags" (DX:444, RG:172) at the start of a run; the PH release snapshots list only `ema_decay` | motivating measurement, no A/B of `ema_power`; raw vs EMA never scored at one step locally |
| G32 | R1/R2 penalty in bf16 vs fp32: PASS within predeclared tolerances; 2k dev deltas 0.005 dB / 0.0015 TOPIQ / 0.0023 LPIPS | `docs/refactor/stage3/r3gan/RESULTS.md:42-53`, `:69-86` | consistent with "bf16 is safe" (PH:1071) for this pass only | no replicate; ratio 0.82-0.83 near the 0.8 bound |
| G12 | R3GAN health: G clipped 95-100 % in the first 2k, D accuracy ~0.9 later; such a clearly winning D added only LPIPS over no-D. The template D (metagan@0.3) sits near chance at 20k (`out_d_acc` ~0.5, `grad_sr_ratio` ~0.02), yet no-D (A1) is a LOSS at 20k | `docs/refactor/stage3/r3gan/RESULTS.md:114-117`, `:128-131`; `experiments/s10/runs/R3_patch_g1/train.jsonl`; `experiments/s10/consult/brief.md:34`, `:38-39` | bears on C4 (D strength) | **study computation** for the 20k R3 diagnostics (read from `train.jsonl`), not an owner-accepted verdict |
| wavelet | `wavelet_init = 0` (SWT-view D from the first GAN update, P0 = 20k L1) vs B0 (RGB D through 8k): LOSS at 8k (PSNR -0.11, LPIPS +0.0032 = 1.7x floor, DISTS +0.0032, TOPIQ inside). An acceptance run switching at update 500 gave a D transient (`grad_norm_d` 7 -> 47 in bf16) that settled in ~100-200 updates | `docs/refactor/stage2/DECISIONS.md:91-92`, `:99-100`; `experiments/s9_ab/gate_phase2.txt:17`; `docs/refactor/stage2/CLOSURE.md:87`; `docs/refactor/analysis/acceptance.md:122-124` | weakly against early wavelet D; consistent with NL:16 (enable after ~40k: P0 had 20k) | one seed, 8k, marginal LPIPS; the template's switch at 80000 is untested; within <= 20k the template **is** the wavelet-off arm (D view is the identity before `wavelet_init`, `neosr/losses/wavelet_guided.py:42-44`) |
| resume | A0 (template 0 -> 20k straight) vs B0 (same config, 8k stop, then resume to 20k: weights, optimizers, EMA exact, data order restarted, `neosr/models/base.py:383-386`): TIE on all four judges | `experiments/s10/consult/brief.md:36-37` | consistent with "resume is safe for an unchanged config" (HB:700-704) | one seed |
| G40 | Late G-gradient spikes: counts vary 6x between seeds of one config; the reference A0 reached 34.6; no skipped update anywhere | `docs/refactor/stage2/DECISIONS.md:74-82`; `experiments/s9_ab/NOTES.md:201-202` | cautions against the community habit of reading one grad-norm spike as instability (PH:1071) | 8k counts documented; **20k maxima are a study computation** |
| G45 | Costs: 0 -> 20k template 37 min (an earlier estimate with validation: 42-45 min); 8k -> 20k extension 20 min (28 min unet, 27.5 min DISTS); 80k arm ~3 h | `experiments/s10/consult/brief.md:24-25`; `experiments/s9_ab/consult/brief.md:40-41` | community budgets (139k-500k) are 4-15 h per arm here | nothing ran past 20k |
| grad share | Per-term G gradient share at P0: FDL 81.2 %, GAN 16.5 %, MS-SSIM 1.4 %, L1 0.53 %, consistency 0.35 %, LDL 0.000 % | `docs/refactor/stage3/PROGRESS.md:429`; `experiments/s9_ab/NOTES.md:25` | first real instance of PG X13 ("nobody logged these norms", PG:261); the L1 anchor carries almost none of the update, which is relevant to C3. At B0's 8k weights: FDL 74.3 / GAN 16.6 / MS-SSIM 7.1 / L1 1.7 / consistency 0.44 / LDL 0.000 % (`docs/refactor/stage3/PROGRESS.md:430`) | a share is not an effect: LDL has 0.000 % share, yet seed 2 without it had a G-gradient spike of 248 and a LOSS on LPIPS (`docs/refactor/stage2/DECISIONS.md:72-83`); attribution is disputed (a ~2.5k-update D outage, `docs/refactor/DECISIONS.md:190`) |
| owner recipe | Redux chain of 38k: fidelity pretrain 7k, then six resumed stages; constant 2e-4, EMA 0.999 / power 0.75, D lr 5e-6; the preset's 1e-4 was never applied (resumed optimizer kept 2e-4); effective Charbonnier 0.01 with GAN x6 collapsed (PSNR 21.3 -> 17.7), Charbonnier 0.1 + HSLuv 0.1 recovered | `docs/refactor/stage3/owner_recipe.md:15-17`, `:38-43`, `:47`, `:58`, `:70-80` | supports "keep a pixel/colour anchor" (DX:323, PH:491) against the pixel-free reading of C3; shows the resume-vs-new-stage pitfall of HB:700-711 in practice | best reading of confounded logs (val8 = 8 photos, one seed, stochastic validation), **not an A/B** |

Net: the local record agrees with the community on AdamW, on fidelity-first as a starting assumption and on keeping an
anchor; it has **no** two-seed 20k verdict on pretrain length, GAN start/ramp, LR decay, EMA settings, batch size,
wavelet timing or an active clip (bundle `local_open_questions`), so it can neither confirm nor refute the community
numbers on those.

## 4. neosr-dev support

| community practice | status | config keys | evidence (path:line) | gap or caveat |
|---|---|---|---|---|
| New stage = new run from G weights (fresh D, optimizer, EMA) | present | `path.pretrain_network_g`, `strict_load_g`, `param_key_g` | `neosr/models/image.py:218-234`; `neosr/models/checkpoint.py:80-99`; `docs/training_guide.md:24` | exports store the evaluated G (the EMA when on) under `params` (`base.py:296-299`, `image.py:685-693`), so a pretrain seed is the EMA; `read_weights` tries `params_ema`, `params-ema`, `params`, bare (`checkpoint.py:88-97`); a `.state` is refused as a pretrain (`checkpoint.py:85-87`) |
| x2 -> x4 hand-off with a key audit | present | `strict_load_g = false` | `neosr/models/checkpoint.py:141-181`; `docs/training_guide.md:43-44` | relaxed load logs loaded share + capped lists (10 keys); strict load gives torch's error only; no arch/scale check against the export's `neosr` meta at training load |
| Exact resume vs warm start kept apart | present | `--auto_resume`, `path.resume_state` | `neosr/models/base.py:331-387`; `neosr/models/phase.py:276-331` | saved hyper-parameters win over the config **with a warning** (`phase.py:307-312`): the owner's silent "1e-4 never applied" case is surfaced here |
| Reuse or reset D on a hand-off | present | `path.pretrain_network_d`, `strict_load_d` | `neosr/config.py:181-185`; `neosr/models/image.py:223` | optimizer state is always fresh in a new run; no "reset moments on resume" |
| Step-0 validation / parity with standalone inference | **missing** | none | `train.py:266` (validation only after an update with `u % val_freq == 0`) | workaround `test.py` on the pretrain; training validation is fp32 (`image.py:571-572`) while `test.py` honours AMP (`image.py:589-594`), so parity needs `use_amp = false` in the test config |
| Raw and EMA scored and saved at the same step | **partial** | `train.ema` | `neosr/models/image.py:484-490` (validation = EMA when on); `:685-693` (export = EMA); `neosr/models/base.py:317-329` (state holds both) | no raw validation, no raw export; raw G lives only in the `.state`, which cannot be a pretrain |
| EMA 0.999 / 0.9995, updated per successful G step | present | `train.ema`, `train.ema_power` | `neosr/models/phase.py:349-406`; `neosr/models/image.py:417-418`; `neosr/config.py:449-454` | template `ema_power = 0.75` is a decay warm-up (ema_pytorch style; redux has it too: the owner's redux run used 0.75, `owner_recipe.md:47`; the PH release snapshots list only `ema_decay`); with 0.75 the decay reaches 0.999 after ~10k EMA updates (`phase.py:371-374`); a new run re-seeds the EMA from G at its first update; no `ema_switch` / `update_after_step` |
| GAN start, ramp, perceptual-first prefix in one run | present | `start_iter`, `ramp_iters`, `end_iter` on every `[train.<slot>_opt]` | `neosr/losses/base.py:73-115`; `neosr/losses/gan_loss.py:73-74` | linear 0 -> weight only, one window per term; no `final_weight` (a decaying pixel weight such as RG:100 needs two runs); D trains exactly while the G term is active and is never ramped; units are updates |
| GAN ladder (each rung a new run) | present by workflow | `pretrain_network_g` (+ `_d`) | `neosr/models/image.py:223` | nothing gates a rung on a metric (by design: nothing adapts, `losses/base.py:84-85`) |
| Wavelet-guided D with a late start | present | `train.wavelet_guided`, `train.wavelet_init` | `neosr/config.py:147-148`; `neosr/losses/wavelet_guided.py:8-49` | hard switch of D's input from RGB to the SWT high bands of Y (`wavelet_guided.py:42-49`); cross-fade deferred (`docs/refactor/stage2/CLOSURE.md:87`, `:294`); template 80000 at constant LR, never crossed locally; `wavelet_init = 0` screened once (8k LOSS) and a switch at 500 measured (D transient), section 3 |
| AdamW betas 0.9/0.99 wd 0; separate G/D LR and wd | present | `[train.optim_g]`, `[train.optim_d]` | `neosr/models/base.py:33-40`; `options/train_realplksr.toml:64-78` | template D is `adan_sf` 1e-4 schedule-free, not AdamW as in the release configs; no SOAP / StableAdamW (the upstream wiki's `soap_sf`, NO:116-145, is not in neosr-dev's table); SAM/FSAM removed (`neosr/config.py:204-205`) |
| LR warm-up | present | `[train.optim_*] warmup_steps` | `neosr/models/image.py:42`, `:304-314`; `neosr/config.py:210` | `warmup_iter` removed; template G has no warm-up, D 600 |
| MultiStep (any milestones, gamma) / cosine; G-only schedule | partial | `[train.scheduler] type`, `milestones`, `gamma`, `T_max`, `eta_min`, `nets` | `neosr/models/base.py:41-44`, `:66-73`; `neosr/config.py:554-586` | no scheduler by default (constant LR); the clock counts attempted updates, including skipped and idle ones (`phase.py:170-174`, `:204-205`), unlike NP:110; no WSD / restarts (`docs/refactor/stage3/CONSENSUS.md:41`) |
| One integer optimizer-step clock | present | `datasets.train.accumulate`, `logger.*`, `val.val_freq` | `train.py:182-183`; `neosr/models/image.py:357-438`; `tests/test_model.py:266`, `:521` | resolves C14 for neosr-dev; warm-up, schedule, EMA, save and validation all count updates |
| bf16 / fp16 / fp32 | present | `use_amp`, `bfloat16` | `neosr/models/phase.py:127-135`; `neosr/config.py:493-497` | training validation is always fp32, so validation scores stay comparable across precision arms; a precision change on resume warns (`base.py:372-378`) |
| Gradient clipping on/off/threshold | present | `train.grad_clip = true \| <float> \| false \| "spike"` | `neosr/config.py:458-465`; `neosr/models/phase.py:168-213` | default `true` = max-norm 1.0 on G **and** D (1x releases run clip off); no per-net key (D5); pre-clip `grad_norm_*` and `clipped_*` are logged (`image.py:430-433`) |
| Finite checks on losses and gradients, per term | present | `train.max_skip_streak` (10), `detect_anomaly` | `neosr/models/phase.py:188-213`, `:409-489`; `neosr/models/image.py:447-475`; `tests/test_model.py:301` | a non-finite gradient skips the update (bounded, logged `skipped_*`, `nonfinite_<key>`), a non-finite loss in an applied step raises naming the term; `detect_anomaly` is a top-level key (`neosr/config.py:92`); no sample-id bundle and no batch replay tool |
| fp32 smoke test / overfit-tiny preset | partial | `use_amp = false`, `--debug` | `neosr/config.py:328-338` | assembled by hand; no preset or command |
| Realised LR in the log | present | none | `train.py:236-241`; `neosr/models/image.py:51-55`, `:477-480` | realised loss weights during a ramp are not logged as such |
| Dense checkpoint history for forks | present | `logger.save_checkpoint_freq`, `keep_states` (3), `keep_exports` (0 = all) | `neosr/models/base.py:121-123`, `:283-309`; `neosr/models/checkpoint.py:233-236` | by default only 3 exact states survive; older forks start from exports (EMA G + D) |
| Best-checkpoint export by a metric | present | `val.save_best` | `neosr/models/base.py:240-265`; `docs/training_guide.md:59-60` | one metric, EMA weights, off in the templates |
| Tiled validation against OOM | present | `val.tile`, `val.tile_overlap` | `neosr/config.py:489-492`; `neosr/models/tiling.py:20-40` | cores are copied, not feather-blended; exact only when the receptive field fits the overlap |
| Per-term generator gradient norms (PG X13) | present, offline | `scripts/term_grads.py -opt <toml> [--net_g] [--net_d] [--batches] [--iter]` | `scripts/term_grads.py:1-12`, `:32-36` | norms over all G parameters on training batches, not "a trunk layer on fixed probes" (RG:59); online only GAN health `grad_sr_gan`, `grad_sr_ratio` (`image.py:58-98`) |
| Colour-drift levers | present | `[train.consistency_opt]`, `train.match_lq_colors` | `neosr/config.py:146`; `options/train_realplksr.toml:89-91` | HSLuv and cosim are redux-only |
| L1 + MS-SSIM fidelity mix | present | `[train.pixel_opt]` + `[train.mssim_opt]` | `options/train_realplksr.toml:81-87`; `docs/training_guide.md:15` | two independent terms; redux "MS-SSIM L1" is one combined loss, so 0.6 / 0.4 is a magnitude question, not a value |
| ICNR init against checkerboard | present (arch helper) | none | `neosr/archs/arch_util.py:33-44`; used by realplksr (`neosr/archs/realplksr_arch.py:154`) | arch scope; no synthetic-probe check for a new upsampler |
| Layer freezing / per-layer LR (PH:526, SKR:44, generic only) | **missing** | none | `docs/refactor/stage3/competitor/training_infra.md:76` ("missing in both", skip) | generic sources only; not proposed |
| Weight interpolation between checkpoints | **missing** | none | `neosr/models/checkpoint.py:46-58` (only strips swa's `n_averaged`) | no tool in `scripts/`, `convert.py` or `neosr/` |
| Region-aware sampling, mask-weighted losses, identity loss/metric | **missing** | none | `neosr/metrics/calculate.py:24`, `:113`, `:429`, `:449`, `:473` (psnr, ssim, dists, topiq, lpips only) | needs a face parser and ArcFace weights (external) |
| Mixed sources with a replay share | **missing** | none | `neosr/config.py:113-114` (one `dataroot_gt` / `dataroot_lq`) | data group's item |
| Artifact diagnostics (invention residual, anchor diff, flat/ramp probes) | **missing** | none | same metric table | validation group's item |
| Stage engine, metric-triggered stage switch, adaptive D / loss scheduler, ReduceLROnPlateau | absent by owner rule | none | `docs/training_guide.md:3`; `docs/refactor/stage3/competitor/training_infra.md:77` | a plateau-triggered switch (PH:423-427) is an automation; schedules here are pure functions of the update |

## 5. Framework work this implies (strategy-independent)

None of these changes a default; each is opt-in or an offline tool. "Prereq" names the part-6 questions it unblocks.

| rank | feature | why (anchors) | size | risk | default change | how to verify | prereq for |
|---|---|---|---|---|---|---|---|
| 1 | **Raw-G access**: opt-in validation of the live G next to the EMA at the same update, and a raw export (or a tool that extracts raw G from a `.state`); the offline tool alone already unblocks G4-Q1 on the kept A/B states | score raw and EMA at one step, seed from either (PF:58, RG:63, RP:174, DX:442, HB:1123) | S-M | validation time doubles when on; schedule-free + EMA interplay (`docs/refactor/DECISIONS.md:37`) | no | EMA path bit-identical with the key off; raw export equals `state["nets"]`; exact-resume tests unchanged | G4-Q1, G4-Q2, G4-Q9 |
| 2 | **Step-0 validation** (opt-in key or CLI flag): validate the loaded weights before update 1 and log the metric vector | prove the load, keep the parent baseline (DX:273, DX:289, HB:1069, FG:242) | S | must leave every training RNG untouched (`base.py:202` already preserves it) | no | trajectory bit-identical with and without; step-0 metrics equal `test.py` at fp32 | every finetune question (Q1, Q3, Q4, Q9, Q10) |
| 3 | **Checkpoint interpolation tool** (`scripts/`): checks type, keys, shapes, buffers and the exports' `neosr` meta, writes an alpha grid of exports | the fidelity/texture dial (RG:65, PF:184, FG:250-252, RP:56); no photo-GAN result exists | S | low (offline); erratic blends are a finding, not a bug | no | alpha 0 and 1 reproduce the endpoints bitwise; refuses mismatched lineage | G4-Q10 |
| 4 | **Load audit that fails loudly**: counts in strict mode too; compare the export's `neosr` meta (arch, scale) with the config at pretrain load | "a loaded log line is not proof" (DX:273, FG:242, HB:81) | S | low; warnings only unless a mismatch is certain | no | tests with a wrong-arch and a wrong-scale export | hygiene for all hand-offs |
| 5 | **Loss schedule: second weight** (`final_weight`-style ramp between two weights, still a pure function of the update) and logging of realised weights at transitions | staged vs fixed weights at the same final weights, in G updates (RG:170, DX:307); practitioner schedulers (PH:1027, RG:100) | S-M | must stay "nothing adapts"; bitwise-unscaled at full weight as today (`losses/base.py:113`) | no | extend `test_gan_start_and_ramp`; resume exactness | G4-Q4 (decaying-anchor arm only) |
| 6 | **Wavelet switch cross-fade** (`wavelet_fade`, default 0 = today) | the abrupt RGB -> SWT switch gives a measured D transient (`grad_norm_d` 7 -> 47, settled in ~100-200 updates) with no measured quality effect (`docs/refactor/stage2/CLOSURE.md:87`, `:294`); CLOSURE names wavelet-from-the-GAN-start and a D warm-up as alternatives; late objectives need care (DX:303) | S | low | no | fade 0 bit-identical; D-input continuity test | one arm of G4-Q7 |
| 7 | **Artifact diagnostics as opt-in validation outputs**: invention residual `SR - bicubic(LQ)` energy on flat vs textured regions; diff against a fidelity sibling; a flat/ramp/line/impulse probe script with a pixel-phase residual for new generators | SKT:31, SKT:47, SKA:58, RG:77, HB:1063; directly serves "a base for new architectures" | M | metric design is unvalidated (calibration belongs to the validation group) | no | known-bad case: random PixelShuffle init vs `icnr_` | G4-Q12, G4-Q13 |
| 8 | **Per-bucket validation report** (tags per image -> per-bucket means and paired deltas in `analyze.py`) | faces/skin/hair and negative controls need buckets (PF:246-250, RG:146-150, HB:95, DR:75) | M | tagging effort is the owner's or a parser's | no | bucket means sum to the dataset mean | G4-Q12, G4-Q13 |
| 9 | **`term_grads.py` options**: fixed probe crops and a chosen trunk layer; batch over several checkpoints | RG:59, PG:261 (X13) | S | none (offline) | no | reproduces the P0 shares in `experiments/s9_ab/NOTES.md:25` | G4-Q11 |
| 10 | **Smoke command / preset**: fp32, pixel-only, overfit N pairs, save and reload, step-0 parity | HB:33, HB:673, DR:17, NP:306 | S-M | low | no | runs on CPU in a test | none (onboarding of new archs) |
| 11 | **Failure record on a non-finite event**: sample paths of the offending micro-batch, last good update, per-term counters, LRs, in one JSON | NP:123-127, DX:424, HB:761 | S-M | diagnostic only; must not add any damping (owner rule) | no | extend `tests/test_model.py:301` | none |
| 12 | **EMA horizon in the log** (current decay and half-life at start and after warm-up) | verify the actual update frequency and decay (PF:58, RG:172) | S | none | no | unit test on `Ema.current_decay` | G4-Q2 |
| 13 | **Scheduler row with a cooldown (WSD / linear)** | only if the decay question is positive (`docs/refactor/stage3/CONSENSUS.md:41`) | S | resume exactness | no | replay test | conditional on G4-Q5 |
| 14 | **Mixed sources with a fixed replay share; region-aware sampling; mask-weighted losses; identity metric** | SKR:43, FG:232, RG:61-62, SKT:56-57 | M-L | external weights and licences (ArcFace, parser); unsourced numbers | no | n/a until the owner wants it | G4-Q9 (replay), G4-Q12 (regions) |

Not proposed: a stage engine or metric-triggered switches (NP:63, PH:423-427). The two-run way plus per-term schedules
already cover the documented curricula, and triggers would be automation.

## 6. Research questions for the strategy phase

Cost unit: one template arm 0 -> 20k from P0 = ~37 min (95-110 ms/update); an 8k screen 16-19 min; 100k = 2.6-3.1 h;
DISTS as a loss +37 %. "x2" = two seeds, as D10 requires for anything that could become a default. Every verdict is
limited to clean x4, 64 px tiles, realplksr.

| id | question | single-factor arms | prerequisites | GPU cost (4090) | blocked by | priority |
|---|---|---|---|---|---|---|
| G4-Q1 | Are raw and EMA weights different candidates on our judges, and which is the better seed for a next stage? (HB:1123 E9, PF:316 round H; C1) | none to train first: score raw vs EMA at the kept states of A0/B0; then finetune from raw vs from EMA | rank-1 feature (raw scoring or the offline extraction tool); states exist at 5k/10k/15k/20k (A0) and 8k/12k/16k/20k (B0) with `keep_states = 0` (`experiments/s10/cfg/A0.toml:105-108`, `B0.toml:105-108`), not at every 2k validation; early states compare raw with a short-horizon EMA (G26) | validation only (minutes per checkpoint); seed test 2 arms x 37 min x2 | missing raw-G access | **high**: cheap, and it decides what every later "checkpoint" means |
| G4-Q2 | Does the EMA horizon matter in a 20k finetune? (C2) | `ema_power` 0.75 vs none first; `ema` 0.999 vs 0.9995 only with `ema_power` off (with 0.75 the two arms are identical until ~10k EMA updates and 0.9995 is reached only at ~25k, past 20k) | optional rank-12 logging | 2-3 arms x 37 min x2 | nothing | low: affects selection more than training; Q1 first |
| G4-Q3 | Does the pretrain help, and how long must it be (local E5)? (C17) | GAN template from scratch vs from P0 20k vs from a longer P0; second P0 seed | a longer / second L1 pretrain | pretrains (L1-only, cheaper per update, unmeasured here) + 3 arms x 37 min x2 | nothing; D6 already schedules E5 at 40k | medium: universal community prior, zero local measurement |
| G4-Q4 | GAN from update 1, delayed, ramped, or after a perceptual-only prefix, at the same final weight? (PG X4 ramp half; C5, C6) | `start_iter` 0 vs N; `ramp_iters` 0 vs M; prefix with `gan_opt.start_iter` only | none (keys exist); rank-5 only for a decaying-anchor arm | 3-4 arms x 37 min x2 | nothing | medium: the stage-2 one-seed 8k read showed no gain |
| G4-Q5 | Constant LR or decay, and 2e-4 or 1e-4 for a finetune (local E2)? (C16) | lr 2e-4 vs 1e-4 at 20k; constant vs MultiStep (G only, `nets = ["g"]`) on a long run | a long-run slot | LR level 1 arm x 37 min x2; decay 2 x ~3 h x2 | decay half needs >= 100k | medium: every release decays, local templates never do |
| G4-Q6 | Does the 12k-20k window rank arms the way a long run does? (SKM:39 "HF arrives late"; SKR:52's generic 30k-80k proxy bracket) | A0 vs one challenger to 100k; compare window verdicts at 4k-8k, 12k-20k, 80k-100k | none | 2 x ~3 h (x2 for seeds) | crossing `wavelet_init` 80000 confounds unless Q7 is decided first | **high**: it tests the D10 protocol itself |
| G4-Q7 | Does wavelet-guided help at all, when should it start, and does the hard switch hurt? (PG X6 timing half; C8) | the template (= wavelet off within 20k) vs a switch inside the run (e.g. 8000, before the 12k-20k window); `wavelet_init = 0` needs only a second seed and the 20k extension (W_wavelet0 exists, 8k LOSS); optional cross-fade arm | none; rank-6 for the fade arm | 1-2 new arms x 37 min x2, plus the W0 extension | nothing (a switch inside 20k avoids the 80k wait) | **high**: the template's late switch is untested and the only screen of the wavelet D lost (one seed, marginal) |
| G4-Q8 | How much pixel anchor does the GAN stage need? (C3; DX:323 vs PH:707-716) | `pixel_opt` weight 1.0 vs lower vs off, MS-SSIM and consistency fixed | `term_grads.py` shares per arm as a magnitude check; the redux 0.1 is a question, not a value | 2 arms x 37 min x2 | nothing | medium-high: cheap; local share of L1 is 0.53 % at P0 and 1.7 % at 8k, owner's run collapsed near zero anchor (confounded) |
| G4-Q9 | Detail first, then detune: does a low-LR finetune on cleaner HR keep detail and remove halos? (PG X10) | detailed parent vs clean-data scratch vs adapted, checkpoints every 2.5k | a second, cleaner HR variant of the data; ranks 1-2; replay share (rank 14) optional | 1 pretrain + 2-3 arms of 50k (~1.5 h each) | **data**: no sharp/clean HR pair of pools exists locally | low now (blocked); relevant once HR curation is decided |
| G4-Q10 | Is weight interpolation between P0 and a GAN checkpoint a usable fidelity/texture dial here? (PG section 6.4) | no training: alpha grid between P0 and A0@20k (and between two late checkpoints) | rank-3 tool | validation only (one dev pass per alpha) | missing tool | **high**: zero training cost, direct knob for the owner's goal |
| G4-Q11 | Do per-term gradient shares predict what an arm does? (PG X13) | none: shares at P0, 8k and 20k for every existing arm against its verdict | rank-9 options (optional) | minutes per checkpoint | nothing | medium: informs every weight translation; LDL (0.000 % share, seed-2 divergence) is the counter-example to explain |
| G4-Q12 | Do the aggregate judges hide region effects on skin, hair, eyes and teeth; is region machinery needed at all? (PG X12; C11, C12) | first no training: bucketed deltas of existing arms; later region sampling / weights / identity loss | rank-8 buckets, an identity metric, a person-disjoint holdout check; rank-14 for the training arms | analysis only at first | missing buckets, parser, identity metric; unsourced weights | medium: central to the owner's content, but the measurement must exist first |
| G4-Q13 | Does the template model sharpen or texture what it should leave alone (bokeh, smooth skin, flats)? (RG:142, PF:232) | none: negative-control bucket scored for existing arms | rank-7 diagnostics, rank-8 buckets, tagged negatives | analysis only | missing diagnostics and tags | medium |
| G4-Q14 | bf16 or fp32 training, and is an fp32 tail worth anything? (C13) | A0 bf16 vs fp32 (20k); 20k + 2k tail bf16 vs fp32 | none | fp32 arm cost unmeasured (slower than 37 min) x2; tails 2 x ~4 min x2 | larger-patch half blocked (64 px tiles) | low: G32 already passes bf16 for the penalty pass; no local failure |
| G4-Q15 | Clip mode where the clip is active (fixed 1.0 vs off vs spike)? (local G24) | one D regime with an active clip (R3GAN or unet), three clip modes | none | 3 arms x 37 min (28 min unet-like extension) x2 | belongs with the D choice (GAN group) | low-medium |
| G4-Q16 | Batch 8 vs 16 (and true batch vs accumulation, HB E3)? (C18) | batch 8 vs 16 at matched images seen; batch 16 vs 8 x accumulate 2 | none | batch 16 about twice the time per update (estimate, not measured) | nothing | low-medium: the owner's batch-16 stages are unproven |
| G4-Q17 | How far is the GAN from a diffusion reference on the same harness? (PG X16) | inference only: detail model, fidelity model, one diffusion SR | scoring of external SR folders (validation group); a reference model | no training | harness feature; the owner's choice of reference | medium: defines the target, changes no training |

PG experiments outside this group: X1, X9, X11 (data and degradation; X1 and X11 are blocked by the parked OTF track),
X2, X3, X14 (metrics and selection), X5, X7, X8 (losses), X6 (discriminator; its timing half is G4-Q7), X15 (generator
noise injection: an arch question, deferred by the owner's arch rule; the owner's own generator already has a noise head,
`docs/refactor/stage3/owner_recipe.md:45-46`).

What the parked OTF track would need before C9, C10, C17 (adaptation stage) and Q9-style questions can be asked: degraded
training data with a clean bucket, held-out real LQ, and per-severity validation strata (HB:1064, NP:308). Reported
only; no work is planned on it here.

## 7. Questions for the owner

1. **Raw vs EMA**: may validation and exports gain an opt-in raw-G path (rank 1)? Today a stage can only be seeded from
   the EMA and only the EMA is ever scored.
2. **Long run before or after the wavelet question**: the planned paired 100k run crosses `wavelet_init = 80000`. Should
   G4-Q7 (wavelet inside a 20k window) run first so the long run tests one thing? The only wavelet screen so far
   (`wavelet_init = 0`, 8k, one seed) was a marginal LOSS.
3. **Is LR decay a strategy question now or still last (E2)?** Every release config decays; no neosr-dev template does.
4. **Pixel anchor**: is a GAN stage without L1 (as in the 1x release stacks) a legitimate arm for you, given the step-4
   collapse in your own redux run, or is "anchor >= some magnitude" a fixed constraint?
5. **Faces, skin, hair**: which features are fidelity constraints for you (identity marks, eyes, teeth) and which may be
   re-drawn (pores, strands)? Is an identity metric with external weights (ArcFace) acceptable in the framework?
6. **Region tags**: are you willing to tag (or accept a parser tagging) the dev tiles by region so that skin / hair /
   eye buckets and negative controls can be reported? Without them Q12 and Q13 cannot be asked.
7. **Interpolation and two checkpoints**: do you want a fidelity model and a detail model (plus an alpha dial) as the
   standard output of a run, or one final model?
8. **Detune track**: is there, or will there be, a cleaner / less-sharpened HR variant of your data? Without it "detail
   first, then detune" stays untestable.
9. **Diagnostics scope**: should the artifact probes (rank 7) target *new architectures* first (flat/ramp/impulse
   upsampler checks) or the photoreal goal first (invention residual, negative controls)?
10. **Skip policy**: the trainer skips a non-finite update by default (bounded by `max_skip_streak`, default 10,
    `neosr/models/image.py:185`; stage-2 decision 7, `docs/refactor/stage2/DECISIONS.md:33-35`). The NeoSR proposal
    argues the opposite default, `abort` with skip as an explicit opt-in (NP:127, a proposal). Does the current default
    match your "never automatic damping" rule, or should "abort on the first non-finite update" be selectable?

## Verification

Adversarial pass, 2026-11-13, read-only (no GPU, no runs). Only this file was edited.

**What was checked**

| scope | count | how |
|---|---|---|
| KB anchors opened at the cited line | ~150 | TG 1-33; PH 383-393, 471-545, 553-904, 950-956, 1000-1100, 1192-1201; RP 50-60, 158-186, 355-366; DX 8-56, 268-326, 420-456, 512-590, 865-937; PF 25-60, 175-190, 226-272, 312-318; HB 1-35, 78-96, 218-223, 305-309, 596-601, 668-763, 1056-1080, 1118-1126; RG 6-106, 140-174, 216-222; FG 13-58, 164-184, 230-253, 413-420; NL 1-20; NO 1-190; UH3 153-158; RX 180-185, 270-275; SI 1-20; SKR 10-53; SKM 8-40; SKT 5-69; SKA 56-72; SKL 50-53; DR 15-18, 53-56, 120-188, 541-552; NP 61-64, 104-130, 304-310, 439-442, 594-597; PG 6-9, 35-45, 214-292, 434-470; FOCUS 276-292; SX 143-224; AN 273-276 |
| bundle | all | evidence mix 63/60/31/28/16/15/13 and counts 226/17/20/14/10 confirmed; all 100 relevance-3 claims, 17 conflicts, 20 gaps and 10 local findings compared with the text |
| code and local-doc claims | ~65 | every path:line in part 4 (`train.py`, `neosr/models/{image,base,phase,checkpoint,tiling}.py`, `neosr/config.py`, `neosr/losses/{base,gan_loss,wavelet_guided,__init__}.py`, `neosr/archs/{arch_util,realplksr_arch}.py`, `neosr/metrics/calculate.py`, `scripts/term_grads.py`, `options/train_realplksr.toml`, `tests/test_model.py`, `tests/test_s2_losses.py:416`); every local number in part 3 against its doc or run file |

**Corrections** (before -> after; source)

| # | where | before -> after | source |
|---|---|---|---|
| 1 | 1.1 pretrain | "matched fidelity (non-GAN) checkpoint", "Helaman from official DRCT-L / DAT2 x4 pretrains (RP:164, RP:166)" -> "matched pretrain"; x4 parents per the release table; RealPLKSR / 1x releases start from his own earlier releases, objective not stated | PH:1196-1201, RP:166 |
| 2 | 1.1 GAN weight | "[discord_cited, neosr channel]" -> neosr.md + training-06.md, weight scale not stated; release constants and the template's 0.3 added, scale-tagged | PH:1004 refs, PH:608, PH:713, PH:822, `options/train_realplksr.toml:102-105` |
| 3 | 1.1 HF late | "[paper via SK*]" -> generic SK* text; SKR:52's 30k-80k proxy bracket added | SKM:39, SKR:52 |
| 4 | 1.1 Compact | "under 40k (carried anchor)" -> typically < 40k, some 5k-10k from an on-point pretrain | RX:273 |
| 5 | 1.2 schedules | MultiStep anchors PH:561/666/771/891 (EMA lines) -> PH:588/693/798/893 | PH |
| 6 | 1.2 cosine | "Cosine is not the default [discord_cited]" -> one confounded EMA + cosine report; "not default" is DR's design | RP:172, DR:186 |
| 7 | 1.2 warm-up | ~2 % listed as "Hypothesis/design" -> discord_cited via HB (NeoSR maintainer [D09]) | HB:742 |
| 8 | 1.2 LR rule | "lower LR only on repeatable divergence" -> "or a stable plateau" (synthesis) | HB:748 |
| 9 | 1.3 detune | "on cleaner HR" -> own less-oversharpened HR, DX `reported_test` with a positive interim | DX:279 |
| 10 | 1.3 interpolation | "no photo-GAN result" -> one release practice (RealWebPhoto v4), no measured result, no alpha | RP:56 |
| 11 | 1.4 bf16 | fp16/bf16 deltas qualified: "across many SR models", apparently inference | RP:182 |
| 12 | 1.5 labels | process and grain rows "discord_cited" -> synthesis; DX:446 "hypothesis" -> `reported_test` (grid patterns in several RealPLKSR models) | PF:177, HB:221, FG:415, PF:229, HB:1065, DX:446-448 |
| 13 | 1.6 PF:27 | "advises little or no GAN for faces, against the owner's goal" -> PF:27 is the archival-accuracy row; the owner's row is PF:28 | PF:27-28 |
| 14 | 1.7 / C3 | pixel-free reading without counter-evidence -> TG:29 "MS-SSIM L1" (official) and Charbonnier 0.8 -> 0.3 (RG:100) added; posted redux block labelled discord_cited | TG:29, RG:100, PH:1005 |
| 15 | 1.7 anchor | "[discord_cited] keep a reconstruction anchor" -> DX synthesis action; evidence is that one tester kept pixel loss | DX:321-325 |
| 16 | C17 | "100k-400k (PH:388, HB:674)": HB:674 gives no bracket -> HB:690 (100k-500k+, planning example); HB:681 >= ~30k added | HB:674, HB:681, HB:690 |
| 17 | C18 | "effective batch >= 8 (TG:15, SKR:25)" -> TG:15 is "larger batch for fidelity from scratch" (example 32); >= 8 is SKR:25 (generic) / DR:18 (D) | TG:15, SKR:25, DR:18 |
| 18 | part 3 grad share | "seed 2 without [LDL] diverged" -> G-gradient spike 248 + LPIPS LOSS, attribution disputed | `docs/refactor/stage2/DECISIONS.md:72-83`; `docs/refactor/DECISIONS.md:190` |
| 19 | parts 3, 4, 6 wavelet | "never exercised locally" / "no local run has exercised" -> `wavelet_init = 0` screened (8k LOSS, one seed); switch at 500 gave a D transient; within 20k the template equals wavelet off | `docs/refactor/stage2/DECISIONS.md:91-92`, `:99-100`; `experiments/s9_ab/gate_phase2.txt:17`; `docs/refactor/stage2/CLOSURE.md:87`; `docs/refactor/analysis/acceptance.md:122-124`; `neosr/losses/wavelet_guided.py:42-44` |
| 20 | G24 | "near idle on the template" -> regime-dependent: stage 2 "clips almost every update", session 9 ~2 % | `docs/refactor/stage2/DECISIONS.md:26`, `:68` |
| 21 | part 4 load row | "key order tries `params_ema` first: the seed is the EMA" -> exports store the EMA under `params` | `neosr/models/base.py:296-299`; `neosr/models/image.py:685-693`; `neosr/models/checkpoint.py:88-97` |
| 22 | part 4 EMA row | "`ema_power` ... redux releases do not have" -> redux has it (the owner's run used 0.75); only the PH snapshots lack it | `docs/refactor/stage3/owner_recipe.md:47`; `docs/refactor/DECISIONS.md:40` |
| 23 | G4-Q1 | "states at the validated updates" -> states only at 5k/10k/15k/20k (A0), 8k/12k/16k/20k (B0) | `experiments/s10/cfg/A0.toml:105-108`, `B0.toml:105-108` |
| 24 | G4-Q2 / C2 | 0.999 vs 0.9995 arms -> identical until ~10k EMA updates under `ema_power = 0.75`; 0.9995 reached only at ~25k | `neosr/models/phase.py:371-374` (derived) |
| 25 | G4-Q7 | arms and priority rewritten per row 19 | as row 19 |
| 26 | G45 | 80k cost had no source -> added | `experiments/s9_ab/consult/brief.md:40-41` |
| 27 | 1.3 | Real-ESRGAN 0.999 "[official, basicsr]" -> upstream config via RG | RG:63 |

**Additions** (missing items that matter for the goal): HB:681 maintainer >= ~30k pixel-only; Siax tiles equal the
local 256/64; the 1x releases' OTF + `lq_usm` + `lq_size` 96; RealPLKSR knobs (PH:1090, HB:742 [D10]); UltraSharp clip
note (DX:424); local rows for the wavelet screen and transient, A0-vs-B0 resume TIE, template D near chance, 8k grad
shares, GAN 0.1 vs 0.3 "mixed", the guide's own 1e-4 finetune advice (`docs/training_guide.md:45`) and TG:30; part 4
rows for layer freezing (missing, generic-only, not proposed), `soap_sf` absent, ICNR used by realplksr,
`detect_anomaly` location; owner questions 2 and 10 extended (W0; NP:127's abort-by-default proposal).

**Rule checks**: no part proposes a default change (part 5 is opt-in or offline; part 6 lists research arms); no work
is planned on the parked OTF track (the closing paragraph of part 6 only reports needs); conflicts are listed by
position, not averaged (C3 and C18 now carry the official positions separately); every weight is scale-tagged or marked
"scale not stated"; local evidence is limited to clean x4, 64/256 tiles and <= 20k updates throughout.

**Unverified**: redux `mssimloss` vs "MS-SSIM L1" semantics (TG:29 read by name only; redux code not read); the scale
of PH:1004's 0.1 -> 0.3; whether the RealPLKSR / 1x release parents were GAN models; whether RP:182 concerns inference
or training precision; HB's [D04] / [D09] / [D10] footnotes not traced to the Discord logs; the 20k maxima in G12 and
G40 (study computations from `train.jsonl`) not recomputed (the 8k "6x" is U 31 vs Us2 5 windows,
`docs/refactor/stage2/DECISIONS.md:96`); the PH:1196-1201 runs' trainer and patch semantics; part 6 GPU costs are
extrapolations from ms/update (fp32 and batch-16 costs unmeasured); KB sources cited but not opened: SKD, SKG, UH1,
S1/S2, DR:75, HB:805-811, NP:322, PH:1013.
