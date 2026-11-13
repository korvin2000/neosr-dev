# Training-strategy study (2026-11-13)

**Status: a study, nothing decided.** The study proposes, the owner decides; the hold on tactic selection
(`docs/refactor/STATUS.md:68`, `next_tasks.md:135`) stays in force until then. No default changes anywhere below.

| | |
|---|---|
| What was read | The Trainings knowledge base (`/home/korvin/apps/Trainings`, 37 aliases in `tools/aliases.tsv`): ~990 extracted claims in five slices. neosr-dev code: 93 capability rows. Local runs and docs: 46 local measurements |
| How | Extraction into claim JSON, per-topic synthesis (five sections, G1-G5), one adversarial verification pass per section, a completeness critic on this document, then an independent read-only review by Codex (section 8). No GPU, no training |
| Verification counts (from the sections' "Verification" parts) | ~740 KB anchors or anchor spans opened at the cited line (G1 ~110, G2 ~150, G3 ~170, G4 ~150, G5 157); ~300 neosr-dev code and doc claims checked at `path:line` (52 / ~55 / ~60 / ~65 / ~70); the D10 windows of the s10 arms recomputed with `scripts/ab/analyze.py` (G2, G3); 108 corrections applied (16 / 21 / 20 / 27 / 24). The Discord originals behind the syntheses are not in the KB and were not checked |
| Where the sources live | KB: `/home/korvin/apps/Trainings` (guide `CLAUDE.md`). Topic sections: `docs/refactor/stage3/strategy/topics/g1..g5_*.md`. Claim JSON: `experiments/study/claims` and `experiments/study/bundles` (gitignored, not committed) |

Reading rules. Community anchors are `ALIAS:LINE` of the KB and keep their evidence label: **official** (trainer
wiki, model card), **release** (a config of a released model), **discord** (a message-cited report, usually one
person, one run), **paper**, **generic** (LLM boilerplate, numbers not empirical), **hypothesis**, **design** (a
choice of a KB synthesis, PG or DR). Every weight carries its trainer scale (redux, neosr, basicsr); weights are
never converted. neosr-dev statements carry `path:line`. "(study)" marks a computation made during this study
from existing run files: a reading, not an owner-accepted verdict. Cost unit: one arm = one seed 0 -> 20k from
the P0 pretrain = ~37 min = ~0.6 GPU-h on the 4090 (~0.85 with a DISTS loss; the `unet` D x1.30-1.34); a long
arm 20k -> 100k = ~2.2-3 h. All costs are the sections' estimates, not new measurements.

## 1. Summary

1. **The community evidence has a low ceiling.** No source ran a controlled ablation with numbers on photo data
   (RG:8, FG:15, DX:10, PG:8). The hardest facts are three release configs of one author on the redux scale: two
   1x stacks trained on OTF degradations for 250k iterations and one 2x stack run for 500k, its degradation not
   printed (release: PH:707-716, PH:604-611, PH:814-823, PH:559, PH:664, PH:769). None is x4; both 1x stacks
   trained on degraded data and the 2x stack's degradation is unknown (PH:765). *Strength: confirmed in all five sections.*
2. **Weights do not transfer; the only portable method is to measure.** Community: discord (HB:499, RP:139-141), hypothesis
   (PF:213), design (DR:6). Local: FDL at 0.75 held ~98-100 % of the G gradient and went to 0.02
   (`docs/refactor/stage2/DECISIONS.md:39-43`); the owner's redux run trained an effective GAN weight of
   0.0016-0.01 under a nominal 0.04-0.1 (`docs/refactor/stage3/owner_recipe.md:53-63`). Nobody in the corpus
   logged per-term gradient norms (PF:213, PG:261); neosr-dev has the tool (`scripts/term_grads.py:33-85`).
   *Strength: the clearest agreement between community and local evidence; a rule, not a number.*
3. **By gradient, the template objective is FDL + GAN.** Share of the summed squared G-gradient norms at P0: FDL
   81.2 %, GAN 16.5 %, MS-SSIM 1.4 %, L1 0.53 %, consistency 0.35 %, LDL 0.000 %
   (`docs/refactor/stage3/PROGRESS.md:428-430`). *Strength: measured once, one template; a share is not an effect
   (LDL has a zero share, yet seed 2 without it had a G-gradient spike of 248 and a LOSS on LPIPS, attribution
   disputed: `docs/refactor/stage2/DECISIONS.md:72-83`, `docs/refactor/DECISIONS.md:190`).*
4. **Locally the GAN is worth having, and no tested departure from the template GAN won**: no GAN
   (`experiments/s10/NOTES.md:40-42`), weight 0.03 neosr (`experiments/s10/NOTES.md:41`), `unet` D at 0.1
   (`docs/refactor/DECISIONS.md:235-236`), the R3GAN recipe (`docs/refactor/stage3/r3gan/RESULTS.md:112-136`) and
   the HF view from the GAN start (`docs/refactor/stage2/DECISIONS.md:91`) lost or traded one judge for another; a
   delayed start + ramp tied at 8k, "settles nothing" (`docs/refactor/stage2/DECISIONS.md:47-48`). Consistent with
   "no uniformly best D" (discord, HB:817). *Strength: mostly one seed, clean x4, <= 20k.*
5. **DISTS is the only added perceptual term with a local signal**: TIE under D9; in the 12k-20k window LPIPS
   -0.0060, TOPIQ +0.0025 (study; `experiments/s10/consult/synthesis.md:6-10`; D10's tau does not bind this arm,
   `docs/refactor/DECISIONS.md:222`). It is the largest perceptual term of the detail release (release, redux
   0.45, PH:714) and costs +37 % time. A metric trained on is not a judge (hypothesis FG:405; D10 rule 7,
   `docs/refactor/DECISIONS.md:227`). *Strength: one seed; seed 2 is held.*
6. **The pixel anchor is where the sources split.** Both 1x releases have no L1/Charbonnier term (release, redux,
   PH:707-716, PH:604-611); the 2x release keeps Charbonnier 0.1 (release, PH:816), the official redux GAN stage
   names "MS-SSIM L1" (official, TG:29, read by name only) and a DX synthesis keeps a reconstruction baseline
   (discord synthesis, DX:321-323); the often-quoted warning is uncited (generic, PH:491). Local evidence and the
   owner's own run favour an anchor (`docs/refactor/stage2/DECISIONS.md:44`, `docs/refactor/stage3/owner_recipe.md:21-23`):
   the stage-2 arm is small deltas, one seed, 8k; the owner's run is confounded with a x6 GAN weight.
   *Strength: weak on every side.*
7. **Discriminator regime is unmeasured everywhere.** The official guideline says G = D = 1e-4 (official, TG:30);
   the 2x release runs D 5x below G with soft labels (release, PH:781, PH:790, PH:822); the owner's redux run ran
   D 40x below G (`docs/refactor/stage3/owner_recipe.md:38`). The template D sits at chance and still helps
   (`experiments/s9_ab/NOTES.md:49-50`). *Strength: untested locally; config-only arms.*
8. **`wavelet_guided` ships in 82 templates and its switch has never been tested**
   (`docs/refactor/stage3/discriminator/RESEARCH_REVIEW.md:150`; no run has crossed `wavelet_init = 80000`).
   Official advice: enable after ~40k (official, neosr, NL:16); no photo result exists (PG:457). Here it is only a
   D-input view (`neosr/losses/wavelet_guided.py:8-49`); the WGSR method also has a subband reconstruction term
   (paper, PF:223). The planned long pair crosses the 80k switch in both arms: a fair comparison under that
   schedule that cannot isolate the switch's own effect, so its windows before and after 80k are reported apart. *Strength: one marginal 8k
   screen (`docs/refactor/stage2/DECISIONS.md:91`, `:99-100`).*
9. **Schedules and run length differ from every release.** All release configs decay the LR with MultiStep and
   run 250k-500k (release, PH:694-700, PH:799-807); no neosr-dev template decays
   (`docs/refactor/stage3/CONSENSUS.md:19`), every arm fine-tunes at 2e-4 although the project's own guide says
   1e-4 for fine-tunes (`docs/training_guide.md:45`; official TG:24), and nothing local ran past 20k. Whether the
   12k-20k window ranks arms as a long run does is open: "HF arrives late" is generic text (SKM:39); locally 8k
   ranks predicted 20k ranks but 3 of 4 early TIEs ended LOSS (`experiments/s10/consult/synthesis.md:22`,
   `experiments/s10/NOTES.md:40-44`). *Strength: untested past 20k.*
10. **The release configs separate "detail" from "cleanup" mainly in the degradation** (release PH:722-763,
    PH:617-658; design reading PG:121), and that is exactly what the local data cannot test: it is a clean
    antialiased downscale (`docs/refactor/stage3/r3gan/RESULTS.md:10-17`). With the OTF track parked
    (`docs/refactor/STATUS.md:91`), the blur, clean-share, noise and kernel questions are blocked (section 5,
    tier D). *Strength: reported, not planned.*
11. **Data.** The leak the community warns about happened here: 27 % of held-out tiles had a train twin
    (`docs/refactor/stage3/r3gan/RESULTS.md:10-17`; hypothesis PF:52). Crop equals tile
    (`neosr/data/paired_dataset.py:113-123`), so the official "larger crop in the GAN stage" (official, TG:31)
    needs the untiled originals. *Strength: measured locally; the crop advice is untested.*
12. **Validation.** Official wikis call metrics loose triage (official, UH3:121, RX:115); no source has a
    selection rule (DX:688, PG:454) or shows which metric tracks the eye on micro-texture (DX:492, PG:453).
    neosr-dev's D10 protocol with a measured null is stricter than anything in the KB
    (`docs/refactor/DECISIONS.md:212-216`). Open locally: dev and val8 rank the arms differently (study; `experiments/s10/runs/*/metrics.jsonl`).
    *Strength: the null is measured (high); judge validity is not.*
13. **Faces, skin, hair: nothing is measured anywhere** (FOCUS:279, PG:463, RG:146). The owner's content needs
    buckets, tags and the owner's own tolerance rules before any region question can be asked.
14. **Framework backlog.** The five sections propose 65 items; deduplicated they are 34 rows plus one item set
    aside (section 4). The first six (F1-F6: tools and reports that leave training untouched, sizes S to M) serve
    34 of the 72 research-question ids and need no GPU time. None changes a default.
15. **Research plan.** The suggested order (section 5) costs about 14-16 GPU-h: ~9.4-10.3 of one-seed 20k arms
    and one matched long pair (5-6 h), behind a calibration gate that must be recorded before new selection runs;
    second seeds of promoted arms come on top. The rest of tiers A and B is not in that sum. We agree with the
    playbook's own order (X2/X3 first, PG:469), but only its full-reference half and a blind owner session on the
    existing visual set are runnable today.

## 2. How strong the community evidence is

**Evidence ceiling.** Four facts bound everything the KB can support.

- **No controlled ablation.** Not for any loss weight, GAN weight, D architecture, blur dose, crop size, EMA or
  schedule (RG:8, FG:15, DX:10, PG:8; gaps PH:707, DX:137, PG:452). The nearest things are a numberless
  perceptual-weight ablation on DF2K/Urban100 (discord, PF:209) and per-term on/off tests on Compact/anime
  (discord, DX:337). Every number is a setting someone used, not a setting someone showed to be better.
- **Single-author release configs.** The three stacks and the two degradation configs come from one section of
  one playbook, an LLM-written extract whose original config files are not in the KB (PH:551-908, PH card). They
  are 1x (OTF-degraded, 250k) and 2x (500k), redux scale (PH:559, PH:664, PH:769). The snapshots lack keys (1x:
  no batch size; 2x: no D, degradation or crop, PH:765), and two printed keys are not what they look like: redux
  reads `blur_kernel_size` and never uses it, and an absent `blur_prob2` defaults to 0 (code, two local redux
  checkouts read for key semantics only; the release's own redux version is unknown; G1 verification).
- **Scale non-transfer.** A perceptual or GAN weight of 1.0 is "super high" on the redux scale (discord, PH:1005,
  PH:1024) and normal in basicsr and NeoSR doc examples (generic SKL:15; official NL:101). In the owner's redux
  checkout a loss with `start_iter` acts with its weight squared (`docs/refactor/stage3/owner_recipe.md:56`), so
  even nominal redux weights inside one trainer can mislead.
- **LLM-written syntheses.** HB, PF, DX, RG, FG, NP, PH, RP are message-cited but the Discord logs are not in the
  KB; the verification could check the KB line, not the message. PG and DR are designs built on them. The
  generic set (SK*, AN, SL) and the Siax reconstructions (SX, S1-S4) carry no empirical numbers. Evidence mix of
  the two bundles whose sections print it: G4 (226 claims) has 63 discord, 60 hypothesis, 31 official, 28
  generic, 16 paper, 15 design, 13 release claims; G5 (300 claims, the largest) has 132 hypothesis, 61 discord,
  41 design, 26 generic, 19 official, 18 paper, 3 release: about two thirds of G5 is proposal, not observation.

| source class | trustworthy for | not for |
|---|---|---|
| Official wikis and cards (TG, NL, NO, RX, UH1-UH5, SI) | key semantics, documented defaults, stage order (TG:7-33), stated prerequisites (NL:16, NL:182) | weights: doc values are syntax examples (NL:1-253); they contain no measurements |
| Release configs (PH:551-908) | what one author actually ran at 1x / 2x under degradations; which terms were on together | x4, clean data, neosr values, causality; the redux version is unknown |
| Message-cited Discord reports (PH cheat sheet, RP, DX, HB, PF, FG, RG) | the direction of single observations, debugging order, failure anecdotes | numbers, generality: mostly one person and one run, often confounded, often anime or Compact (DX:337, DX:345) |
| Inspected code facts (FG:165-167; NP C01/C02, NP:80-97) | trap lists (noise selector vs gate, upstream metric mean) | the upstream state today: unpinned revisions; neosr-dev does not share C01/C02 (`neosr/models/image.py:655-660`, `:484-516`) |
| Paper statements via the syntheses (RG:57-58, FG:392, PF:223) | properties of a loss or metric by design (DISTS tolerance, LDL false positives) | this setup: none was re-run on photo SISR-GAN training |
| Generic text (SK*, AN, SL, PH sections 4-6 such as PH:491) | vocabulary and checklists | every number (SKL:57, SKR:14, PH:508-509 are not empirical) |
| Designs and hypotheses (PG, DR, SX, NP proposals) | experiment designs (X1-X16, PG:452-469) and question lists | evidence: PG states that no source gives a threshold (PG:343) |
| Local evidence (this repository) | evaluation noise and the gate (high), gradient shares, costs, code facts | degradations, x2 / 1x, other generators, anything past 20k; most verdicts are one seed |

## 3. Community vs local evidence

Local limits for every row: clean antialiased x4, GT 256 / LQ 64 tiles of people, one generator (realplksr),
fine-tunes from one 20k L1 pretrain, at most 20k updates, mostly one seed.

| topic | community position (anchors, label) | local result (path) | agree / disagree / untested | limit |
|---|---|---|---|---|
| Weight portability | weights do not transfer, magnitude-match (discord, HB:499, RP:139-141; design, DR:6) | FDL 0.75 -> 0.02; effective vs nominal GAN weight 10-200x apart in the owner's redux run (`docs/refactor/stage2/DECISIONS.md:39-43`, `docs/refactor/stage3/owner_recipe.md:53-63`) | agree | stage-2 gate (one seed, 8k); log reading of one run |
| Balance heuristic | weighted terms within about one order of magnitude (generic, SKL:57) | in gradient norm every term but LDL is within ~15x of FDL; LDL ~1,600x below (study arithmetic; `docs/refactor/stage3/PROGRESS.md:428-430`) | mostly agree | shares over all G parameters on training batches |
| GAN on or off | GAN adds perceptual quality and costs PSNR (synthesis / generic, HB:783, SKR:15) | no GAN: LOSS on all three judges, PSNR +0.46 (`experiments/s10/NOTES.md:40-42`) | agree | one seed for the no-GAN arm |
| D health target | a 50 % rate is not a target (hypothesis, PF:240, DR:172) | template D: `out_d_acc` ~0.50, `grad_sr_ratio` ~0.02, and it still helps (`experiments/s9_ab/NOTES.md:49-50`) | agree | a diagnostic, not an A/B |
| GAN weight | redux 0.14-0.15 at 1x, 0.05 at 2x (release, PH:713, PH:608, PH:822); neosr doc example 0.3 (official, NL:142) | neosr 0.03 vs 0.3: LOSS in the window; 0.1 vs 0.3 "mixed" at 8k (`docs/refactor/stage2/DECISIONS.md:47-48`, `experiments/s10/NOTES.md:41`) | redux band: not comparable; on the neosr scale it sides with NL:142 over a much lower weight | one seed; 0.1 and 0.2 have no D10 verdict |
| D architecture | U-Net SN as the default (generic SKL:23; official log RX:225) vs no uniformly best D (discord, HB:817) | `unet` at 0.1: LPIPS better, TOPIQ worse beyond tau on both seeds = LOSS (mixed) (`docs/refactor/DECISIONS.md:235-236`) | agrees with HB:817; does not confirm the default | D type and weight changed together |
| D LR vs G LR | equal (official, TG:30; release, PH:676, PH:685) vs D 5x lower (release, PH:781, PH:790) | none; the template D is `adan_sf` 1e-4 under AdamW 2e-4 (`options/train_realplksr.toml:72-78`) | untested | - |
| A clearly winning D | R3GAN as a direction only (PH:453-455) | R3 recipe: `out_d_acc` 0.87-0.90, LOSS on DISTS alone (`docs/refactor/stage3/r3gan/RESULTS.md:112-136`) | no community result to compare | multi-change recipe, one seed |
| DISTS as a loss | largest perceptual term of the detail release (release, redux, PH:714) | T_DISTS: TIE under D9; window LPIPS -0.0060, TOPIQ +0.0025 (study; `experiments/s10/consult/synthesis.md:6-10`) | agree in direction | one seed; +37 % ms/update |
| DISTS replacing the D | GAN adds HF beyond DISTS (design, PH:713-714, PG X5) | without a D, DISTS loses to GAN + DISTS: LPIPS +0.0085, TOPIQ -0.0072 (`experiments/s10/consult/synthesis.md:24-25`) | agree | one seed |
| Metric used as a loss | it inflates itself (generic SKL:16; hypothesis FG:405, PF:203) | DISTS moved most (-0.0087); the owner's val8 DISTS followed the loss on and off (`docs/refactor/stage3/owner_recipe.md:18-20`) | agree; already D10 rule 7 | - |
| LDL | default-on (generic, SKL:29); 0.35-0.8 in releases (release, redux, PH:715, PH:610, PH:820); negligible alone on Compact/anime (discord, DX:337) | G-gradient share 0.000 %; seed 7 TIE, seed 2 LOSS with a gradient spike of 248 (`docs/refactor/stage2/DECISIONS.md:72-83`, `docs/refactor/DECISIONS.md:190`) | agrees with DX:337 on the gradient; brake role unresolved | seeds disagree |
| Pixel anchor in the GAN stage | none in the 1x releases (release, PH:707-716) vs Charbonnier 0.1 at 2x (release, PH:816), "MS-SSIM L1" in the official GAN stage (official, TG:29, by name) and keep a baseline (discord synthesis, DX:323; generic PH:491) | + L1 1.0: 28.34 / 0.1879 vs 28.17 / 0.1905 at 8k; the owner's run fell 29.5 -> 17.7 dB without anchors (`docs/refactor/stage2/DECISIONS.md:44`, `docs/refactor/stage3/owner_recipe.md:21-23`) | agrees with "keep"; silent on the 1x no-pixel regime | stage 2: small deltas, one seed, 8k; the owner's run: confounded with a x6 GAN weight |
| Wavelet-guided D timing | at least ~40k first (official, NL:16); while LR remains (discord, DX:303) | `wavelet_init = 0`: LOSS at 8k (`docs/refactor/stage2/DECISIONS.md:91`, `:99-100`) | weakly consistent with NL:16 | one seed, marginal; the 80k switch was never crossed |
| GAN ramp | all losses at once (official, TG:29) vs ramps (hypothesis SKR:14; generic PH:508-509) | start 1000 + ramp 1000: no gain at 8k, "settles nothing" (`docs/refactor/stage2/DECISIONS.md:47-48`) | untested in effect | arms differed in more than the start |
| Optimizer | AdamW, betas 0.9 / 0.99, wd 0 (official, NO:27-28, TG:14) | AdamW 2e-4 beat SF-Adan at 8k (`docs/refactor/stage2/DECISIONS.md:20-25`) | agree | stage-2 gate |
| LR decay and level | MultiStep in every release (release, PH:694-700, PH:799-807); 1e-4 for fine-tunes (official, TG:24) | no template decays; every arm fine-tunes at a constant 2e-4 (`docs/refactor/stage3/CONSENSUS.md:19`), although the project's own guide says 1e-4 for fine-tunes (`docs/training_guide.md:45`) | differs; untested | the level fits a 20k arm; the decay needs a long run |
| Gradient clipping | no universal threshold (design, DR:122, DR:551); off at 1x, on at 2x (release, PH:562, PH:667, PH:772) | TIE wherever tested; clip activity depends on the regime (`docs/refactor/stage2/DECISIONS.md:26-28`, `:66-71`) | agree | open where the clip is active |
| EMA | score raw and EMA, EMA can lag (paper / discord, RG:63, PF:58, RP:172-174) | EMA without warm-up 6.55 dB vs live 23.51 dB at update 1000 (`docs/refactor/DECISIONS.md:40-42`); raw never scored | consistent; the selection question is untested | motivating measurement only |
| Precision | bf16 over fp16, bf16 is safe (discord, PH:1006, PH:1071) | R1/R2 penalty bf16 vs fp32: PASS (`docs/refactor/stage3/r3gan/RESULTS.md:42-53`) | consistent for this pass | no replicate |
| Short screens | a screen can rank (hypothesis, DX:565) vs HF arrives late (generic, SKM:39) | 8k ranks predicted 20k ranks; 3 of 4 4k-8k TIEs ended LOSS (`experiments/s10/consult/synthesis.md:22`, `experiments/s10/NOTES.md:40-44`) | both partly supported | nothing past 20k |
| Validation set size | at most 8-10 images (official, UH3:115, RX:109) | a single-checkpoint gate called LOSS in 9 of 36 same-config comparisons (`docs/refactor/DECISIONS.md:212-216`) | untested: the measured 219-tile null does not show that an 8-10 image set fails (pending G5-Q7) | null from five runs of two configs |
| Seeds | at least 2 (hypothesis, HB:1132) or 3 (hypothesis, NP:483) | D10: two seeds plus a long pair, on a measured null (`docs/refactor/DECISIONS.md:219-230`) | agree with HB:1132 | - |
| Source split | split by source before tiling (hypothesis PF:52; paper-derived RG:42) | 164 of 600 held-out tiles (27 %) had a train twin (`docs/refactor/stage3/r3gan/RESULTS.md:10-17`) | agree | independence is only overlap-verified |
| Resample kernel | kernel and antialias are part of the degradation model (synthesis text, PF:108-112, FG:95) | the owner's LQ matches an antialiased cubic at 52.7 dB, plain bicubic at 31.2 dB (`docs/refactor/stage3/otf/DESIGN.md:28`) | agree | a probe on one set |
| Batch augmentation | used (discord, RP:209) vs abandoned (release, DX:511) | off: DISTS 0.1862 vs 0.1879 at 8k (`docs/refactor/stage2/DECISIONS.md:45-46`) | agrees with "flip / rot only" | low: the augmented arm had an aliasing ResizeMix (`neosr/data/augmentations.py:117-118`) |
| Crop and tiles | 512 px tiles, random crop inside (official tooling RX:129-137; discord DX:237-241); larger crop in the GAN stage (official, TG:31) | crop = tile, no random position (`neosr/data/paired_dataset.py:113-123`) | departs; untestable on this set | needs the originals |
| Ported recipes | a recipe does not transfer as weights (design, DR:6; hypothesis, PF:213) | OWN port: LOSS on dev, best on val8 (study; `experiments/s10/runs/OWN/per_image.jsonl`) | illustrates | not a screen under D10 rule 4 |
| Degradation (blur dose, clean share, noise, USM) | release configs and hypotheses only (release PH:741-745; hypothesis FG:173, PF:139-145); never always-on, keep a clean bucket (PH:328, PF:50, RG:38) | none: the data is the clean stratum at 100 %. Code reading only: the default OTF chain degrades every sample (blur1 always, noise and JPEG twice, no clean sample, no antialiased resampler; `docs/refactor/stage3/otf/DESIGN.md:22-30`) | untested | local evidence cannot speak to it; the chain finding is reported, not planned (OTF parked) |
| Spectral norm in the D | SN keeps the D stable, "mandatory" with a U-Net D (generic, SKL:23, SKR:36) | GroupNorm cancels the SN of the template D (metagan); its input gradient to G at init is ~300x the SN U-Net's (`docs/refactor/stage3/discriminator/RESEARCH_REVIEW.md:103-104`, `:117-126`) | the advice describes a D the template does not have | one batch at init; the opt-in fix DR-08 is untested (G3-Q6) |
| Soft labels | hard 1/0 as the documented default (official, NL:143-144) vs 0.9 / 0.1 in the 2x release, with its lowest weight and D LR (release, PH:822) | none; static keys exist for bce / mse / huber (`neosr/losses/gan_loss.py:93-94`) | untested | whether static smoothing counts as damping is open (owner, section 6, decision 7) |
| Resume vs new stage | exact resume only for an unchanged config; a stage is a new run (discord, HB:599, HB:700-711) | A0 (straight) vs B0 (8k stop + resume): TIE on all four judges (`experiments/s10/consult/brief.md:36-37`); the owner's redux run kept 2e-4 after a 1e-4 preset because the resumed optimizer won (`docs/refactor/stage3/owner_recipe.md:58`); neosr-dev warns in that case (`neosr/models/phase.py:307-312`) | agree | one pair |
| Gradient-norm spikes | on `grad_norm = inf`: clip, try bf16, disable AMP to confirm (discord, PH:1071-1072) | late G-gradient spike windows vary 6x between seeds of one config (unet 31 vs 5, `docs/refactor/stage2/DECISIONS.md:96`); one seed without LDL spiked to 248 (`:74-82`); no skipped update in any s10 arm (`experiments/s10/runs/*/train.jsonl`) | a caution against reading one spike as instability | 8k counts documented; 20k maxima are a study computation |

## 4. Framework backlog (strategy-independent)

The five "Framework work" parts hold 65 items (10 + 14 + 10 + 14 + 17); merged and deduplicated they are the 34
rows below, and G2 item 14 (NR-IQA training loss, blocked) is set aside under "not proposed". Ranking rule: how
many research questions a row unblocks and how much it serves the framework goal, then size; at equal value,
offline tools before changes to the trainer (a prerequisite row stays above its dependants, as F4 above F5).
**No row changes a default**; objective or dynamics items are opt-in and bitwise-off. Whether any row starts, and
where it sits relative to Studio, is the owner's call (section 6). Source = section and its own item number.

**Two lenses (review note).** This rank counts the research questions a row serves, which favours the study's own
question list. Ranked by the owner's first priority, a stable and reliable core, the first tranche would be the
correctness and reproducibility rows: F14 (load audit, run manifest), F7 (step-0 validation), F21 (resume with a
changed crop or batch), F22 (smoke preset, failure record), F28 (export parity), F4 (per-image rows and the
metric-protocol record), F8 (config warnings), F9 and F10 (data preflight, leak detector), plus one performance
item outside the sections: why `scripts/bench.py`'s eager step is slower than the trainer's loop (`next_tasks.md:28`).
F1 and F2 stay early enablers; F3 (interpolation) and F5 (time-series report) wait for a question that needs them.
Which lens applies is the owner's call (section 6, decision 3).

| rank | feature | serves (topics, research questions) | size | risk | default change | how verified | source |
|---|---|---|---|---|---|---|---|
| F1 | **Loss scale card and gradient read-out**: `scripts/term_grads.py` prints raw value, weight, weighted value, norm and share per term, JSON output, a sweep over checkpoints; later a probe mode (named trunk module, fixed probe crops, pairwise cosine). Today only the weighted value is logged (`neosr/losses/base.py:115-120`) | losses, GAN, dynamics: G2-Q1, Q2, Q5, Q6, Q7; G3-Q1, Q4, Q5, Q11, Q12; G4-Q8, Q11; G5-Q8 (X13) | S, probe mode S-M | low: offline, trainer untouched | none | raw x weight x ramp == weighted; reproduces the P0 shares (81.2 / 16.5 / 1.4 / 0.53 / 0.35 / 0.000 %); all-parameter mode bit-identical on a fixture | G2 1-2, G3 1, G4 9 |
| F2 | **Offline folder scorer** (SR folder vs GT folder, same metric code, per-image rows) **plus a state-to-weights extraction** of the raw G (`neosr/models/base.py:317-329`) | validation, EMA: G5-Q2, Q3, Q4, Q6, Q10-Q13; G4-Q1, Q10, Q17 | S-M | low | none | scoring the PNGs a validation wrote reproduces the trainer's values to rounding (`tests/test_s2_metrics.py:338-351`); extracted G equals the state's | G5 2-3, G4 1 (offline half) |
| F3 | **Checkpoint interpolation tool** (alpha grid, key / shape / lineage check) | fidelity-texture dial: G4-Q10 = G5-Q6 | S | low: offline | none | alpha 0 and 1 reproduce the endpoints bitwise; mismatched lineage refused | G4 3, G5 6 |
| F4 | **Per-image rows and a metric-protocol record from plain `train.py` / `test.py`** (today harness runs only, `scripts/ab/run_ab.py:1-10`) | selection: G5-Q1, Q2, Q5, Q7 | S | low | none | mean of the rows equals the logged mean to fp64 | G5 1 |
| F5 | **Checkpoint time-series report**: metric vector per checkpoint, non-dominated set, window means, no composite score | selection: G5-Q1, Q2 | M | over-design: no threshold exists anywhere (PG:343); a report first | none | deltas equal `scripts/ab/analyze.py` on the same rows | G5 5 |
| F6 | **Bucket tags in the A/B analysis**: per-stratum paired deltas, median, worst image | data, people content: G1-Q5, Q6, Q8-Q10; G4-Q12, Q13; G5-Q2, Q5 | S-M (the tags are the owner's or a parser's work) | small strata widen the CIs; tau per stratum | none | per-stratum means recombined by tile counts equal the pooled mean; the pooled verdict is computed on its own (verdicts do not add) | G1 7, G4 8, G5 9 |
| F7 | **Step-0 validation switch** (validation runs only after an update today, `train.py:266`) | every fine-tune question (G4-Q1, Q3, Q4, Q9, Q10); proves a load | S | must leave the training RNG untouched | none | trajectory bit-identical with and without; step-0 metrics equal `test.py` at fp32 | G4 2, G5 7 |
| F8 | **Config warnings**: a validation or `save_best` metric is also a loss; `ff_loss` / `kl_loss` without a pretrain; `wavelet_init` below ~40k | hygiene; D10 rule 7 | S | none (warnings, nothing automatic) | none | config tests | G2 7, G5 4 |
| F9 | **Data preflight**: pair audit (decode, scale, bit depth, PSNR of the LQ against each `resample.py` kernel, sub-pixel shift) and a contact sheet of what the loader yields | trust in any new set: G1-Q7, Q12 | S-M | low: read-only | none | fixtures with a 1 px shift and a wrong kernel are flagged; the owner's set reports ~52.7 dB | G1 1 |
| F10 | **Leak / duplicate detector and group-safe splitter** as a shipped tool | every new held-out set: G1-Q4-Q7, Q12, Q13; G5-Q5 | M | thresholds are data-specific; it reports, never drops | none | reproduces the R5 exclusion list (`docs/refactor/stage3/r3gan/RESULTS.md:10-17`) | G1 2 |
| F11 | **Exposure accounting** in the log and `done.json`: effective batch, LQ pixels and images per update | matched-budget arms: G1-Q1, Q4; G4-Q16 | S | none | none | values equal batch x accumulate x patch^2 | G1 4 |
| F12 | **Raw-G validation pass and raw export**, opt-in (validation and exports are EMA-only, `neosr/models/image.py:484-490`, `:685-693`) | EMA: G4-Q1 (seed test), Q2, Q9; denser points for G5-Q4 | S-M | low-medium: touches `eval_weights()` and export naming; validation time doubles when on | none | EMA path bit-identical with the key off; raw export equals the live G | G4 1, G5 3 |
| F13 | **Blind crop-sheet generator** (fixed coordinates, shuffled labels, key file) | the human half of G5-Q2 (X3) | S | none | none | - | G5 12 |
| F14 | **Load audit that fails loudly** (counts in strict mode, arch / scale against the export's meta) **and a run manifest** (code revision, metric protocol, data hash; never the remote URL) | hygiene for all hand-offs and comparisons | S, S-M | low | none | tests with a wrong-arch and a wrong-scale export; manifest test on a temp run | G4 4, G5 8 |
| F15 | **Tile scorer** writing `meta_info` selections and strata (Laplacian variance, blockiness, noise sigma, flat share, optional NR columns) | data: G1-Q5, Q6; the measured profile of real inputs | M | NR scores reward sharpening (discord, PF:83): it outputs scores, never decides | none | unit tests per score on synthetic tiles; deterministic CSV | G1 3 |
| F16 | **Colour-drift report**, offline through F2 (per-channel bias, flat-patch delta, saturation shift) | anchors: G2-Q5, Q11; G4-Q8 | S | low: report only | none | synthetic colour casts move it | G5 16 |
| F17 | **Exact Charbonnier under a plain name with an `eps` key** (today `chc_loss` clips by default and fixes eps 1e-12, `neosr/losses/basic_loss.py:170-172`, `:204`) | the Charbonnier arm of G2-Q5 | S | none if exact; a non-default eps is an opt-in objective change | none | alias at the default eps == `chc_loss` with overrides, bitwise | G2 3 |
| F18 | **LDL artifact-map dump** | G2-Q4 | S | low | none | dumped map equals the loss's internal map (`neosr/losses/ldl_loss.py:71-96`) | G2 4 |
| F19 | **Piecewise loss-weight schedule** (a second weight, still a pure function of the update; one linear segment today, `neosr/losses/base.py:108-116`) and logging of realised weights | the plateau variant of G3-Q3; the decaying-anchor arm of G4-Q4 | S | low; must stay "nothing adapts" | none | full weight stays bitwise unscaled; resume exactness | G2 11, G3 5, G4 5 |
| F20 | **`wavelet_guided` documented against WGSR, with a band-statistics probe** | reading G3-Q7 / G4-Q7 | S | low | none | band-energy probe on real vs SR batches | G3 3 |
| F21 | **Resume with a changed crop or batch: test and document** | staged variant of G1-Q4 | S | none if it already works | none | resume test `patch_size` 64 -> 96, `batch_size` 8 -> 4 | G1 8 |
| F22 | **Smoke preset, failure record on a non-finite event, EMA horizon in the log** | onboarding of new archs; G4-Q2 | S-M each | low; diagnostic only, adds no damping | none | CPU test; extends `tests/test_model.py:301` | G4 10-12 |
| F23 | **Fork mode**: a recorded branch from a full state that applies changed hyper-parameters (today saved optimizer settings win, `neosr/models/phase.py:276-312`) | ladders and one-knob branches: G3-Q2, Q9 | M | medium: must not weaken strict resume (`neosr/models/base.py:331-352`); touches D10 rule 6 | none | a fork with no change = resume bitwise; provenance test | G3 2 |
| F24 | **DISTS loss cost**: exact speed / memory rewrite | cost of G2-Q2, Q3, G3-Q11 | M | low if exact | none | parity with `dists_loss` on fixtures; ms/update and VRAM before / after | G2 5 |
| F25 | **Weighted / stratified tile sampler with a realised-composition log** (uniform today, `neosr/data/data_sampler.py:45-58`) | G1-Q5; region sampling later (G4-Q12) | M | resume and DDP determinism; the default path must stay untouched | none | no weights = today's permutation, bitwise | G1 5 |
| F26 | **Scripted guard diagnostics, offline and report-only**: flat-region HF residual, invention residual, edge overshoot, radial spectrum; a flat / ramp / impulse probe for new upsamplers | artifacts, new archs: G4-Q12, Q13; G1-Q6, Q10; G2-Q8, Q10; full gates of G5-Q2 | M-L | medium: unvalidated measures; out of any gate until G5-Q3 calibrates them | none | synthetic defects move each measure; random PixelShuffle init vs `icnr_` | G4 7, G5 11 |
| F27 | **Tiler manifest and multi-scale option** in `scripts/extract_subimages.py` | re-tiling for G1-Q4; needs the originals | S | multi-scale stays opt-in | none | manifest round trip; default output bitwise unchanged | G1 10 |
| F28 | **Export parity on probe images** (torch vs ONNX fp16, full vs tiled, as metric deltas) | deployment: G5-Q12 | S-M after F2 | low | none | - | G5 13 |
| F29 | **NR metrics**, opt-in or offline-only with a pinned dependency (five FR metrics today, `neosr/metrics/calculate.py:24-473`) | G5-Q11; a precondition of X14 | M | medium: footprint, bias toward oversharpening (discord observation / hypothesis, DX:490) | none | reference-value tests as for LPIPS (`tests/test_s3_metrics.py:136`) | G5 10 |
| F30 | **Objective-side opt-in keys, each needed by one arm**: per-objective sharpened-GT target; high-pass band wrapper; consistency low-pass width; relativistic-average GAN type (RaGAN / RaHinge) | G1-Q10 = G2-Q10; G2-Q8; G2-Q11; G3-Q10 and an exact port of the owner's recipe | S-M each | medium: objective changes, opt-in only | none | off = bitwise today's losses; formula tests against a reference | G1 9 = G2 6, G2 8-9, G3 4 |
| F31 | **D-side core changes**: static D cadence keys; multi-head D contract (DR-11); `metagan` without GroupNorm (DR-08, arch-specific) | G3-Q9; the multiscale arm of G3-Q5; G3-Q6 | M, M, S | medium-high: core loop, R1/R2 probe, health log; cadence needs the owner's ruling | none | 1:1 and single-tensor paths bitwise unchanged | G3 6-8 |
| F32 | **One HR, several LQ variants** (paired dataset drawing one of N LQ roots) | G1-Q8-Q11 if run offline; blocked with them | M | variant leakage into validation | none | deterministic draw per (seed, epoch, index) | G1 6 |
| F33 | **People-domain machinery**: identity probe (offline), region-aware sampling, mask-weighted losses, mixed sources with a replay share | G4-Q12 (training arms), G4-Q9 | M-L | external weights and licences (face parser, ArcFace); unsourced numbers (generic, SKT:56-57) | none | n/a until the owner wants it | G4 14, G5 17 |
| F34 | **Single-arm or conditional items**: ConvNeXt-Tiny perceptual, HSLuv loss, contextual loss, VGG-style D, scheduled module parameters, scheduler with a cooldown, wavelet cross-fade, safetensors export, online per-term norms | one arm each (G2-Q6 ConvNeXt, G2-Q8 CX, G3-Q13, G4-Q5, G4-Q7 fade) | S to M-L | medium to high (CX: memory, NaN under AMP, discord DX:424); the cross-fade is ST-14, deferred by the owner (`docs/refactor/DECISIONS.md:165-169`) | none | per item, in the sections | G2 10, 12, 13; G3 9-10; G4 6, 13; G5 14-15 |

**Deliberately not proposed.**

| item | why |
|---|---|
| Adaptive loss weighting, adaptive D balancing, AGC, metric-triggered stage switches, ReduceLROnPlateau, a stage engine | the owner's rule: stability from correctness, never automatic damping (`docs/refactor/stage3/competitor/losses_metrics.md:120`, `docs/training_guide.md:3`); per-term schedules and the two-run way cover the documented curricula |
| Gradient-variance loss, A-DISTS, the EMA-refined LDL map | removed from NeoSR for instability (discord, DX:385); fragile implementation; one more G forward for a term that is already inert (`docs/refactor/stage3/competitor/losses_metrics.md:28-30`, `:108`) |
| A discriminator "zoo", D feature matching, LR-conditioned or projected Ds | no KB claim asks for them; the KB's own advice is the simple stable D first (discord, DX:347; NP:393) |
| Generator-side noise or SFT hooks as trainer features (X15), arch comparisons (G3-Q14, G3-Q16) | architecture work; archs rank last in the owner's goal |
| Proxy-generator screens | not adopted in D10: the saving is 13 % and transfer is unknown (`docs/refactor/DECISIONS.md:232-234`) |
| Layer freezing, per-layer LR | generic sources only (PH:526, SKR:44) |
| NR-IQA training loss on an unpaired stream (X14; G2 item 14) | G2 lists it last (size L, risk high) and blocked three ways: no NR metric, no real unpaired LQ, OTF parked; the sources warn against NR as a target (hypothesis, HB:863) |

**Blocked by the parked OTF track** (`docs/refactor/STATUS.md:91`; reported so the cost of the parking is visible,
no work is planned): the bake tool `scripts/degrade.py` (O4, not built: previews and fixed validation sets,
`docs/refactor/stage3/otf/DESIGN.md:165-177`) and the O6 cost check; named degradation buckets with shares; a
configurable kernel-size range (`neosr/data/otf_dataset.py:173-174`); lens-disc and motion kernels; JPEG chroma
subsampling, multi-cycle chains and real codecs; a GT-scale `lq_usm` arm; a realised-diversity log; a
pinned-revision parity test against upstream Real-ESRGAN; on the validation side a real unpaired LQ set and
frozen-seed paired buckets per degradation. Not decided here: whether a resize-only GT -> LQ generator over the
existing `resample.py` kernels (no degradation chain) counts as part of the parked track (section 6, question 1).

## 5. Research plan for the strategy phase (proposal)

The five sections hold 72 questions; merged across sections they are the rows below (section ids kept; "=" joins
the same question asked by two sections; "half" or "arm" splits one question across tiers). G3-Q14 and G3-Q16
(generator and arch comparisons) are left out as architecture work (section 4, not proposed). Costs are GPU-h on
the 4090 for one seed unless "x2" is written; a D10 window verdict is one seed, seed 2 only for promoted arms,
and a default needs two seeds and the long pair (`docs/refactor/DECISIONS.md:225`). Every training arm is under
the owner's hold; the zero-GPU rows train nothing and select nothing. Within a tier, zero-training rows come
first, then training arms by priority. Priorities are the sections', with their reasons; where two sections rate
one question differently both are given.

Statistical scope of every row (review note). A D10 verdict is conditional on the shared P0 and the dev set; its
bootstrap CI over source groups measures evaluation noise, the null RMS from same-config pairs is what covers seeds
(`docs/refactor/DECISIONS.md:212-216`). A question that changes the training LQ, the tile selection or the held-out
set needs its own reference arm and, for a new held-out set, a new same-config null pair
(`docs/refactor/stage3/strategy/topics/g1_data_degradation.md:254-258`). "x2" costs are contingent: under D10
rule 5 a second seed runs only for an arm promoted at its screen. Arms "at a matched share" compare calibrated
configurations, not equal fidelity or intrinsic merit: "equal fidelity" needs a fidelity tolerance fixed in
advance, and quality and cost are reported together (PF:213 warns against forcing equal norms).

### Tier A: runnable now on the current data with existing features

| id | question | arms | cost (GPU-h) | prerequisite | priority (reason) |
|---|---|---|---|---|---|
| G2-Q1 = G3-Q12 = G4-Q11 (X13) | Raw magnitude and gradient share of each term per unit weight, at P0 and along the runs; do shares predict the verdicts we have? | none: `term_grads.py` over existing checkpoints (P0, A0, T_DISTS and the other arms) | zero training; GPU minutes per checkpoint | none to start; F1 makes it complete | **high** (G2, G3) / medium (G4): every weight question needs the scale, and nobody in the corpus has it (PF:213, PG:261) |
| G5-Q1 | Do TOPIQ, LPIPS and DISTS pick the same best checkpoint in one run; are the differences larger than the null? | none: the s10 per-image rows | zero (CPU) | an ad hoc script (shipped form: F5) | **high**: `save_best` has no agreed meaning |
| G5-Q7 | How do null RMS and tau scale with set size (30 / 60 / 100 / 219 tiles) and cadence? Tests the community sizes (official UH3:115, RX:109; hypothesis PG:301) | none: subsample existing rows | zero (CPU) | none | medium |
| G2-Q2 (X5) | Does DISTS help beyond noise, and at what weight? Weights chosen from G2-Q1 magnitudes, never from redux 0.25 / 0.45 (PH:609, PH:714) | T_DISTS seed 2 (held); then one lower and one higher weight | 0.85; ~1.7 per further weight (x2) | the hold | **high**: strongest local signal |
| G2-Q3 | FDL and DISTS: redundant, complementary, or can DISTS replace FDL? | 2x2 FDL {0, 0.02} x DISTS {0, 0.15}; two cells exist, T_DISTS_noFDL is held; every contrast judged on TOPIQ + LPIPS (DISTS cells train on the judge) | ~1.5; ~3 (x2) | the hold | **high**: FDL holds 74-81 % of the squared-norm share and much of the step time |
| G3-Q4 | D LR relative to G: equal, 5x lower, 40x lower? (official TG:30 vs release PH:781, PH:790) | `optim_d.lr` 2e-4, 4e-5, 5e-6 vs 1e-4 | ~1.9 | none; the template D is `adan_sf`, not AdamW | **high**: config only; the release pair and the owner's run both point at a slow D, on unlike setups |
| G3-Q11 (X5, GAN axis) | Which D regime works best once DISTS is in the objective? | 2x2: D {template, slow D of G3-Q4} x DISTS {off, on} | ~0.85: one new cell (slow D + DISTS); template x DISTS on/off exist (A0, T_DISTS) and slow D without DISTS is G3-Q4's arm | G2-Q2 seed 2; the G3-Q4 result | **high**: the owner's own recipe question |
| G4-Q7 (short form of G3-Q7; X6 timing) | Does `wavelet_guided` help at all, when should it start, does the hard switch hurt? (official NL:16; discord DX:303) | template (= off within 20k) vs a switch at 8000; `wavelet_init = 0` second seed and its 20k extension | 1.2-2.5 (x2) + ~0.9; the long form G3-Q7 (RGB always / 40k / 80k) is three long arms, ~7 h | none (the fade arm: F34); extending `wavelet_init = 0` needs its D10 screen eligibility shown or an owner exception (D10 rule 5: no extension of clear 8k losers, `docs/refactor/DECISIONS.md:225`) | **high** (G4): untested in 82 templates; no longer a prerequisite of the long pair (both arms cross the switch) |
| G2-Q5 = G4-Q8 (X5) | How strong must the pixel anchor be in the GAN stage? (release PH:707-716 vs discord synthesis DX:321-323) | `pixel_opt` weight 1.0 / lower / 0, MS-SSIM and consistency fixed | ~2.5 (2 arms, x2) | shares per arm; F16 as the colour check; the Charbonnier arm needs F17 | **high** (G2) / medium-high (G4): the split sits where the owner's goal is |
| G2-Q6 (X8) | Which feature term gives the best texture per cost: FDL-DINOv2, FDL-VGG, VGG19 perceptual, DISTS (a calibrated configuration per term, not "equal fidelity")? | swap the term at a matched share | ~1.8; ~3.6 (x2) | the G2-Q1 numbers; the owner's matching rule | **high**: the corpus has no backbone comparison (PH:418, RG:111) |
| G4-Q6 = G5-Q9 | Does the 12k-20k window rank arms the way 100k does? (generic SKM:39 vs hypothesis DX:565) | A0 and one challenger, 20k -> 100k, paired | 5-6 per pair; 10-12 (x2) | a challenger with two seeds; the 80k switch hits both arms (fair under the schedule; windows before / after 80k reported apart) | **high** (G4) / medium (G5): it tests short-window transfer for one finalist, not D10 in general |
| G5-Q8 (X13, arm half) | Does moving one term to a target share move the judges as predicted? | template; one arm with one weight changed | 0.6-1.25 | the share measurement above | medium-high |
| G3-Q1 (X4, fixed arms) | Which GAN weight does the template D want on the neosr scale, read as adversarial share? | `gan_opt.loss_weight` 0.1 and 0.2 vs 0.3 | ~1.2 | none | medium: 0.3 has only an 8k "mixed" result against 0.1 |
| G1-Q1 = G4-Q16 | Batch 8 vs 16 vs 8 x accumulate 2 at fixed crop (official TG:15 vs discord PH:1009) | three arms | ~0.9 + ~1.2 | the owner's matched-budget rule; F11 for the report | medium (G1) / low-medium (G4): removes a confound of the owner's recipe |
| G4-Q5 (level half) | 2e-4 or 1e-4 for a fine-tune? (official TG:24) | one arm | ~1.2 (x2) | none; the decay half needs a long run (2 x ~3 h, x2) | medium: every release decays, no template does |
| G4-Q3 | Does the pretrain help, and how long must it be? (official TG:7-33; discord HB:681) | from scratch / P0 20k / a longer P0; a second P0 seed | ~3.7 (3 arms, x2) + the pretrains | a longer or second pretrain (compute only) | medium: universal prior, no local measurement |
| G3-Q3 = G4-Q4 (X4 ramp half) | GAN from update 1, delayed, ramped or after a perceptual prefix, at the same final weight? (official TG:29 vs hypothesis SKR:14) | `start_iter`, `ramp_iters`, a prefix arm | ~1.2 (G3: 2 arms); 3.7-4.9 (G4: 3-4 arms, x2) | none (plateau variant: F19) | low (G3) / medium (G4): only hypotheses behind it |
| G2-Q4 | LDL: inert, artifact brake or hidden stabiliser? | LDL {0, 1.0} past 20k on two seeds; a weight with a non-zero share | 4.5-6 (long pair); ~1.2 (weight arm, x2) | F18 for the map question | medium: removal saves ~5 % time |
| G3-Q10 (hinge arm) | GAN type on one D: bce vs hinge | one arm | ~0.6 | none (RaHinge: F30) | medium: needed to explain the OWN gap |
| G2-Q8 (FF arm; X7) | Does focal-frequency supervision add micro-texture with less grain than more GAN weight? | base + `ff_loss`, magnitude-matched | ~0.6 per arm-seed | none for the arm; the "less grain" reading needs a flat-region measure (F26) | medium |
| G1-Q4 (small-crop side) | LR 48 x batch 8 vs 64 x batch 8, which also brings back random crop positions (hypothesis RP:190-192 vs generic SKR:23) | one arm | ~0.4-0.5 (unmeasured) | fit of G and D at GT 192 not checked | runnable half of a high question |
| G3-Q8, G4-Q15, G3-Q2 | Fresh vs loaded D; clip mode where the clip is active; a gated weight ladder vs a fixed weight | 2 arms; 3 arms x2; one ladder | ~1.2; ~3.7; ~0.9-1.2 | the ladder conflicts with D10 rule 6 (owner) | low-medium |
| G1-Q2, G1-Q3, G2-Q7, G2-Q9, G2-Q13, G4-Q2, G4-Q14 | Single batch augmentations; fewer tiles; VGG layer set; FF settings; FDL `num_proj`; EMA horizon; bf16 vs fp32 | config only, one key each | ~2.5; ~1.2; ~1.2; ~0.6; ~0.6; ~2.5-3.7; unmeasured | G2-Q7, Q9 only if their parent question is positive; G2-Q13 only near the end of a long run | low (G2-Q7 low-medium) |

### Tier B: needs a framework item of section 4

| id | question | arms | cost (GPU-h) | prerequisite | priority (reason) |
|---|---|---|---|---|---|
| G4-Q1 = G5-Q4 | Are raw and EMA weights different candidates on our judges; which is the better seed? (discord RP:172-174) | score raw G from the kept s10 states against the EMA rows; then a fine-tune from each | zero for the read; ~2.5 for the seed test (x2) | F2; F12 for the seed test | **high** (G4) / medium (G5): decides what "a checkpoint" means |
| G4-Q10 = G5-Q6 | Is interpolation between P0 and a GAN checkpoint a usable fidelity-texture dial? (discord RP:56) | alpha grid between existing exports | zero training | F3, F2 | **high** (G4) / medium (G5): a direct knob at no training cost |
| G5-Q3 (X2, FR half) | Which judges react in the right direction to small known defects (halo, blur, stipple, wax) on the owner's HR? | sensitivity table on dev GT | zero training | F2 and a small defect generator; NR half: F29; human half: owner time | **high**: D10 rests on these judges |
| G5-Q2 (X3) | Does a metric shortlist reproduce the owner's blind top-5 on one finished run? | every saved checkpoint of one 20k run | zero training; one owner session | F2, F4, F5, F13 (full gates: F26, F29) | **high**: nobody has shown which metric matches the eye (PG:453) |
| G1-Q13 | Are the current verdicts sensitive to residual train / held-out leakage? | re-analysis with a stricter exclusion list | zero | F10 | medium: protects every other verdict |
| G4-Q12, G4-Q13 (analysis half) | Do the aggregate judges hide region effects on skin, hair, eyes; does the model texture what it should leave alone? | bucketed deltas of existing arms | zero training | F6 and tags; F26 | medium: central to the owner's content |
| G5-Q13, G5-Q12, G5-Q11 | Goodhart check with independent judges; ONNX fp16 and tiling against the null; NR tri-score HR / resized LR / SR | re-scoring only | zero training | F2; F28; F29 | low; low-medium; medium |
| G3-Q5 (X6) | Which D architecture gives photo micro-texture at identical G and matched adversarial share? | `unet`, `patchgan`, `dunet` vs `metagan`; multiscale later | ~0.6-0.8 per D, + one per D-LR retune | F1; the multiscale arm: F31 | **high**: "the D is the real lever" is unmeasured everywhere |
| G2-Q11 | Is the colour anchor needed with the current objective; does its low-pass width matter? (discord DX:361-365) | consistency weight {1.0, 0}; width as a second arm | ~1.2 per arm (x2) | F16; the width arm: F30 | medium |
| G1-Q5 (X9, quota half) | Does a quota of smooth / defocus tiles remove false texture without costing texture? (design PG:101) | smooth share as-is / 25 % / 40 % | ~1.9 + ~1.2 for the bucket null | F15, F25 (or repeated list lines), F6, a smooth / bokeh bucket | medium: all positions are hypotheses |
| G1-Q6 (X10) | Does HR tile filtering teach detail or a sharpened look? (discord DX:197-201 vs DX:279) | unfiltered / defect-only / top-score; then a detune | ~1.2 + ~0.3 per detune arm | F15, F6, F26 (halo score), an unfiltered held-out set | medium: a filter that selects what the judges reward is a Goodhart risk |
| G2-Q10 = G1-Q10 (target half; X11) | Do USM-sharpened L1 / perceptual targets add detail or only edge acutance? (official RG:56 vs discord DX:231-233) | U0 / U1 / U2, D target original | ~0.6 per arm-seed | F30, F26 (FR judges penalise a sharpened target by construction) | low-medium (G2): the judges are not ready / medium (G1, the whole of G1-Q10) |
| G3-Q6, G3-Q9, G3-Q13, HP arm of G2-Q8, RaHinge arm of G3-Q10 | Acting spectral norm in the template D; static D cadence; dropout after ~60k; HP-band term; RaHinge | one arm each | ~0.6 each; G3-Q13 one long arm | F31; F31 and the owner's ruling; F34; F30; F30 | medium; low; low; medium; medium |
| ConvNeXt arm of G2-Q6, CX arm of G2-Q8 | ConvNeXt-Tiny as the feature term; contextual loss as HF supervision | one arm each, magnitude-matched | ~0.6 each | F34 (both) | a later arm of a high question; low |

### Tier C: needs new data

| id | question | arms | cost (GPU-h) | prerequisite | priority (reason) |
|---|---|---|---|---|---|
| G1-Q4 (X9, crop half) | Does a larger LR crop help oriented texture, separately from batch? (official TG:31; discord DX:403) | 2x2: LR 64 / 96 x batch 8 / 4 | ~2.5 for three arms + ~1.9 for a new reference and null pair (pixel-scaled, unmeasured) | the originals re-tiled to >= 384 px GT with a group-safe split; F27, F10, F11 | **high if the originals exist** |
| G5-Q5 | dev (219 tiles) vs val8 (8 photos): which ranking follows the owner's eye? | none; a larger source-disjoint tagged photo set with its own null | minutes of scoring | new data, owner time, F6, F10 | **high**: decides what every later verdict means |
| G1-Q12 | Native x2 vs x4 followed by a downscale (hypothesis PF:90; nobody measured it) | native x2 model vs existing x4 + downscale | ~1.2 (pretrain + arm, unmeasured) | an x2 LQ set from the same GT (generation: the owner's call, question 1), F9 | medium: the owner trains x2 as well |
| G4-Q9 (X10) | Detail first, then detune at a low LR on cleaner HR? (discord DX:279, one interim message) | detailed parent / clean scratch / adapted | 1 pretrain + 2-3 arms of ~1.5 h | a cleaner HR variant of the data; F7, F12 | low now |
| G4-Q12 (training arms; X12) | Is region machinery needed at all? | region sampling / weights / identity loss | not estimated | a face parser, an identity metric, a person-disjoint holdout; F33 | medium, after the analysis half |
| G4-Q17 = G5-Q10 (X16) | How far is the GAN from a diffusion reference on the same harness? | inference only | no training | an external reference model; F2 | medium (G4) / low now (G5) |
| G2-Q12 (X14) | Does a small NR-IQA loss on real unpaired inputs help? | template; + NR loss | unknown | real LQ data, F29, a second data stream | low: blocked three ways, one of them the parked OTF track |

### Tier D: blocked by the parked OTF track (reported, no work planned)

| id | question | arms | cost (GPU-h) | prerequisite | priority (reason) |
|---|---|---|---|---|---|
| G1-Q8 (X1) | Which blur dose makes the model re-draw detail before bokeh breaks? (release PH:741-745) | `blur_prob` {0.5, 0.85, 1.0} x sigma max {1.5, 2.2, 3.0} | >= 5.6 at 20k, or >= 2.2 as an 8k screen (OTF step cost unmeasured) | degraded train LQ, bucketed dev sets, real blurred inputs | high for the stated goal; the largest gap of the corpus |
| G1-Q9 (X1, shares) | How large must the clean share be so that good inputs are not redrawn? | `clean_prob` 0 / 0.15 / 0.30 / 0.40 | >= 1.9 | as G1-Q8, plus a near-identity clean bucket | high once G1-Q8 is open |
| G1-Q10 (X11) | Do synthetic noise, target USM and LQ-side sharpening each raise wax or halos? | 2x2 noise x target USM; `lq_usm` later | >= 1.9 | noisy LQ, F30, a GT-scale `lq_usm` (absent) | medium |
| G1-Q7 | Is the LR kernel a sharpness lever or a hidden deblur task? | `aa_bicubic` / `aa_mitchell` / mixture / cubic + mild blur, cross-evaluated | ~2.5 + cross-evaluation | LQ generation per kernel (a resize-only generator would do: question 1), a real-input set | medium |
| G1-Q11 | Do frozen offline variants and OTF sampling train differently? | paired on N baked variants vs OTF | >= 1.2 | the bake tool, F32 | low |
| G3-Q15, the 1x half of G2-Q5, the U3 arm of G2-Q10, G4 C9 / C10 / C17 | GAN at 1x cleanup; the no-pixel 1x regime; sharpened LQ source; data-first fixes and an adaptation stage | - | - | 1x or degraded data | cannot be asked locally |

**Suggested order for tiers A + B (proposal).** GPU-h figures are sums of the rows above (study arithmetic).

| step | content | GPU-h | why here |
|---|---|---|---|
| 0 | Zero-training reads: G2-Q1 = G3-Q12 = G4-Q11, G5-Q1, G5-Q7 now; with F1-F3 also G4-Q1 = G5-Q4, G4-Q10 = G5-Q6, G5-Q3 (plus a defect generator), G5-Q13; with F4, F5, F13 and one owner session the FR-only G5-Q2; with F10 G1-Q13 | no training; GPU minutes for gradient probes and learned-metric scoring | they set the scale, the meaning of "best" and the trust in the judges for everything after |
| gate | **Calibration record before new selection runs**: the FR half of G5-Q3 (do TOPIQ / LPIPS / DISTS move the right way on small known defects of dev GT) and one blind owner comparison of existing arms on the visual set (16 dev tiles + 4 real-LQ crops, `docs/refactor/stage3/r3gan/RESULTS.md:17`; crop sheets: F13), which also probes the dev-vs-val8 disagreement as a reduced G5-Q5. Steps 3-4 wait for an owner-accepted record; steps 1-2 continue arms already planned under D10 | ~0 | X2 / X3 first (PG:469) only matters if later steps depend on its outcome (review note) |
| 1 | G2-Q2 (T_DISTS seed 2, 0.85) and G2-Q3's two missing cells (T_DISTS_noFDL; FDL off without DISTS), one seed each (~1.5) | ~2.4 | the first two are the held arms (`experiments/s10/queue2.held.txt`); the third is new; cheapest high-value runs |
| 2 | Confirm the strongest signal before broad sweeps: seed 2 of the step-1 finalist if step 1 did not run it (~0.85), then the matched long pair G4-Q6 = G5-Q9 (A0 + the finalist, 20k -> 100k) | 5-6.85 | a default needs two seeds and the long pair anyway (D10 rule 5); both arms cross the 80k switch, so the pair stays fair; it tests short-window transfer for this finalist |
| 3 | D regime: G3-Q4, G3-Q1, then G3-Q11's one new cell, one seed each | ~4.0 | config only; after the calibration gate |
| 4 | Anchor and feature term: G2-Q5 = G4-Q8 (~1.2), G2-Q6 (~1.8), one seed each | ~3.0 | needs the step-0 scale and the owner's matching rule |
| optional | G4-Q7: wavelet inside a 20k window, only if the owner wants the switch's own effect isolated; it reopens D6 / D10, under which an early wavelet fork was not adopted (`docs/refactor/DECISIONS.md:195`, `:201`, `:232-234`): owner decision 8 | ~1.2-2.5 | no longer a prerequisite of the long pair |

Total: steps 1-4 ~14.4-16.3 GPU-h, of which ~9.4-10.3 are one-seed 20k arms and 5-6 the long pair. Not counted:
second seeds of promoted arms (0.6-0.85 each), the optional wavelet step and G3-Q5 (D architecture, after F1). The
rest of tiers A and B, and tiers C and D, are outside this sum. The previous draft ran the wavelet step before the
long pair and allotted two seeds up front (~19-27 GPU-h with the long pair); the Codex review showed both were not
needed (section 8).

**The order the playbook proposes, and our view.** PG runs X2 and X3 first, "since every later verdict depends on
the metrics", then X1, X5, X4, the rest by observed defect (design, PG:469).

| PG step | maps to | our view |
|---|---|---|
| X2 (metric sensitivity) | G5-Q3 | Agree it comes first. Its full-reference half is a zero-GPU table once F2 exists; the NR half needs F29 and the human half the owner's time |
| X3 (gates + Pareto vs a blind top-5) | G5-Q2 | Agree in principle; runnable only in a reduced FR-only form (no guards, no NR metrics, no crop-sheet tool yet) and it needs one blind session of the owner |
| X1 (blur ablation) | G1-Q8, G1-Q9 | Cannot be next: the data is clean and the OTF track is parked. It stays tier D until the owner reopens it |
| X5 (loss attribution) | G2-Q2-Q5, G3-Q11 | Agree; partly run already (T_DISTS, NoGAN_DISTS). We put X13 (the scale measurement) before it, which PG leaves to "the rest" |
| X4 (gated GAN ladder) | G3-Q1, G3-Q2, G3-Q3 | Only its fixed-weight arms fit D10; the ladder conflicts with rule 6 (straight 0 -> 20k) and its gates do not exist |
| X6 (D architecture; `wavelet_guided` after 40k) | G3-Q5, G3-Q7 = G4-Q7 (timing half) | PG leaves X6 to "the rest"; so do we: the long pair crosses the 80k switch in both arms, so the timing question is optional (owner decision 8) |
| not in PG | G4-Q6 = G5-Q9 | The window-vs-long-run check is a local protocol question PG does not have |

## 6. Decisions only the owner can make

Merged from the 49 owner questions of the five sections; the smaller ones (batch augmentation, PSNR convention,
`save_best`, skip policy, diagnostics scope) stay in the topic sections, part 7.

| # | decision | options |
|---|---|---|
| 1 | **Data scope of the strategy phase** | (a) stay on clean x4; (b) also admit a resize-only GT -> LQ generator (unblocks the kernel and x2 questions); (c) reopen the offline bake tool of the OTF track (unblocks blur, clean share, noise). Includes whether x2 / 1x questions are queued or dropped, and which matters more: staying faithful on clean inputs or repairing damaged ones |
| 2 | **The hold** | (a) lift it for the zero-GPU reads only; (b) also for the two held arms; (c) for the order of section 5; (d) keep it and reorder |
| 3 | **Framework items vs Studio** | (a) none before Studio; (b) the reliability tranche of section 4 first (F14, F7, F21, F22, F28, F4, F8-F10, the bench gap); (c) the research tools F1-F6 first; (d) also the opt-in trainer keys (F7, F12, F19, F23); whether the data tools ship in `scripts/` or stay ad hoc in `experiments/`; and whether Studio exposes them |
| 4 | **Final judge and deliverable** | metrics alone, or a metric shortlist with one blind sign-off by eye; one model, or a fidelity and a detail model from one lineage with an alpha dial. Also: how was the redux model picked, by eye or by val8 DISTS? |
| 5 | **Which judge decides a trade** | a mixed result (LPIPS better, TOPIQ worse) stays a LOSS as today, or one judge outranks the other for re-drawn texture; and whether D10 (two seeds + the long pair) stays the promotion rule |
| 6 | **Fidelity tolerance** | "anchor of at least some magnitude" as a fixed constraint, or a weak-pixel / no-pixel GAN stage as a legitimate x4 arm |
| 7 | **D regime and the damping boundary** | working hypothesis: the near-chance template D, a slow weak D, or a D that clearly wins; are static label smoothing and a static D cadence admissible opt-in changes or "damping"? Would a recipe without a D be acceptable if it tied the GAN arms? |
| 8 | **Long pair and schedules** | let the long pair cross 80k as planned (fair under the schedule; windows before / after 80k reported apart), or isolate the switch first with G4-Q7 inside 20k (D6; D10 rule 8, an early wavelet fork not adopted: `docs/refactor/DECISIONS.md:195`, `:228`, `:232-234`); LR decay as a strategy question now, or still last (E2); whether LDL gets its own long pair (~4.5-6 GPU-h, G2-Q4) or D1 "keep" stands without further runs |
| 9 | **Protocol extensions** | forks and ladders next to D10 rule 6, or straight runs only; the matched-budget rule for batch and crop arms (updates, images or LQ pixels); "magnitude-matched" as equal weighted contribution (discord, RP:139) or equal gradient share (hypothesis, RG:169) |
| 10 | **Data assets** | do the untiled originals exist; are 50+ real LQ samples of the target inputs available; is there a cleaner HR variant; may a larger tagged held-out photo set be built, and does the confirmation set stay sealed? |
| 11 | **People content** | which features are fidelity constraints (identity marks, eyes, teeth) and which may be re-drawn (pores, strands); who tags the dev tiles (owner or a parser); is an identity metric with external weights acceptable? Is "close-ups of people" the target domain, or will other content be added? |
| 12 | **New objective and D features before evidence** | build only after the measurements show a gap the existing terms cannot fill (the sections' default), or now: RaGAN / RaHinge for an exact port of the owner's recipe, the DISTS cost rewrite (or is +37 % time, +47 % steady VRAM acceptable without it), NR metrics, HP band, multi-head D, HSLuv, a per-loss USM target (only to test the conflict: neosr-dev already has no USM anywhere), DR-08 (metagan without GroupNorm, opt-in, while D3 keeps metagan@0.3); and whether the strategy phase stays on `realplksr` or moves to the owner's own arch |

## 7. Sources

| kind | where |
|---|---|
| Topic sections (each ends with its Verification part) | `docs/refactor/stage3/strategy/topics/g1_data_degradation.md`, `g2_losses.md`, `g3_gan_d_g.md`, `g4_dynamics_artifacts.md`, `g5_validation_framework.md` |
| Claim JSON (gitignored) | `experiments/study/claims/` (slices A-E: 222 + 212 + 189 + 194 + 174 claims; `F_capabilities.json`: 93 rows; `G_local_evidence.json`: 46 findings), `experiments/study/bundles/g1..g5_*.json` |
| KB guide and index | `/home/korvin/apps/Trainings/CLAUDE.md`, `tools/aliases.tsv`, `index/FOCUS.md` |
| KB, official | TG `redux/Training-Guidelines.md`; RX (redux training wiki); NL `neosr/Losses.md`; NO `neosr/Optimizer-Options.md`; UH1-UH5 `upscale-hub/`; SI `Siax/info.txt` |
| KB, release configs and cited reports | PH `SISR_SR_GAN_Training_Playbook_Phhofm.md`; RP `RealPhoto_SISR_Texture_Playbook_aggregated.md`; DX, PF, FG, RG, HB, NP under `ee/` |
| KB, designs | PG `synthesis/Photoreal_Texture_GAN_Detail_Playbook_2026-11-13.md` (experiments X1-X16, run order PG:469); DR `synthesis/Default_SISR_Training_Recipe_2026-11-12.md` |
| KB, low trust | SK* (`skill/sr-restoration-engineer/`), AN, SL, SX, S1-S4 |
| neosr-dev status and rules | `docs/refactor/STATUS.md`, `next_tasks.md`, `docs/refactor/DECISIONS.md` (D1-D10: `:188-198`, `:206-239`), `AGENTS.md` |
| neosr-dev evidence | `docs/refactor/stage2/DECISIONS.md`, `docs/refactor/stage2/CLOSURE.md`, `docs/refactor/stage3/PROGRESS.md`, `docs/refactor/stage3/owner_recipe.md`, `docs/refactor/stage3/r3gan/RESULTS.md`, `docs/refactor/stage3/discriminator/RESEARCH_REVIEW.md`, `docs/refactor/stage3/otf/DESIGN.md`, `docs/refactor/stage3/competitor/`, `experiments/s10/` (runs, `NOTES.md`, `consult/`; gitignored) |
| neosr-dev tools | `scripts/term_grads.py`, `scripts/ab/analyze.py`, `scripts/ab/run_ab.py`, `docs/training_guide.md` |

Not verified by this study (carried from the sections): the PH config values against the original config files;
the Discord messages behind every cited report; upstream Real-ESRGAN and NeoSR behaviour at a pinned revision;
the stage-2 8k numbers and the owner's run numbers (read from their docs, not re-measured); every GPU-h figure
(extrapolated from ms/update; crop, fp32, batch 16 and OTF step costs are unmeasured).

## 8. Review record

Codex (gpt-6.1-sol, read-only) reviewed the draft after the critic pass. Each finding was checked against the
sources before it was applied.

| # | finding (severity) | checked | applied |
|---|---|---|---|
| 1 | the backlog rank favours experiment convenience over framework reliability; interpolation sat above load audit, smoke tests and export parity (high) | yes: the rank rule counts questions served | partly: a second, reliability-first lens and the bench gap added to section 4; the 34-row order is kept and the choice of lens left to the owner (decision 3) |
| 2 | X2 / X3 are first in name only: no later step depends on their outcome (high) | yes | a calibration gate before steps 3-4, using the existing visual set (`docs/refactor/stage3/r3gan/RESULTS.md:17`); steps 1-2 stay, they continue arms already planned under D10 |
| 3 | the long pair is not confounded by the 80k switch: both arms cross it (high) | yes (`docs/refactor/DECISIONS.md:228`, `:234`) | the wavelet step left the order (now optional); summary item 8, G4-Q6, the PG table and decision 8 reworded |
| 4 | some runs conflict with D10 rule 5 (extending an 8k loser, blanket second seeds) (medium) | yes (`docs/refactor/DECISIONS.md:225`) | eligibility note on G4-Q7; second seeds contingent on promotion |
| 5 | statistical qualifications lost: reference / null reuse on new data, CI vs seed noise, "strata sum to the pooled verdict" (medium) | yes (`topics/g1_data_degradation.md:254-258`) | statistical-scope paragraph before tier A; F6 check reworded |
| 6 | a matched gradient share is not "equal fidelity" (medium) | yes (PF:213) | G2-Q6 reworded; estimand note before tier A |
| 7 | G3-Q11 double-counted a cell; step 0 is not zero-GPU (medium) | yes | G3-Q11 ~0.85; step 0 "no training, GPU minutes"; totals recomputed (~14.4-16.3 GPU-h with the long pair) |
| 8 | three statements exceed their anchors: "none is clean data", PF:213 / DR:6 labels, the 8-10 image set (medium) | yes (PH:765; PF:213 untagged, so [H]; DR:6 is the DR reading contract) | fixed in summary items 1-2 and three rows of section 3 |

