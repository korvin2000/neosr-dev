# Validation metrics, checkpoint selection, framework features, deployment

Study section G5. Input: `experiments/study/bundles/g5_validation_framework.json` (300 claims, 11 conflicts, 24 gaps,
121 framework features, 25 capability rows, 12 local findings). Nothing here is decided; the owner chooses.

Reading rules used below:

- Anchors `ALIAS:LINE` point into `/home/korvin/apps/Trainings` (aliases: `tools/aliases.tsv`); `path:line` points into neosr-dev.
- Evidence labels are the bundle's: official, release_config, discord_cited, paper, generic, hypothesis, design.
  PG (the focus playbook) marks its own constructions (D) and unsourced numbers (H).
- Evidence mix of this group: hypothesis 132, discord_cited 61, design 41, generic 26, official 19, paper 18,
  release_config 3. **About two thirds of the group is proposal or design, not observation.**
- "Study reading" = something this study read or computed (code, run files), not an owner-accepted verdict.

## 1. What the community converges on

### 1.1 Official wiki text (strongest label in this group; narrow context)

- **Metrics are loose triage; a worse-scoring checkpoint can look better; metrics mostly flag gross artifacts.**
  Official (UH3:121, RX:115). Context is Compact/anime-era wiki advice with PSNR/SSIM/DISTS only, not a photoreal GAN study.
- **Fidelity stages end at the validation-metric peak; the degraded/GAN stage ends "when it looks good", eyes are the final judge.**
  Official redux guideline (TG:13, TG:23, TG:32). No metric, threshold or stop rule is named for the GAN stage.
- **In-loop validation set: at most 8-10 images or pairs** (the wiki says "no more than", an upper bound, not a target),
  ideally representative of the target sources, optionally one image per ability (UH3:115-117, RX:109-111, official).
  This is a progress monitor, not a statistical test (see 3.1).
- **Stop heuristic:** if validation scores slip or stay flat over several validations the model is likely done; confirm visually.
  Official (RX:273), written for Compact models, not GAN-stage specific, no metric or threshold named. DR:247 repeats it.
- **Cadence:** redux default saves and validates every 1000 iterations; the wiki author uses save 2500 / validation 5000 "to avoid the spam".
  Official (RX:245). Not a quality argument.
- **Shipped metric sets:** redux PSNR/SSIM/LPIPS/DISTS/TOPIQ, upstream NeoSR PSNR/SSIM/DISTS/TOPIQ (NP:59). DISTS is lower-is-better (RX:115, RX:263).
- **Two upstream-NeoSR code findings** (NP label V, read from an indexed web view, never executed, HEAD not pinned: NP:31, NP:45):
  C01 metric mean written as `sum / _idx + 1` (NP:80-88, issue #116) and C02 validation flipping `is_train` so TensorBoard image
  logging and the schedule-free eval switch are skipped (NP:95-97). They describe upstream, not neosr-dev (checked in 3.4).

### 1.2 Release configs

- Only 3 release_config claims touch this group and none of them is about validation or selection: the two 1x snapshots give no
  batch size (they do name D `rvgg`, `lq_size` 96 and the degradation table: PH:553-660), the 2x snapshot lacks its discriminator,
  degradation and crop keys (PH:765), and upstream NeoSR changed the meaning of
  loss names across releases (DX:385-389). **No release documents how its checkpoint was chosen.**

### 1.3 Cited Discord reports (mostly one message from one participant; images never opened: DX:35-46, DX:770-781)

- **Metric vector, not one winner.** Helaman reports LPIPS, DISTS, TOPIQ next to PSNR/SSIM (RP:228; one 2025-11 message,
  TOPIQ variant unstated). In a DF2K-bicubic/Urban100 comparison a U-Net+BCE discriminator raised PSNR/SSIM while LPIPS/DISTS
  got worse (DX:353-357, reported_test). A low perceptual weight improved some metrics while producing severe artifacts, and a
  lower score can look worse even for a perceptual metric (PF:209, PF:54, both [C], one participant; PF:203).
- **Metric conventions create fake differences.** Same stage read 32.25 dB on Y and 30.61 dB without Y-only evaluation
  (DX:480-482; HB:866, HB:1243). Different SSIM tools give different numbers (NP:93, NP:597-598, NP:607). Anomalous metrics came
  from wrong validation geometry (HB:1074, HB:1256).
- **Calibrate the metric stack on HR first.** HR scored against itself (brisque, clipiqa on Urban100) before comparing models
  (PH:1013, PH:1171; RG:125). Many NR metrics did not rank HR first (DX:488-494, reported_observation/hypothesis, mechanism not isolated).
- **Many trainers prefer manual visual checks on diverse unseen images** (HB:878 citing [D31] = HB:1253, discord). The
  "3-5 candidates from the time series, never latest, blinded" procedure around it is HB's own [Proposal] (HB:868-878);
  DR:243-244 is a KB synthesis of the same lines, not an independent report.
- **One image improves while another gets fake texture** (a head fixed, a sign turned into texture): judge per content category,
  no showcase crop (DX:496-502, reported_observation). DX records no automatic rule (`checkpoint_selection_rule` is null: DX:688).
- **With OTF training validate on unpaired real deployment inputs; paired validation only "when you want metrics"**
  (PH:1011, PH:1168; DX:466-470 with RealLR200 as the unpaired set: DX:823, DX:923).
- **Checkpoint interpolation** between two late checkpoints balances sharpness against artifacts; done by eye in chaiNNer,
  alpha unstated (RP:56, RP:325).
- **EMA can be the wrong candidate:** EMA plus cosine annealing from scratch gave over-smooth output; compare EMA with non-EMA
  and use non-EMA as the pretrain if EMA is too soft (RP:172-174).
- **Ship variants from one lineage:** mild and strong models from one pretrain, the strong one loses detail on mild inputs
  (DX:295-297, release_recipe). The stop rule DR:247 cites is the official RX:273 (see 1.1), not a Discord report.
- **Process hygiene:** split by source before tiling (RG:42 paper+derived, PF:52 [H]; source-disjoint milestone set DX:476, advice);
  step-zero validation to prove the pretrain loaded
  (DX:271-275); `strict=false` without an exception does not mean compatible (HB:1069, HB:1234); trainer output can differ from
  chaiNNer/ONNX through arch variant, normalisation, `params` vs `params_ema`, padding (HB:1068, HB:1255); change one knob at a time
  and use loss schedules for ramps (PH:1027-1028).

### 1.4 Paper-backed framing (principles, no SISR-GAN numbers)

- Perception and distortion trade off at distribution level: keep a fidelity checkpoint beside the perceptual one (PF:33, FG:42; FG:51 is (H)).
- A metric optimised as a loss inflates itself (SKL:16 generic; FG:405 sits in FG's [H] selection policy; PF:203). Locally measured in 3.2.
- DISTS tolerates texture resampling by design: a good DISTS does not prove correct fine detail (FG:392, SKL:67, RG:206; HB:861; FG:512).
- NIQE rewards noise and oversharpening; learned NR metrics disagree with each other, use at least two (SKL:70, SKL:71; the +-0.2 noise figure is (H)).

### 1.5 Designs and hypotheses (never run by any source; do not read as evidence)

- **Validation sets V1-V6:** paired synthetic with frozen seeds, real unpaired, texture/face probes, negative controls,
  calibration, sealed test (PG:301-306; FG:379-383; HB:845-849; SKG:75-78). Sizes 30-100 images and about 20 crops per class are (H);
  the SX pack 20 faces/10 hair/10 foliage/10 materials/5 gradients is a speculative reconstruction (SX:243-255).
- **Calibration once per data version:** identity, HR band per NR metric, tri-score HR / resized LR / SR, defect-sensitivity table,
  noise floor across raw/EMA, precision and seed (PG:332-336).
- **Selection:** veto gates G1-G8 relative to the anchor and HR band, per-bucket Pareto front over four axes, 3-5 shortlist,
  stop rule "3 validations without A2 gain or 2 consecutive gate failures" (PG:341-365; FG:403; NP:322; HB:868-878).
  PG states that no source gives a threshold (PG:343).
- **Scripted guards:** flat-region HF residual, edge overshoot/MTF50, radial power spectrum, invention residual `SR - bicubic(LR)`,
  forward-model residual (FG:394-396, RG:207, SKD:104, SKT:47, PF:62, PF:284). All named, none implemented or validated (FOCUS:284).
- **Cadence table:** GAN refinement validates, renders a blind sheet and saves every 2k-5k (HB:882-888); PG picks 2,500 (PG:25).
- **Metric-protocol record and run manifest** (NP:90, HB:866; RG:229-271, FG:450-499, NP:320, HB:905).
- **Seeds:** at least 2 (HB:1132) or at least 3 (NP:483). **Per-term gradient norms** as the portable balance tool (RG:59, PF:56, NP:323, PG:464).
- **Score "detail" under three labels:** preserved (input-supported), reconstructed (closer to HR), synthesized (plausible, unproven);
  an edge can sharpen by a halo as well as by recovery (DX:48-56; DX calls it its own analytical framing). No source turns it into a measure.
- **Identity and anatomy probes** (directly relevant to the owner's close-up people data): eyes, teeth, moles, hairline are fidelity
  constraints; ArcFace cosine to GT, landmark stability, gate on the worst image because means hide rare catastrophic failures
  (RG:47, SKT:60, PG:374-375; the "fixed one image, broke another" report DX:496-502 is the only observation behind it). Not run anywhere.
- **Colour drift:** per-channel mean bias, flat-patch delta colour, saturation shift (HB:830-835 proposal; colour delta-E in NP:322, NP:341;
  RGB PSNR as a colour sensor DR:234 (D)).
- **Deployment:** repeat the probe pass on the exported fp16/int8 model and the tiled path (SKT:27, SKP:40, PG:394); tile overlap
  at least half the receptive field with feathered blending (SKP:48, SKT:42, generic; PG:415 marks it (H)).

### 1.6 How thin it is

- No automatic checkpoint-selection rule, script, threshold or reported run exists anywhere in the corpus (DX:688, HB:868, FG:403, NP:322, RG:223, PG:454).
  The only official stop heuristic (RX:273) names no metric, window or threshold.
- No source shows which metric tracks human preference for synthesised micro-texture (DX:492, PG:453); no NR stack has a calibration result (PH:1013, RP:228, SKL:70-71).
- No test separates plausible re-drawn texture from invented structure (HB:860-863, NP:341).
- RG and FG state that nothing was run (RG:8, FG:15); PF reports no own runs and labels its rows [P]/[C]/[H]/[E] (PF:13-16);
  DX is 51 findings from chat reports, images and links not opened (DX:32, DX:774-775). x2 and 1x are barely covered (PF:90, FG:283, DX:377).

## 2. Open conflicts

| question | positions (anchors, evidence) | what would settle it | can our setup settle it now? |
|---|---|---|---|
| Who is the final judge of a checkpoint? | (a) blind human review cannot be replaced, metrics flag gross artifacts: UH3:121, RX:115 official; PF:282, HB:878 hypothesis/discord. (b) LPIPS/DISTS/TOPIQ primary, PSNR/SSIM guardrails, grids last: RP:234-238, a compiler recommendation built on RP:228 discord. (c) metrics shortlist, eyes sign off once: PG:442-443 design. (d) "trust your eyes", no metric: SX:199, SX:258 speculative | X3: gates + Pareto shortlist against a blind top-5 by eye on one finished run (PG:454) | Partly. FR axes on existing 20k runs yes; needs the owner's eye once, a crop-sheet tool and (for full gates) scripted guards and NR metrics that do not exist |
| Role of NR metrics in selection | capped naturalness axis after calibration: PG:318, PG:350 design. Never a selection target: DR:236 design. Can rank SR above HR, favour hallucination: DX:488-494 discord, HB:863, NP:341 hypothesis | Tri-score HR / resized LR / SR per NR metric on the owner's data; drop metrics that fail (PG:334) | No. No NR metric in neosr-dev (registrations `neosr/metrics/calculate.py:24-473`: five FR metrics), no real unpaired LQ set |
| Can a metric that is a training loss select checkpoints? | DISTS is a loss at 0.25-0.45 (redux scale) in release stacks: PH:714, PH:609 release_config; SX:151 proposes an LPIPS-like loss. Use an independent metric: SKL:16, FG:405 paper | A loss-on/off pair judged by independent metrics | Yes, already measured locally (3.2); the rule is in D10 |
| What does a good DISTS mean? | largest perceptual term and a texture judge: PH:413, PH:714, NL:123-127, SKT:7. Tolerant by design, proves nothing about fine detail, pair with geometry checks: RG:206, FG:392, FG:512, PF:203 paper | Defect-sensitivity table: does DISTS move on stipple, wax, shifted weave (X2) | Partly: FR sensitivity table on dev GT is cheap; the human-agreement half needs the owner |
| Seeds needed to promote a result | at least 3, mean +- std: NP:483, NP:487, NP:384 hypothesis. At least 2 if compute permits: HB:1132 hypothesis | Measured null of same-config pairs | Yes, measured: D10 uses two seeds plus a long run (3.1) |
| Short proxy screens vs late-arriving HF quality | screen on small models, a negative proxy result speaks about the proxy only: DX:565 hypothesis. HF arrives late, do not judge texture from short runs: SKM:39 generic | 8k vs 20k vs 100k verdict agreement on the same arms | Partly: 8k vs 20k measured (3.3); nothing ran past 20k |
| In-loop validation size | at most 8-10 images or pairs: UH3:115, RX:109 official. 10-30 panel, a diagnostic not statistics: DX:472-478 discord. 30-100 frozen images: SKG:75 generic, PG:301 (H) | Null RMS as a function of set size | Yes: subsample the existing per-image rows (219 dev tiles) |
| Validation / save interval | 1000/1000 default, 5000/2500 by one author: RX:245 official. GAN stage 2k-5k: HB:887 hypothesis. 2,500: PG:25 design | Cost and noise of window means at each cadence | Yes for cost (3.5); cadence sensitivity needs re-scoring of saved exports |
| PSNR on Y or RGB | Y with border = scale: SKL:62 generic, DR:234 design. RGB in the official Real-ESRGAN template: UH4:112. Never compare across: DX:482, HB:866 | Not an experiment: a convention plus a stored protocol record | Trivially; needs an owner decision on the reported default (section 7) |
| Evaluate EMA, raw, or both | evaluate EMA: SKL:93 generic. EMA can over-smooth, score both and treat both as candidates: RP:172-174 discord; RG:63, PF:58, PG:341 design | Raw vs EMA deltas against the null at several checkpoints | Partly, offline: neosr-dev validates and exports EMA only (`neosr/models/image.py:484-490`, `:685-693`), but raw G is in the kept s10 states (4k-5k steps to 20k), so existing runs can be scored (G5-Q4) |
| Keep a pixel term in the GAN stage? (objective, shared with the loss group) | none in both 1x release stacks, MS-SSIM + colour terms anchor: PH:707-716, PH:604-611 release_config. Charbonnier 0.1 in the 2x stack: PH:816-817. Keep it unless ablated: DX:321-325 discord, PH:491, SKL:52-53 generic | One-factor arm: template with the pixel term removed | Yes on clean x4 (one arm); weights in the anchors are redux scale and do not transfer |
| Is issue #116 still in upstream NeoSR? | still visible in the indexed file: NP:45, NP:85, NP:88. A Dec 2025 comment says updating may fix it: NP:45 | Pinned upstream HEAD check | Irrelevant for neosr-dev: its mean is sum / count (3.4) |
| NeoSR or redux better at equal config | no universal conclusion: NP:74, NP:601. Mixed single reports both ways, speed reports historical: NP:602, NP:353 | A matched cross-trainer run | No, and not a goal: redux is a competitor, ideas only |

## 3. What our own evidence says

Scope of every row: clean antialiased x4, GT 256 / LQ 64 tiles of people, one generator (realplksr), fine-tunes from one
20k L1 pretrain, at most 20k updates, EMA weights only, dev = 219 tiles in 211 source groups.

### 3.1 Evaluation noise and the gate (G2, confidence high)

- Null delta RMS of same-config pairs (PSNR / TOPIQ / LPIPS / DISTS): one checkpoint 0.036 / 0.0012 / 0.0019 / 0.0006;
  window 12k-20k 0.033 / 0.0006 / 0.0008 / 0.0003 (`experiments/s10/consult/synthesis.md:14-21`, `docs/refactor/DECISIONS.md:212-216`).
- A single-checkpoint gate called LOSS in 9 of 36 same-config comparisons, all on LPIPS; window means halve LPIPS noise. This is the basis of D10.
- **Agreement:** the designs' "differences inside the noise floor are ties" (PG:336) and "two seeds" (HB:1132). The community gives no number; we have one.
- **Disagreement in use, not in fact:** an 8-10 image set (UH3:115, RX:109) is a monitor. `docs/training_guide.md:59-62` already states that 67 tiles and one seed do not separate 0.05-0.1 dB.
- Limit: the null comes from five runs of two configs. The gate's `--null` uses three pairs (A0/B0, B0/B0s2, U/Us2:
  `experiments/s10/NOTES.md:40`); the RMS table and the 9-of-36 count use four (A0/B0s2 added, which is implied by the other two;
  36 = 4 pairs x 9 validations 4k-20k: `experiments/s10/consult/adv/null.py:19`, `adv/null2.py:16-29`).

### 3.2 A metric used as a loss is not a fair judge (G4, confidence high)

- T_DISTS (template + DISTS 0.15, neosr scale): DISTS moved most (-0.0087). In the owner's redux run val8 DISTS fell only while
  the DISTS loss was on (0.160 -> 0.1245, 0.161 -> 0.1107) and rose 0.1245 -> 0.181 within about 2k iterations when it was off
  (`docs/refactor/stage3/owner_recipe.md:18-20`, `:134-137`; `docs/refactor/DECISIONS.md:198`, `:227`).
- Adopted rule: arms trained on DISTS are judged by TOPIQ + LPIPS, DISTS reported (`docs/training_guide.md:57`; `scripts/ab/analyze.py:7`, `--report-only`).
- **Agrees** with SKL:16, FG:405, PF:203. Consequence for the owner's recipe: it validated on PSNR/SSIM/DISTS of val8 while training on DISTS,
  and how the final model was picked is not recorded (`docs/refactor/stage3/owner_recipe.md:48`, `:144`).

### 3.3 Short screens (G3, confidence medium; G44)

- 8k ranks predicted 20k ranks over 7 arms (Spearman TOPIQ 1.0, DISTS 1.0, LPIPS 0.75, swaps inside the null). No arm dropped at 8k was fine at 20k,
  but 3 of 4 window-4k-8k TIEs ended LOSS at 12k-20k (`experiments/s10/consult/synthesis.md:22`, `experiments/s10/NOTES.md:40-44`).
- Reading: supports "a screen can rank" and also "effects keep growing" (SKM:39). It says nothing about 20k -> 100k, where the template's
  `wavelet_init = 80000` switch lies.
- A lighter proxy generator saves only 13 % per update (spanplus 82.4 vs 94.9 ms: `experiments/s10/consult/brief.md:19-20`,
  `docs/refactor/stage2/PROGRESS.md:86`); both advisors rejected proxy-G screens for recipe questions
  (`docs/refactor/DECISIONS.md:214-216`, `:232-234`). This is local evidence against applying DX:565 here: the step cost is FDL, D and their backward.

### 3.4 Upstream defects the KB reports: status in neosr-dev (study reading of code, no run)

- C01 (NP:80-88): not present. The mean is `sum(row) / len(row)` over per-image fp64 values (`neosr/models/image.py:655-660`);
  a 3-image test compares it with the numpy route to 1e-10 (`tests/test_s2_metrics.py:338-351`).
- C02 (NP:95-97): not present. `eval_weights()` yields the EMA, or swaps schedule-free weights to x and restores parameters, optimizer
  flags and per-module training flags (`neosr/models/image.py:484-516`); round trip tested (`tests/test_model.py:358`). TensorBoard images
  depend on `logger.save_tb_img`, not on a flipped flag (`neosr/models/image.py:583-587`).
- LPIPS double normalisation (FG:399): not present. Quantised input is divided by 255 once (`neosr/metrics/calculate.py:379-380`) and mapped to
  [-1, 1] plus the LPIPS shift/scale once (`neosr/metrics/lpips.py:55-57`); reference-value test at `tests/test_s3_metrics.py:136`.

### 3.5 Costs (G34, G35, G30; confidence high)

- Validation on dev 219 + val8 with 4 metrics costs about 9 s per event (`experiments/s10/consult/brief.md:18`); training validation
  stays fp32 because bf16 validation shifts metrics for a 5 % gain (`docs/refactor/stage2/CLOSURE.md:312`, GP-13;
  `neosr/models/image.py:589`). The community 2k-5k cadence is therefore cheap at our image sizes.
- 4K validation: DISTS 8.59 GiB, TOPIQ pipal 5.97 GiB above their nets, full pass 3.0 s, peak 9.02 GiB (`docs/refactor/STATUS.md:156-166`,
  `next_tasks.md:29`). A 24 GB card fits it; on a 12 GB card 4K DISTS would take the CPU fallback (inferred, not measured) and TOPIQ
  is marginal (`docs/refactor/STATUS.md:166-168`).
- A checkpoint save costs 0.0-0.9 s; `val.save_best` exists but no template enables it because no metric was chosen to mean "best"
  (`experiments/s10/NOTES.md:34-36`, `docs/refactor/stage2/CLOSURE.md:127`).
- Template 95 ms/update in runs, 109.9 in `bench.py`; +DISTS 0.15 is 130 ms in runs, +37 % (`experiments/s10/consult/brief.md:15-17`,
  `docs/refactor/stage3/PROGRESS.md:425-427`).

### 3.6 Other framework findings

- torch.compile: TIE on quality at 8k and on the 12k-20k window; -18 to -21 % in `bench.py` but about 3 % in real runs
  (`docs/refactor/stage2/DECISIONS.md:102-105`, `experiments/s10/NOTES.md:4-7`, `:37-39`). The bench-vs-loop gap is open.
- Resume at 8k does not bias a 20k verdict: A0 vs B0 inside the null on all four metrics, one pair
  (`experiments/s10/consult/synthesis.md:11-12`). Resume is a stochastic continuation, not order-exact.
- No skipped non-finite update in any R5 arm to 20k: `skipped_g` / `skipped_d` are 0 in every log window of all 16 arms
  (`experiments/s10/runs/*/train.jsonl`, re-read by the verifier; `done.json` holds no skip count).
- Per-term generator gradient shares already measured offline (`scripts/term_grads.py`, share of sum of squared norms over all
  trainable G parameters, not a trunk layer): at P0 FDL 81.2 %, GAN 16.5 %, MS-SSIM 1.4 %, L1 0.53 %, consistency 0.35 %, LDL 0.000 %;
  at B0's 8k weights 74.3 / 16.6 / 7.1 / 1.7 / 0.44 / 0.000 % (`docs/refactor/stage3/PROGRESS.md:428-430`). One template, neosr scale.
- R1/R2 penalty: exact rewrite took the recipe from 732.2 to 122.5 ms/update (`docs/refactor/stage3/r3gan/RESULTS.md:91-110`).
  Community cost claims about R1/R2 depend on the discriminator's conv types.

### 3.7 dev vs val8 (G46: a study computation, confidence low, not an owner-accepted verdict)

- At 20k the dev set puts the owner's ported recipe (OWN) in the bottom group on LPIPS (13th of 16 arms, 0.1517; best T_DISTS 0.1421)
  while val8 ranks it first (0.1756); R3 has the best val8 TOPIQ and the worst dev PSNR (`experiments/s10/runs/*/metrics.jsonl`, iter 20000,
  re-read by the verifier). OWN, T_DISTS and NoGAN_DISTS, the three arms with `dists_loss`, lead DISTS on both sets: the G4 effect again.
- n = 8, no CI; val8 is 512 px GT / 128 px LQ photos (`docs/refactor/stage3/r3gan/RESULTS.md:16`); that its LQ is the same synthetic
  downscale type, not camera LQ, is the bundle's assumption, not documented there. It matches the community warning that sets can disagree
  (DX:496-502) but cannot say which set tracks the owner's eye.

### 3.8 Limits

- Local evidence cannot speak to: unpaired real validation, NR metrics, degradation buckets, large-image tiling seams, x2/1x, runs past 20k.
  Raw-vs-EMA and interpolation are unmeasured but readable offline from the kept states and exports (G5-Q4, G5-Q6).
- OTF realism is parked by the owner. For the record only: its validation side would need a real unpaired deployment LQ set,
  calibrated NR metrics on `single` sets (item 10) and frozen-seed paired buckets per degradation (PH:1011, DX:466-470, SKG:75-76);
  no work is planned on it here.
- The owner's own generator keeps noise injection on in eval, so its validation is stochastic (`docs/refactor/stage3/owner_recipe.md:45-46`, `:140`); the local null does not cover that.
- Train/held-out independence is only overlap-verified.

## 4. neosr-dev support

| community practice | status in neosr-dev | exact config keys | evidence path:line | gap or caveat |
|---|---|---|---|---|
| FR panel PSNR, SSIM, LPIPS, DISTS, TOPIQ-FR (RP:228, NP:59) | present | `[val.metrics.<k>] type = "calculate_psnr" / "calculate_ssim" / "calculate_lpips" (net = "alex" or "vgg") / "calculate_dists" / "calculate_topiq" (dataset = "kadid" or "pipal")` | `neosr/metrics/calculate.py:210-286`, `:385-410` | Template ships every metric commented out. TOPIQ here is FR only |
| PSNR-Y, border = scale (SKL:62, DR:234) | present, defaults differ | `crop_border` (4), `test_y_channel` (false) | `neosr/metrics/calculate.py:210-216`, `:257-263` | Default is RGB with a 4 px border at every scale; values are computed on the 8-bit quantised output |
| NR metrics on real inputs (SKL:71, HB:863, PH:1011) | missing | none | `neosr/metrics/calculate.py:24`, `:113`, `:429`, `:449`, `:473` (the five registrations; `neosr/metrics/__init__.py:17-42` is `build_metric`) | Registry holds 5 FR metrics; redux has TOPIQ-NR (`docs/refactor/stage3/competitor/losses_metrics.md:58`) |
| Unpaired (LQ-only) validation set (PH:1168, DX:468) | partial | `[datasets.val_N] type = "single"` | `neosr/models/image.py:576` | Images are written, metrics are skipped entirely |
| Several named validation sets / buckets (PG:301-306, FG:379-381) | present | `[datasets.val]`, `[datasets.val_1]`, ... with unique `name` | `neosr/config.py:244-248`, `:537-543` | One mean per set; no tag grouping inside a set, no median or worst image |
| Per-image results (SKL:93, FG:403) | partial | none in `train.py`; `scripts/ab/run_ab.py` writes `per_image.jsonl` | `scripts/ab/run_ab.py:1-10`, `neosr/models/image.py:655-664` | Only runs started through the A/B harness have per-image rows |
| Count-based metric mean (NP:90) | present | implicit | `neosr/models/image.py:655-660`, `tests/test_s2_metrics.py:338-351` | Plain mean; group-aware statistics live in `scripts/ab/analyze.py` |
| Metric protocol stored with results (NP:90, HB:866, DX:484) | partial | none | `neosr/models/base.py:250-256` | A best export records dataset, key, type, direction, value; not border, Y/RGB, weights variant, code revision |
| Raw and EMA both scored and saved (RG:63, RP:172-174, PG:341) | missing | none | `neosr/models/image.py:484-490`, `:685-693` | With EMA on, validation, `save_best` and exports are EMA only; raw G lives only in `training_states/<u>.state` |
| Side-effect-free validation, schedule-free eval weights (NP:99-101) | present | implicit | `neosr/models/image.py:484-516`, `tests/test_model.py:358` | none found |
| Fixed validation precision (PF:280, RP:182) | present | training validation fp32; `test.py` follows `use_amp` / `bfloat16` | `neosr/models/image.py:566-572`, `:589-594` | Trainer metrics (fp32) and a bf16 `test.py` run are different protocols |
| Tiled validation (DR:238, HB:586) | present | `[val] tile` (-1 = whole image), `tile_overlap` (16) | `neosr/models/tiling.py:20-44`, `neosr/config.py:489-492`, `tests/test_model.py:207` | Cores are copied, not blended; overlap is a fixed pixel count set by the user, not derived from the receptive field (community: at least half of it, SKP:48, SKT:42 generic); the docstring states the tiled path is exact only if RF <= overlap (`neosr/models/tiling.py:30-36`) |
| Separate validation and save cadence (HB:882-888) | present | `[val] val_freq`, `[logger] save_checkpoint_freq` | `train.py:266-267` | Template uses 1000 / 1000 |
| Dense checkpoint history for step sweeps (PF:235, FG:423, HB:878) | present | `[logger] keep_exports` (0 = all), `keep_states` (3) | `neosr/models/base.py:121-123`, `:283-309` | Exports (EMA G and D) are all kept; full resumable states only for the last 3 saves |
| Best tracking and best export | present, single metric | `[val] save_best = "<metric key>"`, `[val.metrics.<k>] better` | `neosr/models/base.py:220-265`, `neosr/config.py:470-488` | One key per run; ties count as improvement (`base.py:237`); one checkpoint, no window mean; off in every template |
| Multi-metric shortlist: gates, Pareto, 3-5 candidates (FG:403, NP:322, PG:356-363) | missing | none | `scripts/ab/analyze.py:1-20` | The analyser compares arms against a reference, not checkpoints within a run |
| Guard: selection metric is also a loss (SKL:16, FG:405) | partial | `scripts/ab/analyze.py --report-only <metric>` | `docs/training_guide.md:52-57`, `scripts/ab/analyze.py:330-349` | Documentation and harness flag only; no config warning for `save_best` on a trained metric |
| Step-zero validation (DX:271-275, HB:1069) | missing in the loop | none | `train.py:266` | Validation runs at `u % val_freq == 0` and at the end; offline equivalent is `test.py` on the pretrain. The other half of HB:1069 is present: a relaxed (`strict_load_g = false`) load logs the loaded share and the skipped, missing and unexpected keys, and a file matching no tensor raises (`neosr/models/checkpoint.py:140-179`) |
| Fixed image grid per validation (SKG:78) | present | `[val] save_img`, `save_lq`; `[logger] save_tb_img` | `neosr/models/image.py:583-587`, `neosr/config.py:159-167` | Writes every validation image, no crop sheet |
| Loss weight schedules (PH:1027) | partial | per loss `start_iter`, `ramp_iters`, `end_iter` | `neosr/losses/base.py:73-97` | Units are optimizer updates. Off, linear ramp 0 -> full, hard off; no move between two non-zero weights (redux `final_weight` / `final_weight_iter` in PH:1027); the owner's loss-decay keys were not portable (`docs/refactor/stage3/owner_recipe.md:29`, `:132`) |
| Per-term loss logging, raw and weighted (RP:139-141) | partial | `[logger] print_freq`, `use_tb_logger` | `neosr/losses/base.py:108-121`, `neosr/utils/logger.py:137-172` | The logged value is the weighted (and ramped) one; raw magnitude must be derived |
| Per-term gradient norms (RG:59, NP:323, PG:464) | partial, offline | `python scripts/term_grads.py -opt cfg [--net_g] [--net_d] [--batches] [--iter]` | `scripts/term_grads.py:1-11`, `neosr/models/image.py:430-436`, `:58-70` | Offline: norm, share of the summed squared norms and cosine with the total, over all trainable G parameters (not a trunk layer on fixed probes as RG:59 proposes), training batches, no step. Online only total norms and the adversarial share at the output (`grad_sr_gan`, `grad_sr_ratio` on log updates) |
| Seeds and determinism (HB:1129-1134, NP:482) | present | `manual_seed`, `deterministic` | `neosr/config.py:87-88` | Template leaves `manual_seed` unset; resume restarts the saved epoch's data order |
| Exact resume vs warm start (RG:64, HB:1072) | present | `[path] resume_state`, `--auto_resume`, `pretrain_network_g`, `strict_load_g` | `neosr/models/base.py:331-393`, `neosr/models/phase.py:276-341` | none for this group |
| Run manifest (RG:229-271, FG:450-499, NP:320) | partial | none | `train.py:46-51`, `neosr/models/image.py:695-720` | Config copy with command line, a resume record, export metadata; no code revision, dataset hash or metric protocol |
| Checkpoint interpolation (RP:56, FG:252) | missing | none | no tool under `scripts/` | n/a |
| Scoring external image folders (DR:11, PG:337, HB:1074) | missing | none | `neosr/metrics/calculate.py:24-75` (numpy references exist) | `test.py` scores a model, not a folder of images |
| Calibration runner and scripted guards (PG:332-336, FG:394-396) | missing | none | n/a | No source implemented them either |
| A/B verdicts with paired bootstrap and null (HB:1129-1134) | present | `scripts/ab/analyze.py --window FROM:TO --null X/Y --report-only M` | `scripts/ab/README.md:25`, `scripts/ab/analyze.py:1-20` | D10; stricter than any community protocol in the KB |
| ONNX export with verification (HB:892, NP:327) | present | `convert.py --input net_g.pth -onnx [-fp16] [-static C H W] [-optimize]` | `convert.py:198-229`, `:271-293` | Numeric tolerance check on synthetic shapes; no metric check on probe images; 12 archs refused |
| safetensors export | missing (read only) | none | `neosr/models/checkpoint.py:69-74` | redux has it |
| Identity / face probes (RG:47, SKT:60) | missing | none | no face-embedding or landmark code under `neosr/metrics/` | Design only in the KB; would add a face-recognition dependency |
| Colour-drift measure (HB:830-835, NP:322) | missing | none | `neosr/metrics/calculate.py` (FR panel only) | RGB PSNR without Y is the only colour-sensitive number today |

## 5. Framework work this implies (strategy-independent)

All items are opt-in tooling or reporting. **None changes a default, an objective or training dynamics.** Ranked by how many
research questions they unblock per unit of work; the owner decides whether any is scheduled.

1. **Per-image rows and a metric-protocol record from plain `train.py` / `test.py`.**
   - Why: means hide bimodal failures, conventions silently mix (SKL:93, FG:403, DX:484, NP:90, HB:866). Today only harness runs have them (`scripts/ab/run_ab.py:1-10`).
   - Size S. Risk low: the values already exist as a tensor per image (`neosr/models/image.py:620-660`).
   - Verify: mean of the written rows equals the logged mean to fp64; record lists type, border, Y/RGB, weights variant.
   - Prerequisite for G5-Q1, Q2, Q5, Q7.
2. **Offline folder scorer** (SR folder against GT folder, same metric code, per-image output).
   - Why: recompute metrics from saved images as a cross-check (HB:1074), baseline resizers and existing models first (DR:11, HB:28),
     HR-vs-HR calibration (PH:1013), external references and exported models through one harness (PG:337, DR:248).
   - Size S-M. Risk low. Verify: scoring the PNGs a validation wrote reproduces the trainer's values to rounding level, since metrics run on
     the quantised images (`neosr/models/image.py:566-572`); the existing numpy-route test bounds the difference at 1e-10
     (`tests/test_s2_metrics.py:338-351`); learned metrics on another device may differ at float level.
   - Prerequisite for G5-Q2, Q3, Q6, Q10, Q11, Q12.
3. **Raw-G candidate beside the EMA** (validation pass and export, off by default).
   - Why: raw and EMA are separate candidates, EMA can lag or over-smooth (RG:63, PF:58, RP:172-174, DX:442-444, PG:341).
   - Size S-M. Risk low-medium: touches `eval_weights()` and export naming; doubles validation time when on (about 9 s per event locally).
   - Verify: EMA path bit-identical to today (parity test); raw export equals live `net_g`; mode round trip as in `tests/test_model.py:358`.
   - Not a prerequisite for a first G5-Q4 read: raw G already sits in every kept `training_states/<u>.state` (`neosr/models/base.py:316-328`),
     and the s10 arms kept them (A0 5k-20k; B0, B0s2, U 8k-20k; T_DISTS, OWN 4k-20k, every 4k), so a small extraction script plus item 2
     scores raw vs EMA offline.
4. **Config warning when a validation or `save_best` metric is also a training loss.**
   - Why: SKL:16, FG:405, PF:203; local G4. Today only documented (`docs/training_guide.md:57`).
   - Size S. Risk none (a warning). Verify: config test with `dists_loss` + `calculate_dists`.
5. **Checkpoint time-series report: metric vector per checkpoint, non-dominated set, window means, no composite score.**
   - Why: every selection chapter asks for a shortlist, not one number (FG:403, NP:322, HB:878, DX:500, PG:356-363); `save_best` takes one key (`neosr/config.py:484-488`).
   - Size M. Risk: over-design, because no threshold exists anywhere (PG:343). A report first; gates only as explicit user-set margins.
   - Verify: unit tests on synthetic tables; deltas equal `scripts/ab/analyze.py` on the same rows. Needs item 1.
   - Prerequisite for G5-Q1, Q2.
6. **Same-lineage weight interpolation tool** (alpha grid, key/shape/lineage check, exports with metadata).
   - Why: RP:56, FG:252, PF:184, RG:65, SKT:70. Size S. Risk low (offline).
   - Verify: alpha 0 and 1 reproduce the endpoints bitwise; outputs load in `test.py` and `convert.py`.
   - Prerequisite for G5-Q6.
7. **Step-zero validation switch.**
   - Why: proves the pretrain loaded and gives the parent row for every fine-tune (DX:271-275, FG:242, HB:1069, DR:205).
   - Size S. Risk low. Verify: step-zero metrics equal `test.py` on the same weights.
8. **Run manifest** (resolved config, code revision, metric protocol, data manifest hash) next to the run.
   - Why: RG:229-271, FG:450-499, NP:320, HB:905, DR:28; loss names changed meaning across upstream releases (DX:385-389).
   - Size S-M. Risk low; the revision only, never the remote URL. Verify: test on a temp run folder.
9. **Bucket tags with median and worst image in the offline analyser.**
   - Why: PF:282, SKT:60, DX:500, NP:321, RG:254. Size S (the analyser already groups by source: `scripts/ab/analyze.py:1-20`).
   - Prerequisite for bucket-level reads in G5-Q2 and Q5; needs the owner's tags (skin, hair, eyes, fabric, flats).
10. **NR metrics, opt-in, usable on `single` sets.**
    - Why: SKL:71, HB:863, DX:490, PH:1011, PG:318. Missing (`neosr/metrics/calculate.py:24-473` registers five FR metrics; `single` sets skip metrics: `neosr/models/image.py:576`).
    - Size M. Risk medium: footprint and weights, implementation drift (NP:168), known bias toward oversharpening (DX:490, SKL:70).
      Two shapes: registry entries, or offline-only through item 2 with an optional pinned dependency. The competitor having TOPIQ-NR is not a reason.
    - Verify: reference-value tests as for LPIPS (`tests/test_s3_metrics.py:136`). Prerequisite for G5-Q11.
11. **Scripted guard diagnostics, offline and report-only** (flat-region HF residual, radial spectrum vs HR, edge overshoot, forward-model residual).
    - Why: FG:394-396, RG:207, SKD:104, PF:62, PG:320-324. Size M-L. Risk medium: unvalidated measures can mislead; keep them out of any gate until G5-Q3 calibrates them.
    - Verify: synthetic defects (halo, blur, stipple, wax) move each measure in the stated direction.
12. **Blind crop-sheet generator** (fixed coordinates, N candidates, shuffled labels, key file).
    - Why: HB:878, PF:283, DR:244, PG:383-386. Size S. Risk none. Prerequisite for the human half of G5-Q2.
13. **Export parity on probe images** (torch vs ONNX fp16, full vs tiled, as metric deltas).
    - Why: SKT:27, SKP:40, PG:394, HB:1068, NP:327. Today: tolerance check on synthetic shapes (`convert.py:198-229`), tiled-vs-full test (`tests/test_model.py:207`).
    - Size S-M once item 2 exists. Prerequisite for G5-Q12.
14. **Online sparse per-term gradient norms and raw loss values.**
    - Why: NP:323, RG:59, PF:56. The offline script already answers the question (`scripts/term_grads.py:1-11`), so this is convenience. Size S-M. Risk: extra backward passes; opt-in.
15. **safetensors export.** Size S. No claim in this bundle argues for it beyond deployment convenience.
16. **Colour-drift report, offline** (per-channel mean bias, flat-patch delta colour, saturation shift against GT).
    - Why: HB:830-835 (proposal), NP:322, NP:341; in the owner's redux run PSNR fell 29.5 -> 17.7 dB in one stage when Charbonnier was cut
      to 0.01 and HSLuv dropped (pixel and colour anchors removed together, cause not separated: `docs/refactor/stage3/owner_recipe.md:20-22`);
      PSNR alone does not say whether it was colour. Size S, through item 2. Risk low (report only). Verify: synthetic colour casts move it.
17. **Identity probe, offline and report-only** (face-embedding cosine and landmark drift on face crops, minimum over images).
    - Why: RG:47, SKT:60, PG:374; the owner's data are close-ups of people. Design only, never run in any source.
    - Size M. Risk: a face-recognition weight dependency (footprint, licence); keep it outside the trainer. Needs a face-crop set (G5-Q5 data).

## 6. Research questions for the strategy phase

Cost unit: one template arm 0 -> 20k is about 37 min on the 4090 (+37 % with DISTS); one validation event on dev + val8 with
4 metrics is about 9 s. "Offline" means scoring existing exports or per-image rows, no training. PG experiments in this group:
X2, X3, X13, X16. X4 (metric-gated GAN ladder) and X14 (NR metric as a loss) belong to the GAN and loss groups but need the tools of this one.

| id | question | single-factor arms | prerequisites (data, feature) | rough GPU cost | blocked by | priority for the owner's goal |
|---|---|---|---|---|---|---|
| G5-Q1 | Within one run, do TOPIQ, LPIPS and DISTS pick the same best checkpoint, and are between-checkpoint differences larger than the null? Would a window mean or a non-dominated set be the stable definition of "best"? | none; analysis of the existing s10 per-image rows | section 5 item 5 as a script | none | nothing | high: `save_best` has no agreed meaning (G35) |
| G5-Q2 (X3) | Does a metric shortlist reproduce the owner's blind top-5 on one finished run? | one finished 20k run: every saved checkpoint (EMA exports; raw G from the kept states) scored, owner ranks blind | items 1, 2, 5, 12; owner time; full gates also need 10, 11 | minutes of scoring | owner's eye; guards and NR missing, so a reduced FR-only version first | high: nobody has shown which metric matches the eye (PG:453) |
| G5-Q3 (X2) | Which judges react in the right direction to small known defects on the owner's HR (halo, blur, stipple, wax), and what do they read on HR vs HR? | none; sensitivity table per metric on dev GT | item 2 plus a small defect generator; 60 blind pairs for the human half | minutes | NR half by item 10; human half by owner time | high: cheap, and D10 rests on these judges |
| G5-Q4 | Do raw and EMA weights differ beyond the null at 8k and 20k (EMA 0.999, power 0.75, constant LR)? | existing s10 arms: raw G from the kept states (4k or 5k steps to 20k) vs the EMA rows; null from B0/B0s2 raw pairs | item 2 plus a state-to-weights extraction; item 3 only for denser points | minutes of scoring (offline); 37 min only for a new arm with item 3 | nothing for the offline read | medium: the community report is from a different regime (cosine, from scratch: RP:172) |
| G5-Q5 | dev (219 tiles) vs val8 (8 photos): which ranking follows the owner's eye, and is the disagreement real? | none; needs a larger full-image held-out set with a null | a new source-disjoint photo set with content tags; item 9 | minutes of scoring | data; owner time; the confirmation set must stay sealed | high: local open question, decides what every later verdict means |
| G5-Q6 | Is interpolation between the fidelity pretrain and a GAN checkpoint a usable dial: monotone on PSNR vs perceptual judges, any alpha dominating an endpoint? | alpha in {0, 0.25, 0.5, 0.75, 1} between two existing exports | items 2, 6 | minutes | missing tool | medium: cheap route to a conservative and a detail model from one lineage (PF:181, DX:295) |
| G5-Q7 | How do null RMS and tau scale with validation-set size (30, 60, 100, 219 tiles) and cadence (2000, 4000, 6000)? | none; subsample the existing per-image rows (every 2000 updates; A0: 3080 rows = 10 events x 308 images) | nothing for size and coarser cadence; cadences finer than 2000 need new runs (exports exist only every 5000, e.g. A0 `net_g_5000..20000`) | minutes (CPU) | nothing | medium: tests the community sizes (8-10, 30-100) against our effect sizes |
| G5-Q8 (X13) | Does a term's gradient share predict which weight changes move the judges, so that weights from other trainers can be matched by share instead of by value? | template; one arm with a single term's weight changed to a target share measured by `scripts/term_grads.py` | none for the local half | 1-2 arms, 37-75 min | cross-trainer half needs a redux run of the same stack | medium-high: weights do not transfer (HB:797); every other group needs this translation |
| G5-Q9 | Do 4k-8k and 12k-20k verdicts hold at 100k, across the `wavelet_init = 80000` switch? | reference and one candidate, 20k -> 100k, paired | a candidate worth the time | about 2.5-3 h per arm (80k updates at 95-110 ms plus validation; the D input changes at 80k), 5-6 h per pair, 10-12 h with a second seed | tactic selection on hold | medium: protocol validity for all groups (DX:565 vs SKM:39) |
| G5-Q10 (X16) | How far is the detail model from a diffusion SR reference on the same harness, per content class? | detail model, conservative model, diffusion reference (inference only) | item 2; an external diffusion SR model and outputs with fixed seeds | no training; diffusion inference unknown | owner's choice of reference; clean x4 LQ is not the regime such models target | low now: defines the target, needs calibrated judges first |
| G5-Q11 | Which NR metrics rank HR above resized LR on the owner's data, and where is the HR band? | none; tri-score HR / resized LR / SR | item 10 or an offline pinned dependency | minutes | missing feature; no real unpaired LQ set | medium: required before any NR axis or X14 |
| G5-Q12 | Do the ONNX fp16 export and tiled inference move the judges beyond the null? | full fp32 vs ONNX fp16; full vs tiled at one tile size | items 2, 13; images larger than a tile (val8 is 512 px) | minutes | nothing for fp16; seams need larger images | low-medium: cheap, but after the strategy |
| G5-Q13 | Is the Goodhart effect confined to the trained metric: do independent judges (LPIPS vgg, TOPIQ kadid) agree with the D10 judges on DISTS-trained arms? | none; re-score existing T_DISTS, NoGAN_DISTS, OWN and reference exports | item 2 | minutes | nothing | low: the rule already exists (3.2); this only widens the judge panel |

## 7. Questions for the owner

1. **Final judge.** Metrics alone, or a metric shortlist with one blind sign-off by eye? Is one blind top-5 session (G5-Q2) acceptable?
2. **Meaning of "best".** Should `val.save_best` stay off with selection done offline, or should one metric (or a window mean) define it?
3. **Past practice.** How was the final redux model chosen: by eye or by val8 DISTS? (`docs/refactor/stage3/owner_recipe.md:144`)
4. **Deliverable.** One model, or a conservative and a detail model from one lineage? It decides whether selection keeps two ends of a front.
5. **Target inputs.** Is there a real deployment LQ set (and which inputs is the model for)? Without it the unpaired/NR half of the community protocol has no data.
6. **NR metrics.** Wanted at all, given their bias? If yes: inside the trainer, or offline only with an optional pinned dependency (footprint)?
7. **Raw + EMA candidates.** Worth the doubled validation time and export count, or is an offline read from the kept states (G5-Q4) enough first?
8. **PSNR convention.** Keep RGB with a 4 px border as the reported default, or also report Y with border = scale? (Defaults change only by the owner.)
9. **Held-out data.** May a larger tagged photo set be built for dev-vs-val8 (G5-Q5)? Does the confirmation set stay sealed until a default is proposed?
10. **Promotion rule.** Keep D10 (two seeds + a 20k -> 100k run) against the community's "2 or 3 seeds"?
11. **Faces.** PF's objective table ([H]-level guide text, not a Discord report) places faces under "little or no GAN" (PF:27) while PG places the goal between natural upscaling and presentation realism (PG:69, PG's own reading). Which tolerance applies to skin, hair and eyes?
12. **Scheduling.** Which items of section 5, if any, go before Studio work, and should Studio expose them?

## Verification

Adversarial pass, 2026-11-13, read-only on code, KB and run files (no GPU, no runs). Only this file was edited.

**Checked.** 157 KB anchor spans opened (all release-config, official and part-6-driving anchors, plus the
DX/HB/PH/PF/FG/RG/SKL/SKG/SKT/PG/NP/RP/SX anchors of parts 1-2); about 70 neosr-dev claims (code `path:line`, template keys and
defaults, docs, run files: `metrics.jsonl` and `train.jsonl` of all 16 s10 arms, kept states and exports, `adv/null*.py`).
Bundle counts (300 claims, evidence mix, 11 conflicts, 24 gaps, 121 features, 25 capabilities, 12 findings) reproduce exactly.
Confirmed unchanged: C01/C02/LPIPS status in 3.4, metric defaults (`crop_border` 4, RGB), `save_best` tie rule (`base.py:237`) and
absence from every `options/*.toml`, template val/save 1000/1000 and commented metrics, `NO_ONNX` = 12 archs, D10 numbers, G4 numbers.

**Corrections (before -> after, source).**

| # | before | after | source |
|---|---|---|---|
| 1 | in-loop set "8-10 images" | "at most 8-10" (an upper bound), representative / per-ability at UH3:117, RX:111 | UH3:115-117, RX:109-111 |
| 2 | stop "when scores slip or stagnate" as a Discord report via DR:246-247 | official redux text, Compact-era, no metric; moved to 1.1 | RX:273 (DR:247 cites it) |
| 3 | "3-5 candidates, blind" as a Discord report | HB's [Proposal]; only "trainers prefer visual checks on unseen images" is Discord | HB:868-878, HB:1253 |
| 4 | 1x/2x snapshots "omit batch, D, degradation, crop" | 1x: no batch only (D rvgg, lq_size 96, degradation present); 2x: no D, degradation, crop | PH:553-660, PH:765-780 |
| 5 | "a perceptual metric improved while images showed artifacts" | "a low perceptual weight improved some metrics" [C], one participant | PF:209, PF:54 |
| 6 | source split cited to DX:478 | DX:476 (DX:478 is about not tuning on the held-out set); RG:42 / PF:52 labels added | DX:476-478, PF:52 |
| 7 | tile overlap >= RF/2 cited to SKT:27, SKP:40, PG:394 | SKP:48, SKT:42 (generic), PG:415 (H) | SKP:48, SKT:42, PG:415 |
| 8 | "nothing in PF was run (PF:19)" | RG/FG say so (RG:8, FG:15); PF only labels [P]/[C]/[H]/[E] (PF:13-16) | PF:13-19 |
| 9 | null "from three same-config pairs" | five runs of two configs; gate uses 3 pairs, RMS table and 9/36 use 4 (36 = 4 x 9) | `NOTES.md:40`, `adv/null.py:19`, `adv/null2.py:16-29` |
| 10 | fp32-validation reason cited to `STATUS.md:156-169`, `next_tasks.md:29` | `stage2/CLOSURE.md:312` (GP-13), `image.py:589` | same |
| 11 | 4K "learned metrics do not fit a 12 GB card" | 24 GB fits; on 12 GB DISTS would fall back to CPU (inferred), TOPIQ marginal | `STATUS.md:164-168` |
| 12 | +37 % and spanplus ms cited to PROGRESS / DECISIONS lines that lack them | `consult/brief.md:15-20`, `stage2/PROGRESS.md:86` | same |
| 13 | skips 0 cited to `A0/done.json` | `train.jsonl` of all 16 arms (`skipped_g/_d` 0); `done.json` has no skip field | run files |
| 14 | dev ranks OWN "worst" on LPIPS | 13th of 16 (0.1517; M, B2s2, A1_nogan worse); DISTS lead of the 3 `dists_loss` arms added | `s10/runs/*/metrics.jsonl` @20k |
| 15 | val8 LQ "the same synthetic downscale" stated as fact | flagged as an undocumented assumption | `r3gan/RESULTS.md:16` |
| 16 | metric registry at `neosr/metrics/__init__.py:17-42` | that is `build_metric`; registrations at `calculate.py:24/113/429/449/473` (3 places) | code |
| 17 | loss schedules "present" | "partial": no move between two non-zero weights (redux `final_weight`) | `losses/base.py:79-97`, `owner_recipe.md:29,132` |
| 18 | per-term norms row | term_grads uses all G params (not a trunk layer); online `grad_sr_ratio` anchored | `scripts/term_grads.py:1-11`, `image.py:58-70` |
| 19 | G5-Q4 needs item 3 and a 37-min arm | offline from kept states (raw G in `training_states`), minutes | `base.py:316-328`, s10 state listings |
| 20 | G5-Q7 "exports if still on disk (not verified)", cadence 500/1000 | verified: rows every 2000, exports every 5000; finer cadence needs new runs | run files |
| 21 | item 2 "reproduces exactly" | to rounding level (test bound 1e-10) | `tests/test_s2_metrics.py:338-351` |
| 22 | G5-Q9 "about 3 h per arm" | 2.5-3 h (80k x 95-110 ms + validation) | 3.5 costs |
| 23 | noise-in-eval anchor `owner_recipe.md:45-46` only | adds `:140` (the caveat line) | doc |
| 24 | "the community places faces under little or no GAN (PF:27)" | PF's own objective table, guide text | PF:27 |

**Added.** Detail labels preserved/reconstructed/synthesized (DX:48-56); identity/anatomy probes (RG:47, SKT:60, PG:374-375) and
colour drift (HB:830-835, NP:322) in 1.5, section 4 rows and section 5 items 16-17; relaxed pretrain-load report already present
(`neosr/models/checkpoint.py:140-179`); local term_grads shares (3.6); tiled-path exactness condition (`tiling.py:30-36`); an
OTF-parked record of what its validation would need (3.8, no work planned); G5-Q13 extended to NoGAN_DISTS and OWN.

**Rule check.** No section proposes a default change, objective or dynamics change, or OTF work; redux weights stay labelled redux
scale; local evidence is not claimed for degradation, NR metrics or runs past 20k. No averaged conflict found.

**Unverified.** Upstream NeoSR C01/C02 (NP read web views; not re-checked against upstream). How val8 LQ was produced. The 12 GB
CPU-fallback for 4K DISTS (inferred in STATUS). That the PH release snapshots are redux configs (taken from the bundle's scale
tag). The B95 note that RP:234-238's refs are release cards. Design anchors opened only by spot check (FG:394-396, SKD:104,
SKT:47, PF:284, FG:450-499, RG:229-271). Part 6 cost figures are estimates; no part-5 size is measured. The DX/HB/NP Discord
registers cannot be resolved to raw logs (the KB holds none).
