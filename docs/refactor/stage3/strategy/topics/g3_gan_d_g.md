# GAN, discriminator and generator

Study section for topic group g3 (bundle `experiments/study/bundles/g3_gan_d_g.json`: 99 claims, 12 conflicts,
11 gaps, 15 capability rows, 14 local findings). Nothing here is a decision: the owner chooses.

Conventions used below:

- KB anchors are `ALIAS:LINE` (aliases in `/home/korvin/apps/Trainings/tools/aliases.tsv`). Evidence labels are the
  bundle's: **official** (trainer docs, wiki logs, model cards), **release_config** (a published model's config),
  **discord_cited** (a message-cited community report), **paper**, **design** (a KB synthesis choice, PG/DR),
  **hypothesis**, **generic** (LLM boilerplate or the BasicSR-centric skill; numbers are not empirical).
- Weight scale is named per number: **redux** (traiNNer-redux `ganloss`), **neosr** (`gan_loss`), **basicsr**.
  Loss weights do not transfer across trainers; no number below is a value for neosr-dev.
- Local cost unit: **1 arm** = one seed, 0 -> 20k from `P0_pretrain` at the template step (~95-110 ms/update,
  ~37 min); `unet` D x1.18 in `bench.py` (RESEARCH_REVIEW.md:17-19) but x1.30-1.34 in the s10 20k arm logs
  (8k-20k medians 127.4 vs 95.1-97.8 ms/update, U_unet01/U_unet01s2 vs B0/B0s2/A0 `train.jsonl`), +DISTS x1.37;
  **long arm** = 20k -> 100k (80k updates, ~2.2-2.5 h, eager).

## 1. What the community converges on

How thin the evidence is, stated first: the corpus holds **no controlled ablation on photo data** for any item in
this group (D architecture, GAN weight, ramp, label smoothing, wavelet-guided D, generator-side levers). The
hardest facts are three release configs (two 1x, one 2x, all redux scale), two official wiki logs, one trainer
guideline and one loss doc. Nothing at x4 prints a GAN block except a posted redux default loss block
(PH:1032-1054, not scale-specific). The two 1x release configs train on OTF high-order degradations
(`high_order_degradation: True`, `lq_size` 96, PH:633-635, PH:738-740) for 250k iterations (PH:559, PH:664); the
2x config runs 500k (PH:769), and the redux guideline's GAN stage is the "with degradations" stage (TG:26-28).
Their weights were set for degraded pairs and long runs, neither of which the local set has. No source gives a
numeric GAN health indicator (DR:172), and RG, DX, PF, FG, NP and the HB troubleshooting chapters give no GAN
weight, D LR, ramp length or D:G ratio of their own (RG:148, NP:310, NP:312, HB:819-824).

### 1.1 Official docs and release configs (best evidence)

- **GAN stage = all losses on at once, from a fidelity pretrain, G and D both at lr 1e-4.** redux guideline:
  MS-SSIM-L1 + perceptual + HSLuv + GAN enabled together, lr 1e-4 for both optimizers, larger `lq_size` in this
  stage (TG:29, TG:30, TG:31; official). D optimizer AdamW 1e-4, betas [0.9, 0.99], its own block (DR:168,
  HB:619; official as carried). No trainer doc documents an in-run GAN ramp (DR:166; NL:139-147 has no such key).
- **GAN weight in release stacks, redux scale:** 0.15 and 0.14 in the two 1x GaterV3 configs, with D lr = G lr =
  1e-4 and `grad_clip: False` (PH:608, PH:713, PH:676, PH:685, PH:562, PH:667; release_config); 0.05 in the 2x
  RealPLKSR config, with soft labels 0.9/0.1, G 5e-5 over D 1e-5, weight decay 0.01 on both and `grad_clip: True`
  (PH:822, PH:781, PH:790, PH:772; release_config); 0.1 in a redux default loss block Helaman posted (PH:1054; the
  bundle labels it release_config, but it is a posted template, not a release). The two official Compact logs print no
  weight; `l_g_gan` 0.075 (redux, RX:239) and 0.069 (NeoSR, UH3:212) divided by BCE ~0.7 gives ~0.1, an inference
  by DR (DR:509; design on official logs).
- **NeoSR doc example:** `gan_type = "bce"`, `loss_weight = 0.3`, labels 1.0/0.0; types `bce`, `mse`, `huber`
  (NL:139-147; official, neosr scale). It is a doc example, not a recommendation; DR does not adopt it (DR:492).
- **GAN type: vanilla/BCE in every printed release config** (PH:608, PH:713, PH:822, PH:1053; release_config).
  Hard labels 1/0 are the documented default (NL:143, NL:144; official); D:G 1:1 is DR's choice, no doc states a
  ratio (DR:169; design; 1G:2D "temporarily" is generic, PH:497).
- **Discriminator in templates and releases:** U-Net with spectral norm is the redux template D
  (`UNetDiscriminatorSN`, RX:225; official log), the D of the NeoSR Compact wiki log (`unet`, UH3:198; official
  log) and the Real-ESRGAN default (SKL:23; generic text of a paper fact). The RealPLKSR release names a DUnet
  in prose only (PH:443); the 1x GaterV3 restore release uses `rvgg` (PH:555; release_config).
- **Released D may be loaded:** the Real-ESRGAN finetune recipe downloads and loads `RealESRGAN_x4plus_netD.pth`
  (UH4:184; official).
- **Wavelet-guided D (WGSR) in NeoSR:** `wavelet_guided = true`, `wavelet_init = 80000` in the doc example;
  "works better for finetuning", train at least ~40k before enabling (NL:6, NL:12, NL:16; official). Its stated
  purpose is to stabilise GAN training and reduce artifacts (NL:6), not to add detail. The WGSR method itself is a
  D on HF wavelet subbands **plus** a subband reconstruction objective; its results concern that method, "not any
  checkbox named `wavelet`" (PF:223; paper). No photo result exists anywhere in the corpus (gap, PG:457).
- **Siax card:** Contextual loss + PatchGAN, and "added multiscale patchgan to increase the detail generation"
  (SI:17, SI:23; official model card, no ablation, no weights).
- **GAN may be off:** Jaywreck3 (1x JPEG cleanup, ESRGAN Lite, 320k iterations, batch 4, HR 112) shipped with the
  GAN disabled (DX:377, DX:379, RP:119; release_config). It says nothing about generative x4 detail.
- **Generators in releases:** RealPLKSR (dysample variant in x4 releases) and RRDB nf 64 / nb 23, ~16.7 M params
  as the reference GAN generator (DR:39, DR:40, SKA:27, SI:5); DAT2/ATD/DRCT-L finetunes ran 140k-200k iterations
  from official x4 pretrains, loss stack not stated (PH:1199-1201; release facts).

### 1.2 Message-cited community reports (single reports, mostly one person each)

- **Start the GAN weight low and raise it late:** 0.1 (or 0.2), then 0.3 once validation outputs stop drifting;
  0.5 used once in an OTF config, "more unstable", pretrain important (PH:1004, PH:1021; discord_cited, Helaman;
  the messages span the neosr and redux channels, so the scale is not stated per message). "1.0 is super high"; redux-style
  defaults perceptual 0.01, GAN 0.1 (PH:1005, PH:1024). One appendix report of 0.5 under bf16 + amp (RP:355).
- **No uniformly best D; multiscale gave small or negligible gain for its cost** (HB:817, NP:393; discord_cited:
  musl 2024-05-17, Zarxrax 2024-05-19, two anecdotal tests).
- **PatchGAN ~ U-Net and faster; a2fpn slowest and worst** in a Compact/anime test; in a seeded DAT2 manga run
  AdamW + U-Net was stable, AdamW + a2fpn hit a D NaN near 7k, Adan + a2fpn kept patterns at 50k (DX:345;
  discord_cited `reported_test`, anime/manga, not photo). Action drawn from it: benchmark the simple stable D
  first and retune the D LR whenever D changes (DX:347).
- **PatchGAN/VGG-style D gives sharper micro detail, with an aliasing risk when the LR is too clean** (RP:151,
  RP:41; nmkd 2020); **UNetDiscriminatorSN gives better control on photos**, used with AdamW + BF16 (RP:153,
  RP:253; Helaman). Usage reports; the "why" column is the compiler's. RealWebPhoto v3 also ran OTF JPEG ~Q40 +
  recompress (RP:253): a degraded-data context.
- **If D dominates:** confirm GT/LQ colours and normalisation first, then lower D LR or the GAN weight, one at a
  time; check D/crop compatibility; start from a better G; abandon the GAN if it stays unstable (HB:819-824; no
  numbers; one reader labels it discord_cited, another hypothesis). The only number for the first D-LR step is
  0.5x (SKR:33; generic, carried by DR:170 as design). The 2x release pair (G 5e-5 / D 1e-5, soft labels,
  clipping) is what PG and DR use as the fallback (PH:441, PH:442; using it as a fallback is design).
- **NaNs after lowering the batch:** suspect a too-high D LR rather than the batch; try D LR 1e-4 (PH:1087;
  discord_cited, Helaman 2024-07-30, one message).
- **Debug false detail on a low-GAN or pixel-only branch from the same full state** before adding losses; for
  grain or halos fork from an earlier full checkpoint and lower GAN weight or D LR one at a time (PF:57, PF:182;
  discord_cited, Helaman 2024-02-20; the PF:229 symptom row is uncited). Whether a pretrained GAN carries its
  original target's sharpening is posed as an open question (PF:374; [C/H]); FG:241 (I/H) says existing GAN
  checkpoints "may carry" inherited smoothing or texture artifacts.
- **Fresh D when only G was pretrained** (DR:51, HB:484, PH:526; discord_cited/design). HB:484 itself keeps a D
  "if you have an exactly relevant D checkpoint", which is the Real-ESRGAN finetune case (UH4:184, G and D from
  the same released run), so the "conflict" with UH4:184 is mostly a difference of condition.
- **D-update-every-N as a collapse mitigation:** one mention, N unspecified, no result (NP:312; discord_cited,
  umzi 2025-02-08).
- **Texture comes from the training distribution and the objective, not the backbone;** size is no protection
  from hallucination (DX:23, FG:280, FG:281; discord_cited). Small generators ring quickly under edge-pushing
  losses (PH:397; DX:337 is anime/Compact).
- **RealPLKSR knobs:** dropout 0.3-0.5 only after ~60k iterations, lr 1e-4 with warmup from a pretrain, larger GT
  crop later (PH:1090; one message).

### 1.3 Designs, hypotheses and generic text (not evidence)

- PG's chosen start per scale, redux: 0.14 at 1x, 0.05 + soft labels at 2x, 0.1 at 4x; band 0.05-0.15; ceiling
  0.3 only through a metric-gated ladder of new runs (PG:251, PG:253, DR:165; design built on the release
  configs above). DR's NeoSR choice of 0.1 is a design, too (DR:154).
- Ramps: 0 -> target over ~5k (SKR:14; hypothesis); 0-0.02 for 10k, then to 0.05-0.2 by 50k (PH:508, PH:509;
  generic); "low weight ramped from zero", no number (NP:310; hypothesis); GAN from ~10k with Charbonnier 0.8 ->
  0.3 (RG:100; discord_cited but confounded, trainer not stated); loss schedulers `start_iter` / `final_weight` /
  `final_weight_iter` mentioned in one message (PH:1027). RG:100 and PH:1027 cite the **same** message (Helaman,
  2025-11-27 17:45): one report, not two. Local fact on the redux side: `start_iter` exists in redux, and in the
  owner's checkout a loss with `start_iter` or decay keys is wrapped so that its weight acts **squared** (GAN 0.07
  -> 0.0049; `docs/refactor/stage3/owner_recipe.md:56`, `:59`; that checkout only, other versions unverified). Any
  redux weight printed next to `start_iter` (e.g. SL:36-37, 0.08 with `start_iter: 10000`) may therefore be
  nominal only.
- TTUR with D faster than G (PH:477; generic) is contradicted by the release config (PH:479).
- D loss or a 50 % rate is not a health target; log D(real)/D(fake) per `gan_type` (PF:240, SKR:31, DR:172;
  hypothesis). Warning signals and symptom tables: HB:826, HB:1060, PF:227-238, RG:171 (all hypothesis).
- Generic: GAN 0.05-0.1 against L1 1.0 + perceptual 1.0 (SKL:26; basicsr scale); RaGAN "converges a bit more
  gracefully" (SKL:20); "D design is the real lever" (SKL:21-25); spectral norm "mandatory" with a U-Net D
  (SKR:36); add a coarser D only when texture is right at 100 % zoom but wrong at arm's length (SKT:16).
- Not used by any synthesis: 0.003-0.005 (SX:148, S1:18; speculative Siax reconstructions) and 0.001-0.01
  (AN:139; generic), both uncited and on another scale (DR:539, PG:251).
- Untested ideas: StyleGAN-style noise inputs, SFT/FiLM conditioning maps (SKD:79, SKD:83, PG:475-477); R3GAN
  head, direction only, with an unattributed 0.08 / r1 = r2 = 10 block (PH:453-455, SL:33-39); multiscale
  PatchGAN with patches ~70/35/17 (SX:134-138); 1x needs an explicit 1x head (FG:283).

## 2. Open conflicts

| question | positions (anchors, evidence) | what would settle it | can our setup settle it now? |
|---|---|---|---|
| GAN weight for a detail model | 0.14-0.15 at 1x, 0.05 + soft labels at 2x, 0.1 template (PH:713, PH:608, PH:822, PH:1054; release_config, redux) / 0.1-0.2 -> 0.3, 0.5 unstable (PH:1004, PH:1021; discord_cited) / 0.3 doc example (NL:142; official, neosr) / 0.05-0.1 (SKL:26; generic, basicsr) / 0.003-0.005, 0.001-0.01 (SX:148, AN:139; hypothesis, generic) | per trainer: a weight sweep read against the adversarial gradient share, not the nominal weight (X13, X4) | Partly: on the neosr scale, clean x4, template D. 0.03 is already a LOSS, 0.1 is "mixed" at 8k/one seed (section 3). The redux numbers themselves cannot be checked here |
| In-run GAN ramp | none: all losses at once (TG:29, DR:166; official) / ramp ~5k, or 0-0.02 for 10k then target by 50k, or "from zero" (SKR:14, PH:508-509, NP:310; hypothesis, generic) / raise by new runs from gated checkpoints (PG:253; design) | fixed vs staged at the same final weights and enough exposure (RG:170) | Short ramps yes (config only, `ramp_iters`). The 10k + 40k schedule with weight 0 for the first 10k is also config only (`start_iter = 10000`, `ramp_iters = 40000`; D then idles too) but needs a run past 50k; only its 0.02-plateau variant needs a two-level schedule (missing) |
| D LR relative to G LR | equal (TG:30, PH:676, PH:685; official, release_config) / D 5x lower (PH:781, PH:790; release_config) / D higher, TTUR (PH:477; generic) / equal, lower only when D dominates (DR:531, PG:434; design) | one-factor D-LR arms with D health logged | Yes, config only; caveat: the template D uses `adan_sf` (schedule-free), not AdamW, so the ratio is not like for like |
| D architecture for micro-texture | U-Net SN default (SKL:23, RX:225, PH:443) / PatchGAN similar, faster, sharper micro detail (DX:345, RP:151; anime, 2020) / multiscale PatchGAN for detail (SI:23) vs small gain (HB:817, NP:393) / wavelet-guided (NL:6) / rvgg in the 1x release (PH:555) | identical G, budget, retuned D LR, compared at similar sharpness (RG:168, DX:347) = X6 | Partly: `unet`, `patchgan`, `dunet`, `metagan`, `r3gan`, `ea2fpn` exist, and `resgan` / `r3gan_scalar*` give one logit per image (global Ds, not VGG-style); multiscale and rvgg do not. One confounded unet arm exists (section 3) |
| Fresh D or a released/loaded D | fresh when only G was pretrained (DR:51, SKB:149, SKR:12; design) / the Real-ESRGAN finetune loads the released netD (UH4:184; official). Weak conflict: HB:484 keeps "an exactly relevant D checkpoint", which UH4:184 is | branch from one G checkpoint with and without the D | Yes (`pretrain_network_d`); only with our own D checkpoints, no released D matches our archs |
| Soft labels | hard 1/0 as the documented default (NL:143-144, DR:169) / 0.9-0.1 in the only release that prints a stabiliser, together with the lowest weight and lowest D LR (PH:822, PH:825-829) | a one-factor label arm | Yes, config only (bce); needs the owner's ruling on whether static smoothing counts as damping |
| First remedy when D dominates or collapses | lower D LR or GAN weight, one at a time; return to the best G (HB:819-824, HB:1060) / D update every N (NP:312) / fresh D, lower GAN, lower D LR as separate branches (RG:171) | forks from one failing full state, one knob each | Not now: the template D sits at chance, it does not dominate (section 3); cadence keys are missing |
| Over-smooth vs artifact-ridden output: what first | raise GAN (0.05 -> 0.1, SKT:35; 0.1 -> 0.3, PH:1004) / reduce adversarial pressure or fix data and degradation first; prefer LDL or HF-subband pressure over more global GAN (RG:171, FG:352, PF:229, SKR:37) | paired branches from one state with defect gates (G1, G2 of PG:347-348) | Partly: weight branches yes; the defect gates are not built (validation group) |
| Where re-drawn detail comes from | the degradation distribution (PH:741, PH:726; design) / loss-side levers: CX + PatchGAN, DISTS, LDL, FF, higher GAN (RP:41, PH:714, SKT:68) | X1 and X5 on degraded data | No: local data is a clean downscale and OTF realism is parked |
| GAN in 1x work | off for JPEG cleanup (DX:377) / 0.14-0.15 for 1x detail and restore (PH:713, PH:608) | a GAN-off control per 1x task | No: no 1x data, no degradations |
| GAN type | vanilla in all releases (PH:822, PH:713) / RaGAN smoother (SKL:20; generic) / R3GAN direction only (PH:453, SL:36) | one-factor `gan_type` arms on one D | Partly: `bce`, `mse`, `huber`, `hinge`, `rpgan` exist; RaGAN does not |
| Gradient clipping in the GAN stage | off in the 1x releases (PH:562, PH:667) / on in the 2x release (PH:772) / "sometimes negligible" (RP:367) | belongs to the dynamics group (g4); listed because it travels with the D settings | Yes (`train.grad_clip`), see g4 |

## 3. What our own evidence says

All local results: clean antialiased x4 (GT 256 / LQ 64, photo tiles of people), one generator (`realplksr`,
7.39 M), fine-tunes from one 20k L1 pretrain, at most 20k updates, mostly one seed, batch 8. No run crossed
`wavelet_init = 80000`. Verdicts marked *(study)* were computed during this study and are **not** owner-accepted.

| id | question | result | agrees with the community? | limits |
|---|---|---|---|---|
| G5 | Does the template D (metagan, bce 0.3) help vs no GAN? | no-GAN is a **LOSS** on all three judges at 8k, 20k and in the 12k-20k window (TOPIQ -0.0093, LPIPS +0.0123, DISTS +0.0038; PSNR +0.46); no-D is 14-18 % faster (84.3 vs 97.6 ms in R4; 80.0 vs 95.1-97.8 in the s10 8k-20k logs) (`docs/refactor/stage2/DECISIONS.md:87-92`, `experiments/s10/NOTES.md:40-42`, `experiments/s10/consult/brief.md:34`; window numbers recomputed by the study and again by the verifier) | Agrees that the GAN adds perceptual quality and costs PSNR (HB:783, SKR:15). HB:824 abandons the GAN only under persistent instability; the template GAN is stable here, so this is consistent with HB, not against it | one seed for the no-GAN arm; objective already has FDL + MS-SSIM + LDL |
| G6 | In what regime does the template GAN run? | `out_d_acc` ~0.50, `grad_sr_ratio` ~0.02, GAN share of the G gradient 16.5 % at P0, 16.6 % at B0's 8k weights (`scripts/term_grads.py`); on an **untrained** G (GP-06, one seed, not the P0 fine-tune regime) the D saturates by ~250-350 updates (logits +-8.7), then sits at chance from ~1.7k (`experiments/s9_ab/NOTES.md:49-50`, `docs/refactor/stage3/PROGRESS.md:428-430`, `docs/refactor/stage2/analysis/gpu_profile.md:347-352`) | Consistent with "50 % is not a health target" (PF:240, DR:172): a near-chance D still helps (G5). It is the first logged data for the gap DR:172 | diagnostic, not an A/B. The two "shares" measure different things: `grad_sr_ratio` is a norm ratio on the SR pixels, the term_grads share is a **squared**-norm share over G's parameters (`scripts/term_grads.py:7-8`, `:58-66`; 16.5 % = a norm ~0.44x the root-sum-square of the other terms' norms); which one to match across Ds is itself open (G3-Q12) |
| G7 | Lower GAN weight on the template D (0.03 vs 0.3, neosr scale)? | **LOSS** in the 12k-20k window (TOPIQ -0.0027, LPIPS +0.0051); its 8k LOSS was noise-level; stage 2: 0.1 vs 0.3 "mixed", 0.3 kept (`docs/refactor/stage2/DECISIONS.md:47-48`, `:88-89`, `experiments/s10/NOTES.md:41`, `experiments/s10/consult/synthesis.md:26`) | Neither confirms nor refutes the redux band 0.05-0.15 (other scale). On the neosr scale it sides with the doc example 0.3 (NL:142) over a much lower weight | one seed; the weight change also moved D clip occupancy to 99 %; 0.1 and 0.2 have no D10 verdict |
| G8 | SN U-Net D (`unet` @0.1) instead of metagan @0.3? | 8k and 20k checkpoints (D9 gate): seed 7 TIE, seed 2 WIN; D10 window vs pooled, both seeds: **LOSS (mixed)**: LPIPS -0.0046/-0.0048 better, TOPIQ -0.0026/-0.0018 worse beyond tau (recomputed by the verifier); +18 % ms/update and +848 MiB in `bench.py`, +30-34 % in the 20k arm logs (`docs/refactor/stage2/DECISIONS.md:89-101`, `docs/refactor/DECISIONS.md:235-236`, `experiments/s10/consult/synthesis.md:24-25`) | Does not confirm "U-Net SN is the default choice" (SKL:23, PG:183) as a clear gain here; agrees with "no uniformly best D" (HB:817) | D type and weight changed together; the verdict depends on which judge counts |
| G9 | Is the unet D's large gradient norm a defect? | `grad_norm_d` 22-64, clipped in 100 % of updates; decision D3: it is its parametrisation, `clipped_d = 1.0` is no health signal (`docs/refactor/DECISIONS.md:192`, `experiments/s9_ab/NOTES.md:201-209`) | Relevant to "gradient clipping as a D stabiliser" (PH:442): under `adan_sf` a fixed clip rescales, it does not damp | short probe; spike counts differ 6x between seeds |
| G10 | Does spectral norm act in the template D? | No: GroupNorm cancels it in metagan; input gradient to G at init ~300x the SN U-Net's (5.4e-2 vs 1.8e-4); unet is not cheaper on the GPU (129.5 vs 109.9 ms) (`docs/refactor/stage3/discriminator/RESEARCH_REVIEW.md:103-104`, `:117-126`, `:17-19`) | The community's "SN keeps the D stable / is mandatory" (SKL:23, SKR:36) describes a D the template does not have | one batch at init; the opt-in fix (DR-08) is untested |
| G11 | R3GAN recipe (r3gan D, rpgan, R1/R2 gamma 1, 0.3, D lr 2e-4) vs template? | **LOSS** on DISTS alone (window: DISTS +0.0019 beyond tau; LPIPS +0.0028 at tau, not beyond; TOPIQ inside; PSNR -0.69 reported, not gated). Unlike the template, its D wins clearly (`out_d_acc` 0.87-0.90 over 2k-8k) at a similar `grad_sr_ratio` (~0.03) (`docs/refactor/stage3/r3gan/RESULTS.md:112-136`, `experiments/s10/NOTES.md:41`; window recomputed by the verifier) | Consistent with PG leaving R3GAN unused (PG:190); the KB has no result either. The one local "D that clearly wins" did not beat the near-chance template D | multi-change recipe, one seed, far from the owner's regime |
| G13 | The owner's redux recipe ported (OWN) *(study)* | **LOSS** on dev vs the pooled template (TOPIQ -0.0093, LPIPS +0.0062, PSNR -0.28), TIE vs no-GAN, yet best of all arms on val8 LPIPS/DISTS (8 photos, never decisive; DISTS is trained on) (`experiments/s10/cfg/OWN.toml`, `experiments/s10/runs/OWN/per_image.jsonl`, `docs/refactor/stage3/owner_recipe.md:96-130`) | n/a; shows a weak slow regularised D does not win on dev here | not a screen under D10 rule 4 (`docs/refactor/DECISIONS.md:224`); different arch, loss and length than the real recipe |
| G15 | Is the GAN still needed with DISTS (2x2)? | Yes: NoGAN_DISTS vs T_DISTS LPIPS +0.0085, TOPIQ -0.0072 (`experiments/s10/consult/synthesis.md:25`, `:31-32`); vs the pooled template *(study)*: LOSS on TOPIQ | First local data for the GAN axis of PG's X5 (PG:456): DISTS does not replace the D on this data | one seed |
| G19 | HF view from the GAN start (`wavelet_init = 0`)? | **LOSS** at 8k (LPIPS +0.0032, DISTS +0.0032); the switch transient is `grad_norm_d` 7 -> 47 (`docs/refactor/stage2/DECISIONS.md:91-92`, `docs/refactor/stage2/CLOSURE.md:87`) | Same direction as "enable after >= ~40k, better for finetuning" (NL:16); it does not test `wavelet_guided` itself | single 8k checkpoint, low confidence; no run reached 80k |
| G20 | Delayed start + ramp (1000 + 1000)? | TIE, "settles nothing" (`docs/refactor/stage2/DECISIONS.md:47-48`, `docs/training_guide.md:27-29`) | Leaves the ramp conflict open | arms differed in more than the start; 8k |
| G33 | R1/R2 on a map D | mean-logit R1 is 0.03-0.05x the per-position R1 for the r3gan patch D, ~1/P for metagan; neosr-dev penalises the per-position mean (`docs/refactor/DECISIONS.md:54-73`) | Community gammas (SL:38-39: 10/10) are not comparable across implementations | design + CPU measurement |
| G36, G38 | What the owner's successful redux run trained | effective GAN weight 0.0016-0.01 (nominal 0.04-0.1, applied squared), D lr 5e-6 (40x below G), RaHinge + lazy R1/R2, D near chance; GAN alone took val8 DISTS 0.192 -> 0.164 at -1.7 dB (`docs/refactor/stage3/owner_recipe.md:24-28`, `:38`, `:53-63`) | Direction of "D LR below G" (PH:790) but far beyond it. The strongest local proof that nominal weights do not transfer (10-200x gap inside one trainer). Its numeric closeness to AN:139 / SX:148 is a coincidence of scale, not a confirmation | log reading, one run, val8 only, no no-GAN control |

Reading across the rows: (a) locally the GAN is worth having, and every tested departure from the template GAN
(lower weight, other D, other GAN type, HF view from the start) lost or traded one judge for another; (b) nominal
weight is a poor coordinate; the logged adversarial share (`grad_sr_ratio`, term share) is the candidate
comparable one, but whether it predicts outcomes is untested (G3-Q12), and the two measures differ (G6);
(c) none of this speaks to degraded inputs, x2/1x, long runs or other generators.

## 4. neosr-dev support

| community practice | status | exact config keys | evidence (path:line) | gap or caveat |
|---|---|---|---|---|
| Vanilla GAN, weight on the G term | present | `[train.gan_opt] type = "gan_loss"`, `gan_type = "bce"`, `loss_weight` | `neosr/losses/gan_loss.py:60-172`, `:88`; `options/train_realplksr.toml:102-105` | class default 0.1, template 0.3; the D term is unweighted; neosr scale only |
| Other GAN types | present / missing | `gan_type = "mse" \| "huber" \| "hinge" \| "rpgan"` | `neosr/losses/gan_loss.py:99-110`, `:167-169`, `:256-270` | no RaGAN / relativistic-average hinge (the owner's redux "r3gan" is RaHinge: `docs/refactor/stage3/owner_recipe.md:124`); no WGAN |
| Soft labels 0.9 / 0.1 (PH:822) | present (static) | `real_label_val`, `fake_label_val` | `neosr/losses/gan_loss.py:93-94`, `:137`, `:154-155` | bce/mse/huber only, refused for hinge/rpgan (`:108-110`); G's target is the real label too (`:169`); close to "damping" per `docs/refactor/stage3/discriminator/RESEARCH_REVIEW.md:151` (owner ruling open) |
| Delayed GAN start, linear ramp | present | `[train.gan_opt] start_iter`, `ramp_iters`, `end_iter` | `neosr/losses/base.py:80-121`; `neosr/losses/gan_loss.py:186-187`; `neosr/models/image.py:366-410` | linear ramp-in and hard stop only; D trains exactly while the G term is on and D's term is never ramped; default off (GAN from update 1). Not redux's `final_weight` semantics (PH:1027, unverified); neosr applies the weight once (`neosr/losses/base.py:113-115`), unlike the squared weight of the owner's redux checkout (`owner_recipe.md:56`) |
| Two-level or decaying GAN weight (PH:508-509) | missing | none | `neosr/losses/base.py:108-116` | one linear segment per term |
| Separate D optimizer, LR, scheduler | present | `[train.optim_d] type, lr, betas, weight_decay, warmup_steps`; `[train.scheduler] nets` | `neosr/models/image.py:298-333`; `options/train_realplksr.toml:72-78` | template D is `adan_sf` 1e-4 (schedule-free, warmup 600) under an AdamW 2e-4 G: neither "equal" nor the release pair; a schedule-free D is not scheduled unless listed |
| Gradient clipping on G and D (PH:442, PH:772) | present | `[train] grad_clip = true \| false \| <max-norm> \| "spike"` | `neosr/config.py:458-464`; `neosr/models/phase.py:180-186`; `neosr/models/image.py:179-185` | **default on**, max-norm 1.0, applied per net with the same setting; no D-only key (decision D5, `docs/refactor/DECISIONS.md:194`); the 1x releases run with clipping off (PH:562, PH:667) |
| D update ratio, D warm-up, D freeze (NP:312, PH:497, HB:823) | missing | none | `neosr/models/image.py:395-428` | exactly one D step per G update window; lazy `reg_every` is the only cadence knob |
| U-Net SN discriminator | present | `[network_d] type = "unet"` (`num_feat`, `skip_connection`) | `neosr/archs/unet_arch.py:9-24` | measured locally: +18 % ms/update in `bench.py`, +30-34 % in the 20k arm logs; clipped 100 % at the default clip (section 3) |
| PatchGAN | present | `type = "patchgan"`, `num_layers` (receptive field), `use_sn`, `norm_type` | `neosr/archs/patchgan_arch.py:34-66` | BatchNorm variant cannot take R1/R2; `use_sigmoid` + bce is wrong (`RESEARCH_REVIEW.md:83`) |
| DUnet, ea2fpn, R3GAN Ds | present | `type = "dunet" \| "ea2fpn" \| "r3gan" \| "r3gan_scalar128" \| "r3gan_scalar256" \| "resgan"` | `neosr/archs/dunet_arch.py:22-43`; `neosr/archs/ea2fpn_arch.py:183-188`; `neosr/archs/r3gan_arch.py:145-215` | dunet/ea2fpn refuse R1/R2 (dunet by name in `neosr/config.py:437-445`; any DySample or BatchNorm D at model build, `neosr/losses/gan_loss.py:22-28`, `:189-193`, `neosr/models/image.py:174-176`); scalar heads need exact GT 128/256; `resgan` (SN residual D, global average pool + linear, `neosr/archs/resgan_arch.py:34-36, :61-63`) gives one logit per image at any size |
| Template D with working SN (SKR:36) | partial | `type = "metagan"` | `neosr/archs/metagan_arch.py:120-124`, `:178-182` | GroupNorm cancels the SN (G10); opt-in fix DR-08 deferred (`RESEARCH_REVIEW.md:153`) |
| VGG-style global D (`rvgg`, PH:555) | missing (a global D exists) | none for VGG-style; `type = "resgan"` or `"r3gan_scalar*"` for one logit per image | `studio/shared/catalog/neosr-surface.json`; `neosr/archs/resgan_arch.py:34-36, :61-63` | `resgan` and `r3gan_scalar*` give one logit per image (not VGG-style). The stage-3 competitor review skipped VGGStyle ("no evidence", `docs/refactor/stage3/REVIEW.md:19`) |
| Multiscale D (SI:23, SX:134-138) | missing | none | `studio/shared/catalog/neosr-surface.json` | needs a multi-head contract (DR-11, `RESEARCH_REVIEW.md:156`); the stage-3 competitor review skipped multiscale PatchGAN (`docs/refactor/stage3/REVIEW.md:19`) |
| Wavelet-guided D (NL:6-16) | present | `[train] wavelet_guided = true`, `wavelet_init = N`, or `[train.gan_opt] wavelet_init = N` | `neosr/losses/wavelet_guided.py:8-49`; `neosr/losses/__init__.py:64-65`; `neosr/losses/gan_loss.py:151-152` | here it is **only the D input view** (level-1 SWT sym19 of Y; LH/HL/HH scaled 0.035/0.025/0.02; LL and colour dropped) for G's adversarial term and D; no other code uses it, so no G-side subband fidelity term (WGSR has one, PF:223). Hard switch at `wavelet_init` (ST-14 deferred, `docs/refactor/DECISIONS.md:165-169`); template 80000, never crossed by any run |
| R1 / R2 penalties | present | `r1_gamma`, `r2_gamma`, `reg_every`, `gamma_decay`, `gamma_iters`, `penalty_fp32` | `neosr/losses/gan_loss.py:75-85`, `:204-254`; `neosr/config.py:437-445` | per-position definition for map Ds (G33): community gammas are not comparable |
| Fresh D or loaded D (DR:51 vs UH4:184) | present | `[path] pretrain_network_d`, `strict_load_d`, `param_key_d`; omit for a fresh D | `neosr/models/image.py:218-234`; `neosr/models/checkpoint.py:141-181` | ignored while resuming; no partial D re-init |
| Branch from a full state with one changed knob (PF:229, PH:1004 ladder, HB:821) | partial | `--auto_resume` / `path.resume_state`, or `pretrain_network_g/_d` | `neosr/models/base.py:331-378`; `neosr/models/phase.py:276-312`; `docs/training_guide.md:24` | resume: an explicit `path.resume_state` may point at another run's state under a new `name` (`train.py:55-69`); a changed loss weight is taken from the config with a warning (read from code, not tested), but saved optimizer hyper-parameters win, so a D-LR change needs `pretrain_network_*` = fresh optimizers, fresh EMA, clock back to 0 (so `start_iter`, `wavelet_init` recount) |
| GAN health log (PF:240, SKR:31, HB:826) | present | automatic with `network_d` on `print_freq` updates | `neosr/models/image.py:58-98`; `neosr/losses/gan_loss.py:256-290`; `readme.md:198-213` | `out_d_real/fake`, `out_d_acc`, `out_d_abs`, `grad_sr_gan`, `grad_sr_ratio`, `l_d_*`, `clipped_d`; ahead of what the KB asks for |
| Per-term gradient shares (X13: RG:59, PF:56) | present (offline) | `python scripts/term_grads.py -opt <toml> [--net_g --net_d --batches --iter]` | `scripts/term_grads.py:1-138` | gradient over G's parameters (not one trunk layer, not fixed probe crops); share = squared-norm share (`:58-66`); the CLI prints a table, `term_grads()` returns a dict; D's update left out, but train-mode forwards still move SN vectors and BN statistics (`:33-36`) |
| Adaptive D balancing, AGC, dynamic loss weights | absent by rule | none | `docs/refactor/stage3/competitor/CONSENSUS_DRAFT.md:158` | evidence rule: no automatic damping |
| D feature matching | missing | none | `docs/refactor/stage3/competitor/losses_metrics.md:105` | no KB claim in this group asks for it |

## 5. Framework work this implies (strategy-independent)

Candidates for the owner, ranked by how many questions of part 6 they unblock per unit of work. None changes a
default; every item is opt-in or a tool. Items that already have an id in
`docs/refactor/stage3/discriminator/RESEARCH_REVIEW.md` keep it.

| # | feature | why (anchors) | size | risk | changes defaults? | how to verify | prerequisite for |
|---|---|---|---|---|---|---|---|
| 1 | **Magnitude-matching read-out:** `term_grads.py` with a machine-readable output and a sweep over checkpoints/configs, reported next to `grad_sr_ratio` | weights are not portable (PF:213, PG:261); nobody has logged per-term norms (RG:59, PF:56 = X13); local G6/G36 show nominal weight misleads. PF:213 [H] also warns against forcing equal norms by default: a read-out, not a balancer | S | low (a tool; trainer untouched; `term_grads()` already returns a dict when imported, the CLI prints a table) | no | same numbers as today's table on the template (FDL 81.2 %, GAN 16.5 %); a unit test on a toy model | G3-Q1, Q4, Q5, Q11, Q12 (matching adversarial pressure across weights and Ds) |
| 2 | **Fork mode:** an explicit, recorded branch from a full state (nets, EMA, optimizer moments, clock) that applies the config's changed hyper-parameters and writes parent + changed keys into the run record | one-knob branches from a common state are the community's main debugging and ladder practice (PF:229, PF:182, PH:1004, HB:821, RG:171, PG:253); today a D-LR change loses optimizer state and EMA (`neosr/models/phase.py:276-312`) | M | medium: must not weaken strict resume (`neosr/models/base.py:331-352`) | no | fork with no change = resume bitwise; a changed lr is applied and logged; provenance test | G3-Q2 (ladder), Q9, the D-dominance remedies; touches D10 rule 6 (owner) |
| 3 | **Document and check `wavelet_guided` against WGSR:** state that it is a D-input view, what the band scales are, and what the paper's design has beyond it | "needs the full design, not a flag; check subband normalisation" (PF:206, RG:57, PG:187); 82 templates ship it without an A/B (`RESEARCH_REVIEW.md:150`) | S (doc + a probe of band statistics) | low | no | band-energy probe on real vs SR batches; doc reviewed against the paper | reading G3-Q7 results |
| 4 | **Relativistic-average GAN type** (RaGAN and RaHinge) | ESRGAN lineage (SKL:20; generic); the owner's own redux recipe ran RaHinge, so an exact port needs it (`docs/refactor/stage3/owner_recipe.md:60`, `:124`) | S-M | low-medium: batch-coupled means interact with the one-forward D path and R1/R2 | no (opt-in type) | formula unit tests against a reference implementation; health keys defined for it; bitwise unchanged for existing types | G3-Q10; an exact OWN port |
| 5 | **Two-level / piecewise loss-weight schedule** | PH:508-509 (two phases), PH:1027 (`final_weight`; same message as RG:100); today one linear segment (`neosr/losses/base.py:108-116`) | S | low: a pure function of the update, like the existing schedule | no | full weight stays bitwise unscaled; schedule unit tests; resume test | only the 0.02-plateau variant of G3-Q3's long ramp (the 0 -> target variant is config only) |
| 6 | **Multi-head D contract (DR-11)** | multiscale PatchGAN (SI:23, SX:134-138), coarse-scale D on a symptom (SKT:16), per-scale weights (SX:215); the practice itself has weak support (HB:817, NP:393), and the stage-3 competitor review already skipped multiscale PatchGAN (`docs/refactor/stage3/REVIEW.md:19`): reopening it is the owner's call | M | medium: touches `scores`, `adv_loss`, the R1/R2 probe, the health log | no | single-tensor Ds bitwise unchanged; per-head logs; R1/R2 per head | the multiscale arm of G3-Q5 |
| 7 | **Static D cadence keys** (D every N, n D steps per G, D-only warm-up; NP:312 also lists a D EMA switch) | NP:312, PH:497, HB:823; today 1:1 only (`neosr/models/image.py:395-428`) | M | medium-high: core loop, accumulate, skip/restore, EMA, lazy `reg_every`, DDP | no (1:1 stays) | 1:1 bitwise parity; cadence unit tests; matched-compute accounting in the log | G3-Q9; needs the owner's ruling (static schedule vs damping) |
| 8 | **metagan `Down` without GroupNorm (DR-08)** | SN is what the community assumes a D has (SKL:23, SKR:36); A-04 (G10) | S | low; arch-specific (archs are deferred by the owner) | no (opt-in key) | init-gradient probe (expected ~27x lower, `RESEARCH_REVIEW.md:153`) | G3-Q6 |
| 9 | **VGG-style global D** | needed only to copy the 1x release stack (PH:555, PG:186); a global one-logit D already exists (`resgan`, `r3gan_scalar*`), and the stage-3 review skipped VGGStyle (`docs/refactor/stage3/REVIEW.md:19`) | S | low; an arch | no | shape and R1/R2 contract tests | none now (1x is blocked by data) |
| 10 | **Scheduled module parameters** (e.g. G dropout on after N updates) | PH:1090, one message | S-M | low | no | schedule unit test, resume test | G3-Q13 |

Not proposed: any adaptive D balancing (owner rule); a D "zoo" for its own sake (NP:393); generator-side noise or
SFT hooks as trainer features (X15 is architecture work, and archs rank last). Diagnostics from PF:227-238 that
belong to validation (raw-vs-EMA delta, flat-patch HF energy, step sweeps) are left to group g5. Existing
RESEARCH_REVIEW candidates not repeated here: DR-10 (joint RGB + HF view, the code-side alternative to the
`wavelet_init` switch; waits on DR-05 being positive, which it was not) and DR-12/DR-13 (LR-conditioned and
projected Ds; no KB claim in this group asks for them) (`RESEARCH_REVIEW.md:155-158`).

## 6. Research questions for the strategy phase

Questions, not recommendations. Costs assume the existing references (A0, B0, B0s2) are reused; a promoted arm
adds seed 2 (x2) and a default needs the long pair (2 long arms, ~4.5-5 h). PG experiments mapped: **X4** (Q1-Q3),
**X5** GAN axis (Q11), **X6** (Q5-Q7), **X13** (Q12), **X15** (Q14). X16 (GAN vs diffusion reference) belongs
to the validation group.

| id | question | single-factor arms | prerequisites (data, feature) | rough GPU cost | blocked by | priority (reason) |
|---|---|---|---|---|---|---|
| G3-Q1 | What GAN weight does the template D want on the neosr scale, read as adversarial share? (X4 fixed arms; D82) | `gan_opt.loss_weight` 0.1 and 0.2 vs 0.3 (0.03 already LOSS) | none; item 1 to report shares | 2 arms (~1.2 h) | nothing | medium: against 0.1, 0.3 has only a stage-2 8k "mixed" result (pre-R5 gate, 67 tiles); against 0.03 a D10 window win; cheap |
| G3-Q2 | Does a gated ladder of new runs (PH:1004, PG:253, X4) beat a fixed weight at equal total updates? | fixed 0.3 vs 0.1 -> 0.2 -> 0.3 with a rung per gated checkpoint | a weight-only rung already works as a resume from the parent's state under a new name (weight from the config with a warning; read from code, not tested); item 2 only for provenance or LR changes; automatic gates (g5) | ~1.5-2 arms per ladder | D10 rule 6 (straight 0 -> 20k); no defect gates yet | low-medium: design-level idea, no community measurement |
| G3-Q3 | Does an in-run ramp change the outcome at the same final weight (RG:170)? | `ramp_iters` 5000 (SKR:14); `start_iter` 10000 (RG:100); the 10k + 40k schedule (PH:508-509) as `start_iter = 10000`, `ramp_iters = 40000` | none for the first two; a run past 50k for the third (item 5 only for its 0.02-plateau variant) | 2 arms (~1.2 h); the long schedule = 1 long arm | long schedule: run length only | low: only hypotheses behind it; local 8k TIE; the template already starts from a pretrain |
| G3-Q4 | D LR relative to G: equal, 5x lower, 40x lower? | `optim_d.lr` 2e-4, 4e-5, 5e-6 vs 1e-4 | none | 3 arms (~1.9 h) | nothing; caveat `adan_sf` vs AdamW is a second factor if the optimizer is changed too | high: the release pair (PH:781/790; 2x, long run) and the owner's own run (redux, RaHinge) both point at a slow D, on setups unlike the template; config only |
| G3-Q5 | Which D architecture for photo micro-texture at identical G, budget and matched adversarial share? (X6) | `network_d` unet, patchgan (`num_layers` 3), dunet vs metagan, each at the weight that matches the reference share; multiscale PatchGAN | item 1; D LR retune per D (DX:347) adds arms; multiscale needs item 6. "Matched share" is a design choice: RG:168 compares at similar sharpness, PF:213 [H] warns against forcing equal norms | ~1 arm per D (unet x1.30-1.34 in run logs), +1 per retune | multiscale: item 6; rvgg: item 9 | high: "the D is the real lever" is unmeasured everywhere; the one local arm is confounded |
| G3-Q6 | Does an acting spectral norm change the template D's result? | metagan with vs without GroupNorm in `Down` | item 8 (DR-08) | 1 arm | item 8 | medium: touches the D of ~85 templates; arch-specific |
| G3-Q7 | Does `wavelet_guided` help at all, and when should it switch? (X6 arm; NL:16) | RGB always vs switch at 40k vs switch at 80k | runs that cross the switch; item 3 for reading | 3 long arms (~7 h), or 2 if the D10 long pair supplies the 80k arm | no run past 20k yet; ST-14 transient unresolved | high for the framework (shipped in 82 templates untested), costly |
| G3-Q8 | Fresh D or loaded D when a GAN stage starts from a GAN checkpoint (DR:51 vs UH4:184)? | from one 20k checkpoint: G only vs G + D | none (`pretrain_network_d`) | 2 short arms (~1.2 h at 20k) | nothing | low-medium: matters once forks and ladders are used |
| G3-Q9 | Does a static D cadence (D every N; 2 D steps early) help at matched compute (NP:312, PH:497)? | N = 2, 4 vs 1:1 | item 7 | 2 arms | item 7; owner ruling | low: no N and no result in the KB; the local D is at chance, not dominant |
| G3-Q10 | GAN type on one D: bce vs hinge vs RaGAN/RaHinge (SKL:20) | `gan_type` hinge; RaHinge | hinge: none (DR-06); RaGAN: item 4 | 1-2 arms | RaGAN: item 4 | medium: needed to explain the OWN gap; all releases use vanilla |
| G3-Q11 | Which D regime works best once DISTS is in the objective? (X5 GAN axis) | 2x2: D regime {template, slow D from Q4} x DISTS {off, on} | T_DISTS seed 2 (held); the winner of Q4 | 2 new arms at x1.37 (~1.7 h) | T_DISTS not yet confirmed on seed 2 | high: it is the owner's own recipe question (G13, G15) |
| G3-Q12 | Can adversarial gradient share predict the outcome across weights and Ds, i.e. is there a portable weight-setting rule? (X13) | no training: `term_grads.py` + `grad_sr_ratio` over the existing arms' checkpoints | item 1; the redux half of X13 needs a redux-side probe | minutes per checkpoint | redux side is out of scope here | high as a prerequisite: cheap and it frames Q1, Q4, Q5 |
| G3-Q13 | RealPLKSR dropout after ~60k and a larger GT crop late (PH:1090)? | dropout 0.3 from 60k vs none | item 10; a long run | 1 long arm | item 10; arch-specific | low: one message; archs rank last |
| G3-Q14 | Do generator-side noise inputs or SFT maps reduce repeated motifs? (X15) | G with vs without noise inputs | an arch with the lever; deterministic validation for a stochastic G | per arch | architecture work; no repetition metric | low for the framework; relevant to the owner's own noise-head arch |
| G3-Q15 | Is the GAN wanted at 1x cleanup / under real degradations (DX:377 vs PH:713)? | GAN on vs off per task | 1x or degraded data | 2 arms per task | parked OTF; clean x4 data only | low now: cannot be asked locally |
| G3-Q16 | RealPLKSR vs RRDB at matched patches seen (DR:43 gap)? | generator swap | a second pretrain; patches-seen accounting | several arms; RRDB is much slower | arch choice is outside the framework goal | low |

What OTF realism would need for this group (parked, not planned): every degradation-dependent claim above
(soft labels and a slow D "on noisy/compressed domains" PH:479, the aliasing risk of PatchGAN with clean LR RP:151,
GAN off for JPEG cleanup DX:377, degradation as the source of detail PH:741) needs degraded pairs; the local set
is the "LR too clean" case by construction.

## 7. Questions for the owner

1. **Damping boundary:** are static label smoothing (0.9/0.1) and a static D cadence (D every N, D warm-up)
   admissible opt-in objective/dynamics changes, or do they fall under "no automatic damping"? DR-06 and
   G3-Q9 wait on this.
2. **Which D regime is the working hypothesis for the strategy phase:** the template's near-chance D (acc ~0.5,
   share ~2 % on SR), a slow weak regularised D (your redux run), or a D that clearly wins? This orders G3-Q4,
   Q5 and Q11. The one local arm with a clearly winning D (R3, `out_d_acc` ~0.9) lost on DISTS and 0.7 dB PSNR
   (G11), but it changed several factors at once.
3. **Which judge decides a trade:** the unet D is better on LPIPS and worse on TOPIQ on both seeds. For re-drawn
   texture, does one of them outrank the other, or does a mixed result stay a LOSS?
4. **Ladders and forks vs D10 rule 6:** may the protocol admit branch runs from a gated checkpoint (the
   community's main practice), and is an opt-in fork mode (part 5, item 2) acceptable next to strict resume?
5. **`wavelet_guided`:** is it worth long arms to test a feature that 82 templates ship and no run has ever
   reached, and which arm matters most (off, 40k, 80k)?
6. **Discriminator scope:** are new D-side features wanted now (RaGAN/RaHinge for an exact port of your recipe,
   the multi-head contract, a VGG-style D), given that archs rank last and the KB's own advice is "simple stable
   D first"? Does DR-08 (metagan without GroupNorm, opt-in) enter the candidate list while D3 keeps
   metagan @0.3 as default?
7. **Generator as a fixed factor:** does the strategy phase stay on `realplksr`, or should some questions be
   asked on your own arch (noise head, stochastic validation), which is where X15 lives?
8. **Scales:** the only release configs are 1x and 2x; the local set is clean x4. Should x2/1x questions wait
   for OTF, or is a clean x2 set wanted so that at least the non-degradation part can be asked?

## Verification

Adversarial check of this file against its sources (KB, code, local docs and run files). No GPU, no training;
the D10 windows were recomputed on the CPU with `scripts/ab/analyze.py` (read-only, `--window 12000:20000
--null A0/B0,B0/B0s2,U_unet01/U_unet01s2`) for A1_nogan, M_metagan003, U_unet01, U_unet01s2, R3_patch_g1, OWN,
T_DISTS and NoGAN_DISTS; per-segment ms/update were read from the arms' `train.jsonl`.

**Checked:** ~170 KB anchors opened at their line (all release-config numbers PH:555-848, PH:1004-1090; TG, NL,
RX, UH3, UH4, SI; every anchor behind part 6 priorities "high"); ~60 neosr-dev code, config and doc claims
(`gan_loss.py`, `losses/base.py`, `losses/__init__.py`, `wavelet_guided.py`, `models/image.py`, `phase.py`,
`models/base.py`, `checkpoint.py`, `config.py`, `train.py`, the nine D archs, `term_grads.py`, the template TOML,
the Studio surface, template counts); 9 local window verdicts recomputed (all matched the file's numbers).

**Corrections (before -> after, source):**

| # | before | after | source |
|---|---|---|---|
| 1 | `unet` D cost x1.18 / "+18 % ms/update" | x1.18 in `bench.py`, x1.30-1.34 in the 20k arm logs (header, G8, part 4, Q5) | U_unet01/U_unet01s2 vs B0/B0s2/A0 `train.jsonl`, 8k-20k medians 127.4 vs 95.1-97.8 ms; RESEARCH_REVIEW.md:17-19 |
| 2 | "Nothing at x4 prints a GAN block except the redux template" | a posted default block, not scale-specific; the 1x releases are OTF high-order, `lq_size` 96, 250k iterations; 2x 500k; the guideline's GAN stage is the degradation stage | PH:1019-1054, PH:633-635, PH:738-740, PH:559, PH:664, PH:769, TG:26-28 |
| 3 | 0.1 "in the redux template (PH:1054; release_config)" | a posted template, not a release | PH:1032-1054 |
| 4 | labels 1/0 **and D:G 1:1** "documented default (official)" | labels documented; 1:1 is DR's design | NL:139-147, DR:169 |
| 5 | 1x clipping off cited as PH:667 only | PH:562 (restore) and PH:667 (sharpen) | PH:553-563, PH:660-667 |
| 6 | wavelet-guided D listed without purpose or method scope | stated purpose = stabilise GAN / reduce artifacts; WGSR = HF-subband D **plus** subband reconstruction | NL:6, PF:223 |
| 7 | "A pretrained GAN can carry its original target's sharpening (PF:374)"; PF:229 as discord_cited | PF:374 is an open question; FG:241 says "may carry"; PF:229 row is uncited | PF:374, FG:241, PF:229 |
| 8 | fresh D vs UH4:184 as a conflict | weak conflict: HB:484 keeps "an exactly relevant D checkpoint", which UH4:184 is | HB:484, UH4:175-187 |
| 9 | RG:100 and PH:1027 as two reports | the same message (Helaman 2025-11-27 17:45) | RG:100, PH:1027 |
| 10 | 10k + 40k ramp "needs a two-level schedule (missing)" (part 2, item 5, Q3) | the 0 -> target form is config only (`start_iter = 10000`, `ramp_iters = 40000`); only the 0.02 plateau needs item 5 | `neosr/losses/base.py:80-115`, PH:508-509 |
| 11 | "only `r3gan_scalar*` give one logit per image" | `resgan` (SN, global avg pool + linear) also does (part 2, part 4, item 9) | `neosr/archs/resgan_arch.py:34-36`, `:61-63`; RESEARCH_REVIEW.md:123 |
| 12 | dunet/ea2fpn R1/R2 refusal at `config.py:437-445` | dunet there; ea2fpn (DySample) at model build | `gan_loss.py:22-28`, `:189-193`; `image.py:174-176` |
| 13 | G5 "no-D 14 % faster"; "against 'abandon the GAN' (HB:824)" | 14-18 %; consistent with HB:824, which abandons only under persistent instability | `train.jsonl` (A1 80.0 ms), HB:824 |
| 14 | G6 "on a fresh G the D saturates by ~300 updates, chance by ~2k" | on an **untrained** G (GP-06), 250-350 updates, chance from ~1.7k; added what each "share" measures | gpu_profile.md:347-352; `term_grads.py:7-8`, `:58-66` |
| 15 | G8 "8k/20k checkpoints: TIE / WIN" | seed 7 TIE, seed 2 WIN at both 8k and 20k | brief.md:32-33, stage2/DECISIONS.md:89-99 |
| 16 | G11 LOSS listed as PSNR, DISTS and LPIPS | LOSS on DISTS alone (LPIPS at tau, PSNR not gated); its D wins clearly (`out_d_acc` 0.87-0.90) | analyze.py recomputation; r3gan/RESULTS.md:128-134 |
| 17 | reading (b): adversarial share "is the comparable one" | a candidate coordinate, untested (Q12) | G6 row; `term_grads.py` |
| 18 | Q2 needs "item 2 (fork) or `pretrain_network_*`" | a weight-only rung already works as a resume from the parent state under a new name (code reading); item 2 for LR changes and provenance | `train.py:54-69`, `models/base.py:371-378`, `phase.py:280-310` |
| 19 | G13 "best of all arms on val8 LPIPS/DISTS" | + val8 is 8 photos, never decisive, and OWN trains on DISTS | bundle G13; D10 rule 7 |
| 20 | G33 ratio without its D | 0.03-0.05x for the r3gan patch D, ~1/P for metagan | refactor/DECISIONS.md:57-60 |

**Additions:** D-LR 0.5x as the only number for the first step (SKR:33 generic, DR:170); PH:1087 (D LR 1e-4 on
NaNs after a batch change); RealWebPhoto v3's OTF context (RP:253); redux `start_iter` makes weights act squared
in the owner's checkout (owner_recipe.md:56, :59), so SL:36's 0.08 may be nominal; 2x release weight decay 0.01;
the stage-3 competitor review already skipped VGGStyle and multiscale PatchGAN (`docs/refactor/stage3/REVIEW.md:19`;
part 4, items 6 and 9); `term_grads.py` caveats (dict API, SN/BN buffers move); PF:213's warning against forcing
equal norms (item 1, Q5); D EMA in item 7; DR-10/12/13 noted as not repeated; Q4's evidence comes from unlike
setups; owner question 2 now notes that the one local clearly-winning D (R3) lost.

**Rule check:** no averaged conflicts, no redux/basicsr weight presented as a neosr value, no work planned on
the parked OTF track (only what it would need), no proposal changes a default (every part 5 item is opt-in or a
tool), and local evidence is stated as clean x4, <= 20k, one generator throughout. Hypothesis vs evidence labels
were kept; the PH:1054 "release_config" label of the bundle is corrected in the text.

**Still unverified:** the Discord originals behind every discord_cited anchor (only the KB's citation lines were
read); whether the PH release configs used `start_iter` (squared-weight risk) and the 2x release's degradation
(not printed); the squared-weight behaviour in redux versions other than the owner's checkout; the branch-by-
`resume_state` path with a changed loss weight (read from code, never run); which ms/update (bench or run log)
the next arms will see; the stage-2 8k numbers (G7's 0.1 vs 0.3 "mixed", G20) and the G9/G10/G36/G38 numbers,
read from their docs, not re-measured; the 8k D9 verdicts (only the D10 windows were recomputed).
