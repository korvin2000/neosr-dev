# The owner's staged photoreal recipe (traiNNer-redux, January 2026): what worked and why

Written 2026-11-11 (session 10, Linux machine). Sources:
- the owner's presets `step1.yml` .. `step5.yml` (`/home/korvin/apps/training/`, dated 2026-01-12);
- the run they drove: `/home/korvin/apps/traiNNer-redux-dev/experiments/Experimental3/` (six logs, 02:19-04:39);
- the redux code as it ran (fork commit d4867434 and its parent ef90e5dd; the loss, D and arch files did not change after it).

Method: two agents read the redux code and mapped it to neosr-dev, then Fable 5.2 and Codex GPT-6-Astra analysed the
recipe independently. I checked the decisive claims myself against the code and the logs. Evidence keys used below:
**code** (read), **log** (the owner's run), **paper**, **reasoning**.

## 1. Summary
The presets look like a five-step curriculum with many losses. What actually trained was simpler:

1. **A short fidelity pretrain, then one chained fine-tune.** Step 1 trained Charbonnier + MS-SSIM-L1 for 7,115
   iterations. The GAN phase was one run of 38,125 iterations in 2 h 20 min: six configs, each resuming the previous
   state (G, D, EMA, optimizers) [log].
2. **DISTS as a loss was the texture driver.** Val8 DISTS fell only in the two stretches where `distsloss` was on:
   0.160 -> 0.1245 (step 3) and 0.161 -> **0.1107** (step 5). When it was off, DISTS rose again (0.1245 -> 0.181 in
   about 2k iterations), although the GAN stayed on [log].
3. **A pixel and colour anchor kept the result on the image.** In step 4 (Charbonnier 0.01, no HSLuv) PSNR fell from
   29.5 to 17.7 dB in 1,556 iterations. Step 5 restored Charbonnier 0.1 + HSLuv 0.1 and recovered to 28.05 dB with the
   best DISTS of the run [log].
4. **The GAN was weak and slow.** Effective G weight 0.0016-0.01 (not 0.04-0.1, see section 3). D lr 5e-6, 40x below
   G. Relativistic-average hinge with lazy R1/R2 (gamma 3). The raw G hinge stayed at 1.03-1.09, where 1.0 means no
   separation, so the D sat near chance [log, code]. Its contribution on top of DISTS is not measured: the chain has no
   no-GAN control.
5. **Half the listed losses did nothing.** DINOv3 and ConvNeXt perceptual losses had no gradient. LDL and HFEN were
   numerically nil. The loss-decay keys were ignored. The G lr of 1e-4 in steps 4-5 was never applied [code, log].

So the "secret" is a fidelity basin first, then a texture-tolerant feature loss (DISTS) held in place by a pixel +
colour anchor, with a gentle GAN on top, in few iterations at a constant lr of 2e-4 on small patches.

## 2. What ran, stage by stage [log]
| stage | iterations | batch | config (nominal weights) | effective G objective | val8 PSNR | val8 DISTS |
|---|---|---|---|---|---|---|
| step 1 (separate run) | 0-7,115 | 8 | Charbonnier 0.6, MS-SSIM-L1 0.4 (alpha 0.1) | as written | - | - |
| step 2 | 0-11.1k | 8 | Charb 1.0, HSLuv 0.1, r3gan 0.1 (start 1000) | Charb 1 + HSLuv 0.1 + GAN 0.01 | 29.10 -> 27.41 | 0.192 -> 0.164 @9k -> 0.172 |
| step 3 | -19.5k | 8 | + ConvNeXt 0.2, DISTS 0.15; HSLuv 0.07, r3gan 0.04 | Charb 1 + HSLuv 0.07 + GAN 0.0016 + DISTS 0.15 (+ ConvNeXt 0.04, no gradient) | 29.06 -> 28.90 | 0.160 -> **0.1245** |
| not in the presets | -24.2k | 16 | Charb 1.0, HSLuv 0.01, r3gan 0.01 | Charb 1 + HSLuv 0.01 + GAN 1e-4 | -> 29.38 | 0.156 -> 0.181 |
| not in the presets | -31.5k | 16 | Charb 0.2, r3gan 0.02, DINOv3 0.15, LDL 0.1, HFEN 0.08 | Charb 0.2 + GAN 4e-4 (+ LDL, HFEN ~0) | 29.47 | 0.178 |
| step 4 | -33.1k | 16 | Charb 0.01, r3gan 0.1, DINOv3, LDL, HFEN, ConvNeXt 0.01, DISTS 0.03 | Charb 0.01 + GAN 0.01 + DISTS 0.03 | **21.3 -> 17.7** | 0.173 -> 0.214 |
| step 5 | -38.1k | 16 | Charb 0.1, r3gan 0.07, HSLuv 0.1, DISTS 0.05 (+ the inert terms) | Charb 0.1 + HSLuv 0.1 + GAN 0.0049 + DISTS 0.05 | 22.7 -> **28.05** | 0.161 -> **0.1107** |

Common to all stages: arch `hyfreqmoe_sr` (12.1 M params, the owner's own design with a noise-injecting texture
head), x4, LQ 64 / GT 256 crops with flips and rot90, bf16 autocast, channels_last, TF32 matmul, no compile, AdamW
(0.9, 0.99) without weight decay, EMA 0.999 with power 0.75, MultiStepLR milestones at 250k+ (so a constant lr),
validation every 1000 on val8 (PSNR, SSIM, DISTS), a checkpoint every 1000. From step 4 on: batch 16 and grad clip
max-norm 1.0 on G and D. Throughput 8.5 it/s at batch 8 with the GAN only; 3.4 it/s at batch 16 with every loss on;
peak VRAM 16.9 GB [log].

## 3. What the presets say vs what the code did [code, verified]
| preset entry | what happened in redux | where |
|---|---|---|
| `DINOPerceptualLoss` 0.15, `convnextperceptualloss` | `_extract_feats` runs under `@torch.no_grad()` for the prediction too: zero gradient to G, only compute cost. The logged `l_g_dinoperceptual` (0.13 of a 0.148 total in step 5) is pure display | `dino_perceptual_loss.py:90`, `convnext_perceptual_loss.py:118` |
| any loss with `start_iter` or `loss_decay*` (ganloss, ldlloss, hfenloss) | wrapped by `IterativeLossWrapper`, which multiplies by `loss_weight`; `sr_model` then multiplies by `loss_weight` again, so the weight acts squared (GAN 0.07 -> 0.0049); HFEN also weights internally (0.08 -> 5e-4) | `iterative_loss_wrapper.py:214-224`, `sr_model.py:648-669` |
| `loss_decay: 0.8`, `loss_decay_inflection: 35000` (LDL) | popped from the options and never used: no decay | `losses/__init__.py:79-104` |
| `lr: 1e-4` (steps 4-5) | not applied: the resumed optimizer state kept 2e-4 (every log line prints `lr:(2.000e-04)`) | logs |
| `start_iter: 1000` / `2000` | only mattered in step 2; the later stages resumed past them. Before it the D does not train either | `sr_model.py:502-505, 733-745` |
| `ganloss gan_type: r3gan` | relativistic-average **hinge**, not R3GAN's paired softplus; R1 on GT and R2 on the detached output every 4th D step, x4, on a random 96 px crop, in fp32; average weight gamma/2 = 1.5. The D-side adversarial term is not scaled by `loss_weight` | `r3gan_loss.py:44-111, 232-248, 284-335` |
| `RpGANDiscriminatorScalar` | R3GAN-style ResNeXt D without any normalisation, widths 32-256, 4 downsamplings, pooled to 4x4, one scalar per image; 1.73 M params | `rpgan_r3gan_scalar_discriminator_arch.py` |
| `hsluvloss` | hue, saturation and lightness terms, 1/3 each; the hue term is zeroed wherever the output is light (`y_l > eps - 1` is always true) | `basic_loss.py:404` |
| `hyfreqmoe_sr` `noise_warmup` | the noise ramp counter is not saved, so it restarts at every resume; noise stays on in eval, so validation is stochastic | `hyfreqmoe_sr_arch.py:474-510` |

Correction to `competitor/recipes_evidence.md` (stage 3 harvest): it gives Experimental3's GAN as "r3gan 0.05-0.07"
(effective 0.0025-0.0049) and blames the Experimental3B PSNR collapse on the DINOv3 term being ~95 % of the loss.
DINOv3 had no gradient, so it cannot have caused that collapse. The pixel weight is the likelier cause: it was low
there, as in step 4 here.

## 4. Why it works: the mechanisms, ranked
| # | element | mechanism | evidence | status |
|---|---|---|---|---|
| 1 | DISTS loss 0.05-0.15 | DISTS compares VGG feature *statistics* (texture) and their correlation (structure), so it accepts a texture with the right statistics at shifted positions (Ding et al. 2020). The G may synthesise grass, skin or fabric detail that an L1 loss would average away, without needing a strong D | DISTS falls only while the loss is on; it rises when it is off, with the GAN still on [log] | strong on val8; DISTS is also the metric, so this is partly Goodhart |
| 2 | pixel + colour anchor (Charb >= 0.1, HSLuv 0.1) | feature losses and a relativistic D see little of the low-frequency colour and luminance; the anchor fixes the point on the perception-distortion curve (Blau and Michaeli 2018) | step 4's collapse at Charb 0.01 without HSLuv, step 5's recovery [log] | the clearest causal evidence; confounded with the GAN weight rising x6 in step 4 |
| 3 | fidelity pretrain first | ESRGAN / Real-ESRGAN: start in the PSNR basin, so the perceptual terms only add high frequencies; 7k iterations sufficed | 29.1 dB at the start; DISTS responds within 1-2k iterations [log, paper] | standard; neosr's A/B protocol already does it (P0) |
| 4 | a weak, slow, regularised D | D lr 40x below G (an inverted TTUR, Heusel et al. 2017), lazy R1/R2, effective weight <= 0.01: the D stays near equilibrium and adds a small sharpening gradient but cannot overpower the G | GAN alone in step 2: DISTS 0.192 -> 0.164, then PSNR -1.7 dB and DISTS rising after 8k [log] | its value on top of DISTS is unproven |
| 5 | the arch's noise-injecting texture head | a stochastic high-frequency source, as in StyleGAN's noise inputs | none isolated [reasoning] | arch-specific; not portable as such |
| 6 | short run, constant lr 2e-4, small patches, bf16 | "fast" means few iterations (38k) at a high constant lr on 64 px LQ crops, not a fast update (3.4 it/s at batch 16) | [log] | the time budget, not a trick |
| 7 | batch 16, clip 1.0 | a better batch mean for the relativistic average and R1/R2 | changed together with the losses [log] | unproven |
| - | DINOv3, ConvNeXt, HFEN, LDL, loss decay, lr 1e-4 | none: no gradient, ~5e-4 weight, ignored keys, never applied | [code, log] | inert; DINOv3 and ConvNeXt only cost time |

## 5. What it means for neosr-dev
**It explains our A/B evidence.**
- Our template's D (metagan, BCE, weight 0.3) runs in the same regime as the owner's: near chance (`out_d_acc`
  ~0.50) with an adversarial gradient of ~2 % of the other terms (`grad_sr_ratio`). Compare GAN regimes by those two
  health keys, not by the nominal weight.
- Our R3GAN screen (R3 LOSS, D wins with `out_d_acc` ~0.9, PSNR -0.81) did not test the owner's regime. The R6 R arms
  use Adam lr 2e-4 for the D (40x the owner's) at GAN weight 0.3 (~60x the owner's effective weight) and also turn
  wavelet guidance off. No planned R6 arm has a slow D at a small weight.
- The no-GAN arm lost to the template, but the template has no DISTS. The owner's run suggests DISTS may supply much
  of what the D supplies; only a no-GAN + DISTS arm can tell.
- LDL was effectively off in the owner's run, so the run says nothing for or against neosr's LDL at 1.0.

**Do not copy the preset weights.** They are not transferable: the nominal and the effective weights differ by 10-200x,
the GAN formulation differs (neosr's `rpgan` is the paired softplus of the R3GAN paper), and the arch is not in neosr.

**The recipe in neosr terms** (effective weights; for an opt-in preset after the A/Bs, not a default):
```toml
[network_d]
type = "r3gan"                # patch head; r3gan_scalar256 is the scalar head (exact 4x4 map at GT 256)
[train]
ema = 0.999
ema_power = 0.75
[train.optim_d]
type = "adamw"
lr = 5e-6
betas = [0.9, 0.99]
weight_decay = 0
[train.pixel_opt]             # Charbonnier
type = "chc_loss"
criterion = "huber"
loss_lambda = 0
clip_min = 0
clip_max = 1e6
loss_weight = 0.1
[train.consistency_opt]       # the colour anchor (Oklab + CIE L*, a stand-in for HSLuv)
type = "consistency_loss"
loss_weight = 0.1
[train.dists_opt]
type = "dists_loss"
loss_weight = 0.05            # 0.15 in the step that gained most
[train.gan_opt]
type = "gan_loss"
gan_type = "rpgan"            # paired softplus; redux's RaHinge has no neosr key
loss_weight = 0.005
r1_gamma = 3.0
r2_gamma = 3.0
reg_every = 4
penalty_fp32 = true
```
Not expressible and not worth porting: the redux scalar D's pool-to-4x4, the 96 px penalty crop, HSLuv (its hue term
is buggy), HFEN, DINOv3 and ConvNeXt perceptual losses (inert in the run), the loss-decay keys.

**Gate rule for arms that train on DISTS.** DISTS is then both the loss and a judge, so a DISTS arm wins that metric
for free. For those arms the verdict uses TOPIQ and LPIPS (LOSS if either is worse, WIN if both are better); DISTS is
reported, not gated. The confirmation set and the visual set check the finalists. This follows `docs/training_guide.md`
section 4 ("DISTS only when no training term uses `dists_loss`").

## 6. Caveats
- Val8 only (8 photos), one seed, a stochastic validation (the arch's noise is on in eval), and several changes in
  every stage: the ranking above is the best reading of confounded logs, not an A/B.
- The `grad_norm_d` of the unnormalised D grew 0.09 -> 41 -> 245 over the first stages (the fixed clip held it), and
  nothing ran past 38k iterations.
- How "trained well" was judged is not recorded: by eye on which checkpoint, or by val8 DISTS.

## 7. Open questions for the owner
1. Was the final model the 38k checkpoint of step 5, and was it judged by eye or by val8 DISTS?
2. Were the two stages that are not in the presets, and step 4's collapse, deliberate? Is "step 3, then step 5" the
   reference recipe?
3. Would a recipe without a D be acceptable if pixel + colour + DISTS ties the GAN arms?

The A/B arms that test this on the R5 protocol (20k, two seeds, the confirmation set) are listed in `next_tasks.md`.
