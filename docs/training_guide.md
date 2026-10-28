# Training guide

Short recipes for the shipped templates. Every recipe is a config edit; there is no second template set. The
numbers come from neosr's own runs (`docs/refactor/stage2/`, realplksr x4, held-out tiles) unless marked as a
paper result.

## 1. Fidelity (L1-only) run

Edit any `options/train_<arch>.toml` GAN template:

| remove | why |
|---|---|
| `[network_d]`, `[train.optim_d]`, `[train.gan_opt]` | no discriminator |
| `train.wavelet_guided`, `train.wavelet_init` | they need `gan_opt` |
| `[train.mssim_opt]`, `[train.consistency_opt]`, `[train.ldl_opt]`, `[train.fdl_opt]` | perceptual and artifact terms (keep `mssim_opt` for a fidelity mix) |

Keep `[train.pixel_opt] type = "L1Loss"`. Judge by PSNR/SSIM (`[val.metrics.psnr]`, `[val.metrics.ssim]`).
Reference: realplksr L1-only, 5k updates, 29.91 dB on 67 held-out tiles.

## 2. Fidelity first, then GAN

| way | config | note |
|---|---|---|
| two runs (Real-ESRNet -> Real-ESRGAN, [paper]) | run 1 as in 1.; run 2 = the GAN template with `path.pretrain_network_g = "experiments/<run1>/models/net_g_<u>.pth"` and the same `[network_g]` | run 2 is a new run: fresh optimizer, and a fresh EMA that starts from the generator after run 2's first update |
| one run | in the GAN template set `start_iter = N` (and `ramp_iters = M`) on `[train.gan_opt]` and on each perceptual term | every `[train.<slot>_opt]` takes `start_iter`, `ramp_iters`, `end_iter`; D trains exactly while the GAN term is on |

Measured so far: in the stage-2 set (8k updates, one seed) the arm "GAN from update 1000 + 1000-update ramp"
(28.69 dB, DISTS 0.1900) showed no gain over the arms that start the GAN at update 1 (28.58-29.36 dB); the arms
differ in more than the start, so this settles nothing. A longer L1 phase on your data is experiment E5.

**R3GAN recipe** (`options/train_realplksr_r3gan.toml`, opt-in, the two-run way): `network_d.type = "r3gan"` (0.87 M
params, one logit per 16 px; `r3gan_scalar256`/`r3gan_scalar128` = the paper's scalar head, needs exactly that GT size),
`gan_type = "rpgan"` (each fake against its own real, per position), `r1_gamma = r2_gamma` on every D update
(`reg_every = 1`), Adam with `betas = [0.0, 0.99]` for D, no `wavelet_guided`. On a map D the gammas are normalised
per position (each output position is a critic), so the number of positions no longer scales them; the A/B grid is
{1, 3} for the patch head and {4, 40} for `r3gan_scalar256` at GT 256. Logs: `out_d_margin`, raw `r1`/`r2` (squared input-gradient norms before gamma), `l_d_r1`/`l_d_r2`,
`gamma_r1`/`gamma_r2`; `gamma_decay`/`gamma_iters` give R3GAN's cosine decay. The penalty pass runs in bf16 when the
run is bf16 (`penalty_fp32 = true` forces fp32). dunet and BatchNorm discriminators cannot take R1/R2 (refused). Whether
the recipe beats the template GAN on your data is the A/B in `docs/refactor/stage3/r3gan/RESULTS.md`.

## 3. Scale change and fine-tuning

- **x2 -> x4** ([paper] EDSR): `pretrain_network_g` = the x2 model, `strict_load_g = false`. Layers whose shapes
  differ are skipped; the load logs one line with the loaded share and the skipped, missing and unexpected keys.
- **Fine-tuning a finished model**: lower the generator lr (e.g. 1e-4 instead of 2e-4) and keep `warmup_steps`.
  Community ESRGAN files in the original `model.N` layout and `.safetensors` files (with the `safetensors`
  package installed) load as pretrained weights.
- **Cooldown or a new schedule on a running job**: change `[train.scheduler]` and resume. A changed scheduler is
  rebuilt from the config and replayed to the saved update, with a warning; `nets = ["g", "d"]` selects the
  optimizers (default: every optimizer that is not schedule-free).

## 4. Which metric judges what

| run | judge | not the judge |
|---|---|---|
| fidelity | PSNR, SSIM | - |
| GAN / perceptual | a learned metric that is no training loss of the run: TOPIQ-FR (`dataset = "kadid"` or `"pipal"`, PIPAL was trained on SR/GAN outputs), LPIPS (`calculate_lpips`, lower is better); DISTS only when no training term uses `dists_loss` | the training loss itself, PSNR alone |

`val.save_best = "<metric key>"` writes `models/net_g_best_<dataset>_<metric>.pth` (the evaluated EMA/SF
weights) whenever that metric matches or improves its best. A small validation set cannot resolve small
differences: 67 tiles and one seed do not separate 0.05-0.1 dB; use hundreds of held-out tiles and a second seed
before changing a default.

## 5. Using an export

Exports describe themselves (`neosr` entry: network options, scale, update; the best export also its metric):
`convert.py --input net_g_best_val_topiq.pth` needs no `-net`/`-s`, and `test.py` takes the network options from
the file (the config's `[network_g]` keys win). A new training run from an export is `pretrain_network_g`; training
builds the network from its own config, so copy the export's `[network_g]` options (`neosr.network_g` in the file)
and keep the scale, or relax the load (`strict_load_g = false`) on purpose.

## 6. 1x restoration (denoise, deblur, artifacts)

`scale = 1`, `[network_g] type = "nafnet"` (or `nafnet_l`), paired data with same-size LQ/GT. `psnr_loss` is the
paper's objective. Official NAFNet feeds the loss the unclamped output: `train.clamp = false` does the same (its
lr, betas and cosine schedule are separate settings). Validate on
full images: NAFNet's channel attention pools globally, so tiled outputs differ.

## 7. Speed

`python scripts/bench.py -opt <config> [--compile --compile_mode reduce-overhead] [--precision bf16]` reports
steady-state ms/update, warm-up/compile time, peak VRAM and inference ms. One GPU job at a time: never next to a
training run.
