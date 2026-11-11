# Discriminator research: review against the current code

Date 2026-11-11, code at `master` b622affe. Read-only review: no code changed, no GPU job, nothing committed.

**Status (end of session 9, 2026-11-11).** The review below is unchanged; this table records what became of it.

| id | state |
|---|---|
| DR-01 | done another way (94c36bde): every r3gan conv takes its input gradient as a `conv_transpose2d`. The reshape of the FIR alone would have kept its wasted weight gradient and an NHWC copy, and left the grouped 3x3 and the 1x1 convs slow. One regularised D update 647 -> 35 ms; the R3GAN recipe 732 -> 122.5 ms/update |
| DR-02 | done (59a6fbc4): GAN health keys on log updates, hooks inside the one real backward |
| DR-03 | 8k screen ran (U_unet01, M_metagan003 vs B0 and A1) <<PENDING: verdicts>> |
| DR-04 | 8k screen ran (R3_patch_g1 vs A1 vs B0) <<PENDING: verdicts>> |
| DR-05 | <<PENDING: W_wavelet0 verdict, or "not run (dropped by the deadline)">> |
| DR-07 | done (cd3b0adb): refused at model build. Correction: the run did not crash before; `config.py` already refused dunet by name and ea2fpn has BatchNorm (refused at the first D call) |
| DR-06, DR-08 to DR-11 | deferred |

**Correction (measured 2026-11-11):** `unet` is not cheaper on the GPU. The op-count argument of P1 and DR-03
("170 vs 807 fwd ops") does not hold: on the template, unet D costs 129.5 vs 109.9 ms/update (+18 %) and 4117 vs
3269 MiB peak (+848 MiB) (`scripts/bench.py`, RTX 4090, bf16 eager).

| key | source (`/home/korvin/apps/arch_research/docs/research/`, dated 2026-10-29) |
|---|---|
| AUD | `discriminator-independent-code-audit.md`: static audit of 6 traiNNer-redux + 6 neosr Ds, with a CPU probe |
| MOD | `discriminator-modern-design-review.md`: 2025-26 work and a critique of the hybrid design |
| SRC | `discriminator-primary-sources-2026.md`: 18 primary sources, transfer limits |
| CTD | `SISR_DISCRIMINATOR_DESIGN_2026_RU.md`: the "CTD-SR" design built on neosr `unet` |

**Base of the documents:** neosr 36054fd (2026-10-28), 65 commits behind HEAD. That base predates the R3GAN
discriminator (a71a7a4), the R3GAN penalty path (9c4cca2: per-position R1/R2, bf16 pass, gamma cosine) and the lazy
registries. The docs' probe ran on Windows with torch 2.7.1 CPU. The neosr rows in this review are re-checked on
torch 2.9.1 (CPU, `CUDA_VISIBLE_DEVICES=`). The traiNNer-redux claims are not re-checked: it is a competitor, used
for ideas only.
**Labels:** *fact* = read in the code (path:line) or measured here (probes P1-P3, section 2.3); *judgement* = this
review's opinion, marked as such.

## 1. The documents

**AUD (code audit).**
- neosr `unet` is the Real-ESRGAN SN U-Net. The stem and the last conv have no SN, and the skip additions need H and W to be multiples of 8.
- `patchgan`: SN by default, no norm, no sigmoid. `use_sigmoid` together with a logit loss is a wrong pair.
- `ea2fpn`: the trainable pretrained ResNet18 keeps its BN layers and gets no ImageNet normalisation. It is hard-wired to 6 output channels, and `l2_norm` has no eps.
- `dunet` and `ea2fpn` use DySample (`grid_sample`). Their R1 double backward fails.
- `metagan`: stride 64, NCHW output, `n_class` honoured, partial SN. `resgan`: `.squeeze()` changes the output rank at B=1.
- neosr: a D with BN gets separate real/fake forwards; the penalty runs in an fp32 eager pass; `AdvView` switches D's input to SWT high-frequency bands.
- traiNNer-redux only: feature matching calls a method that no D has, and its "R3GAN" loss is RaGAN-hinge.
- Proposal: compare one dense D (unet) and one local D (patchgan) with the same crop, loss and view; try `metagan` as the context candidate.

**MOD (design review).**
- 2025-26 precedents: SFD (CLIP feature D), FSGAN (region and global semantic D), StrSR (CLIP-ConvNeXt multi-level patch heads plus a frequency loss, approximate R1), HYPIR++ (latent D), FA-PWSR and WGSR (wavelet HF D).
- Proposed carrier: the SN U-Net plus one coarse head, with no full FPN.
- HF critic: signed SWT bands **plus low-pass context**. An independent HF D only matches marginal distributions, so it is an ablation, not the design.
- LR conditioning: the same condition for real and fake, else D finds a pipeline shortcut. Semantic branch later, on whole objects.
- Anti-aliased downsampling is a hypothesis to A/B, not a fix.
- Loss contract: apply the loss per logit, then average (`loss(mean)` is a different functional). R1 of a mean logit is not the mean of per-patch R1 norms. Test exact RpGAN + R1/R2 with a scalar head first. SN + R1/R2 together may over-constrain D.
- A frozen extractor must stay differentiable on the G step: no `no_grad` on the fake path.
- Ablation order: U-Net, coarse head, independent HF, context HF, LR conditioning, semantic branch.

**SRC (primary sources).**
- Frequency: D's weakness is small-amplitude components and lossy downsampling, not "blindness" to high frequencies (S01). BlurPool (S02) helps against aliasing but costs detail.
- Pretrained features in D: Projected GAN, Vision-aided GAN, SeD (an HR condition shared by real and fake), SFD, DINOv3.
- SR uses of pretrained features: OMGSR (DINOv3-ConvNeXt multi-level heads), TUDSR (DINOv3-ViT layers 3/6/9; BCE with real label 0.8), StrSR, HiFiPix (wavelet heads on DINOv3 features; fidelity up, MUSIQ/MANIQA down).
- Objectives: relativistic variants differ from each other. R3GAN = RpGAN + R1 + R2 (formula given). Its guarantee does not carry over to RaGAN, R1-only or arbitrary patch reductions.
- Semantic + wavelet D already exists (WDDSR 2025, HiFiPix 2026), so the combination is not new.
- Evaluation: the perception-distortion trade-off; hallucination scores; structure errors judged apart from texture.
- Order: fix a baseline D, then downsampling, the frequency branch and the frozen semantic branch; loss changes as a separate series.

**CTD (CTD-SR design).**
- Probe table of 12 Ds (parameters, SN layers, output shape). The neosr rows match P1.
- Base: neosr `unet`. Donors: MetaGan2 blocks, `metagan` late attention, `ea2fpn` scale fusion. PatchGAN is the mandatory simple baseline.
- **Core** = U-Net pixel head + coarse head on E3 (/8) + HF critic. The HF critic takes the 2-level signed Haar SWT of Y plus low-pass RGB (9 channels) through a small SN CNN and outputs at /2.
- **Context** = Core + a shared LR condition fused at E3 + a frozen semantic condition + an attention context head with output `[B,1]`.
- Output contract: a dict of raw-logit heads. Loss per head, with fixed weights that sum to 1 (0.5/0.25/0.25).
- R3GAN is a separate axis, tested with a scalar critic. For a multi-head D, R1 must be defined anew.
- neosr integration: `scores()` expects one tensor and `chunk(2)`; a condition must enter as `cat([c, c])`; do not mix with the old RGB-to-HF switch.
- Plan: 15 ablation arms (B0-B4, N1, A1-A5b, X1-X3, L0/L1), at least 3 seeds, blind tests.

## 2. Verification against the code

### 2.1 Claims about neosr
| # | claim (source) | status | evidence |
|---|---|---|---|
| V1 | `unet`: SN on 8 inner convs, not on the stem or the last conv; sizes must be multiples of 8 (AUD, CTD) | holds | `unet_arch.py:26,38` (no SN), `:28-37`, `:48-62` (`scale_factor=2` + add); templates use GT 256 |
| V2 | `patchgan` defaults; `use_sigmoid` + `bce` is wrong (AUD) | holds; `rpgan`/`hinge` take logits too | `patchgan_arch.py:60-69,140-141`; `gan_loss.py:94,245,249` |
| V3 | DySample Ds have no double backward for R1/R2 (AUD, CTD) | **holds on torch 2.9.1** (P1); nothing refuses the config, so it crashes at the first regularised update | `arch_util.py:162`; `dunet_arch.py:17`; `gan_loss.py:179-186` checks BatchNorm only |
| V4 | `ea2fpn`: 6 channels hard-wired, trainable pretrained BN, no ImageNet norm, `l2_norm` without eps (AUD, CTD) | holds (`class_num` not re-probed) | `ea2fpn_arch.py:12-15,201,208-210,248-251` |
| V5 | `metagan`: /64, NCHW `[B,n_class,h,w]`, RMSNorm, GroupNorm in `Down`, DropPath 0.02, head Dropout 0.2 (AUD, CTD) | holds | `metagan_arch.py:120-124,155-159,179-182` |
| V6 | `metagan` has "partial SN" (AUD, CTD) | **incomplete**: the GroupNorm after the SN conv is scale invariant, so the SN has no Lipschitz effect (A-04) | `metagan_arch.py:123`; P3 |
| V7 | `resgan.squeeze()`: shape `[1]` at B=1, `[B,1]` otherwise; harmless in the D step (AUD, CTD) | holds | `resgan_arch.py:63`; `gan_loss.py:36,189` |
| V8 | `rpgan` = pairwise `softplus(f - r)`, per position for maps (AUD, CTD) | holds | `gan_loss.py:150-154` (G), `:245` (D) |
| V9 | the penalty runs in an fp32 eager pass (AUD, CTD section 5) | **outdated**: eager D and MATH SDPA still hold, but the pass now follows a bf16 phase; fp32 only with `penalty_fp32` or an fp16/fp32 phase | `gan_loss.py:21-23,200-216`; `r3gan/RESULTS.md` section 3 (bf16 PASS) |
| V10 | the penalty is R1 of a per-sample spatial mean; per-patch R1 is "much more expensive" (CTD sections 9.2 and 13, MOD) | **outdated**: neosr penalises the per-position mean through a Rademacher probe, unbiased with added variance, at one backward's cost | `gan_loss.py:31-39,225-229`; `r3gan/DESIGN.md` section 1.2 item 2 |
| V11 | a D with BN gets separate real/fake forwards (AUD, CTD) | holds; R1/R2 on a BN D is now a `ConfigError` | `gan_loss.py:16-18,179-188` |
| V12 | `AdvView` replaces RGB with 3 sym19 SWT high-frequency bands of Y; LL and colour are gone (AUD, CTD) | holds; hard switch at `wavelet_init` (80000 in 83 templates) = ST-14 | `wavelet_guided.py:8-49`; `losses/__init__.py:64-65` |
| V13 | the loss is applied per logit and then averaged (MOD, CTD) | holds in neosr | `gan_loss.py:145,245,249-250` |
| V14 | neosr has no exact RpGAN + R1/R2 scalar critic (CTD 9.2 points to traiNNer-redux) | **outdated**: `r3gan`, `r3gan_scalar128` and `r3gan_scalar256`, with `rpgan`, R1/R2 and a gamma cosine, and a template | `r3gan_arch.py:82-152`; `gan_loss.py:65-74`; `options/train_realplksr_r3gan.toml:105-114` |
| V15 | filtered downsampling is absent (CTD X1, MOD, SRC S02) | partly outdated: the `r3gan` `Down` is a [1,3,3,1] FIR before stride 2 | `r3gan_arch.py:67-79` |
| V16 | `scores()` takes one tensor and `chunk(2)`; multi-head outputs and a condition need new plumbing (CTD section 12) | holds; `Context` already carries `lq` | `gan_loss.py:176-189`; `image.py:326-334` |
| V17 | a real-label smoothing arm (TUDSR uses 0.8) needs code (SRC, implied) | not needed: `bce` takes `real_label_val` | `gan_loss.py:57-59,127` |
| V18 | DySample builds its coordinates in at least fp32 under autocast (CTD 4.10) | holds | `arch_util.py:138` |
| V19 | "six neosr discriminators" (all four) | outdated: 7 registered (`r3gan` family added) | `@ARCH_REGISTRY.register` in `unet_arch.py:9`, `patchgan_arch.py:34`, `ea2fpn_arch.py:183`, `dunet_arch.py:22`, `metagan_arch.py:149`, `resgan_arch.py:8`, `r3gan_arch.py:82,143,149` |

### 2.2 What the documents miss (facts)
- **M1, `metagan`'s gradient to G.** The SN in `Down` is cancelled (V6, P3). At init, `||dBCE(D(sr),1)/dsr||` is 5.4e-2 for `metagan` against 1.8e-4 for `unet`, about 300x (P1). A-04 measured 250-800x. The templates run it at GAN weight 0.3 (`options/train_realplksr.toml`, `[train.gan_opt]`).
- **M2, cost.** `metagan`'s forward is 807 aten ops; `unet` is 170 and `r3gan` 79 (P1). The GPU profile gives `metagan` 52 % of the kernel launches and 27 % of the step (`stage2/analysis/gpu_profile.md:10-13`). None of the four documents measures cost.
- **M3, the R1/R2 penalty-pass cost (R4: ~614 ms per regularised update).** The double backward of a grouped conv runs one iteration per group (1 conv + 2 slices each, P2).
  - The `r3gan` R1 pass has 572 such iterations. 480 come from the depthwise FIR (groups = C = 32+64+128+256), and 92 from the grouped 3x3 convs (groups 4-32).
  - Running the FIR as one `groups=1` conv over `(B*C, 1, H, W)` gives the same math (parameter gradients equal within 1.7e-7) and removes 84 % of these iterations.
  - The CUDA milliseconds are not re-measured.
- **M4, R1/R2 on `metagan` is costly.** Its R1 pass is 8,958 aten ops, against 466 for `unet` (P1). Its depthwise 7x7 `DConv` (groups 48/96) takes the same per-group path; this is inferred from the op histogram, not isolated.
- **M5, D runs in train mode in G's adversarial pass** (`gan_loss.py:148`). `metagan`'s Dropout(0.2) and DropPath therefore act on the adversarial signal G receives, and also in the penalty pass.
- **M6, the D dynamics are already measured.** D saturates by ~300 updates (logits about +-8.8) and is at chance by ~1.7-2.4k (`gpu_profile.md:204-212`, GP-06). A GAN start with a ramp gave no gain at 8k (`stage2/DECISIONS.md:47-48`).
- **M7, a prior decision exists.** The stage-3 competitor review adopted only the R3GAN scalar D from traiNNer-redux. MetaGAN2/3, MUNet, VGGStyle and multiscale PatchGAN were skipped (`stage3/REVIEW.md:19`). The CTD donor list re-proposes MetaGan2 blocks.

### 2.3 CPU probes (torch 2.9.1, fp32, train mode; scratch scripts, not committed)
**P1**: each D at defaults, B 4, 256x256 uniform input. The aten op count is a proxy for CUDA launches, not for time. The gradient column is one batch at init weights: an order of magnitude only.

| D | params (trainable) | SN layers | output | aten ops fwd / bwd | R1 pass (`probe` + double backward) | `\|\|dBCE/dsr\|\|` at init |
|---|---|---|---|---|---|---|
| unet | 4.377 M | 8 | [4,1,256,256] | 170 / 152 | ok, 466 ops | 1.8e-4 |
| patchgan | 2.765 M | 5 | [4,1,30,30] | 98 / 94 | ok, 264 ops | 7.7e-4 |
| metagan | 6.017 M | 8 | [4,1,4,4] | 807 / 1,556 | ok, 8,958 ops | 5.4e-2 |
| dunet | 3.231 M | 8 | [4,1,256,256] | 276 / 229 | **fails**: `grid_sampler_2d_backward` has no derivative | 2.3e-4 |
| resgan | 0.215 M | 4 | [4,1] | 88 / 96 | ok, 263 ops | 2.2e-3 |
| r3gan (patch) | 0.869 M | 0 | [4,1,16,16] | 79 / 155 | ok, 2,190 ops | 8.9e-4 |
| r3gan_scalar256 | 1.546 M | 0 | [4,1,1,1] | 110 / 212 | ok, 4,849 ops | 2.6e-3 |
| ea2fpn | 12.232 M (11.710 M) | 10 | [4,6,256,256] | 824 / 874 | **fails**: same error | 4.9e-1 |

**P2**: the `r3gan` R1 pass at B 4, 128x128.

| FIR form | aten ops | slice | conv | CPU time |
|---|---|---|---|---|
| depthwise (today) | 2,221 | 1,144 | 635 | 268 ms |
| one `groups=1` conv on `(B*C,1,H,W)` | 817 | 188 | 159 | 231 ms |

The parameter gradients of the two forms differ by at most 1.7e-7 (relative). In a single-layer test the forward is bitwise equal and the op count drops from 232 to 48 at C=64. The CPU is compute-bound, so the time gap there says little about CUDA.

**P3**: `metagan` `Down(48 -> 96)`. Multiplying `weight_orig` by 10 changes the output by 5e-7 (relative): GroupNorm removes the scale the SN sets. SN and no-SN differ by 4.7 % before GroupNorm, and only through the conv bias.

## 3. Candidate tasks
Effect: Q = quality, St = stability, Sp = speed. GPU: 0 = none, S = one measurement up to 15 min, A8 = 8k-update A/B
arm(s) from the R6 pretrain (~15-25 min each), L = hours. "Exact" = ships as a default behind a parity test;
"opt-in" = waits for an A/B.

| id | what | effect | evidence | files | size | GPU | kind | risk | depends on |
|---|---|---|---|---|---|---|---|---|---|
| DR-01 | Penalty pass: run the `r3gan` `Down` FIR as one `groups=1` conv over `(B*C,1,H,W)`; then profile one regularised update on CUDA and `bench.py` the R1 template. If the grouped 3x3 (92 iterations) still dominates, record the alternatives and the `reg_every` 4/16 costs | Sp: R1/R2 becomes affordable | strong for the op count (P2, M3); CUDA ms unknown | `r3gan_arch.py:67-79`, tests (parity, CPU + CUDA double backward) | S | S | exact | a channels_last forward copies on reshape; the 614 ms may partly sit elsewhere | none (= `next_tasks.md` rank 3, lane P) |
| DR-02 | GAN health log, on log updates only: `\|\|d(w * l_g_gan)/dsr\|\|` and its ratio to the other G terms (one extra `autograd.grad` over `sr`, no RNG, no buffer writes), plus D's sign accuracy and mean `\|logit\|` | St: makes every A/B arm readable | A-04 recommendation 1 still open (`CLOSURE.md:239`; `archs_cnn.md:200-201`); `scripts/term_grads.py` is offline only | `gan_loss.py`, the log cadence in `image.py`, tests | S-M | 0 (+S overhead check) | exact (logs only) | the extra D backward costs time on log updates | none |
| DR-03 | A-04 screen at 8k (config only): `metagan` @0.3 (A0, shared with `next_tasks` G2) vs `metagan` @0.03 vs `unet` @0.1 | Q/St; Sp for `unet` (170 vs 807 fwd ops) | medium: synthetic probe + P1 (~300x at init) + the GP-06 trajectory; no quality data yet | configs only | 0 | A8 x2 | opt-in | 8k is a screen; the A-04 rule asks for >= 20k | DR-02 helps |
| DR-04 | R3GAN screen at 8k: `R1_patch_g1` (`r3gan` + `rpgan` + R1/R2, gamma 1) vs A0 and A1 (no D) | Q/St (GP-06) | medium: the R4 2k run kept D off chance (margin 0.08) and bf16 passed; no A1 control at 2k | `options/train_realplksr_r3gan.toml` | 0 | A8 x1-2; fits a window only after DR-01, or with `reg_every = 16` (0.14 s/update, ~19 min) | opt-in | 8k vs R6's 20k; the clip rate (D 50 %, G 90 %) is a confounder | DR-01 |
| DR-05 | HF view from the GAN start: `wavelet_init = 0` vs the template (RGB until 80k) at 8k | Q: does `wavelet_guided` (82 templates) help at all; ST-14 (its "no switch" alternative) | weak: WGSR-type precedent, no neosr A/B | configs only | 0 | A8 x1 | opt-in | the view drops LL and colour (V12): D cannot see colour casts | none |
| DR-06 | Bounded or softened D objective on the template: `gan_type = "hinge"`, or `bce` with `real_label_val = 0.9` | St (GP-06 logit runaway) | weak-medium | configs only | 0 | A8 x1-2 | opt-in | label smoothing is close to "damping" (owner rule); hinge keeps `metagan`'s input gradient | DR-02 |
| DR-07 | Fail fast: refuse `r1_gamma`/`r2_gamma` > 0 for a D without a double backward (`dunet`, `ea2fpn`), as is already done for BatchNorm | St: a `ConfigError` instead of a crash at update `reg_every` | strong (V3, P1) | `gan_loss.py:179-186`, a flag on the DySample Ds (or a one-time tiny probe), tests | S | 0 | exact (it refuses a config that cannot run) | a flag list can go stale | none |
| DR-08 | `metagan` opt-in `Down` without GroupNorm, so the SN acts (A-04's code item) | St | medium: A-04 probe, init gradient 27x lower, but D learns slowly | `metagan_arch.py:120-146` | S | A8 x1 | opt-in | it changes D's learning speed; arch-specific | DR-03; the owner's pick (`CLOSURE.md:239`) |
| DR-09 | R1 on the template D (`reg_every = 16`, small gamma) as a direct bound on D's input gradient | St (A-04, GP-06) | weak | configs (+ maybe `DConv`) | 0-S | S (penalty cost, M4) + A8 | opt-in | the interplay of R1 with the cancelled SN is untested; cost | DR-01 findings, DR-02 |
| DR-10 | Joint view: an `AdvView` mode that feeds `cat([RGB, LH, HL, HH])` from the GAN start (D `in_ch` 6), giving HF its context (MOD, CTD 8.3) and removing the switch | Q/St (ST-14) | weak: hypothesis; precedents are other setups | `wavelet_guided.py`, `gan_loss.py`, `config.SCHEMA` + Studio surface | M | A8 x1-2 | opt-in | band scaling vs RGB | DR-05 positive |
| DR-11 | Multi-head D contract: D may return a list or dict of logit maps; loss per head with fixed weights summing to 1; per-head probe for R1/R2; per-head logs | enabler | design only | `gan_loss.py` (`scores`, `adv_loss`, `probe`), tests | M | 0 | exact for single-tensor Ds | API surface for unproven Ds | the owner wants new Ds |
| DR-12 | LR-conditioned D: `D(x, lq)` with `cat([c, c])`; the condition is held fixed in R1/R2 | Q (fidelity) | weak | `gan_loss.py`, the D signature | M-L | A8 x2+ | opt-in | shortcut cues if the condition differs between real and fake | DR-11 |
| DR-13 | Projected D head on the DINOv2 features FDL already computes | Q | weak-medium (Projected GAN, OMGSR, TUDSR; not this G) | new loss/arch, `fdl_loss` | L | S + A8 xn | opt-in | VRAM; D too strong; G's gradient through the backbone | DR-11 |
| DR-14 | CTD-SR Core as a new D (U-Net + coarse head + context HF critic) | Q | hypothesis | new arch | L | L (15-arm plan) | opt-in | many confounders | DR-11, DR-05/DR-10 |
| DR-15 | D frequency-response probe tool: sinusoids, noise and grids; score and input gradient vs frequency and amplitude (SRC S01, CTD 11.3) | diagnostic | literature only | `scripts/` | M | 0-S | exact (a tool) | low payoff on its own | none |
| DR-16 | Arch contract fixes: `resgan` `flatten(1)`; `ea2fpn` eps and channel wiring; a `patchgan` sigmoid guard; odd sizes in `unet`/`dunet` | St (edge cases) | holds (V2, V4, V7) | those arch files | S each | 0 | exact | none | deferred: arch-specific (owner, 2026-11-10) |

## 4. Mapping to the open items
| open item | what the documents add | candidates | judgement |
|---|---|---|---|
| A-04 (`metagan`: SN cancelled, gradient 250-800x, 85 templates at 0.3) | nothing on GroupNorm vs SN; they call `metagan` "partially SN" and a context donor | DR-02, then DR-03, then DR-08 or DR-09 | the documents do not change A-04; DR-03 screens the arms `CLOSURE.md` already names, at 8k |
| R3GAN cost gate (719 vs 105 ms) | none: no cost data; their probe only checks that the double backward succeeds | DR-01, then DR-04 | the CPU evidence points at the per-group double-backward loop, 84 % of it in the FIR, which has an exact rewrite |
| GP-06 (D saturates, then sits at chance) | label smoothing (TUDSR), exact RpGAN + R1/R2 (SRC S09), caution on SN + R1 (MOD) | DR-04, DR-06, DR-09, DR-02 | the GAN start and ramp are already A/B'd with no gain at 8k (M6) |
| ST-14 (abrupt RGB-to-HF switch at 80k) | "HF in context, not instead of RGB" (MOD, CTD 8.3); "do not combine with the old switch" (CTD section 12) | DR-05, DR-10 | DR-05 is config only and is one of ST-14's listed alternatives; DR-10 only if DR-05 shows that HF helps |

## 5. Shortlist for a session with 15-25 minute GPU windows
| rank | task | GPU | why (judgement) |
|---|---|---|---|
| 1 | DR-01 penalty-pass FIR rewrite + CUDA profile | S | exact; it unblocks every R1/R2 arm (R6, DR-04, DR-09), and the CPU already shows the cause and an exact fix for 84 % of the loop |
| 2 | DR-02 GAN health log | 0 (+S) | exact; each 8k arm then explains its result (adversarial share, saturation) instead of giving only a metric delta |
| 3 | DR-03 A-04 screen (`metagan` @0.03, `unet` @0.1 vs A0) | A8 x2 | config only; it screens the arms A-04 names, and the `unet` arm also prices the 27 % D cost |
| 4 | DR-04 R3GAN screen vs A0/A1 | A8 x1 | the config exists; it gives the first R3GAN quality signal once DR-01 lands (or with `reg_every = 16`) |
| 5 | DR-05 HF view from the GAN start | A8 x1 | config only; the first evidence for or against `wavelet_guided` (82 templates) and ST-14's no-switch option |

**Defer:**
- DR-06: wait until DR-02 shows the saturation persists, and the owner rules on label smoothing.
- DR-07: needs no GPU, so it fits any CPU lane; low urgency, since no template pairs DySample with R1.
- DR-08: the owner's pick after DR-03.
- DR-09: needs DR-01's penalty-cost findings.
- DR-10: only after DR-05.
- DR-11 to DR-14: new D features; they rank after the core and Studio in the owner's priorities.
- DR-15: a tool with no consumer yet.
- DR-16: arch-specific, deferred by the owner's goal.

## 6. Open questions for the owner
1. Is the template D (`metagan`, 85 templates) framework, in scope now, or an arch, ranked last? This decides DR-08 and DR-09 and any `DConv` penalty work.
2. May 8k screens (DR-03, DR-04, DR-05) prune arms before the >= 20k runs that A-04, GP-06 and R6 require? Or do they not count at all?
3. Are label smoothing and hinge acceptable A/B arms under "stability from correctness, never damping"?
4. Are new D features (the multi-head contract, an LR-conditioned D, a DINOv2 projected head, CTD-SR) wanted at all before Studio? The documents propose them; nothing in `STATUS.md` asks for them.
5. Should ST-14 start with DR-05 (config only) before any fade code?
