# R3GAN in neosr: design (Fable 5.1 synthesis, 2026-10-28; critique round: Opus approved with changes, applied; Codex pending)

Inputs: `audit.md` (Opus: the owner's implementations vs paper and reference), `neosr_fit.md` (Opus: neosr side,
efficiency, A/B plan), `codex.md` (Codex Astra: independent review + efficiency addendum). Disputes settled by evidence
(probes and code), listed in section 3. Evidence lives in the three reports; this file holds the decisions.

## 0. Verdict
- **Nothing of the owner's R3GAN code is ported as code.** The loss that actually trained in traiNNer (`alternative/`) is
  relativistic-average hinge with lazy penalties on random crops and a fallback that would silently disable R1/R2; every
  logged run also trained with the GAN weight squared (0.1 was 0.01). True pairwise R3GAN was never run. The minimal
  `v2` loss is the reference math but keeps a Python counter and crashes on scalar outputs.
- **The best R3GAN loss implementation available is neosr's own `gan_loss.py`** (`gan_type="rpgan"` + R1/R2): pairwise
  logistic, one concatenated forward serving loss and penalties, per-sample penalty, `backward(inputs=...)`-safe,
  schedule by `ctx.it`. R3GAN = that path at `reg_every = 1` plus a new discriminator, three penalty-path fixes and a
  gamma schedule. No new loss class.
- **The owner's `RpGANDiscriminatorScalar` is the base for the D**: a faithful port of the reference block mechanics
  (fix-up init, zero-init last conv, FIR, no normalization). Rewritten, not copied: group width 16 (3.2x fewer FLOPs than
  its dense high-resolution stages), FIR as a grouped conv, no StyleGAN ops, no `stage_dtype`, one head rule, 4-D output.
- **The incumbent template GAN is inert.** Stage-2 logs: metagan saturates to logits +-8.7 by update 350, then collapses to
  chance (`l_d = 2 ln 2`, logits ~0, `l_g_gan = 0.3 ln 2`) by ~2.4k updates (`stage2/analysis/gpu_profile.md:204-213`,
  Q8 run `train.jsonl`). The GAN templates have in effect trained without a GAN. The A/B therefore carries a no-GAN control,
  and R3GAN is the candidate fix, not an optional extra.

## 1. What gets built (all default-preserving; `bce` + metagan stay the template default until the A/B wins)
### 1.1 `neosr/archs/r3gan_arch.py`, class `r3gan` (~110 lines), `ARCHS` rows `r3gan`, `r3gan_scalar128`, `r3gan_scalar256`
`r3gan(in_ch=3, widths=(32,64,128,256,256), group_width=16, expansion=2, blocks=1, head="patch")`: 1x1 stem (no act);
per stage `blocks` residual blocks `x + lin3(act(lin2(act(lin1(x)))))` with 1x1 expand, grouped 3x3 (groups =
`e*c // group_width`), 1x1 project zero-initialised; biased leaky ReLU 0.2 in place (forward gain 1; `sqrt(2/(1+0.04))`
and `(blocks*n)^(-1/4)` are init gains only); FIR `[1,3,3,1]/8` outer-product stride-2 depthwise conv (non-persistent
buffer) + 1x1 projection between stages. Heads: `patch` = 1x1 conv to one logit per position (stride 16, RF 108 px);
`scalar` = exact 4x4 depthwise basis + 1x1 (paper head), `ValueError` naming the right GT size if the final map is not 4x4
(no silent pooling). Plain state-dict keys, parameters only. 0.87 M params / 2.42 GMAC per image at GT 256 (metagan: 0.63
GMAC, 6 M params; owner defaults 7.68 GMAC; dunet 23 GMAC). `widths=(16,...)` halves activation memory if VRAM-bound.
### 1.2 `gan_loss.py` changes (inside `gan_loss`/`gan_d_loss`, ~25 lines)
1. `out.flatten(1)` -> `out.reshape(len(out), -1)`: a `(B,)` output no longer crashes the penalty.
2. **Per-position penalty for map outputs** (Rademacher probe): `score = sum_i sum_p eps_ip D_p(x_i) / sqrt(P)`, `eps` = +-1
   from the device RNG, one draw per regularised update. `E||grad score_i||^2 = sum_p ||grad D_p||^2`, so the penalty is
   the mean per-position `gamma/2 ||grad_x D_p||^2` in expectation, at the cost of today's single `autograd.grad`. P = 1
   draws nothing and is bitwise today's penalty. Replaces the current mean-logit penalty, which is 1/P weaker per position
   (measured 0.03-0.05x for the patch D, ~1/P for metagan) and makes gamma depend on the patch size. Documented as a
   design choice beyond the paper (which has scalar critics only); the scalar head is the paper-exact control.
3. **Penalty pass follows the phase autocast when it is bf16** (fp32 leaves, logits and squared norms accumulated in fp32);
   fp16 and no-AMP runs keep today's fp32 eager pass (the inner `autograd.grad` is not loss-scaled). Measured parity:
   per-sample penalty error <= 2.3 %, D-parameter gradient cosine 1.0000, norm ratio 0.99. Halves the live graph (4.0 GB ->
   1.55 GB with the in-place activation at GT 256, B 8) and doubles tensor throughput vs TF32 on Ada. `penalty_fp32: bool =
   False` forces the old path (the parity check and a fallback; no other knob).
4. **Gamma schedule as a pure function of `ctx.it`**: `gamma_decay` (final/initial ratio, default 1 = constant) and
   `gamma_iters` (cosine length from the first GAN update, default 0). No state, no sync, resume-exact.
5. Logs: raw per-sample squared gradient norms `r1`, `r2` (before gamma; their ratio is the paper's stability diagnostic),
   weighted `l_d_r1`, `l_d_r2` (`l_d_reg` kept as their sum).
6. Build-time checks through one generic hook on `Loss` (called once after nets and losses are built; no loop branch):
   a gamma > 0 with a batch-coupled D (BN) -> `ConfigError`; a gamma > 0 under fp16 AMP -> `ConfigError` ("use bf16": the
   paper reports fp16 cripples R3GAN training, and the inner grad is not loss-scaled); a 32 px double-backward probe through D
   (`autograd.grad(create_graph=True)` + backward on zeros) so a D without double backward (dunet: grid_sample) fails at
   build, not at the GAN start.
7. Validation: `gamma_decay` in [0,1], `gamma_iters >= 0`, a schedule with both gammas 0 is an error, gammas finite.
### 1.3 Recipe arm `options/train_realplksr_r3gan.toml` (opt-in)
`pretrain_network_g` from the non-GAN run (E5; D and optimizers fresh), `network_d.type = "r3gan"`, `[train.optim_d] type =
"adam", lr = 2e-4, betas = [0.0, 0.99], weight_decay = 0`, `gan_opt`: `rpgan`, `loss_weight = 0.3`, `r1_gamma = r2_gamma =
1` (per position; grid in section 4), `reg_every = 1`, `wavelet_guided = false` (the penalty acts on D's RGB input), template
`grad_clip` in round 1 (`clipped_d` is logged; a per-net clip arm follows only if D is clipped often). Not ported: beta2 0.9->0.99 schedule (fixed 0.99; serves from-scratch generation), ADA augmentation (paired data),
EMA half-life ramp (`ema_power` exists), lr decay inside the A/B horizon (E2 decides).

## 2. Efficiency (owner criterion)
| item | effect | exact? |
|---|---|---|
| group width 16 instead of the owner's dense high-res 3x3 | D FLOPs 7.68 -> 2.42 GMAC/img at GT 256 (3.2x), same memory | different model (the paper's choice) |
| FIR as grouped conv (not `reshape(B*C,1,H,W)`) | no layout copy under channels_last; same values | exact |
| in-place biased leaky ReLU | -39 % forward saved bytes, -24 % penalty graph | exact (gradgradcheck passes) |
| one `cat` forward for loss + both penalties (today) | 12 D-forward units per regularised D update (the minimal v2 loss: 12 with two forwards; traiNNer's alternative: 16-20) | exact for per-sample Ds |
| bf16 penalty pass | graph 4.0 -> 1.55 GB at GT 256 B 8; ~2x tensor throughput vs TF32 | 1-2 % per-sample error, gradient direction unchanged |
| every-step R1+R2 | D-related work 15 units vs 9 without penalty: +~230 GFLOP on a ~3 TFLOP update (+8 %; G fwd+bwd ~1.45 TFLOP, FDL ~1.3 TFLOP); FLOP-based estimate +0-10 ms on a ~100 ms step (bf16), +10-25 ms (fp32); the CPU wall ratio of a penalty update was 3.4x at a 2.0x FLOP ratio, so this is optimistic and `bench.py` decides | the reference objective |
| lazy `reg_every` 4/16 | average time only; peak VRAM unchanged (set by the regularised update); not exact under Adam + shared clip; paper: worse | measured option, not the recipe |
| launches | r3gan every step ~1010 aten ops per update vs metagan without penalty ~3400: less host time than today | - |
| real-logit reuse across phases | would need a phase-order change and a `Context` write for 6.7 % of D work; not bitwise | rejected |
| penalty on random crops, relativistic average, mean-logit penalty | change the objective; the first two are how traiNNer's loss got cheap | rejected |
All GPU numbers are estimates from FLOP/byte counts and `gpu_profile.md`; `scripts/bench.py` (A5) measures ms/update and
peak VRAM per arm before the quality A/B.

## 3. Disputes and decisions
| question | audit (Opus) | neosr_fit (Opus) | Codex | decision |
|---|---|---|---|---|
| head for SR | patch at 1/16, scalar option | patch first, scalar control | scalar first (reference-exact) | both heads in one arch; round 1 runs both (`r3gan` patch, `r3gan_scalar256`) |
| penalty on a map | `||grad sum_p D_p||^2 / P` (exact only for disjoint RFs) | Rademacher per-position estimator (unbiased; the deterministic sum is 2-3x too high on overlapping RFs) | scalarise before both terms, or define a local objective with its Jacobian penalty | Rademacher estimator: it is the local objective's penalty at one backward's cost; sum/P and mean-logit are biased |
| penalty precision | paper ran bf16 stages | bf16 when the phase is bf16, fp32 leaves | fp32 eager as the reference arm, bf16 mixed as a candidate | bf16-following with fp32 leaves/accumulation as default of this opt-in path; `penalty_fp32` for the parity check |
| lazy regularisation | every step; lazy = paper-rejected deviation | every step; lazy not exact under Adam/clip | every step first, benchmark 4 | every step in the recipe; lazy 4/16 measured in round 2 |
| gamma | /4 for [0,1] inputs: FFHQ-256's 150->15 is 37.5->3.75; patch toy at 10 nearly frozen | patch 10 per position, scalar 150/15 (toy anchors) | screen 0.1/1/10 scalar | grid: patch {1, 3} (+10 optional), scalar {40, 4} at GT 256 (paper values / 4); constant; cosine x0.1 as a round-2 arm |
| D size | gw 16 on owner widths (0.87 M, 4.83 GFLOP) | (32..256) gw 16, 1 block (0.87 M, 2.42 GMAC) | lean (16..256), 1 block (~1.0 M, 1.33 GFLOP) | `widths` kwarg; default (32..256); slim (16..256) only if bench.py exceeds the cost gate |
| D optimizer | Adam (0, 0.99); the paper calls a low initial beta2 crucial at lr 2e-4 | same, fixed beta2 | same; beta2 schedule optional | Adam (0, 0.99); a beta2 0.9 -> 0.99 warm-up only if round-1 logs show early margin runaway (then a 5-line per-update betas rule) |
| beta2 / EMA / ADA schedules | - | not ported | optional, phase-owned hooks | not ported |
| fail-fast | build-time tiny double-backward probe | ConfigError for batch-coupled D only | reject batch-coupled D | all three at build through one `Loss` hook: batch-coupled, fp16 AMP, 32 px double-backward probe |
| real-logit caching | - | no (6.7 %, order change) | no, initially | no |

## 4. A/B on the owner's data (one RTX 4090, one job at a time, memory-capped; ~15 GPU-hours over three rounds)
Round 0 (E0-lite): a fixed held-out split of ~300 tiles from the 10,607 (meta_info lists, never trained on), `val8` (8 real
photos, 512 px) and a fixed visual set (16 tiles: foliage, hair, fabric, text, faces, sky). Judges: TOPIQ-FR pipal, LPIPS,
DISTS (none is a training loss in these arms), PSNR for fidelity sanity; EMA weights every 2k updates.
Round 1 (20k GAN updates after one shared 20k non-GAN pretrain, seed 7): A0 template GAN (metagan, bce, adan_sf D) - confirms
the inert baseline; A1 no D; R1 `r3gan` patch gamma 1, `loss_weight` 0.3; R2 patch gamma 3; R3 patch gamma 1, `loss_weight`
0.1 (the owner's runs trained at an effective 0.007-0.01 and 0.01 already cost 0.17 dB by 7k); R4 `r3gan_scalar256` gamma 40
(4 if D freezes). A2 (metagan + rpgan, no penalty) moves to round 2 if A0 is confirmed inert. Each arm: `bench.py` ms/update and peak VRAM; logs `out_d_margin`, `l_d_r1/r2`, `grad_norm_d`,
`clipped_d`, skips; `scripts/term_grads.py` shares at 0/10k/20k.
Round 2 (best R): lazy 16 and 4; gamma cosine x0.1; D clip off (per-net clip, only if `clipped_d` was frequent); D lr 1e-4;
beta2 warm-up (only on early-margin evidence); A2; bf16 vs fp32 penalty by a 10-batch parity check (no run).
Round 3: top two arms + A0 with a second seed; the best arm past 80k updates.
Acceptance (replace the template GAN only if all hold): two seeds and the long run agree; TOPIQ-pipal and LPIPS better than A0
by more than A0's seed spread, DISTS not worse; better than A1 (the GAN must do something); PSNR within 0.3 dB of A0; owner's
visual sign-off (no checkerboards, ringing, colour shifts, false texture on faces/text); D margin bounded and above chance in
most windows, penalties finite, no skipped updates; cost within +30 % ms/update and +1.5 GB peak vs A0 under `bench.py`, or
the owner accepts the cost.
Note: the owner's LQ is a clean area-like downscale of real-photo GT; real degradations enter with Phase B (OTF realism).

## 5. Risks
Inert baseline (every arm is in effect vs no GAN: controls A0/A1 settle it); gamma is D-, head- and size-specific (grid);
cuDNN speed of group-width-16 NHWC bf16 convs and their double backward unmeasured (bench.py first); the Rademacher penalty is
beyond the paper (scalar control arm); behaviour changes of an opt-in never-A/B'd path (map penalty meaning, penalty dtype:
record in DECISIONS); DDP + compile + double backward only CPU-tested (owner trains on one GPU); resume exactness of the probe draw on multi-GPU
needs every rank's RNG state saved (today rank 0 only) or a draw seeded by (seed, it, rank).

## 6. Tests
`neosr_fit.md` section 7 (20 tests) is the list: golden keys/param counts/shapes, zero-init and linearity at init, FIR equivalence,
gradgradcheck (in-place act), compiled-vs-eager and `eager()` on regularised updates, pairing order (batch and position),
penalty on known functions (disjoint RFs exact; exhaustive sign patterns on overlapping ones), P = 1 bitwise today's, gammas 0
bitwise plain, lazy k-scaling, gamma schedule values and validation, bf16 vs fp32 parity, double backward under MATH SDPA incl.
the documented dunet refusal, batch-coupled ConfigError, log keys, `(B,)` output, resume parity across a schedule and RNG draws,
DDP 2-rank gloo, Adam (0, 0.99) build/resume + template sweep; CUDA: bf16 channels_last double backward finite, 10-batch parity,
bench.py numbers.
