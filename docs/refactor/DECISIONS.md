# Decisions

## Owner decisions (2026-10-20)
1. Delete legacy model types `default`, `simple_otf`, `eqr_otf`, `pancam_otf` and their exclusive datasets/archs
   (`pcpaired`, `pancamotf`, `pancamsingle`, `eqrotf`, `simple_otf` datasets, `otfgray` empty file, `pancam`/`eqrsrgan` archs).
   Only `image` and `otf` remain (used by all configs in `options/`).
2. Aether: register `aether_small/medium/large` AND make quantization-aware training (QAT) work end to end.
3. Git: branch `refactor/stabilize`, one commit per verified step. Never stage the owner's uncommitted files in the main
   checkout (`.codex/*`, `AGENTS.md` deletion, `info.txt`, `install.sh`, `options/train_esrgan2.toml`).
4. Old checkpoints: old `net_g/net_d *.pth` must load as pretrained weights; old `.state` resume is best effort;
   the new checkpoint format is the source of truth.
5. Loss pipeline requirement: every loss takes the data it needs from a shared context itself (and uses a model such as
   the discriminator when needed); no if-chains or per-loss `if/isinstance` branching in the training loop.

## Consensus decisions
See `analysis/consensus/*.md` (verdict tables) and `design/T*.md` (architecture). Summary of settled points is kept in
`PLAN.md`; disputed-then-resolved items are marked `majority` or `judge` in the consensus tables.

Notable refutations (do NOT "fix"):
- EMA `AveragedModel(use_buffers=False)` already copies live buffers (BN stats, spectral-norm u/v) on every update (F.1).
- `soap_sf.py` `parse_options` call lives inside a string literal and never runs (F.14).
- DropPath re-seeding does not clone weights in swinir/dat (they re-init after construction) but DOES in metagan (F.3).

## Competitor policy (owner, 2026-10-20)
traiNNer-redux-dev (/home/korvin/apps/traiNNer-redux-dev) is a newer sibling project: check it for already-fixed bugs, better
implementations and useful features in every step; adapt and improve what is taken, never port bad code as-is.
Port candidates per area: `analysis/competitor/H1..H4.md`.
6. DISTS (owner, 2026-10-20): restore ImageNet input normalisation for BOTH the dists training loss and the validation metric
   (correct per paper and competitor; metric values shift vs old logs). Apply after Phase L1 (L1 was told to leave it) — G5.M4.
7. Model allocation (owner, 2026-10-20): Fable only for strategic/global/conceptual questions at effort high/medium (limits);
   Opus implements (algorithms, optimisation, math); Codex gpt-6-astra high = code reviewer + advisor for every step.
8. Session 2 (2026-10-21) — design open questions resolved (owner answered #a; the rest take the designs' recommended defaults):
   a. `logger.keep_states = 3` by default (0 = keep all); exports in models/ never pruned (owner).
   b. `train.max_skip_streak = 10`; the live adversarial G term ships without a legacy flag (T1 Q1, Q2).
   c. DDP is best-effort: gloo-tested on 1 GPU, NCCL untested; DataParallel removed (T1 Q3, T2 D5).
   d. Stochastic resume continuation (not order-exact); `.state` not accepted as pretrain; same-shape net option change on
      resume = warning; EMA keeps averaging the schedule-free y sequence (T2 Q1, Q3, Q4, Q5).
   e. Unknown config keys warn with did-you-mean (not an error); convert.py CPU-only, scale from -s (T3 Q2, Q3).
   f. Loader defaults pin_memory/persistent_workers on (already landed in 9360403; T3 Q1).
9. EMA warmup (session 2): new optional `train.ema_power` (ema_pytorch-style decay ramp, competitor recipe 0.75); code default = constant
   decay (backward compatible); shipped templates get `ema_power = 0.75` next to `ema = 0.999`. Evidence: acceptance runs, EMA PSNR 6.55
   vs live 23.51 at update 1000.

## Owner decisions, 2026-10-28 (stage 3 goals)
- 1x restoration is a goal: Phase A item A7 (NAFNet) and `psnr_loss` stay.
- Real-world photographs are the main target (not anime or synthetic images): Phase B OTF realism is approved in
  principle, after the R3GAN track below.
- R3GAN track first: deeply and critically analyse the owner's R3GAN experiments in `/home/korvin/apps/r3gan/` (three loss
  variants, three traiNNer `sr_model` integrations, the paper) plus traiNNer's `r3gan_loss.py` and
  `rpgan_r3gan_scalar_discriminator_arch.py`, with Opus 5.5 and Codex Astra; pick the most workable base, fix and improve
  it, migrate a complete, correct R3GAN (loss + discriminator + recipe) into neosr and test it on the owner's real data.
  Docs: `stage3/r3gan/`.

## R3GAN loss path, 2026-10-28 (stage 3 R2; design `stage3/r3gan/DESIGN.md` 1.2, approved by Opus and Codex)
Two deliberate behaviour changes of the opt-in R1/R2 path (`train.gan_opt.r1_gamma`/`r2_gamma` > 0; no shipped template
sets them, so default runs are unchanged):
1. **Map Ds: the penalty is per position.** Before: `||grad_x mean_p D_p(x)||^2` (the mean logit). Against the per-position
   mean its ratio is `sum_pq A_pq / (P tr A)` with `A_pq = <grad D_p, grad D_q>`: 1/P when the position gradients are
   orthogonal (disjoint receptive fields), 1 when they are identical, ~0 when they cancel; measured 0.03-0.05x for the
   r3gan patch D and ~1/P for metagan, so gamma depended on the patch size and the D. Now each output
   position is a critic: `gamma/2 * mean_p ||grad_x D_p(x)||^2` per sample, estimated with one backward through
   `sum_p eps_p D_p / sqrt(P)` (fresh Rademacher signs per sample, position, side and micro-batch from the device RNG;
   unbiased, extra variance). A scalar D (P = 1) draws nothing and is bitwise the old penalty. Old configs with a map D
   and a gamma get a stronger penalty at the same gamma; no switch back (one knob fewer; Codex's opt-in proposal not taken).
2. **The penalty pass follows a bf16 phase.** Before: always fp32 eager. Now: bf16 when the phase autocasts to bf16 (fp32
   input leaves, squared norms in fp32), fp32 for fp16 and no-AMP runs (fp16's inner `autograd.grad` is not loss-scaled);
   `penalty_fp32 = true` forces fp32. CPU evidence: norms within 3 %, gradient cosine > 0.999; recipe default only after the
   CUDA qualification (R4).
Also new: `gamma_decay`/`gamma_iters` cosine (pure function of the update), raw `r1`/`r2` + weighted `l_d_r1`/`l_d_r2` +
effective `gamma_r1`/`gamma_r2` logs, `(B,)` outputs accepted, a batch-statistics D with a gamma raises `ConfigError` at
the first D call, `validate_train` refuses dunet with a gamma (grid_sample has no double backward). Multi-rank runs draw
rank-specific signs from each rank's RNG, but only rank 0's RNG is saved: resume is bitwise on one GPU only (as the data
order already is not on several).
