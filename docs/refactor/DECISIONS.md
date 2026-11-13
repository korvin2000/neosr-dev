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

## Owner decisions, 2026-11-10 (working setup)
- **The GitHub token in the `origin` remote URL stays.** Both machines are private and single-user; this is a
  deliberate choice, not an oversight. Agents do not report it as a finding again and never print the remote URL
  (mask it in command output).
- **Two machines, two roles.** The Linux workstation with the RTX 4090 (`/home/korvin/apps/neosr-dev`) runs every
  real test, training run, A/B, benchmark and profile. The Windows workstation (`C:\work2000\neosr-dev`, no torch)
  is for code analysis, documentation, small core changes that need no GPU run, and the Studio prototype. A core
  change made on Windows is unverified until the suite runs on Linux and is listed in `STATUS.md`.
- **Owner decision 3 of 2026-10-20 is narrowed:** the owner's local files (`.codex/*`, `AGENTS.md`, `CLAUDE.md`) are
  committed when the owner asks for it (done on 2026-11-10); otherwise they are still left alone.
- **`STATUS.md` is the single status page**; the per-stage `PROGRESS.md` files stay chronological logs.
- **Studio follows the trainer through a generated contract** (`scripts/studio_surface.py` ->
  `studio/shared/catalog/neosr-surface.json`, tested on both sides) instead of a hand-maintained snapshot.
- **Windows machine, addendum (same day):** it has a weak CPU as well as no GPU. A CPU-only venv exists for targeted
  tests, but work there is prioritised by compute: no-compute tasks first, then tasks one test file can check; no
  suite runs, sweeps or training (`STATUS.md`, "Machines").

## Owner decisions, 2026-11-10 (priorities)
- Order of work: (1) a stable, reliable, fast core: close the audit findings and the small core items; (2) finish
  NeoSR Studio; (3) only then new features. OTF realism (Phase B) is parked after its steps O1-O3; it was started
  on an agent's recommendation, not on the owner's request.
- Registration usability: the lazy tables in `__init__.py` (stage 1, T3) were right for start-up cost but lost
  the convenience of `@ARCH_REGISTRY.register()` / `@LOSS_REGISTRY.register()` for the person adding a module.
  Bring the decorator style back in a form that keeps the lazy import; design first.
- Agents ask before opening a track.

## Owner clarification, 2026-11-10 (what the project is; the competitor)
- neosr-dev is a new, reworked and optimised project of its own: clean and well-thought-out code, speed, a small
  footprint, good structure, compactness.
- traiNNer-redux (the-database; the local fork `traiNNer-redux-dev`) is a **competitor / alternative project**,
  often with badly written, bloated code and god classes. It is a source of ideas and examples only: an idea is
  taken when it is good and is then re-thought and rewritten here; its structure is not imitated and its code is
  not copied. This narrows the competitor policy above ("newer sibling project"): checking it is optional
  research, never a requirement, and its having a feature is no argument for adding it here.
- An earlier note of the same day called it "upstream" and told agents to reuse its work before writing their own:
  that was an agent's misreading of the owner and is withdrawn.
- Agents make no hasty decisions or conclusions: disputed or open questions go to the owner.

## Owner decision, 2026-11-10 (registries)
The five registries come back: `ARCH_REGISTRY`, `LOSS_REGISTRY`, `METRIC_REGISTRY`, `DATASET_REGISTRY`,
`MODEL_REGISTRY`, with `@X_REGISTRY.register()` on the class or function. This supersedes design T3, decision D3
("explicit lazy tables are the single source; `Registry` is deleted") in its form, **not in its goals**: the
new registries must not import every module to resolve one type and must do no work at import time (no argv,
TOML, CUDA or RNG), which is why the old ones were removed. It is a rework with the needed improvements and
optimisations, not a revert. Old code for reference: this repository's history (`git show
8da8ae0^:neosr/utils/registry.py`, `10876ed^:neosr/archs/`) and the owner's older fork
https://github.com/delafer/neosr. The mechanism (how the lookup stays lazy) is not decided: design note
`design/T7_registration.md` first, then the owner chooses. `tests/test_hygiene.py` bans `_REGISTRY` today; that
rule changes with the implementation.

## Owner decisions, 2026-11-10 (session 7: registry mechanism, small core items)
- **Registry mechanism: A1 of `design/T7_registration.md`.** `@X_REGISTRY.register(...)` is a declaration and
  returns the object unchanged; a registry reads the decorators of its package from the source text on first
  use (no import) and imports only the module of the requested type. One accepted decorator form; every other
  spelling fails the scan by file name; an `ast` reading in `tests/test_registry.py` is the independent check.
- **Decorator arguments:** `scale=` is required on archs (the constructor kwarg that receives the run scale, or
  `None`); metrics carry `choices=`. The tables `ARCHS`, `LOSSES`, `METRICS`, `CHOICES`, `DATASETS`, `MODELS`
  and the test list `NON_SR_ARCHS` are gone.
- **ONNX export gaps stay in `convert.py`** (`NO_ONNX`, `STATIC_ONLY`): they describe one exporter at one
  opset, not the arch. This closes the registry half of follow-up S.11.
- **Pad-to-window at trace time:** the reflect/replicate choice is frozen by a trace in every arch that pads
  its input this way, not only in hma. The owner chose one shared trace-invariant helper for all of them.
- **Commits:** one local commit per step in this session; pushed at the end on the owner's word.
- Implemented as proposed without a separate question, open to revision: no aliases for type names, no
  user plugin folders yet, `OPTIMIZERS` / `SCHEDULERS` stay tables (torch's own classes).

## Owner clarification, 2026-11-10 (the goal is the framework, not the archs)
The goal is the trainer itself: the training framework and its ecosystem (losses, metrics, schedulers,
optimizers, the GAN implementations, utilities and helpers), its speed and its stability: a solid, stable,
flexible and convenient foundation for testing and training **new, future** architectures. The existing
architectures rank near last: some are experimental or outdated, and most of them the owner will never train.
Fixing or polishing them is not a goal.

Consequences:
- Arch-specific findings and ideas are **deferred**, not rejected: they wait unless a change in the framework
  needs them (a contract every arch must meet, e.g. `scale=`, export, checkpoint loading) or the owner trains
  that arch. In the stage-2 audit this covers A-12, A-17, W-09, W-12, W-16, MX-15 and MX-19
  (`stage2/CLOSURE.md`).
- Work on the framework is preferred over work on one arch; a framework fix found through one arch is made
  generic where that costs little (e.g. `convert.py --prune` calls any network's `switch_to_deploy`).
- Small, exact fixes in the core may be made without asking first (owner, the same day); the house rules for a
  code step (targeted tests, Codex review of the diff, one commit) still hold.

## Owner decisions, 2026-11-11 (session 9: two open audit findings)
- **T-11 (host syncs per update): rejected.** A step without the `isfinite` sync was measured and is not faster
  (stage 2 GP-09: 116.2 -> 115.9 ms realplksr, 6.3 -> 6.3 ms compact L1, the launch-bound case with 57 % GPU
  idle). Merging the G and D reads into one would split `Phase.step` and couple the two phases' skip
  transactions; the fp16 half relies on `GradScaler` internals, and no template uses fp16. One sync per phase
  stays the design (T2 D13). Revisit only if a whole-step CUDA graph becomes a goal (that removes every sync, a
  different design).
- **ST-14 (abrupt D input switch at `wavelet_init`): deferred.** The transient is real but no harm to quality
  is shown, no run has crossed the templates' `wavelet_init = 80000`, and a new key is a feature. Which fix is
  right is open too: a cross-fade mixes RGB and SWT bands in the same channels; `wavelet_init` at the GAN
  start (config only) or a short D warm-up are alternatives, and `wavelet_guided` itself has no A/B. Trigger:
  a run past 80k or the R6 A/B shows a dip after the switch, or the templates' `wavelet_guided` is decided.
- **L-16 (fp32 generator tail) and the pyvips lazy-read idea (`next_tasks.md` 6.1): deferred** (owner, the
  same day).
- **Run decisions delegated (owner, the same day):** within the session plan, which A/B arms and measurements
  run, which tasks are swapped for cheaper or more useful ones as time runs out, and the gate on the 8k A/B
  results are decided by the main session after consulting Fable (t5, effort high), without asking the owner.
  Every such decision is logged with its evidence; default flips stay marked provisional (8k, one arch).
- **Session plan:** a 4-hour session on the Linux machine with parallel lanes and no multi-hour GPU runs
  (`next_tasks.md`); core quality, GAN modes, speed and stability first, Studio in parallel.

## Delegated decisions, 2026-11-11 (session 10: after Fable 5.2 and Codex GPT-6-Astra)
**Delegation (owner, the same day):** decide the open questions after consulting Fable and Codex, for the owner's
goal: train new SOTA photoreal SR models (high-frequency detail, textures) on a stable, fast base that fits one
RTX 4090 24 GB, with compact code; priorities (1) stability, quality and efficiency of training, (2) speed,
(3) photorealism and textures. Long GPU runs are allowed from now on.
**Inputs:** a brief (`experiments/s9_ab/consult/brief.md`), an Opus probe of the unet D's gradient scale, two
independent answers and a cross-review round (raw text in `experiments/s9_ab/consult/`, gitignored), and the
analysis of the owner's own redux recipe (`stage3/owner_recipe.md`).

| # | question | decision | why |
|---|---|---|---|
| D1 | LDL in the templates | **keep**; extend B2/B2s2 to 20k as evidence; removal needs a long run without LDL | both advisors; B2s2's LOSS sits on a ~2.5k-update D outage (Fable), which an LDL removal may itself have caused (Codex) |
| D2 | `compile = true` in the CNN templates | **not yet**; gate = a CUDA parity check (one update eager vs compiled; metagan, unet, R1/R2), then a two-seed 20k compile pair; A/B and acceptance runs stay eager | compile is an exact rewrite, so it ships behind parity tests; one 8k seed is not enough; the challenger's compiled path is untested |
| D3 | the template D (A-04) | **metagan@0.3 stays default and reference**; unet@0.1 is the challenger: 20k on two seeds, then the paired 100k runs | unet's grad norm of 30-70 is its parametrisation (high-res encoder, the logit-magnitude term), not a defect; under adan_sf the fixed clip rescales each step but does not damp it (resume probe: separation 0.66 / 0.79 / 0.69 with clip 1 / 50 / off); `clipped_d = 1.0` is no health signal |
| D4 | R6 (R3GAN) | **shrunk**: R3 gamma 1 extended to 20k; gamma 3 only if that leaves a question; scalar arms and round 2 deferred; a new arm OWN replaces them | the R arms were far from the regime that worked for the owner (D lr 40x higher, GAN weight 30-60x higher, wavelet off) |
| D5 | gradient clipping | **fixed max-norm stays the default**; one sensitivity arm (unet extended with `"spike"`); no D-only key | both advisors |
| D6 | Phase 0 and the long run | **E4 (DISTS) promoted** to the first new arms (T_DISTS, NoGAN_DISTS, OWN); then the paired 100k run (A0 and the challenger, both eager); then E5 at 40k with a second P0 seed; E2 last, as a G-only scheduler fork after the wavelet switch | the owner's recipe shows DISTS as its texture driver; the long run must cross `wavelet_init = 80000` (CONSENSUS) |
| D7 | metric memory at 4K | **document the limit** for cards under 24 GB; one measured 4K validation pass on the 4090; no chunked DISTS/TOPIQ | 24 GB is the target card; the 4K numbers are extrapolated so far |
| D8 | Studio | **real mode stays experimental** until one GPU lifecycle run passes; preset-bound parameters stay visible and disabled, with inline "fixed by preset" text. Lifted: the GPU run passed 2026-11-12, label dropped by the owner 2026-11-13 | both advisors |
| D9 | the A/B protocol for 20k+ | seeds 7 and 2; matched restart boundaries (new arms stop at 8k and resume, like the extensions); endpoints 20k and 100k fixed in advance; floors recomputed at 20k from B0/B0s2; the gate unchanged, deltas against the B0/B0s2 mean reported as a sensitivity read; for arms trained on DISTS the verdict uses TOPIQ and LPIPS; health medians per 2k window from `train.jsonl`; the confirmation set once, for the frozen finalist; states every 4k in long runs; patch stays 64 (the split's LQ tiles are 64x64) | both advisors, plus the DISTS rule from `docs/training_guide.md` section 4 |

Where the advisors disagreed: compiled long runs (Fable) vs eager (Codex): eager. The perception-distortion line
and the pooled reference (Fable): reported, not gated (Codex). An early wavelet fork at 20k (Fable): not run, the
long run crosses the real switch (Codex). R3 gamma 3 fresh (Fable) vs conditional (Codex): conditional.
**Queue:** `experiments/s10/queue.txt` (ranks 1-4, ~8 GPU-h), then the compile pair and the short checks, then the
paired long run once the 20k verdicts name the challenger.

## Owner decision, 2026-11-12 (D10: a faster, less noisy A/B protocol)
**Question (owner):** check concepts as fast as possible (a lighter G such as SPAN++, other settings, fewer
runs); the goal is the concept, not a trained model. **Inputs:** a brief and two independent answers (Fable 5.2,
Codex GPT-6-Astra; `experiments/s10/consult/`, gitignored: `brief.md`, `synthesis.md`, Fable's scripts in `adv/`,
re-run by the main session). **Accepted by the owner** as proposed in `synthesis.md`.

Evidence: on same-config pairs (B0/B0s2/A0, U/Us2) the single-checkpoint gate called LOSS in 9 of 36 comparisons
(all LPIPS: its 1 % floor is below one run-to-run sigma); window means halve the LPIPS noise (null RMS 0.0019 at
one checkpoint, 0.0008 over 12k-20k); 8k ranks predicted 20k ranks over 7 arms. A lighter G saves little:
spanplus is -13 % on the same objectives (`stage2/PROGRESS.md:86`); the step is FDL, D and the backward through
them (`stage2/analysis/gpu_profile.md` section 2). Corrections: T_DISTS was a TIE under the D9 DISTS rule, not a
WIN; its 12.75 GiB was a start-up peak (steady 4.70 GiB vs 3.19).

| # | rule |
|---|---|
| 1 | verdicts use window means of the paired deltas (`analyze.py --window`): screen 4k-8k, decision 12k-20k; never one checkpoint |
| 2 | tau per metric = max(1 % of the reference, 3 x null RMS of the window) (`--null`); TOPIQ: 3 x null RMS (its 1 % is ~10x its noise); fixed before new arms, binding only for arms started after 2026-11-12 |
| 3 | screen at 8k: drop a clear loser (a gated metric worse, none better); promote a better arm; a no-effect arm only if cheaper or simpler; stopped checkpoints are kept; if a queued 8k-to-20k extension flips its window verdict, the screen lengthens to 12k |
| 4 | one factor per arm, or a 2x2; multi-change recipes (OWN) are not screens |
| 5 | no extension of clear 8k losers; seed 2 only for promoted arms; a default needs two seeds and the long pair |
| 6 | new arms run straight 0 -> 20k (no artificial restart); A0 is the primary reference, the pool A0+B0+B0s2 the gate's |
| 7 | metrics an arm trains on are reported, not gated (`--report-only`) |
| 8 | long run: reference + one finalist 20k -> 100k, compiled if the compile pair passes the window gate and `compile_parity.py` passes on the finalist (a prospective revision of D2), health checks at 40k/60k; stop on a numerical failure or two consecutive LOSS windows |
| 9 | every verdict reports ms/update and the steady VRAM, the start-up peak separately (`run_ab.py` records `mem_gib`) |
| 10 | the confirmation set once per finalist, with the visual set and the owner's eye |

Not adopted: a proxy G (both advisors: transfer to larger archs is unknown and the saving is small); a mid-run stop
at 12-14k (saves <= 10 min per arm, 25 % false LOSS at one checkpoint); a compressed wavelet fork at 22k (not
acceptance; both long arms cross the real switch at 80k). Deferred: E2, E5, R3 gamma 3 and round 2, spike clip
from a fresh G, an LDL long run, the U+DISTS screen (unet is LOSS (mixed) under rule 1: LPIPS -0.005 better,
TOPIQ -0.002 worse beyond tau on both seeds).
**Plan:** `experiments/s10/queue2.txt` after the running queue: T_DISTS seed 2, then T_DISTS_noFDL (DISTS replaces
FDL); then the short measurements (batch 8/16/32 with and without DISTS, the DISTS profile, one 4K validation, the
Studio GPU run); then the long pair A0 + the finalist; then the confirmation set.
