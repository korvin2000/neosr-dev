# neosr refactor: index and resume protocol

Goal: bring neosr-dev to a stable, fast, compact, reference-quality state (see `PLAN.md`).
Started 2026-10-20. Everything up to stage 3 Phase A and the R3GAN code is merged into `master` (05954e0, together
with the NeoSR Studio prototype); `refactor/stabilize` is the working branch of the Linux machine.

**Current state, open work and what runs on which machine: `STATUS.md`. Read it first.**

## Resume in a fresh session
1. Read `STATUS.md` (tracks, open work, machines), then the log of the track you continue: `stage3/PROGRESS.md`
   (newest: the R3GAN pause and its resume commands), `stage2/PROGRESS.md`, `PROGRESS.md` (stage 1); `DECISIONS.md`.
2. On the Linux machine (tests, GPU): `cd /home/korvin/apps/neosr-dev/.worktrees/refactor` (create it if missing:
   `git worktree add .worktrees/refactor refactor/stabilize`), `git pull` and merge `master` first: the Windows
   machine commits docs, Studio and small changes there.
3. On the Windows machine (`C:\work2000\neosr-dev`, `master`): no torch; analysis, docs, Studio, small core changes
   only (`STATUS.md`, "Machines").
4. `git log --oneline` and `git status` show what landed; log the outcome in the track's `PROGRESS.md`, update
   `STATUS.md` when a track changes state, commit per step.

## Stage 2 (2026-10-27/28)
Speed / reliability / memory / training-efficiency pass: `stage2/PLAN.md` (lanes, interfaces), `stage2/PROGRESS.md`
(log, A/B numbers, benchmark), `stage2/DECISIONS.md` (what changed and why), `stage2/analysis/` (10-lane audit).

## Stage 3 (2026-10-28)
What to take from traiNNer-redux: `stage3/CONSENSUS.md` (3-model verdict), `stage3/REVIEW.md` (Fable re-check, final
phased list), `stage3/TASK_OPUS.md` (Phase A implementation brief), `stage3/PROGRESS.md`, `stage3/competitor/` (lane reports).
Phase A (A1-A9) is done (commits and follow-ups at the end of `stage3/PROGRESS.md`); user docs: `docs/training_guide.md`.
R3GAN track (`stage3/r3gan/`: `DESIGN.md`, `TASK_R3GAN.md`, `RESULTS.md`): discriminator, loss path and template are
in; the GPU qualification failed the cost gate and the A/B is paused (`STATUS.md`). Phase 0 (A/Bs) is not started.
Phase B: the OTF realism design is `stage3/otf/DESIGN.md` (draft 2, no code); the rest is not started.

## NeoSR Studio (2026-10-28, synced 2026-11-10)
`studio/` is a web UI prototype (Angular + Node) that edits a recipe and compiles it to a NeoSR TOML; training is
simulated. Concept: `design/NeoSR_Studio_UI_Concept_v2.md`. Its catalog is tested against the trainer's tables
through `scripts/studio_surface.py` (`STATUS.md`, "Studio contract").

## Files
| file | content |
|---|---|
| `STATUS.md` | the single status page: machines, tracks, open work by priority, closed follow-ups |
| `PLAN.md` | ordered implementation steps, file ownership, acceptance checks, status |
| `PROGRESS.md` | chronological log: done, verified, dead ends (and why), next |
| `DECISIONS.md` | owner decisions + consensus decisions (Opus/Fable/Codex) with rationale |
| `analysis/reviews/` | the three independent reviews (`opus.md`, `fable.md`, `codex.md`) and the shared `brief.md` |
| `analysis/findings_round*.md` | consolidated finding lists that went through consensus verification |
| `analysis/consensus/` | final verdict tables per group (G1..G8 = round 1, F = Fable-unique round 2) |
| `design/` | (planned: `T7_registration.md`, the return of the five registries, see `DECISIONS.md` 2026-11-10 and `../../next_tasks.md`) final architecture designs T1, T2, T3, T5 (loss pipeline/step, model+checkpoint, config/imports, aether+QAT). T4 (triage+perf) exists only as the draft `design/drafts/T4_feature_triage_perf.md`; T6_stability was never written (stability evidence: `analysis/stability/raw_summary.md` and `analysis/acceptance.md`) |

## Process used
Three independent reviews (Opus 5.5 main thread, Fable 5.1 xhigh, Codex gpt-6-astra high) -> consolidated findings ->
per-group verification by Fable and Codex -> Opus reconciliation -> dispute round -> final verdicts.
Architecture: Fable + Codex proposals -> Opus synthesis -> Fable + Codex critique -> Opus final.
Codex consultation thread for follow-ups: `01a15ad5-bdae-7183-8287-29ba5ecbab9e` (gpt-6-astra, high).
