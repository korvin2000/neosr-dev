# neosr refactor: index and resume protocol

Goal: bring neosr-dev to a stable, fast, compact, reference-quality state (see `PLAN.md`).
Started 2026-10-20. Branch `refactor/stabilize`, checked out as the worktree `.worktrees/refactor`
(the main checkout stays on `master` and holds the owner's uncommitted files; `.worktrees/` is excluded locally via `.git/info/exclude`).

## Resume in a fresh session
1. `cd /home/korvin/apps/neosr-dev/.worktrees/refactor` (create it if missing: `git worktree add .worktrees/refactor refactor/stabilize`).
2. Read, in this order: `PROGRESS.md` (last state + next step), `PLAN.md` (step table with status), `DECISIONS.md`.
3. `git log --oneline master..refactor/stabilize` and `git status` show what landed and what is uncommitted.
4. Continue the first step in `PLAN.md` whose status is not `done`; log the outcome in `PROGRESS.md` and commit per step.

## Stage 2 (2026-10-27/28)
Speed / reliability / memory / training-efficiency pass: `stage2/PLAN.md` (lanes, interfaces), `stage2/PROGRESS.md`
(log, A/B numbers, benchmark), `stage2/DECISIONS.md` (what changed and why), `stage2/analysis/` (10-lane audit).

## Files
| file | content |
|---|---|
| `PLAN.md` | ordered implementation steps, file ownership, acceptance checks, status |
| `PROGRESS.md` | chronological log: done, verified, dead ends (and why), next |
| `DECISIONS.md` | owner decisions + consensus decisions (Opus/Fable/Codex) with rationale |
| `analysis/reviews/` | the three independent reviews (`opus.md`, `fable.md`, `codex.md`) and the shared `brief.md` |
| `analysis/findings_round*.md` | consolidated finding lists that went through consensus verification |
| `analysis/consensus/` | final verdict tables per group (G1..G8 = round 1, F = Fable-unique round 2) |
| `design/` | final architecture designs T1, T2, T3, T5 (loss pipeline/step, model+checkpoint, config/imports, aether+QAT). T4 (triage+perf) exists only as the draft `design/drafts/T4_feature_triage_perf.md`; T6_stability was never written (stability evidence: `analysis/stability/raw_summary.md` and `analysis/acceptance.md`) |

## Process used
Three independent reviews (Opus 5.5 main thread, Fable 5.1 xhigh, Codex gpt-6-astra high) -> consolidated findings ->
per-group verification by Fable and Codex -> Opus reconciliation -> dispute round -> final verdicts.
Architecture: Fable + Codex proposals -> Opus synthesis -> Fable + Codex critique -> Opus final.
Codex consultation thread for follow-ups: `01a15ad5-bdae-7183-8287-29ba5ecbab9e` (gpt-6-astra, high).
