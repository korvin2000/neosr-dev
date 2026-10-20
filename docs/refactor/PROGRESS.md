# Progress log (newest last)

## 2026-10-20 session 1 (Opus 5.5)
- Read core code; wrote own review (`analysis/reviews/opus.md`). Verified empirically: G-side GAN loss under
  `inference_mode` gives zero generator gradient; D always in `eval()` (spectral norm frozen). Logs in
  `experiments/train_esrgan/*.log` show D saturation (logits +-40, l_d ~1e-10).
- Launched in parallel: Fable 5.1 xhigh review, Codex gpt-6-astra high review, runtime smoke test (t1) of all archs and
  short training runs (R1..R7).
- Consensus verification workflows: round 1 (8 groups G1..G8, Fable + Codex + Opus reconcile + dispute round),
  round 2 (Fable-unique findings F.*, Codex + Opus verifier) -> `analysis/consensus/F.md` done.
- Architecture consensus workflow (T1..T4) and aether/QAT design (T5) running.
- Owner decisions recorded in `DECISIONS.md`. Branch `refactor/stabilize` + worktree `.worktrees/refactor` created.
- Next: collect G1..G8 consensus + T1..T5 designs into `analysis/` and `design/`, write `PLAN.md`, start step 1.
