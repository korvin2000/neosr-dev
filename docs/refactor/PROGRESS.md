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
- tests/ safety net written (uncommitted until the loss step): 65 pass, 7 strict-xfail (known G5.1-G5.4), 5 fail = consistency_loss
  hidden dependency on torch.set_default_device("cuda") (consistency_loss.py:42,132 build tensors without device) -> fix in loss step.
  G5.1 trigger is broader: any zero-variance residual (constant or anticorrelated inputs), not only identical inputs.
- Owner reminder: traiNNer-redux-dev may contain fixes/features; port ideas/code only after adaptation, never as-is.
  Competitor-harvest workflow launched (H1 losses, H2 data/OTF, H3 loop/checkpoint/config, H4 archs/optimizers/metrics;
  Fable t5 + Codex per area, Opus merge) -> analysis/competitor/H*.md. PLAN.md must consult these before each step.
- Smoke baseline (analysis/smoke/baseline_report.md): 85/99 archs pass (real failures: grformer*/krgn rely on global cuda default
  device, esc_fp/omnisr invalid default kwargs, hit_srf_medium/large OOM at 64px); R1 bf16 NaN abort at iter 1 in ~50% of starts,
  R3 fp16 NaN ~iter 24 always; speed kill-switch fires at iter 10-20 in every run; OTF configs crash (poisson_scale_range KeyError,
  then sinc_prob None); resume works mechanically (eta negative). NaN root-cause investigation launched (t4 + Codex).
- Codex NaN analysis (analysis/nan/codex.md): iter-1 NaN most likely MS-SSIM fractional power of negative cs/ssim (CPU probe:
  8/10 random-init esrgan seeds NaN, also in fp32); metagan spectral norm never updated because D is always eval() -> logits up to
  25k at init (one train-mode forward fixes it); adan_sf warmup not applied (warmup 1600 vs 16000 give identical updates);
  the smoke "fp16 NaN at iter 24" was actually CUDA OOM (dr3.log:58-61), l_g_gan 624 of 640 total. Empirical t4 repro pending.
- Owner: traiNNer-redux trains stably despite bad code -> its stability experience must shape the fixes. Evidence: its real run
  experiments/4x_ESRGAN (ESRGAN7.yml) 43k iters stable with AdamW G 1e-4 / D 5e-6, warmup 1000, EMA ema_power 0.75 warmup,
  r3gan weight 0.03 start_iter 1000 + R1/R2, grad-norm + scaler logging; neosr templates: adan_sf 8e-4 / 1e-4, gan 0.3 from iter 0,
  metagan. Stability-playbook workflow launched (Fable + Codex static extraction, empirical traiNNer runs E1/E2 + neosr E3)
  -> design/T6_stability.md.
- T5 aether+QAT design done -> design/T5_aether_qat.md (GPU-verified: QAT from scratch works with Q/DQ stubs, plain FakeQuantize
  under bf16/fp16 autocast, strict save/load round trip diff 0, QDQ ONNX parity 5e-3). Note: its D18 keeps the old checkpoint
  scheme; reconcile with T2's new checkpoint format in PLAN.md.
- Round-1 consensus done (48 agents): 119 items, 89 TRUE / 28 PARTIAL / 2 FALSE; 75 unanimous, 40 majority, 4 judge ->
  analysis/consensus/G1..G8.md + round1_final.json. Competitor harvest done -> analysis/competitor/H1..H4.md
  (notable new: FlexNet attention under no_grad by default, HAT/DRCT/SwinIR window padding, DISTS input normalisation).
- PLAN.md v1 written. Phase L (L1 losses, L2 optimizers, L3 data, L4 metrics/utils/archs) and Phase A (A1-A3 arch fixes) running
  as workflows in the worktree (uncommitted; review + commit per step when they return).
- Phase L committed: L1 155f677 (losses), L2 14e3b2b (optimizers), L3 9360403 (data), L4 52137a8 (metrics/utils/small archs).
  Each: Opus t3 implement -> Codex review -> fix pass -> main-thread review. Main-thread corrections: consistency cosine term kept
  with its original gated semantics (agent had made it unconditional = objective change for 85 configs) via torch.where (no sync);
  DISTS ImageNet normalisation on loss + metric per owner. 185 tests pass (losses/optimizers/data/metrics).
  Lint: remaining ruff findings are pre-existing style (quotes, blank lines) -> C8.
- Session 1 closed at the owner's request (usage limit ~92%): all background tasks stopped (design T1-T4 workflow, stability
  playbook, Phase A, NaN repro). Saved: design/T1-T3 finals, design/drafts/* (T4 synthesis, proposals, Codex reviews),
  analysis/stability/raw_summary.md, Phase A WIP patch in wip/. Scratch worktree nanwt removed; experiments/stab_* and smoke_*
  deleted in both repos. Owner's uncommitted files in the main checkout untouched. Next steps: PLAN.md "Session-1 end state".

## 2026-10-21 session 2 (Opus 5.5)
- Resumed from PLAN "Session-1 end state". Read design/T1-T3 finals (consistent, cross-referenced). Owner: keep_states=3.
  Other design open questions take recommended defaults (DECISIONS #8). PLAN v2 = waves W0..W5. Seed c7a2e7a (ConfigError).
- W0 launched (workflow phase-C-wave0): A-verify (apply wip Phase A patch + T1-A4), T3-0, T1-A3 phase.py, T2-A1 checkpoint.py,
  T2-A2 tiling.py, T2-A4 stop.py. If interrupted: the workflow can be resumed with its run id, or redo lanes from PLAN v2.
- W0 committed (6 commits): Phase A arch fixes verified (state_dict keys unchanged; swinirdense 3conv the one intended shape change)
  09ffccc; T3-0 config/runtime/cli/factory de925d4; phase.py 5c7e8ee; checkpoint.py 998ce6c; tiling.py 2e18ffe; stop.py 6968a2c.
  Full suite 598 passed / 2 skipped / 4 xfailed. Main-thread review of the new modules: no changes needed.
- W1 launched (workflow phase-C-wave1): L-A (fixture on old code -> ARCHS table/build_network(scale=) -> 6 parallel R1 batches -> Codex
  review -> fix), L-B metrics table, L-C RNG atomic switch + data tables, T1-A1 loss contract/builder (not wired until W3).
  Pending main-thread chore after W1: shrink tests/test_hygiene.py / test_imports.py allowlists.
- W1 committed: archs table + scale injection + net_opt removal 10876ed (fixture equivalence atol 1e-6; neosr/neosr.archs/data/models
  import side-effect free; hygiene allowlists shrunk), metrics table 4363b79, RNG switch + data tables 0c7b5b3 (seed_all seeds the shared
  rng), loss contract/builder c48252d. Main-thread fixes: a2/a3 arch tests ported to the table, patchloss test expects ConfigError.
  Suite: 912 passed + imports/hygiene green. Noted for C8: archs/losses build_* share one 5-line shape -> factory.build helper.
- W2 launched (phase-C-wave2). KNOWN TRANSIENT: after W2's B1 commit the old trainer cannot build losses (kernels lose loss_weight)
  until W3 (C1 integration) lands; train.py is not usable on commits between W2 and W3.
