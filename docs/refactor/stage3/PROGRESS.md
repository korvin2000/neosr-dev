# Stage 3 (port from traiNNer-redux) progress

## 2026-10-28 session 3
- Owner asked which traiNNer-redux functionality is worth porting, decided by consulting Fable 5.1 and Codex Astra
  (gpt-6-astra). Workflow wf_7caf74ef-acb (resume: Workflow({scriptPath, resumeFromRunId}); script
  ~/.claude/projects/-home-korvin-apps-neosr-dev--worktrees-refactor/c1b40b02-f454-4498-9d89-be0afa4e2973/workflows/scripts/neosr-port-strategy-consensus-wf_7caf74ef-acb.js):
  5 Opus research lanes (archs, losses_metrics, data_degradation, training_infra, recipes_evidence) -> independent
  strategies from Fable 5.1 and Codex Astra -> Opus consensus draft -> cross-critique. Reports: competitor/*.md.
- Consultation done: lanes + Fable 5.1 + Codex Astra proposals, Opus draft, both critiques -> final CONSENSUS.md.
- Fable 5.1 re-check of CONSENSUS.md against the full C option/module inventory and neosr code: `REVIEW.md` (inventory
  complete; placement corrections: generic loss schedule, LPIPS, load/export items, ONNX hygiene and NAFNet into Phase A;
  i-LN and R3GAN D into Phase B). Implementation brief for the Opus 5.5 xhigh session: `TASK_OPUS.md` (Phase A, A1-A9).

## 2026-10-28 session 4: Phase A (Opus 5.5 implements, Codex gpt-6-astra reviews each diff)
Tests run CPU-only with the main checkout's venv (`/home/korvin/apps/neosr-dev/.venv/bin/python -m pytest`; the
worktree `.venv` has no torch), each file in its own capped scope (`systemd-run --user --scope -p MemoryMax=4G`).

### A1 exports and loading
- Done: exports are `{"params", "neosr": {network_<g|d>, scale, update}}`; `val.save_best = "<metric>"` writes
  `models/net_g_best_<dataset>_<metric>.pth` (exactly the evaluated EMA/SF weights, `neosr.metric` = dataset/key/type/
  better/value) via `atomic_save` after `validate()` returns; `read_weights` reads `.safetensors` (optional package,
  ConfigError hint); relaxed load logs one summary (loaded share, capped 10-key lists; shape filter in a root
  load_state_dict pre-hook, so it sees an arch's own remap); `esrgan` remaps the original `model.N` community layout
  (two upconvs; an x8 file fails strict as unexpected); `convert.py` defaults `-net`/`-s`/kwargs from the file (flags
  win; flag-only refusals still come before the read, MT-17); `test.py` merges the file's `network_g` under the
  config's; `--strip-qat` output carries `network_g` with `fused_init = true`.
- Codex review: 4 findings, all fixed: (1) the final state was saved before the final validation, so its best record
  lagged a best export -> with `save_best` the state is saved again when the final validation changed the record;
  (2) an explicit `-s` clashed with the file's native scale kwarg -> `merge_network` drops it (the run scale is
  injected); (3) a TOML date in `network_g` would make exports fail `weights_only` -> `validate_train` rejects
  non-builtin network values; (4) a failed best export could leave an advanced record in the next state -> a failed
  validation event restores the prior record.
- Decisions: ties overwrite the best export (the record is inclusive too, file and record stay equal); `nets`-style
  per-net metadata key is `network_<g|d>`; numbered exports carry no `metric` (only the best export has an evaluated
  value). Test changes: `test_checkpoint` relaxed-load strings and `test_model` export keys updated (obsolete by design).
- Verified: test_s3_exports (22), test_checkpoint, test_entry, test_model, test_resume, test_config, test_qat,
  test_s2_metrics, test_s2_review, test_s2_trainer, test_s2_exportfix, test_imports, test_hygiene: green.
- Open: a best export written after the last state, then a crash (abandoned timeline) is overwritten by the resumed
  run's next improvement over the restored record; DDP rank-0 write/failure of save_best is covered by reasoning only
  (validation() already all-reduces a rank-0 failure).
- Owner: 1x restoration needed (A7 stays); real-world photos are the main goal (Phase B OTF approved after R3GAN);
  R3GAN track started: analysis lanes (Opus audit, Opus neosr-fit, Codex Astra) -> Fable design -> implementation brief.
  Reports in `stage3/r3gan/`.

### A1 fixup (3751cc0)
- `test_metrics::test_validation_parks_metric_nets_on_cpu` drives `base.validation` with a duck-typed model; after A1
  it needs `best_metric_results`, `save_best` and a loader name. Missed in the A1 scoped runs (the file was not among
  them); fixed and verified at HEAD in a temporary worktree. `test_s2_mixed` and `test_losses_pipeline` exceed the 4 GB
  cap at HEAD already (exit 137, memory, not failures): the final suite runs them with a larger cap.

### A2 scheduler targeting, resume, range checks
- Done: `train.scheduler.nets = ["g", "d"]` (explicit targeting; default = every optimizer not in schedule-free mode,
  one info line per optimizer left out). Resume: each scheduler's configuration (state minus clock/base_lrs, compared
  by value) is matched against the saved one; unchanged -> exact restore (as before); changed, added or removed (also
  the trainer's LinearLR warm-up, or a changed `nets`) -> rebuilt from the config and replayed to the saved update from
  the restored `initial_lr` (saved lr wins, as for the optimizer), with one warning naming old/new schedulers and the
  lr before/after. `validate_train`: `nets` range check; ConfigWarning for CosineAnnealing `T_max < total_iter` and a
  MultiStepLR milestone `>= total_iter` (assumption: the brief's "last milestone < total_iter" reads inverted; a last
  milestone below total_iter is the normal case). No `wsd` row (after E2).
- Behaviour change (per the brief/CONSENSUS decision): with a `[train.scheduler]` and no `nets`, a schedule-free
  optimizer is no longer scheduled (no template has a scheduler). A state that scheduled one resumes with a warning and
  the base lr; `nets = ["g", "d"]` keeps the old targeting. Replay costs ~1 us per scheduler step (2 s at 1M updates).
- Codex review: 4 findings, all fixed: string comparison made `eta_min` 0 vs 0.0 (older torch) a change -> structural
  comparison, formatting only for the log; nested `nets` arrays raised TypeError -> element type check; more coverage
  (warm-up + cosine T_max change, idle D clock, changed `nets`, legacy `verbose`/int state, SF migration); the resume
  summary no longer claims an exact scheduler restore after a replay.
- Test changes (obsolete by design): `test_phase_state` scheduler-list raises removed, `test_s2_trainer` warm-up removal
  now replays (lr 0.5) instead of raising, `test_model` state test sets `nets` for its SF G.
- Verified: test_s3_scheduler (25), test_phase_state, test_s2_trainer, test_resume, test_model, test_config,
  test_entry, test_ddp_gloo, test_phase, test_s2_review, test_stop: green.
