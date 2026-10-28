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

### A3 TOPIQ-FR weights, LPIPS
- Done: `calculate_topiq(_pt)` takes `dataset = "kadid" | "pipal"` (default kadid unchanged; PIPAL weights
  `cfanet_fr_pipal_res50-69bbe5ba.pth` on pyiqa's HF mirror, HEAD-checked 200; local override
  `topiq_fr_pipal_weights.pth`); strict load as before. `calculate_lpips(_pt)` (`neosr/metrics/lpips.py`, LPIPS v0.1
  rewritten from the reference): torchvision AlexNet (or VGG16) ImageNet features at the 5 ReLUs, official v0.1 lin
  layers loaded strictly (`lin<k>.model.1.weight`), [0,1] -> [-1,1] -> ScalingLayer, unit-normalised channels,
  spatial mean, summed; both images in one backbone pass; `METRIC_LOWER` (better = lower). Metric-net cache key is
  `(name, *variant)`: one net per weights variant, built once, parked on the CPU as before. Variant values are checked
  when the metric is bound (`metrics.CHOICES`), not at the first validation. 84 training templates gained commented
  rows only (TOPIQ dataset, LPIPS, which metric judges: not a training loss); `options/aether/*` experiment configs
  untouched.
- Parity probe (scratchpad `probe/lpips_parity.py`, pyiqa's own `lpips_arch.py` from `/home/korvin/apps/IQA-PyTorch`,
  imports stubbed): alex with the official weights max |diff| 1.5e-8 over 64x64, 97x131, batch 2; vgg 9.3e-10 with
  shared random lin layers (the official vgg lin file is not cached; not downloaded).
- Codex review: no scoring bug; fixed: the PIPAL strict-load test now forces the hub branch; download-free tests
  (untrained backbones, both nets, odd sizes, batch 2, grayscale) against the algorithm spelled out per image;
  per-layer temporaries freed before the next slice. Open: official VGG-lin parity, real PIPAL file load and 1080p
  peak memory are unverified offline (GPU/owner).
- Test changes: metric-cache tests use the tuple key and variant-taking builders (internal API changed).
- Verified: test_s3_metrics (13), test_metrics, test_s2_metrics, test_s2_review, test_config, test_imports: green.

### A4 loss schedule, hinge, psnr_loss
- Done: `start_iter`/`ramp_iters` moved from `gan_loss` into `Loss` with a new `end_iter`; `build_losses` pops the
  three keys from any `train.<slot>_opt` and calls `Loss.schedule()` (ValueError -> ConfigError naming the section).
  `active()` = `start_iter < it <= end_iter`; the linear ramp is a host float in `compute()` applied before the weight
  (the old gan order, so a full-weight term is bitwise unscaled). gan_d_loss still follows the G term's `active()` and
  is never ramped. `gan_type = "hinge"` (D `relu(1 - r) + relu(1 + f)` in fp32, logged as l_d_real/l_d_fake; G
  `-f.mean()`; no labels). `psnr_loss` (LOSSES row): `10 log10(mse_i + eps)` per image, mean|sum, fp32 (kept: the
  owner made 1x restoration a goal, DECISIONS 2026-10-28).
- Codex review: 4 findings, fixed: `psnr_loss` `reduction = "none"` could not train (a term is a scalar) -> mean|sum
  only; `eps` must be a finite non-bool number > 0; a falsy non-numeric `end_iter` leaked TypeError -> ConfigError;
  added coverage: exact resume just before and at an `end_iter` (logs, weights, optimizer moments), hinge D with
  R1/R2 double backward.
- Test changes (obsolete by design): `test_s2_losses` builds the GAN schedule via `Loss.schedule` (the ctor no longer
  takes it); the two ctor-validation cases moved to builder-level ConfigError tests.
- Verified: test_s3_losses (16), test_s2_losses, test_gan, test_losses, test_losses_pipeline (10 GB cap; it exceeds
  4 GB at HEAD), test_s2_integration, test_s2_review, test_registry, test_resume, test_model: green.
