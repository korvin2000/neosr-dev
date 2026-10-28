# Stage 3 competitor lane: training infrastructure (traiNNer-redux -> neosr)

Date 2026-10-28. neosr `refactor/stabilize` at `e0019ee` (T), traiNNer-redux at `3a2d2549` (C, upstream merged through
2026-10). Read-only for both repos. Scope: C `traiNNer/{optimizers,schedulers,ops}`, `models/{base_model,sr_model,ae_model,
con_sr_model}.py`, `utils/{eco,vram_manager,hardware_detection,zero_config_training,redux_options,options,config,logger,misc,
grid_sample_fix,grid_sample_gradfix}.py`, `train.py`, `test.py`, `convert_to_onnx.py`, `schemas/`, `docs/`, `scripts/{benchmarking,
training}`; against T `neosr/models/*`, `neosr/{config,runtime,cli}.py`, `train.py`, `test.py`, `convert.py`.

Evidence tags: **paper** (published claim, not reproduced here), **C-cfg/C-log** (C's own configs or training logs),
**probe** (tiny CPU probe, capped 4G, scripts in `scratchpad/stage3_infra/`), **source** (read, not run), **reasoning**.
Earlier harvests H3/H4 and `stage2/analysis/stability.md` section 2 are not repeated; their status is in section 1.

## 0. Verdict

Where the two loops overlap, neosr is now ahead on every point that affects correctness: transactional skipped updates,
sync-free logging, exact resume (format 2, RNG, EMA lerp count), atomic saves, compile of G and D after loading, DDP
accumulation, validation OOM handling, ONNX verification. C's training-infrastructure lane is mostly damping automations
(rejected), dead or unused code, and option tables whose entries do not work. It has **one real idea neosr lacks and needs
now** and a handful of cheap usability items:

1. **P1. LR decay for the AdamW templates (closed-form WSD row + an A/B).** Stage 2 switched the generator to AdamW. None
   of the 84 templates has a `[train.scheduler]`, and `total_iter = 1000000`, so every run trains at a constant 2e-4.
   The schedule-free optimizer that made a schedule unnecessary is gone. C contributes the menu (KneeLR, WSD, cosine with
   warm-up), not the code: KneeLR is buggy, WSD erases a chained warm-up, and none of them is used in C's 559 scheduler
   entries or logs. Write one closed-form row, make the config authoritative on resume, and let an A/B decide the default.
2. **P2. Best-checkpoint export**, `val.save_best`. Both repos only *track* the best value.
3. **P2. Self-describing exports.** `convert.py` cannot export a model trained with non-default `network_g` kwargs. Put the
   `network_g` table into the `.pth`.
4. **P2. Scheduler resume semantics.** A resumed run silently keeps the *saved* milestones or `T_max` (probe). This is a
   neosr gap that the P1 work fixes anyway.
5. **P3.** A teacher/ECO target transform (opt-in), safetensors read, portable ONNX simplification, a dynamo-export path, a
   bench script, a JSON schema for TOML editors, pretrained-load UX, small hygiene items, and recipe docs.

Skip: extra optimizers (SOAP/ADOPT/StableAdamW), EMA switch/delay, the other scheduler rows and `restart_scheduler`,
freezing/per-layer LR, all automations/VRAM/zero-config/hardware detection, bf16 ONNX, grid_sample patches, msgspec
config, the logger rewrite, blended tiling, and the dead/LLM-generated files (section 4).

## 1. Status of the earlier harvest (H3/H4) in today's neosr

| H3/H4 item | status in T (verified in code) |
|---|---|
| H3 #1 correct GAN step, #2 one accumulation owner, #8 per-optimizer outcome | done: `Phase.backward/step` (`models/phase.py`), `image.optimize_parameters` (window index `it = update + 1`, idle phases, buffer restore on skip) |
| H3 #3/#4 checkpoint layout, complete atomic state | done: format-2 `.state` (nets, EMA, phases, RNG, best, run record), `checkpoint.atomic_save` (tmp + fsync + replace + dir fsync) |
| H3 #5 fail-closed resume, strict load | done: `base.resume` (no fallback; ConfigError with a hint), `read_weights`/`load_weights` |
| H3 #6 load before compile, compile mode | done: `image.__init__` loads, then `wrap()` (DDP, then compile with `compile_mode`, `dynamic=False`) |
| H3 #7 precision resolved once / scaler only for fp16 | done: `Phase` scaler `enabled = use_amp and fp16`, init scale 2**5 |
| H3 #9/#10 sync-free logging, grad-norm logging | done: `LossMeter` (one cat + all_reduce + tolist per read), `grad_norm_*`, `clipped_*` |
| H3 #11 SIGINT at a boundary; #15 deterministic decoupled from seed | done: `utils/stop.StopFlag` (all-reduced), `runtime.setup_runtime` |
| H3 #12 msgspec typed config | superseded: reviewed `config.SCHEMA` inventory with did-you-mean and range checks; msgspec not needed |
| H3 #14 `fast_matmul` "high" | not done; **probe**: on CUDA "medium" == "high" (both TF32), only mkldnn/CPU gets bf16, so it is harmless on GPU (P3 hygiene) |
| H3 #16 safetensors export | not done (P3, section 3.6) |
| H3 #17 / ST EMA warm-up | done (`Ema` with `ema_power`, lerp count saved) |
| H4 optimizer items | done in stage 2 (SF-Adan rewrite, closure/no_grad, STATE_VERSION) |

## 2. Candidate table

| # | candidate | C where | T status | priority | cost | approach |
|---|---|---|---|---|---|---|
| 1 | Closed-form WSD-style LR schedule + template decay A/B | `schedulers/kneelr_scheduler.py`, `base_model.py:262-319` (`WARMUPSTABLEDECAYLR` = `pytorch_optimizer.get_wsd_schedule`) | missing (no template decays) | P1 | M | rewrite from reference |
| 2 | Scheduler resume and range checks | (C `restart_scheduler`, `base_model.py:825-831`, buggy) | worse (saved params win silently) | P2 | S | idea only |
| 3 | Best-checkpoint export (`val.save_best`) | none (C tracks the best only, `base_model.py:171-195`) | partial (tracks the best) | P2 | S | idea only |
| 4 | Self-describing exports (`network_g` in the `.pth`), used by convert/test | `base_model.py:442-447` (spandrel `hyperparameters` -> `metadata`), `convert_to_onnx.py` (reads YAML) | worse (convert takes only `-net`, `-window`) | P2 | S | port+fix |
| 5 | Teacher target transform (pure KD + ECO curriculum) | `utils/eco.py`, `sr_model.py:75-125,581-633` | missing | P3 | M | port+fix |
| 6 | safetensors read (+ optional export) | `base_model.py:452-462,615-616` | missing | P3 | S | port+fix |
| 7 | Portable ONNX simplification (constant folding, ORT BASIC / onnxslim) | `convert_to_onnx.py:18,175-207` | partial (folding off, ORT EXTENDED output is ORT-only) | P3 | S | port+fix |
| 8 | Dynamo ONNX exporter path | `convert_to_onnx.py:348,433-455` | missing (TorchScript exporter, deprecated in torch 2.9) | P3 | M | rewrite from reference |
| 9 | Train-step / inference benchmark script | `scripts/benchmarking/benchmark_archs.py` | missing in repo (stage-2 harness lives in the scratchpad) | P3 | S | rewrite from reference |
| 10 | Compile mode / CUDA-graph measurement | `redux_options.py:537-544` (mode string only) | equal (mode string exists) | P3 | S | idea only |
| 11 | Small hygiene: `fast_matmul` "high", `detect_anomaly`, free-space log, drop the unused `pytorch-optimizer` dependency | `train.py:273,363,919` | partial | P3 | S | port |
| 12 | Pretrained-load UX: match summary, legacy ESRGAN key remap | `base_model.py:488-560,564-600` (spandrel for ESRGAN) | partial | P3 | S | rewrite from reference |
| 13 | JSON schema for TOML editors | `schemas/redux-config.schema.json`, `scripts/options/generate_schema.py` | missing | P3 | S | idea only |
| 14 | Fine-tuning recipe docs (x2->x4 partial load, fidelity->GAN, lower LR) | `docs/source/training_guidelines.md` | missing (docs) | P3 | S | idea only |
| 15 | SOAP / ADOPT / StableAdamW | `optimizers/__init__.py:6,30` (pytorch_optimizer) | missing | skip | S | skip |
| 16 | EMA `ema_switch_iter`, `ema_update_after_step` | `sr_model.py:250-288` | missing | skip | S | skip |
| 17 | KneeLR, CosineAnnealingWarmupLR, CosineAnnealingRestartLR, the 19-row torch table, `restart_scheduler` | `schedulers/*`, `base_model.py:272-298,825-831` | equal (covered by #1 and the existing LinearLR chain) | skip | S | skip |
| 18 | Layer freezing / per-layer LR | none in C | missing in both | skip | S | skip |
| 19 | Automations, VRAM manager, dynamic batch/patch, OOM recovery, zero-config, hardware detection, dataset analyzer | `utils/{training_automations,vram_manager,zero_config_training,hardware_detection,dataset_analyzer}.py`, `train.py:369-486,757-852` | better (absent by design) | skip | L | skip |
| 20 | bf16 ONNX, `io_dtype`, 5D->4D wrap | `convert_to_onnx.py:209-330` | missing | skip | M | skip |
| 21 | grid_sample_fix / gradfix, con_sr_model, entropy hooks, fp16_static_analysis.md, precision_smoke, train_vgg19 | see section 4 | n/a | skip | - | skip |
| 22 | msgspec config, enhanced logger, blended tiling, crash checkpointing | `redux_options.py`, `logger.py`, `sr_model.py:1061-1132`, `train.py:984-1040` | better | skip | - | skip |

## 3. Details

### 3.1 (#1, P1) LR decay for the AdamW templates: one closed-form WSD row, then an A/B

**The gap (source):** `grep '^\[train.scheduler\]' options/*.toml` finds nothing in 84 templates. Every template has
`total_iter = 1000000` and G `adamw` at lr 2e-4 with no decay (`options/train_realplksr.toml`). Before stage 2 the
generator ran schedule-free `adan_sf`, which needs no schedule. With AdamW the LR now stays at its peak for the whole run.
EMA 0.999 averages over about 1000 updates and recovers part of a decay's effect, but not all of it. The stage-2 optimizer
A/B (8k updates, `DECISIONS.md` #4) compared constant-LR AdamW with SF-Adan. A long run with a decayed AdamW was never
measured, so that decision is open at long horizons too.

**Why a decay matters (paper + convention + reasoning):**
- Every reference SR recipe decays the LR. EDSR halves it every 200k updates. SwinIR and HAT use MultiStepLR halvings at
  about 50/80/90/95% of the run. C's own templates do the same (547 MultiStepLR entries, milestones 100k-400k of 500k-1M).
- WSD (warmup-stable-decay; Hu et al. 2024, MiniCPM; Hägele et al. 2024, "Scaling Laws and Compute-Optimal Training
  Beyond Fixed Training Durations") reports two things. A short final cooldown (about 10-20%, linear or 1-sqrt) matches
  cosine. And the cooldown can be branched from any checkpoint of the constant phase. KneeLR (Iyer et al., "Wide-minima
  density hypothesis", explore at peak, then linear decay) is the same family.
- These results are from LLM and vision work. **No SR ablation was measured here**, so the template change needs an A/B.

**C's implementation quality (probe/source): take none of it.**
- `KNEELR` is `functools.partial(KneeLR, peak_lr=optim_g.lr)` (`base_model.py:290-296`) and is applied to every
  optimizer. It returns `peak_lr` for all groups (`kneelr_scheduler.py:35-41`). **Probe:** D configured at 1e-4 runs at
  2e-4.
- `WARMUPSTABLEDECAYLR` (`get_wsd_schedule`) is a `LambdaLR`, so it is not chainable. C then overwrites the scheduler
  output with its own `warmup_iter` override (`base_model.py:375-396`, H3 #8).
- `CosineAnnealingWarmupLR` is exactly neosr's existing chain. **Probe:** `LinearLR(1/W, W)` + `CosineAnnealingLR(T)`
  equals the closed form `warm(t)*cos(t)` to 4 digits at every checked step.
- `CosineAnnealingRestartLR` raises `ValueError` one step past its periods (**probe**). Its 12 template uses are a single
  period, i.e. a plain cosine.
- Four table rows cannot work from a config: `SEQUENTIALLR`/`CHAINEDSCHEDULER` need scheduler objects and `LAMBDALR`
  needs a callable. `REDUCELRONPLATEAU.step()` raises `TypeError: missing 'metrics'` at the first update (**probe**).
- No C config or log uses KneeLR, WSD or cosine-warmup (C-cfg: 559 scheduler entries = 547 MultiStepLR + 12 single-period
  cosines). No C log reaches a decay milestone (the longest, ESRGAN7, stops at 37k of 1M).

**Design for neosr (rules: explicit table, compact, resume-exact):**
- Add one row `"wsd"` to `models/base.py:SCHEDULERS`: an `LRScheduler` subclass in closed form,
  `lr(t) = initial_lr * w(t) * d(t)`.
  - `d` is 1 until `decay_start` (an update, or a fraction of `total_iter`), then falls over `decay_iters` to
    `min_ratio`, with `shape` in {`linear`, `sqrt` (1-sqrt), `cosine`}.
  - `w` is the linear warm-up. It must live **inside** the row: a closed-form scheduler stepped after the prepended
    `LinearLR` (`image.py:264`) overwrites its value. **Probe:** `LinearLR` + `LambdaLR(1)` gives 2e-4 at update 0, so the
    warm-up is lost. For `wsd`, `_phase` therefore passes `warmup_steps` to the row instead of prepending `LinearLR`.
    That is a construction-time table entry, not a loop branch.
- Make its `state_dict` hold only the clock (`last_epoch`, `_step_count`, `_last_lr`). The config is then authoritative on
  resume: changing `decay_start` or `decay_iters` and resuming starts the cooldown from that checkpoint. That is WSD's
  "branch a cooldown" workflow, with the AdamW moments and EMA kept, and no new run.
  - `run_record` already warns about the change.
  - `initial_lr` and the betas stay saved-state-authoritative (the existing `Phase` rule).
- Validate in `validate_train`: `0 <= decay_start`, `decay_start + decay_iters <= total_iter`, `0 <= min_ratio < 1`.
- Tests: the closed-form values, resume parity at a fixed config, a resume with a changed `decay_start` (lr follows the
  new config from the next update), and DDP-free determinism.
- Cost: about 40 lines plus wiring and tests (S). The resume-semantics change and the A/B make it M.

**A/B (owner protocol, decides the template default):** realplksr, 10k updates. Arms: constant (today) vs WSD with a
2k linear cooldown to 0 vs MultiStepLR halvings at 5k/7.5k/9k. Run the L1-only fidelity setup and the GAN template. Read
EMA and live PSNR/SSIM/DISTS every 1k. D follows the same schedule (the convention). Expect the cooldown gain to show at
the end of the decay and to be partly diluted by EMA 0.999. If it holds, the templates get
`[train.scheduler] type = "wsd"` with the decay tied to `total_iter`, plus a comment explaining how to branch a cooldown
when stopping early.

### 3.2 (#2, P2) Scheduler resume and range checks (neosr's own gap, found while checking #1)

- **Probe:** a `MultiStepLR([100])` state resumed into a config with `[60]` keeps milestone 100. The lr at update 70 is
  still 2e-4. neosr restores the full scheduler state (`phase.py:266`, after a type check only), and `base.resume` prints
  only the generic "changed since ...: scheduler.milestones: [100] -> [60]" (`base.py:337`), which reads as if the new
  value applied. For optimizer groups `Phase` says explicitly "restored ... from the state". The scheduler should either
  say the same or (for closed-form rows, #1) follow the config.
- **Probe:** `CosineAnnealingLR(T_max=100)` climbs back to the peak by update 200. Nothing checks `T_max` against
  `total_iter`, so a too-small `T_max` re-warms the run silently. Add a `validate_train` check.
- Cost S, no behaviour change except the messages and the new error.

### 3.3 (#3, P2) Best-checkpoint export

- **Value (C-log):** in C's ESRGAN GAN run, EMA PSNR peaks at 29.47 @23k and then sits at 28.6-29.2 up to 37k
  (`stability.md` section 2). The last export is not the best one. In neosr `val_freq` and `save_checkpoint_freq` are
  independent, so the update that set the best value often has no export. `keep_exports > 0` then prunes the nearest ones.
  `best_metric_results` only records the value and update (`base.py:212-246`). C is the same (`base_model.py:171-195`).
- **Design:** `val.save_best = "<metric key>"`. When `update_best` reports an improvement for that metric, rank 0 writes
  `models/net_g_best_<dataset>_<metric>.pth = {"params": ..., "update": u, "value": v}` through `atomic_save`.
  - Write it after `validate()` returns, outside `eval_weights()`. Nesting `export_state_dicts()` inside the validation
    context would re-enter the SF swap.
  - `prune_exports` already ignores non-numbered names (`checkpoint.py:166-178`).
  - The saved `best` record makes it resume-correct.
- **Caveats for the docs:** 67 validation images give noisy selection. For GAN runs, select by DISTS/TOPIQ
  (`better = "lower"` exists), not PSNR.
- Cost S.

### 3.4 (#4, P2) Self-describing exports

- **Gap (source):** `convert.py:build_net` builds `{"type": args.network}` plus `window_size` only (`convert.py:81-85`).
  The CLI has no way to pass other `network_g` kwargs (`cli.py:46-113`). A model trained with a non-default `dim`,
  `n_blocks`, `num_feat` etc. fails the strict load in `load_net`. C's converter reads `network_g` from its YAML, and its
  archs store spandrel `hyperparameters` in a `metadata` entry (`base_model.py:442-447`).
- **Design:** the export payload becomes
  `{"params": sd, "neosr": {"network_g": <table>, "scale": s, "update": u}}`, with plain builtins (`weights_only`-safe).
  - Old files still load: `read_weights` picks `params`.
  - spandrel/chaiNNer unwrap `params` first, so they still load the file (reasoning from spandrel's canonicalize order).
  - `convert.py`/`test.py` default `-net`, `-s` and kwargs from the file; explicit CLI values win; old files without the
    entry fall back to the CLI.
- Cost S.

### 3.5 (#5, P3) Teacher target transform: pure KD and the ECO curriculum

- **C (source):**
  - With `network_g_teacher` alone, the target is `gt = teacher(lq)` under `inference_mode` (`sr_model.py:581-585`). That
    is pure output distillation and needs no GT.
  - With `train.eco` (`utils/eco.py`, `sr_model.py:100-125,600-633`) the target is `lerp(teacher_sr, hr, alpha)`. In
    `full` mode the input is also `lerp(bicubic(teacher_sr), lq, alpha)`. `alpha` ramps 0 to 1 over `end_ratio` of the
    run, and D's real class stays the HR.
  - The code is clean and unit-tested (`tests/test_utils/test_eco.py`). The teacher is DDP-wrapped and compiled
    needlessly, and `alpha` is keyed on C's late-incremented `current_iter`.
- **Value (paper, unverified here):** ECO is cited only in code comments (Eq. 14), with no link. Its reported gains were
  not checked. Teacher-centred targets are easier early in training, which is plausible for lightweight students. The
  pure-KD mode is the practical one: distil a slow model into a compact one on LQ-only data.
- **C-cfg:** no C config or log uses either mode.
- **Cost:** one teacher forward per batch; a HAT/DAT-class teacher costs several times a compact student.
- **Design (opt-in):** `[train.teacher]` with `network`, `pretrain`, `end_ratio` (0 = pure KD) and `mode`.
  - The teacher is built once, `eval().requires_grad_(False)`, never DDP-wrapped, runs bf16 under `inference_mode`.
  - The transform runs in `image._set_batch` (shared with `otf`), with `alpha` a pure function of `it` (resume- and
    DDP-safe, like `Loss.active`).
  - `Context` gets `real` (default `gt`), and the GAN pair reads `ctx.real`.
- Cost M. P2 if the owner distils compact models.

### 3.6 (#6, P3) safetensors

- C reads and writes `.safetensors` (`base_model.py:452-462,615-616`). neosr's `read_weights` calls `torch.load`, so a
  `.safetensors` pretrain fails with an unpickling error.
- The value is interoperability (OpenModelDB models), not safety: neosr already loads with `weights_only=True`.
- Add an optional import in `read_weights` (`safetensors.torch.load_file(..., device="cpu")`) with a ConfigError hint
  when the package is missing. Optionally add `logger.export_format = "safetensors"` for the exports, with the same
  metadata as #4 as a JSON string. Resume stays `.state`.
- New optional dependency (not installed in the venv). Cost S.

### 3.7 (#7, P3) Portable ONNX simplification

- neosr exports with `do_constant_folding=qat` (`convert.py:449`), i.e. folding is off for every float model. The setting
  has been `False` since 2024 with no recorded reason. `-optimize` writes an `ORT_ENABLE_EXTENDED` graph
  (`convert.py:466-470`). EXTENDED fusions may emit `com.microsoft` contrib ops that TensorRT or ncnn converters cannot
  parse. C uses `ORT_ENABLE_BASIC` plus onnxslim (`convert_to_onnx.py:18,175-207`).
- Proposal:
  - Turn constant folding on. The existing 3-shape fp32/fp16 verification catches regressions.
  - Make `-optimize` use BASIC (portable), or label the EXTENDED file as ORT-only.
  - Optionally run onnxslim when it is installed.
- Cost S.

### 3.8 (#8, P3) Dynamo exporter path

- neosr pins `dynamo=False` (`convert.py:450`). The TorchScript exporter is deprecated as of torch 2.9 (dynamo is the
  default), so this is a forward-compatibility risk, not a quality gain today.
- The dynamo exporter (opset >= 18) may also close the `aten::rms_norm` gap (flexnet/metaflexnet in `NO_ONNX`).
- C offers `onnx.dynamo` (`convert_to_onnx.py:433-455`) but verifies only fp32 (`:507`). neosr's multi-shape fp32 + fp16
  verification must stay.
- Cost M: `torch.export` dynamic dims, the onnxscript dependency, and a per-arch re-sweep of the export tables.

### 3.9 (#9, P3) Benchmark script

- C's `benchmark_archs.py` measures eager inference FPS and VRAM, with a channels_last comparison and CUDA events.
- neosr's stage-2 GPU numbers came from a scratchpad harness (`analysis/gpu_profile.md`) that is not in the repo.
- A small `scripts/bench.py` would let the owner decide archs and catch regressions. It builds the trainer from a TOML as
  `train.py` does and reports the median ms/update and peak memory over N updates (plus inference ms at a given size),
  for eager/compile and bf16/fp16.
- Cost S. Never run concurrently with training (OOM history).

### 3.10 (#10, P3) Compile mode / CUDA graphs (a measurement, not a port)

- The neosr step is launch-bound: 6.4k-19.5k kernels per update, GPU idle 16-24% on the CNN templates and 60% on DAT
  (`gpu_profile.md` section 0).
- `compile_mode` already exists. `"reduce-overhead"` (CUDA graphs) and `"max-autotune"` have never been measured.
- Risks: cudagraph trees with `backward(inputs=...)`, toggled grad mode, accumulation, `no_sync`, and the foreach EMA.
- C only exposes the string (default `max-autotune-no-cudagraphs`, G only).
- Run one capped GPU probe on workloads a/d before documenting a recommendation.

### 3.11 (#11, P3) Small hygiene

- `fast_matmul`: switch "medium" to "high". **Probe:** identical on CUDA; "medium" only makes mkldnn/CPU matmuls bf16.
  Update the comment (`runtime.py:107-110`).
- `detect_anomaly = true` debug key (C `train.py:273`): one line in `runtime`, documented as slow. neosr's
  `nonfinite_<key>` counters already name the failing loss, so this is for arch-internal NaNs.
- Free disk space in the "saving update" log line (C `train.py:919`): one `shutil.disk_usage`.
- `pyproject.toml` declares `pytorch-optimizer>=2.16.0`, but nothing in `neosr/`, the entry points or `tests/` imports
  it. Drop it.

### 3.12 (#12, P3) Pretrained-load UX

- neosr's relaxed load logs every missing or unexpected key as its own warning (`checkpoint.py:104-108`). It only refuses
  a load that matches nothing.
- C prints a capped list (10) and a matched-keys percentage (`base_model.py:488-560`). Adopt the capped list and
  percentage. Keep neosr's strict default and its training-state guard.
- C loads every ESRGAN pretrain through spandrel (`base_model.py:583-598`), which converts the old `model.N...` community
  layout. neosr has no remap, so old-format OpenModelDB ESRGAN models cannot be fine-tuned. A 20-line key remap in
  `esrgan_arch` (`_load_from_state_dict`, as asid/ditn already do) covers it without the spandrel dependency. Arch-lane
  overlap.
- Cost S.

### 3.13 (#13, P3) JSON schema for TOML editors

- C ships a msgspec-generated JSON schema, including per-arch kwargs read from constructor signatures. YAML templates
  reference it with `yaml-language-server`.
- For TOML the same works through Taplo / Even Better TOML (`#:schema` directive).
- neosr can generate it from `config.SCHEMA` plus `inspect.signature` of the arch/loss tables in about 60 lines, with a
  test that the committed schema is current.
- The value is editor autocomplete and inline typo marks. neosr already catches typos at load with did-you-mean.
- Cost S. Descriptions would need a docs source; types alone are fine.

### 3.14 (#14, P3) Fine-tuning recipe docs

C's `training_guidelines.md` writes down the standard recipes:

- x2 to x4 with `strict_load_g = false`: this is EDSR's x2-initialisation result (paper).
- Fidelity pretrain, then GAN fine-tuning: the Real-ESRNet to Real-ESRGAN two-stage pattern (paper).
- A lower LR when fine-tuning.

neosr supports all three already (relaxed load drops shape mismatches). Only the docs are missing. Note that neosr's
templates train the GAN from scratch, and stage 2 measured that this works (29.36 dB @8k). No template change is needed.

## 4. Skip (with the reason)

- **SOAP / ADOPT / StableAdamW** (`optimizers/__init__.py`, via pytorch_optimizer).
  - No SR evidence in C's configs or logs, or in the papers' domains: SOAP and StableAdamW are LLM/CLIP-scale results,
    and ADOPT's claim is mainly a convergence proof.
  - The stage-2 A/B already found AdamW best on the owner's data.
  - StableAdamW's per-tensor update clipping is the damping style the owner rejects.
  - SOAP's preconditioners cost 2-3x Adam's memory on conv weights. Revisit only as a deliberate A/B.
- **EMA `ema_switch_iter` and `ema_update_after_step`** (`sr_model.py:250-288`). `switch_iter` periodically copies the EMA
  into the live net, a regulariser with no evidence. No C template sets either; `update_after_step` appears once, at 10.
- **The other scheduler rows and `restart_scheduler`** (section 3.1). `restart_scheduler` sets `last_epoch` without
  recomputing chainable schedules (`base_model.py:825-831`). #1's config-authoritative closed form replaces it correctly.
- **Layer freezing / per-layer LR.** C does not have it. For SR fine-tuning, a full fine-tune at a lower LR is the norm,
  and the evidence for freezing is weak.
- **Training automations, VRAM manager, dynamic batch/patch, OOM recovery, zero-config, hardware detection, dataset
  analyzer.** These are rejected by the owner and the evidence agrees (`stability.md` section 2).
  `zero_config_training.py`, `hardware_detection.py` and `dataset_analyzer.py` are not even imported by the training path
  (source).
- **bf16 ONNX, `io_dtype`, 5D->4D wrapping.**
  - SR deployment runtimes (chaiNNer/spandrel, vs-mlrt, TensorRT engines) use fp16.
  - bf16 has no speed edge over fp16 at inference and thin runtime support.
  - 5D wrapping is for video models.
  - neosr's fp16 export (fp32 islands for DySample/GridSample, verified at 3 shapes) is better than C's: C does not verify
    fp16 and falls back to a raw `.half()` export.
- **Dead or generated files:**
  - `utils/grid_sample_fix.py` monkeypatches `F.grid_sample` globally and *silently turns every non-bilinear mode into
    bilinear*. It is imported nowhere.
  - `utils/grid_sample_gradfix.py` is StyleGAN2's double-backward op, imported nowhere.
  - `models/con_sr_model.py` is an orphaned 1036-line SRModel copy, not in `build_model`.
  - `sr_model._setup_feature_hooks` is commented out and finds layers by name.
  - `docs/fp16_static_analysis.md` is an LLM report with false claims, e.g. "fp16 gradients" for fp32 parameters under
    autocast.
  - `scripts/precision_smoke.py` is a toy.
  - `scripts/training/train_vgg19.py` is ImageNet classifier training, outside an SR framework; the loss lane can take
    published weights instead.
  - `models/ae_model.py` is AESOP autoencoder training and belongs to the loss lane.
- **msgspec typed config, the "enhanced" rich logger, crash checkpointing, blended tiling, validation OOM auto-tiling.**
  H3 already rejects most of these. New here: C's tiler uses a triangular weight that is zero at the tile borders, plus
  reflect padding (`sr_model.py:1083-1110`). Its validation `continue`s past any per-image exception and so changes
  the metric denominators (`sr_model.py:1244-1258`). neosr's halo-crop tiler is exact under RF <= halo, and its OOM retry
  is per image and warned (`image.py` `_infer`/`_metric`).

## 5. What neosr already does better (verified in T code)

- `Phase`: per-optimizer objective; idle windows keep the clocks; skipped updates restore buffers; skip-streak guard spans
  resume; strict, named optimizer/scheduler identity on resume; no host sync except one `isfinite` per update.
- `LossMeter`: device sums, one collective per read, non-finite accounting per phase.
- `Ema`: bare eager G, foreach lerp, buffers copied, warm-up lerp count saved (exact resume). C uses ema_pytorch with a
  CPU `step` buffer hack.
- Format-2 training state (self-contained, `weights_only`), atomic saves with directory fsync, `keep_states`/
  `keep_exports` pruning, exports = EMA or SF `x`. C writes directly, retries, and reports success on the 3rd attempt as
  failure.
- Compile of G **and** D (one-forward D), after loading, `dynamic=False`, configurable mode (realplksr 1.31x, `PROGRESS.md`).
- DDP only, correct `no_sync` accumulation and D gradient scale. C still offers DataParallel.
- Precision: scaler only for fp16, fp16 init scale 2**5, bf16 templates, `fp16_accumulation` opt-in.
- Validation: rank 0 with failure all-reduced, preserved RNG, cuDNN autotune off, background PNG writer, exact halo
  tiling, per-image OOM retry, learned-metric CPU fallback.
- Config: reviewed `SCHEMA` with did-you-mean, removed-key messages and range checks; TOML; seed decoupled from
  determinism.
- `train.clamp = "restore"`, NaN-skip policy, GAN start/ramp, lazy R1/R2, rpgan (stage 2).
- QAT for aether (prepare, observer phases, strip, QDQ export); `convert.py` with 3-shape fp32+fp16 verification,
  DySample fp32 islands, and per-arch NO_ONNX/STATIC_ONLY tables. C has no QAT and does not verify fp16.
- Cosine with warm-up is already expressible (`LinearLR` + `CosineAnnealingLR`, exact per the probe), so C's
  `CosineAnnealingWarmupLR` adds nothing.

## 6. Probe index (`scratchpad/stage3_infra/`, CPU, `CAP=4G`)

| file | result |
|---|---|
| inline (matmul precision) | `set_float32_matmul_precision("medium")`: cuda `tf32`, mkldnn `bf16`; "high": both `tf32` |
| `p_sched.py` | LinearLR+Cosine == closed form; LinearLR+LambdaLR loses the warm-up (2e-4 at update 0); C KneeLR sets D lr 1e-4 -> 2e-4; `ReduceLROnPlateau.step()` TypeError; C CosineAnnealingRestartLR ValueError past its periods |
| `p_resume.py` | resumed MultiStepLR keeps saved milestone 100 over config 60 (lr 2e-4 at update 70); CosineAnnealingLR(T_max=100) back to 2e-4 at update 199 |
