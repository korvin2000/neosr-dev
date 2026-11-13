# NeoSR Studio — backend API (prototype)

Express 5 on port **4310**, JSON everywhere, SSE for run events. The Angular dev server proxies `/api` to it (`proxy.conf.json`). By default the backend is a **simulation**: it compiles recipes with the shared compiler, stores recipes as JSON files and runs a deterministic trainer simulator. With `STUDIO_TRAINER=real` new runs launch the actual `train.py` instead and its output is mapped onto the same event shapes (see [Real trainer](#real-trainer)); `/api/environment` then reports the real machine; the library, dataset scan and validation images stay mocked in both modes.

Shared types: `shared/recipe.model.ts`, `shared/run.model.ts`, `shared/compile/compile.ts` (`compileRecipe`), `shared/compile/problems.ts`, `shared/templates.ts`.

| Method | Path | Body → Response | Notes |
|---|---|---|---|
| GET | `/api/environment` | → `EnvironmentInfo` | Simulator: a mock (platform from `process.platform`, one GPU "NVIDIA GeForce RTX 4090" 24 GB cc 8.9, neosrPath `C:\work2000\neosr-dev`, version from `../pyproject.toml` if readable else `"dev"`, experimentsRoot `<neosrPath>\experiments`, freeDiskGb 812, simulated true). Real mode: see [Real trainer](#real-trainer), "Environment" |
| GET | `/api/recipes` | → `RecipeSummary[]` | `{ id, name, updatedAt, templateId?, phases, arch, scale, nodeCount, description }` sorted by updatedAt desc |
| GET | `/api/recipes/:id` | → `Recipe` | 404 if missing |
| POST | `/api/recipes` | `Recipe` → `Recipe` | Stores `server/data/recipes/<id>.json`; keeps the client id |
| PUT | `/api/recipes/:id` | `Recipe` → `Recipe` | Upsert; sets `meta.updatedAt` |
| DELETE | `/api/recipes/:id` | → 204 | |
| POST | `/api/compile` | `{ recipe, platform? }` → `CompileResult` | Thin wrapper over `compileRecipe` (the browser also compiles locally; this exists for parity checks and for the run launcher) |
| POST | `/api/import` | `{ text, name? }` → `ImportResult` | Wrapper over `importToml(text, emptyRecipe(name), name)`; 400 with `{ error }` on parse failure |
| GET | `/api/runs` | → `RunSummary[]` | Newest first |
| POST | `/api/runs` | `{ recipe: Recipe, platform?: 'windows'\|'linux', speed?: number }` → `RunDetail` | Compiles; **400 `{ problems }` if any blocker**. Creates the run (`queued`), starts the simulator. `speed` = simulated updates per real second (default 400). Real mode: compiles for the host platform, ignores `speed`, `simulated: false`. |
| GET | `/api/runs/:id` | → `RunSnapshot` | Detail + full event list (capped, see below) + series + lastSeq |
| GET | `/api/runs/:id/events?after=<seq>` | SSE | Replays events with `seq > after`, then streams live. Each SSE message: `id: <seq>`, `event: run`, `data: <RunEvent JSON>`. Heartbeat comment every 15 s. |
| POST | `/api/runs/:id/stop` | `{ force?: boolean }` → `RunSummary` | Graceful: `stopping` → `saving` (writes a checkpoint event) → `stopped`. Force: `stopped` immediately, no checkpoint. |
| POST | `/api/runs/:id/resume` | → `RunSummary` | Only from `stopped` / `disconnected`: continues from the last checkpoint iteration (NeoSR `--auto_resume`), status `running` |
| POST | `/api/runs/:id/chaos` | `{ kind: 'disconnect' \| 'nan' }` → `RunSummary` | Demo only (400 for a real run): `disconnect` sets status `disconnected` and pauses emission; `nan` emits a warning `NONFINITE` + `skipped/g` scalar spike then continues |
| DELETE | `/api/runs/:id` | → 204 | Stops the simulator |
| GET | `/api/runs/:id/images/:imageId` | `?kind=lq\|sr\|gt&iter=<n>` → `image/svg+xml` | Procedural validation images (see below) |
| GET | `/api/library/models` | → `ModelFile[]` | `{ file, path, arch, scale, iter, sizeMb, runName, variant: 'ema'\|'eval'\|'state', updatedAt }` — mock list of ~8 checkpoints across 3 fake experiments plus the generators of finished simulated runs |
| GET | `/api/library/datasets` | → `DatasetInfo[]` | `{ id, name, gtPath, lqPath?, images, scale?, minWidth, minHeight, format, note }` — 3 fixtures: a clean paired ×4 set, an OTF GT-only set, a small val set |
| POST | `/api/datasets/scan` | `{ gt: string, lq?: string, scale: number, patch: number }` → `ScanReport` | Mock, deterministic from the path string hash: `{ gtCount, lqCount, pairs, unmatchedGt: string[], unmatchedLq: string[], scaleMismatch: {file, gt:[w,h], lq:[w,h]}[], belowPatch: {file, w, h}[], channels: {rgb, rgba, gray}, overlapWithVal: string[], samples: {name, w, h}[] (6), durationMs }`. Paths containing `bad` or `broken` produce problems; otherwise a clean report with 1–2 minor warnings. |

## Simulator

`server/sim/trainer-sim.ts`. One instance per run, deterministic from a seed derived from the run id (mulberry32 PRNG). Ticks every 250 ms; each tick advances `speed / 4` curriculum updates (never past a phase boundary within one tick).

Derive everything from the compiled recipe (`compileRecipe(recipe).phases` and `context`):
- Phases: on entering a phase emit `phase` (`phaseIndex`, `name`, `runName`); for `finetune` phases the **native counter restarts at 0**, for `continue` it is cumulative — `progress.nativeIter` carries it. Curriculum `iter` is cumulative in every event.
- Every `print_freq` updates (from the plan's `logger.print_freq`, default 100): one `scalar` per active loss row `loss/g/<slot>` (value = `weight × base(slot) × decay(iter) × (1 + noise)`; base per loss family: pixel 0.05, mssim 0.08, consistency 0.02, ldl 0.01, fdl 0.4, dists 0.2, perceptual 0.3, gan 0.6; decay = `0.35 + 0.65 × exp(-iter/60000)` with the phase's native clock), `loss/g/total`, and when the phase has GAN: `loss/d/total`, `out/d/real`, `out/d/fake`; `lr/g` (warm-up ramp over `warmup_steps` then constant for schedule-free optimizers; multistep/cosine when an LR Schedule node is wired; `lr/d` when GAN), `grad_norm/g` (≈ 0.3–1.2), `skipped/g` (0, occasionally 1). Also emit one `log` line in the NeoSR style: `[epoch: 12, iter: 12,300, lr:(1.000e-03,)] [performance: 3.91 it/s] [eta: 0:14:32] l_g_mssim: 0.0471 l_g_consistency: 0.0112 l_g_total: 0.0583`.
- Every `val_freq` updates (when a Validation node is wired): `validation` with metrics for the enabled metric names (psnr rising from ~24 towards an arch-dependent ceiling 28–32 with noise, ssim 0.70→0.90, dists 0.30→0.12 falling, topiq 0.45→0.70), and 4 `ValidationImage`s (`lqUrl`, `srUrl`, `gtUrl` pointing at the images endpoint with `iter`), per-image metrics. Keep a `bestMetric` on the run (first enabled metric, honouring `better`).
- Every `save_checkpoint_freq` updates: `checkpoint` with `files: ['net_g_<native>.pth', 'net_d_<native>.pth'?, '<native>.state']`, `sizeMb` from a per-arch table (span 9, compact 5, realplksr 29, hat_s 38, esrgan 64, default 20), `state: true`. Update `lastCheckpointIter`.
- Every tick: `system` (gpuUtil 85–99, vramUsedGb from arch/patch/batch heuristic, dataWaitPct 2–12) — but emit at most every 1 s of wall time.
- `progress` every tick: `itersPerSec` (arch-dependent nominal: compact 12, span 9, realplksr 4.2, hat_s 1.1, default 3, times 0.9–1.1 jitter), `etaSec`, `phaseIndex`, `nativeIter`.
- `warning` occasionally: `NONFINITE` ("non-finite loss, update skipped"), `VAL_COST` ("validation took 9% of the last interval"), and at the first GAN phase start `GAN_START` ("discriminator enabled: expect l_g_total to rise").
- Status lifecycle: `queued` → (0.5 s) `starting` (emit log lines: "Loading dataset…", "Building network span…", "Params: 2.23M") → (1 s) `running` → `completed` at the end (final `validation` + `checkpoint`, log "Training finished"). Stop: `stopping` → `saving` (checkpoint event at the current native iter) → `stopped`. Resume: `running` from `lastCheckpointIter` (emit log "Resuming from …state").

Series: keep `Record<series, ScalarPoint[]>` (validation metrics as `val/<dataset>/<metric>`, system as `sys/gpu_util`, `sys/vram`, `sys/data_wait`). Cap events kept in memory at 4000 per run by dropping the oldest `log` and `system` events first; series are never dropped (downsample to ≤ 2000 points per series with stride if needed).

Seed data: at startup, if no runs exist, create **two finished runs** by fast-forwarding the simulator synchronously (no timers): one from template `4x-span-fidelity` (completed, 150k) and one from `4x-curriculum` (completed, 300k, two phases), plus **one stopped** run from `2x-compact-beginner` stopped at 12,000. Also save the 8 built-in templates as recipes only if `server/data/recipes` is empty? No — recipes folder starts empty; templates come from `shared/templates.ts`.

## Real trainer

The simulator stays the default. Real mode passed its GPU lifecycle run on 2026-11-12 (D8) and is no longer experimental (owner, 2026-11-13); the open points are listed at the end of this section.

`STUDIO_TRAINER=real` or the `--real` flag (`npm run api:real`; anything else: simulator) makes `POST /api/runs` start `server/trainer/trainer-runner.ts` instead of the simulator; runs created before keep their kind (`simulated` on the run). Seed runs stay simulated.

| Variable | Default | Meaning |
|---|---|---|
| `STUDIO_TRAINER` | `sim` | `real` launches the trainer (same as `--real`) |
| `NEOSR_ROOT` | the repository holding `studio/` | cwd of `train.py` (relative paths in the TOML resolve there; `experiments/<name>/` lands there) |
| `NEOSR_PYTHON` | `<NEOSR_ROOT>/.venv/bin/python` (`.venv\Scripts\python.exe` on Windows) | the interpreter |

- **Environment.** `/api/environment` reports `NEOSR_ROOT` (and its `pyproject.toml` version), its default `experiments` folder (a recipe's `path.experiments_root` may differ), the free disk there, `pythonPath`, `torchVersion`, `cudaVersion` and the GPUs, with `simulated: false`. One probe at startup runs `<python> -c <script>` (`server/trainer/environment-probe.ts`; torch is imported only there): `torch.cuda.device_count()`, which honours `CUDA_VISIBLE_DEVICES`, gives the GPU count, and `nvidia-smi` the name, memory, driver and compute capability (no CUDA context on a GPU that may be training; torch's device properties are the fallback, without a driver). The result is cached until the backend restarts. A failed probe (no Python, no torch, a timeout after 120 s, unexpected output) gives null versions, no GPUs and the reason in `warnings`; the backend keeps running.
- **Launcher.** Use `npm run api:real` (`node --import tsx server/index.ts --real`): `tsx watch` (`npm run api`, `npm start`) SIGKILLs the backend 5 s after a reload or Ctrl-C, so it cannot wait for the trainers to save (see Backend shutdown).
- **One run, one namespace.** The frozen TOMLs of a real run get `name = "<compiled name>_<run id without run_>"`, and the `[path]` entries that point into an earlier phase of the same run (`resume_state`, `pretrain_network_g/_d`) follow that name (only the `<experiments_root>/<name>/` segment changes); imported paths stay as written. Each run therefore writes and resumes only its own `experiments/<name>_<id>`. `RunDetail.tomls` holds these TOMLs and `phases[].runName` (and the `phase` event's `runName`) the suffixed name, so output paths built from it point at the run's real folders.
- **Process.** The phase TOMLs are written to `server/data/runs/<id>/<runName>.toml` (git-ignored); each phase runs `<python> train.py -opt <that file>` in `NEOSR_ROOT`, one after another (the next starts, with status `starting`, when the previous exits 0 after `End of training`). stdout and stderr (the trainer logs to stderr) are split into lines (`\n`, `\r\n`, tqdm's `\r`) and parsed by `server/trainer/log-parser.ts`, a pure module tested against a recorded run (`server/trainer/fixtures/p0_pretrain.log`).
- **Lock.** While a trainer process runs, `<experiments_root>/<name>.studio.lock` (next to the folder: a fresh `train.py` start archives an existing folder, which would carry a lock inside away) holds its pid and, on Linux, its start time. It is written right after the spawn; if that fails, the process is killed and the phase fails. It is removed when the process exits; a lock whose process is gone (or, on Linux, whose pid now belongs to another process) is deleted when found. A lock left by a crashed or timed-out backend marks an **orphan**: launch and resume return 409 until it is gone. Stop on such a run signals the orphan's process group (Linux, where the start time verifies the pid: SIGINT, or SIGKILL with `force`) and reports `stopping` until the lock is released, then `stopped`; Delete returns 409 while it lives. On Windows the pid cannot be verified, so Stop and Delete return 409 there and the orphan must be ended by hand; a pid reused by another process keeps such a lock alive (delete the `.studio.lock` file once no trainer runs).
- **Refused launches.** `num_gpu > 1` (400: the adapter runs plain `train.py`, not `torchrun`). Launch or resume while the backend shuts down (503). Launch or resume while a live process holds the lock of one of the run's output folders (409), or, as a guard, while those folders equal or nest in a live real run's (409; case-insensitive on Windows).
- **Persistence.** A real run is saved on every status change and on checkpoint and validation events (at most every 10 s), so after a backend restart it comes back `disconnected` and can resume.
- **Stop.** Graceful: SIGINT to the trainer's process group (its first Ctrl-C: it saves at the next update boundary, logs `Stopped at update N`, exits 0) → `stopping` → `saving` → `stopped`; a trainer that ends normally meanwhile still counts as `completed` (between phases: `stopped`, and a resume starts the next phase); a trainer that dies of the SIGINT during its startup (before `train.py` handles it) counts as `stopped`. Force and `DELETE`: SIGKILL to the group (Windows: `taskkill /T /F`; a graceful stop is a kill there, since Node cannot send SIGINT to it). A deleted run publishes nothing more.
- **Backend shutdown** (SIGINT/SIGTERM): new launches and resumes are refused, every live real run not already stopping gets a graceful stop (a second SIGINT would make the trainer abort its save), the backend waits up to 120 s for the trainers to exit, saves every real run (not one deleted meanwhile), then exits. A repeated signal exits at once only when it comes more than 1 s after the first (one Ctrl-C reaches the backend twice: from the terminal and relayed by npm or tsx). A trainer still running at the deadline keeps its lock. **Windows limitation:** the stop there is `taskkill /T /F`, so a shutdown kills the trainers without a checkpoint (progress since the last periodic save is lost); stop runs from the UI first.
- **Resume.** The same phase from the newest `<update>.state` in this run's `<experiments_root>/<name>_<id>/training_states`: the adapter writes `<phase file>.resume.toml` with `path.resume_state` set to it (an explicit `resume_state`, which the compiler writes for `continue` phases and imported resumes, would otherwise win over `--auto_resume`). Without a state there, the phase TOML runs as written. Series past that update are dropped, and the best metric, last validation and last checkpoint derived again from what is kept, before the request returns.

Log line → event (the clock is the trainer's update, mapped to the curriculum clock of the current phase):

| Trainer output | Events |
|---|---|
| any non-blank line except tqdm bars | `log` (level from the record: INFO → info, WARNING → warn, ERROR/CRITICAL and traceback lines → error; timestamp and colour codes stripped) |
| `Starting at epoch E, update U.` | status `running` (from `starting`), `phase` |
| `nothing to do: update N >= total_iter` | `progress` at N (a resume of a finished phase); the run ends `completed` |
| `[ epoch: … ] [ iter: N ] [ performance: X it/s, data D% ] [ mem: U / T GiB ] [ lr: … ] [ lr_d: … ] [ eta: … ] [ key: v ] …` | `progress` (`itersPerSec` X, `etaSec` = eta + later phases at X), `scalar` per key: `l_<net>_<x>` → `loss/<net>/<x>`, `out_d_<x>` → `out/d/<x>`, `grad_norm_<net>`, `clipped_<net>`, `skipped_<net>` → `<kind>/<net>`, `nonfinite_<key>` → `nonfinite/<series>`, others → `train/<key>`; `lr/g`, `lr/d` (first value; `lr_d` only with a D phase); `sys/data_wait` (D); `mem` (CUDA runs: the window's peak allocated / device total) → `system` with `gpuUtil` null, `vramUsedGb` U, `vramTotalGb` T, `dataWaitPct` D (then no separate `sys/data_wait` scalar) |
| `skipped_<net>` > 0 in that line | `warning` `NONFINITE` |
| `Validation <dataset>` + `# <metric>: <value>…` lines | `validation` (`images: []`); the first dataset sets `bestMetric` |
| `saving update N` (confirmed by the next line; dropped if the process then dies) | `checkpoint` `net_g_N.pth` (+ `net_d_N.pth` when `network_d` was logged), `N.state`; `sizeMb` from `one export set is X MiB` |
| `End of training at update N` / `Stopped at update N` | `checkpoint` with `*_latest.pth`; the run ends `completed` / `stopped` when the process exits |
| `best <metric> on <dataset>: … saved <file>` | `checkpoint` `[file]`, `state: false` |
| `stop requested: saving update N` | status `saving` |
| other WARNING records | `warning` `TRAINER_WARNING` |
| exit ≠ 0 (not 130) | status `failed`, message: `config error: …` text (exit 2), argparse `neosr: error: …`, the traceback's exception line (`FloatingPointError: …`), or the last ERROR record; exit 130 → `stopped` |

Not in the trainer's console output, so absent from real runs: GPU utilisation (`system.gpuUtil` is null; VRAM comes from `mem` on CUDA runs), validation images and per-image metrics, the current update on validation lines (taken from the last progress line), checkpoint sizes before the first save and for `net_g_best_*`.

Open points of the real adapter: Windows has no graceful stop (Node cannot send it SIGINT) and cannot verify an orphan's pid; `num_gpu > 1` is refused instead of launched with `torchrun`; the library, the dataset scan and validation images stay mocked. Runs against the real `train.py`: stop and resume end to end on the CPU (Windows), and one GPU lifecycle run on the RTX 4090 (2026-11-12: realplksr ×4 + metagan with the template objective, 600 updates, launch, live events, graceful stop at 305, resume from `305.state`, completion, exports, lock released).

## Procedural validation images

`GET /api/runs/:id/images/:imageId?kind=…&iter=…` returns an SVG 256×256 (LQ: 64×64 viewBox scaled). Four image ids: `building` (grid of rectangles + text "NeoSR"), `foliage` (feTurbulence green noise), `portrait` (circles + gradient skin tones), `text` (lines of text). `gt` is crisp; `lq` applies `feGaussianBlur stdDeviation=1.6` plus a `feTurbulence` noise overlay and a JPEG-like blockiness (an 8-px checker overlay at 6% opacity); `sr` interpolates blur from 1.4 at iter 0 to 0.15 at the end of training and noise from 0.5 to 0.05 (progress = iter / totalIter of the run). Cache-Control: public, max-age 3600.

## Persistence

`server/data/recipes/*.json` (recipes), `server/data/runs/*.json` (run snapshot written when a run reaches a terminal status; loaded at startup). Both folders are git-ignored except a `.gitkeep`.

## Errors

JSON `{ error: string, problems?: Problem[] }` with 400/404; never HTML.
