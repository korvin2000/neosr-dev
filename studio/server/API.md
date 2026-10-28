# NeoSR Studio — backend API (prototype)

Express 5 on port **4310**, JSON everywhere, SSE for run events. The Angular dev server proxies `/api` to it (`proxy.conf.json`). The backend is a **simulation**: it compiles recipes with the shared compiler, stores recipes as JSON files and runs a deterministic trainer simulator that emits the same event shapes a real NeoSR adapter would.

Shared types: `shared/recipe.model.ts`, `shared/run.model.ts`, `shared/compile/compile.ts` (`compileRecipe`), `shared/compile/problems.ts`, `shared/templates.ts`.

| Method | Path | Body → Response | Notes |
|---|---|---|---|
| GET | `/api/environment` | → `EnvironmentInfo` | Mock: platform from `process.platform`, one GPU "NVIDIA GeForce RTX 4090" 24 GB cc 8.9, neosrPath `C:\work2000\neosr-dev`, version from `../pyproject.toml` if readable else `"dev"`, experimentsRoot `<neosrPath>\experiments`, freeDiskGb 812, simulated true |
| GET | `/api/recipes` | → `RecipeSummary[]` | `{ id, name, updatedAt, templateId?, phases, arch, scale, nodeCount, description }` sorted by updatedAt desc |
| GET | `/api/recipes/:id` | → `Recipe` | 404 if missing |
| POST | `/api/recipes` | `Recipe` → `Recipe` | Stores `server/data/recipes/<id>.json`; keeps the client id |
| PUT | `/api/recipes/:id` | `Recipe` → `Recipe` | Upsert; sets `meta.updatedAt` |
| DELETE | `/api/recipes/:id` | → 204 | |
| POST | `/api/compile` | `{ recipe, platform? }` → `CompileResult` | Thin wrapper over `compileRecipe` (the browser also compiles locally; this exists for parity checks and for the run launcher) |
| POST | `/api/import` | `{ text, name? }` → `ImportResult` | Wrapper over `importToml(text, emptyRecipe(name), name)`; 400 with `{ error }` on parse failure |
| GET | `/api/runs` | → `RunSummary[]` | Newest first |
| POST | `/api/runs` | `{ recipe: Recipe, platform?: 'windows'\|'linux', speed?: number }` → `RunDetail` | Compiles; **400 `{ problems }` if any blocker**. Creates the run (`queued`), starts the simulator. `speed` = simulated updates per real second (default 400). |
| GET | `/api/runs/:id` | → `RunSnapshot` | Detail + full event list (capped, see below) + series + lastSeq |
| GET | `/api/runs/:id/events?after=<seq>` | SSE | Replays events with `seq > after`, then streams live. Each SSE message: `id: <seq>`, `event: run`, `data: <RunEvent JSON>`. Heartbeat comment every 15 s. |
| POST | `/api/runs/:id/stop` | `{ force?: boolean }` → `RunSummary` | Graceful: `stopping` → `saving` (writes a checkpoint event) → `stopped`. Force: `stopped` immediately, no checkpoint. |
| POST | `/api/runs/:id/resume` | → `RunSummary` | Only from `stopped` / `disconnected`: continues from the last checkpoint iteration (NeoSR `--auto_resume`), status `running` |
| POST | `/api/runs/:id/chaos` | `{ kind: 'disconnect' \| 'nan' }` → `RunSummary` | Demo only: `disconnect` sets status `disconnected` and pauses emission; `nan` emits a warning `NONFINITE` + `skipped/g` scalar spike then continues |
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

## Procedural validation images

`GET /api/runs/:id/images/:imageId?kind=…&iter=…` returns an SVG 256×256 (LQ: 64×64 viewBox scaled). Four image ids: `building` (grid of rectangles + text "NeoSR"), `foliage` (feTurbulence green noise), `portrait` (circles + gradient skin tones), `text` (lines of text). `gt` is crisp; `lq` applies `feGaussianBlur stdDeviation=1.6` plus a `feTurbulence` noise overlay and a JPEG-like blockiness (an 8-px checker overlay at 6% opacity); `sr` interpolates blur from 1.4 at iter 0 to 0.15 at the end of training and noise from 0.5 to 0.05 (progress = iter / totalIter of the run). Cache-Control: public, max-age 3600.

## Persistence

`server/data/recipes/*.json` (recipes), `server/data/runs/*.json` (run snapshot written when a run reaches a terminal status; loaded at startup). Both folders are git-ignored except a `.gitkeep`.

## Errors

JSON `{ error: string, problems?: Problem[] }` with 400/404; never HTML.
