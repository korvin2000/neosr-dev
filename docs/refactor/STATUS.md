# Project status (single entry point)

Updated 2026-11-10. Read this first; the per-stage logs (`PROGRESS.md`, `stage2/PROGRESS.md`,
`stage3/PROGRESS.md`) hold the history and the numbers. Update this file whenever a track changes state.

## Machines and what runs where (owner, 2026-11-10)
| machine | checkout | role |
|---|---|---|
| Linux workstation, RTX 4090 | `/home/korvin/apps/neosr-dev` (+ worktree `.worktrees/refactor`), venv with torch | **every real test and measurement**: the pytest suite (CPU and CUDA), training runs, A/Bs, profiling, benchmarks, ONNX verification. The A/B harness, the pretrain and the data live here only (`experiments/`, gitignored) |
| Windows workstation (Ryzen 5 5500U, 6 cores, 31 GB RAM, no GPU) | `C:\work2000\neosr-dev`, CPU-only `.venv` (torch 2.9.1+cpu) since 2026-11-10 | code analysis and review, documentation, small core changes checked by **targeted** CPU tests, NeoSR Studio (the UI prototype). Slow CPU: no suite runs, no sweeps, no training |

Consequences:
- On the Windows machine prefer work that needs no compute at all; run single test files or `-k` selections only
  (measured: 360 config/registry/import tests take 2 min, one `import neosr.*` subprocess ~10 s). Never start the
  full suite, the per-arch sweeps (`test_archs*`, the `test_s2_*` arch files, `test_losses_pipeline`), ONNX export
  sweeps or a training run there.
- A core change made on the Windows machine counts as **verified only for the tests named next to it**; list it
  under "Waiting for the Linux machine" below until the full suite ran there, and keep such changes small.
- Torch-free checks: `python scripts/studio_surface.py` (`--check`, `--validate <toml>`) and everything under
  `studio/` (`npm run typecheck`, `npx ng test --watch=false`, `npm run contract`, `ng build`).
- Paths like `/home/korvin/...` in the logs and in three test fixtures are the Linux machine's and are correct there.

### The Windows CPU environment (not `uv sync`: the lock pins CUDA wheels)
```bash
python -m pip install --user uv
python -m uv venv --python 3.13 .venv
python -m uv pip install --python .venv/Scripts/python.exe torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m uv export --frozen --no-hashes --no-emit-project | grep -vE "^(torch|torchvision|nvidia-|triton|onnxruntime-gpu)" > req.txt
python -m uv pip install --python .venv/Scripts/python.exe -r req.txt onnxruntime pytest ruff
.venv/Scripts/python.exe -m pytest tests/test_config.py -q -p no:cacheprovider
```
Versions follow `uv.lock` except the CPU builds of torch/torchvision and `onnxruntime` instead of `onnxruntime-gpu`.

## Not an open point: the token in the remote URL
The `origin` URL in `.git/config` embeds a GitHub token. The owner keeps it deliberately (private single-user
machines; `DECISIONS.md`, 2026-11-10). Do not report it again and do not print the URL in logs or docs.

## Tracks
| track | state | evidence / where to resume |
|---|---|---|
| Stage 1: stabilisation (phases 0, L, A, waves W0-W7) | done, merged | `PLAN.md`, `PROGRESS.md`; 1192 tests at the end |
| Stage 2: speed, reliability, memory, templates | done, merged | `stage2/DECISIONS.md`, `stage2/PROGRESS.md` |
| Stage 3 Phase A (A1-A9, port from traiNNer-redux) | done, merged | `stage3/PROGRESS.md` "Phase A closed" |
| Stage 3 Phase 0 (E0-E5: evaluation protocol, config-only A/Bs) | **not started** (E0 partly covered by the R5 split) | `stage3/CONSENSUS.md` Phase 0 |
| R3GAN track | code done (R1-R3), data done (R5), qualification R4: bf16 PASS (provisional, no replicate), **cost gate FAIL**; **A/B (R6) not run: paused** | `stage3/PROGRESS.md` "PAUSED", `stage3/r3gan/RESULTS.md` |
| Stage 3 Phase B (OTF realism, i-LN, KD/ECO, dynamo exporter, ...) | OTF realism: **design draft 2** (Codex-reviewed); steps **O1-O3 done** (antialiased resampler and `resize_mode*` keys; per-sample `noise_prob*` / `jpeg_prob*` gates, `blur_prob`, the clean path; the unsharp mask `sharpen_*` at LQ scale; all opt-in, defaults bitwise unchanged), O4 (`degrade.py`) and O5 (readme, templates, Studio fields) open; the owner's answers Q1-Q4 open. The rest not started; the owner ordered Phase B after R3GAN | `stage3/otf/DESIGN.md`, `stage3/REVIEW.md` "Final list" |
| NeoSR Studio (`studio/`) | prototype: real recipe model, TOML compiler and validator; **training is simulated** (no adapter to `train.py`). Catalog synced to the trainer on 2026-11-10 and guarded by a contract | `studio/README.md`, "Studio contract" below |

Last full suite: 2047 passed, 0 failed, 377 skipped (CPU, commit 504ad50). Later commits ran scoped tests only.

## Open work, by priority
### Needs the Linux machine (GPU or the suite)
1. **Full suite on the current head** (CPU + CUDA): covers f0738ee (hma), bbd935c (PLK), the Studio merge and
   everything in "Waiting for the Linux machine".
2. **R1/R2 penalty pass costs ~614 ms per update** (719 vs 105 ms; R4). Profile one regularised D update
   (`torch.profiler`; suspect: double backward of the grouped 3x3 and the depthwise FIR convs), fix it or fall back
   to lazy regularisation (`reg_every` 4 or 16, a measured deviation from the paper). The +30 % cost gate cannot pass
   otherwise, and round 1 of the A/B would take ~21 GPU-hours.
3. **R6: R3GAN A/B** rounds 1-2, the second seed and the confirmation set; commands in `stage3/PROGRESS.md`.
   Includes the missing bf16 replicate of R4 (the raw R1/R2 ratio 0.82-0.83 sits next to the 0.8 bound).
4. **Is the template discriminator worth its cost?** `stage2/analysis/gpu_profile.md:204-213`: metagan is at chance
   from ~2.4k updates, yet all GAN templates pay ~14 % per update for it. The A0 vs A1 arms of R6 answer it.
5. **Phase 0 A/Bs** E5 (L1 pretrain then GAN), E2 (LR decay), E3 (no LDL), E4 (DISTS) on the R5 dev/confirmation
   split with two seeds. Template defaults today rest on one architecture, one seed and 67 validation tiles.
6. **A long run**: nothing has run past 20k updates; no run crossed the templates' `wavelet_init = 80000` switch
   (the stage-1 follow-up S.1 asked for >= 20k before the GAN defaults change; CONSENSUS asks for 80k+).
7. Gradient clipping under R1/R2: D is clipped in ~50 % and G in ~90 % of the updates of the 2k R3GAN run; the
   `noclip` arm of round 2 decides.
8. Opt-in switches waiting for an A/B (`stage2/DECISIONS.md`, last section): `tight_canvas`, `aug_antialias`, CRAFT
   `pad_extra_window = false`, span_fast `fan_in_init`, `"spike"` clip, `rpgan`, R1/R2, `fdl_opt.checkpoint`,
   realplksr norm init, RGT train/eval depth; realplksr `layer_norm`/`icnr` speed and 8k A/B (A6).
9. Unverified offline (A1, A3): `save_best` under DDP (reasoning only), a real PIPAL weights load, official VGG-LPIPS
   lin parity, 1080p LPIPS peak memory; DDP with NCCL (only gloo on one GPU is tested).

### Possible on the Windows machine (ordered by the compute it needs; least first)
**No compute (reading, design, docs, TypeScript):**
1. **Phase B, OTF realism: design** (the owner's main goal: real-world photographs): **done as draft 2**,
   `stage3/otf/DESIGN.md`. Open: the owner's answers Q1-Q4 (above all 50+ real LQ samples for the holdout set R
   and the residual analysis O4b).
2. **Closure matrix of the stage-2 audit**: 167 findings, `stage2/PLAN.md` assigns ~96 ids to lanes; the disposition
   of the rest (done / opt-in / rejected / open) is not recorded.
3. **Penalty-pass cost note** before the profiling session: read `r3gan_arch.py` and the `gan_loss` penalty path,
   list the double-backward candidates (grouped 3x3, depthwise FIR) and the cheap alternatives to try first.
4. **Studio, next steps** (Node only): arch constructor parameters in the surface contract (ast of the arch
   signatures), the real adapter (`server/` spawning `train.py`, parsing the log and TensorBoard events into the
   simulator's event shapes; testable against a recorded log), ONNX capability badges from `no_onnx`/`onnx_static_only`.
5. **A/B harness into the repository** as `scripts/ab/` (it exists only in the gitignored `experiments/r3gan_ab/harness`
   on the Linux machine): needs one copy step there, then review and cleanup can happen here.

**Seconds to minutes of CPU (targeted tests):**
6. **Test infrastructure**: pytest markers (`cuda`, `slow`, `fixture`) and a fast CPU subset in `pyproject.toml`
   (this machine needs it most); the three fixture paths (`tests/test_checkpoint.py:19`, `tests/test_entry.py:27`,
   `tests/test_metrics.py:17`) from an environment variable with today's values as defaults; a GitHub Actions
   workflow for the fast subset plus the Studio checks (stage-1 follow-up S.4).
7. **Windows portability of the tests** (found 2026-11-10 by the first runs here):
   `tests/test_runtime.py::test_two_rank_gloo_shares_rank0_seed` fails with "unsupported gloo device" (the gloo
   backend cannot resolve the Windows host name; the DDP tests need a skip or a loopback setting on win32, and the
   readme should say DDP is Linux-only if that holds); text reads without an encoding break on a non-UTF-8 locale
   (fixed in `test_s3_hygiene.py`; `test_entry.py:171,191`, `test_s3_exports.py:154`, `test_s3_bench.py:69` read
   run-generated files the same way and were not run here); `tests/test_s2_data.py`:
   `test_scandir_skips_hidden_and_non_images` needs the symlink privilege (WinError 1314) and `test_path_list`
   decodes a subprocess's output as UTF-8.
8. **Phase B, OTF realism: code** steps O1-O5 of the design (resampler, probabilities and clean path, sharpening,
   `degrade.py`, schema/readme/Studio): opt-in, default-preserving, unit-tested on tiny tensors. Needs the owner's
   go-ahead (Q3: before or after the R3GAN A/B).
9. **Small core items**: hma `check_image_size` picks reflect/replicate at trace time (dynamic ONNX pads wrong
   for inputs under 8/16 px; `neosr/archs/hma_arch.py:1375`); the dead `rep_rt4ksr` block in `convert.py:32`;
   one defaults table for `total_iter = 1000000` (6 call sites; S.5); per-arch capability metadata (S.11).

### Waiting for the Linux machine (changes not yet covered by a full suite run)
| date | change | check there |
|---|---|---|
| 2026-11-10 | `scripts/studio_surface.py`, `tests/test_studio_surface.py` | passed on Windows CPU (3 tests) together with test_config, test_hygiene, test_imports, test_registry, test_runtime (minus the gloo test), test_s3_hygiene: about 370 passed; there: the full suite |
| 2026-11-10 | `tests/test_s3_hygiene.py`: repo files read as UTF-8 | passed on Windows (6 tests); there: the same file |
| 2026-11-10 | OTF O1: `neosr/data/resample.py`, `resize_mode*` keys in `otf_degradation.py` and `config.py`, `tests/test_s3_otf.py` | passed on Windows CPU: test_s3_otf, test_otf_degradation (frozen reference), test_config, test_studio_surface, test_s2_data (380 passed; 2 Windows-only failures, item 7). There: the CUDA rows of test_otf_degradation, `test_degrader_no_host_sync` with AA modes on, the full suite, and the cost (O6) |
| 2026-11-10 | OTF O2: gates and clean path in `otf_degradation.py`, `blur_prob` in `otf_dataset.py`, schema keys, `tests/test_s3_otf.py` (O2 part) | passed on Windows CPU with the same files as O1 (454 passed). There: `test_no_host_sync_with_every_option_on` (CUDA only, skipped here), the CUDA generator path of `_mask`, a multi-worker loader run with `blur_prob`, the full suite |
| 2026-11-10 | OTF O3: `sharpen_prob` / `sharpen_sigma` / `sharpen_amount` in `otf_degradation.py`, autocast disabled inside the Degrader call, schema keys, tests | passed on Windows CPU (499 passed with the files of O1). There: the CUDA rows, the no-host-sync test (it now includes sharpening), the full suite |

## Studio contract
`scripts/studio_surface.py` writes `studio/shared/catalog/neosr-surface.json` from `ARCHS`, `LOSSES`, `METRICS`,
`OPTIMIZERS`, `SCHEDULERS`, `config.SCHEMA`, the gan types and `convert.py`'s export tables. Two tests hold the
chain together:
- `tests/test_studio_surface.py` (pytest): the JSON is current; the static read equals the imported tables.
- `studio/src/app/core/catalog-contract.spec.ts` (`ng test`): the Studio catalog equals the JSON; `npm run contract`
  also validates every Studio template's TOML with the trainer's `load_config` + `validate_train`.

After changing a trainer table or `SCHEMA`: `python scripts/studio_surface.py`, commit the JSON, then fix what
the Studio contract reports. `design/neosr_config_surface.md` is a prose snapshot from before stage 2; the JSON
is the truth.

## Closed follow-ups (were listed as open somewhere)
| item | closed by |
|---|---|
| adan_sf SF recurrence (stage-1 R4.1) | stage 2: sound SF-Adan, `STATE_VERSION = 2` (`stage2/DECISIONS.md` 3) |
| final save before the final validation (R1.4) | stage 3 A1: the state is saved again when the final validation changed the best record (9cfc1da) |
| `gan_opt.start_iter`, R1/R2 regulariser (S.1, code part) | stage 2 (ST-05, ST-06), generalised in stage 3 A4 (24d07a8) and R2 (9c4cca2); the long acceptance run stays open |
| warm-up for plain optimizers (S.2) | stage 2 ST-10: `warmup_steps` for every optimizer (trainer LinearLR) |
| D compile (S.12, D part) | stage 2 GP-01/ST-08 |
| AMP on a CPU run (S.13) | `runtime.py:67` refuses `use_amp` without CUDA in training |
| PLKConv2d eval-mode in-place write | bbd935c |
| hma / hma_medium dynamic ONNX at batch 2 | f0738ee |
| stage 2 "not merged, not pushed" | merged into `master` with the Studio prototype (05954e0), pushed |
