# Project status (single entry point)

Updated 2026-11-11 (end of session 9). Read this first; the per-stage logs (`PROGRESS.md`, `stage2/PROGRESS.md`,
`stage3/PROGRESS.md`) hold the history and the numbers. Update this file whenever a track changes state.

## Machines and what runs where (owner, 2026-11-10)
| machine | checkout | role |
|---|---|---|
| Linux workstation, RTX 4090 | `/home/korvin/apps/neosr-dev`, venv with torch. The worktree `.worktrees/refactor` (branch `refactor/stabilize`) is fully merged into `master`, as are the `s2/*` branches; it is kept only for its gitignored `experiments/r3gan_ab` (1.4 GB: the pretrain, the R5 splits, the A/B harness) | **every real test and measurement**: the pytest suite (CPU and CUDA), training runs, A/Bs, profiling, benchmarks, ONNX verification. The A/B harness, the pretrain and the data live here only (`experiments/`, gitignored) |
| Windows workstation (Ryzen 5 5500U, 6 cores, 31 GB RAM, no GPU) | `C:\work2000\neosr-dev`, CPU-only `.venv` (torch 2.9.1+cpu) since 2026-11-10 | code analysis and review, documentation, small core changes checked by **targeted** CPU tests, NeoSR Studio (the UI prototype). Slow CPU: no suite runs, no sweeps, no training |

Consequences:
- On the Windows machine prefer work that needs no compute at all; run single test files, `-k` selections or at
  most the **fast CPU subset** `pytest -m "not slow and not cuda"` (measured: 360 config/registry/import tests take
  2 min, one `import neosr.*` subprocess ~10 s). Never start the full suite, the per-arch sweeps (`test_archs*`, the
  `test_s2_*` arch files, `test_losses_pipeline`), ONNX export sweeps or a training run there.
- Test markers (set in `tests/conftest.py`, registered in `pyproject.toml`): `cuda` (needs a CUDA device: read
  from the tests' CUDA skip marks and device parameters) and `slow` (the reviewed `SLOW` list: per-arch sweeps,
  per-arch ONNX exports, the shipped-config constructors, torch.compile, the import subprocesses). The fast CPU
  subset ran 1800 passed, 16 skipped in 98.6 s on the Linux machine at fd4c0e47
  (`CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 .venv/bin/python -m pytest tests -m "not slow and not cuda" -q
  -p no:cacheprovider`; the full CPU suite 7 min). It needs the network once: ~31 tests load VGG16/19,
  LPIPS/TOPIQ (torch hub cache) and the DISTS
  weights (`neosr/losses/dists_weights.pth`). `.github/workflows/tests.yml` runs it plus the Studio checks on
  push/PR (weights cached).
- A core change made on the Windows machine counts as **verified only for the tests named next to it**; list it
  under "Waiting for the Linux machine" below until the full suite ran there, and keep such changes small.
- Torch-free checks: `python scripts/studio_surface.py` (`--check`, `--validate <toml>`) and everything under
  `studio/` (`npm run typecheck`, `npx ng test --watch=false`, `npm run contract`, `ng build`).
- Paths like `/home/korvin/...` in the logs are the Linux machine's. The two machine-specific test fixtures default
  to them and are overridden by `NEOSR_NET_G_2000` (a trained esrgan `net_g_2000.pth`: `test_checkpoint.py`,
  `test_entry.py`) and `NEOSR_VAL_GT` (the validation GT images: `test_metrics.py`); without the file the tests skip.

### The Windows CPU environment (not `uv sync`: the lock pins CUDA wheels)
```bash
python -m pip install --user uv
python -m uv venv --python 3.13 .venv
python -m uv pip install --python .venv/Scripts/python.exe torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m uv export --frozen --no-hashes --no-emit-project | grep -vE "^(torch|torchvision|nvidia-|triton|onnxruntime-gpu)" > req.txt
python -m uv pip install --python .venv/Scripts/python.exe -r req.txt onnxruntime pytest ruff
.venv/Scripts/python.exe -m pytest tests/test_config.py -q -p no:cacheprovider
.venv/Scripts/python.exe -m pytest tests -m "not slow and not cuda" -q -p no:cacheprovider   # fast CPU subset
```
Versions follow `uv.lock` except the CPU builds of torch/torchvision and `onnxruntime` instead of `onnxruntime-gpu`.

## Not an open point: the token in the remote URL
The `origin` URL in `.git/config` embeds a GitHub token. The owner keeps it deliberately (private single-user
machines; `DECISIONS.md`, 2026-11-10). Do not report it again and do not print the URL in logs or docs.

## What the project is (owner, 2026-11-10)
neosr-dev is a reworked, optimised trainer of its own: clean and well-thought-out code, speed, a small footprint,
good structure, compactness, and a stable, reliable core. traiNNer-redux is not its upstream but a competitor with
often bloated code: a source of ideas and examples at most, never something to imitate (`DECISIONS.md`).
Disputed or open questions go to the owner; agents do not settle them alone.

## Owner priorities (2026-11-10): read before picking work
**The next session starts from `next_tasks.md` in the repository root.** Session 7 (2026-11-10) finished the three
tasks the owner had fixed: the registries (3 below), the audit closure matrix (1a) and the small core items (1b).
Session 8 (Linux) ran the full suite and the small core items of the audit. What is left (end of session 9,
2026-11-11): the audit findings still open (`stage2/CLOSURE.md`; A-04 and L-07 wait on >= 20k runs; their 8k screens
ran), the R3GAN A/B (R6, now affordable), then Studio. Session 9 did the ONNX sweep, the GPU measurements, five 8k
screens and the R1/R2 penalty fix (`stage3/PROGRESS.md`, "Session 9").

**What "the core" means (owner, 2026-11-10, `DECISIONS.md`):** the training framework and its ecosystem
(losses, metrics, schedulers, optimizers, the GAN implementations, utilities), its speed and stability, as the
foundation for new architectures. The existing archs rank near last; arch-specific findings are deferred unless
a framework contract needs them. Small, exact core fixes may be made without asking first.

The owner sets the order; an agent does not promote a track on its own reading of the documents. Current order:
1. **A stable, reliable core** ("works like a Swiss watch": fast, reliable, good quality) before any new feature
   work. Concretely: (a) the closure matrix of the stage-2 audit: **done** (`stage2/CLOSURE.md`: 167 findings =
   131 done, 16 opt-in, 7 rejected, 9 deferred, 4 open on 2026-11-11), the 4 open ones remain; (b) the small core items: **done** (the pad-to-window helper for 14 archs instead of hma alone, the
   dead `rep_rt4ksr` block, `config.total_iter`; per-arch capability metadata: the registry half is decided,
   the Studio half is open); (c) what the Linux machine must confirm (full suite: done 2026-11-10, session 9: done
   2026-11-11; the ONNX sweep: done 2026-11-11; items of "Needs the Linux machine").
2. **NeoSR Studio / UI**: finish it (next steps in "Possible on the Windows machine", item 4).
3. **Registration: done** (session 7). The five registries are back (`ARCH_REGISTRY`, `LOSS_REGISTRY`,
   `METRIC_REGISTRY`, `DATASET_REGISTRY`, `MODEL_REGISTRY` with `@...register()`), lazy: the decorator is a
   declaration, a registry reads the decorators from the source text on first use (16 ms for the arch folder)
   and imports only the requested module. The owner chose mechanism A1 of `design/T7_registration.md`; the
   hand-written tables are gone. Waits for the full suite on the Linux machine.

**Parked: OTF realism (Phase B).** Steps O1-O3 are merged, opt-in and change no default; O4 (`degrade.py`), O5
(readme, templates, Studio fields) and the A/B wait until the owner reopens the track. It was started on an
agent's recommendation ahead of the owner's priorities; the recorded order was "Phase B after R3GAN". If it is
reopened, the owner decides its scope first. The competitor traiNNer-redux has comparable degradation options;
they may serve as ideas (`stage3/competitor/data_degradation.md` reviews them, defects included), not as a
model to follow.

## Tracks
| track | state | evidence / where to resume |
|---|---|---|
| Stage 1: stabilisation (phases 0, L, A, waves W0-W7) | done, merged | `PLAN.md`, `PROGRESS.md`; 1192 tests at the end |
| Stage 2: speed, reliability, memory, templates | done, merged | `stage2/DECISIONS.md`, `stage2/PROGRESS.md` |
| Stage 3 Phase A (A1-A9, port from traiNNer-redux) | done, merged | `stage3/PROGRESS.md` "Phase A closed" |
| Stage 3 Phase 0 (E0-E5: evaluation protocol, config-only A/Bs) | **not started** (E0 partly covered by the R5 split) | `stage3/CONSENSUS.md` Phase 0 |
| R3GAN track | code done (R1-R3), data done (R5), qualification R4: bf16 PASS (provisional, no replicate), **cost gate PASS since 2026-11-11** (penalty fix 94c36bde: 122.5 vs template 109.9 ms/update, +11 %; was 732); 8k screen 2026-11-11: R3 LOSS vs B0 on DISTS +0.0017 (1.1x the floor, marginal; PSNR -0.81 dB is not a gate metric), TIE vs A1 (LPIPS -0.0067 better), the fewest late G-gradient spikes of the changed-D arms; an 8k fine-tune does not judge R6; **A/B (R6) not run**: a 20k round now takes ~4 GPU-h (was ~21) | `stage3/r3gan/RESULTS.md` section 4, `stage3/PROGRESS.md` |
| Stage 3 Phase B (OTF realism, i-LN, KD/ECO, dynamo exporter, ...) | **parked by the owner (2026-11-10)**. OTF realism: **design draft 2** (Codex-reviewed); steps **O1-O3 done** (antialiased resampler and `resize_mode*` keys; per-sample `noise_prob*` / `jpeg_prob*` gates, `blur_prob`, the clean path; the unsharp mask `sharpen_*` at LQ scale; all opt-in, defaults bitwise unchanged), O4 (`degrade.py`) and O5 (readme, templates, Studio fields) open; the owner's answers Q1-Q4 open. The rest not started; the owner ordered Phase B after R3GAN | `stage3/otf/DESIGN.md`, `stage3/REVIEW.md` "Final list" |
| Registries (design T7) | **done on Windows, waits for the Linux suite** | `design/T7_registration.md`, `neosr/utils/registry.py`, `tests/test_registry.py` |
| Stage-2 audit closure | matrix done; 2026-11-10: A-11 and the metrics sub-items done, 7 arch-specific findings deferred (owner's goal); 2026-11-11: T-11 rejected, ST-14 and L-16 deferred (owner); session 9 added measurements and 8k screens to T-02, L-03, L-07, L-08, A-04, ST-06, GP-04 (`stage2/DECISIONS.md` 14-17); **4 open** (the LDL flip failed its seed replicate): A-04 and L-07 (8k screens done, both wait on >= 20k runs), L-15, D-16 (parked OTF track) | `stage2/CLOSURE.md` |
| NeoSR Studio (`studio/`) | prototype: real recipe model, TOML compiler and validator. Training is simulated by default; **real-trainer mode experimental** (`STUDIO_TRAINER=real`, e97902ba: spawns `train.py`, streams its log as run events, stop/resume tested end to end on the CPU; `/api/environment` reports the real machine, 587cbdb2). Arch constructor parameters in the surface contract (S.11 done: 176990cc, 2ede3c2e) | `studio/README.md`, `studio/server/API.md`, "Studio contract" below |
| A/B harness (`scripts/ab/`) | done 2026-11-11 (c03c44f0, f2f7d917): `run_ab.py`, `analyze.py` (paired CI over source groups, the gate), `queue.sh`; data and runs stay in `experiments/` | `scripts/ab/README.md` |

Last full suite: 2647 passed, 0 failed, 199 skipped, 5 xfailed (CPU + CUDA on the RTX 4090, 10 min 23 s,
2026-11-11) at 100a5881, the session-9 head (code unchanged after it; docs only). Before:
2583 passed, 0 failed, 199 skipped, 5 xfailed (CPU + CUDA on the RTX 4090, 12 min, 2026-11-10) on the session-8
commits after d1afe9f (the audit's small core items, the Windows test fixes).
Before: 2566 passed at e0f387c, after a test-only fix: `tests/test_s3_losses.py` took CUDA images from `test_gan.images`
with a CPU critic, so its two hinge tests failed wherever CUDA is visible (latent since stage 3 A4); the file now
pins its images to the CPU. Before that: 2047 passed, 377 skipped (CPU only, 504ad50).

## Open work, by priority
### Needs the Linux machine (GPU or the suite)
1. **Full suite on the current head** (CPU + CUDA): **done** at 100a5881, 2026-11-11 (see "Last full suite" above);
   what the suite does not cover is in "Waiting for the Linux machine".
2. **R1/R2 penalty pass: done 2026-11-11** (94c36bde). ATen's conv double backward ran one conv per group (96 %
   of the CUDA time); r3gan's convs now differentiate through `conv_transpose2d` (exact). The R3GAN recipe runs
   at 122.5 ms/update (was 732; template 109.9): the cost gate passes (`stage3/r3gan/RESULTS.md` section 4).
3. **R6: R3GAN A/B** rounds 1-2, the second seed and the confirmation set; commands in `stage3/PROGRESS.md`.
   Includes the missing bf16 replicate of R4 (the raw R1/R2 ratio 0.82-0.83 sits next to the 0.8 bound).
4. **Is the template discriminator worth its cost?** `stage2/analysis/gpu_profile.md:204-213`: metagan is at chance
   from ~2.4k updates, yet all GAN templates pay ~14 % per update for it. The A0 vs A1 arms of R6 answer it.
   2026-11-11: the unet D is no cheaper on the GPU (+18 % per update, +848 MiB vs metagan); 8k screens unet@0.1,
   metagan@0.03 and A1 (no D) vs B0: A1 LOSS (B0's D helps all three perceptual metrics: TOPIQ +0.0088, LPIPS -0.0077,
   DISTS -0.0018 over A1), metagan@0.03 LOSS (LPIPS +0.0027; WIN vs A1), unet@0.1 TIE (LPIPS -0.0080, 4x the floor,
   TOPIQ/DISTS inside; WIN vs A1; seed 2 WIN vs B0s2: LPIPS -0.0079, DISTS -0.0030; its D clipped in 100 % of the
   updates, 31 late G-gradient spike windows; +18 % per update by the bench). At 8k the template D earns its cost; the
   GAN health log (DR-02) is in every run's log.
5. **Phase 0 A/Bs** E5 (L1 pretrain then GAN), E2 (LR decay), E3 (no LDL: screened at 8k on 2026-11-11, TIE; flip not
   merged: seed 2 LOSS vs B0s2 on LPIPS +0.0029, a `grad_norm_g` spike of 248), E4 (DISTS) on the R5 dev/confirmation
   split with two seeds; `wavelet_init = 0` screened at 8k: LOSS (PSNR -0.11, LPIPS +0.0032, DISTS +0.0032); the
   template's 80000 stays. Template defaults today rest on one architecture, one seed and 67 validation tiles.
6. **A long run**: nothing has run past 20k updates; no run crossed the templates' `wavelet_init = 80000` switch
   (the stage-1 follow-up S.1 asked for >= 20k before the GAN defaults change; CONSENSUS asks for 80k+).
7. Gradient clipping under R1/R2: D is clipped in ~50 % and G in ~90 % of the updates of the 2k R3GAN run (the
   8k screen: G 95-100 % in the first 2k); the `noclip` arm of round 2 decides. On the template the fixed clip is
   near idle and `"spike"` ties (`stage2/DECISIONS.md` 14).
8. Opt-in switches waiting for an A/B (`stage2/DECISIONS.md`, last section): `tight_canvas`, `aug_antialias`, CRAFT
   `pad_extra_window = false`, span_fast `fan_in_init`, `"spike"` clip, `rpgan`, R1/R2, `fdl_opt.checkpoint`;
   realplksr `layer_norm`/`icnr` speed and 8k A/B (A6).
9. Unverified offline (A1, A3): official VGG-LPIPS lin parity; DDP with NCCL (only gloo on one GPU is tested).
   Done 2026-11-11: `save_best` under 2-rank gloo DDP (1951d9f3, `tests/test_ddp_best.py`); r3gan + rpgan + R1/R2
   and the health log under 2-rank gloo DDP (c10b89c5); the session's A/B runs scored TOPIQ-pipal with the
   hub-cached `cfanet_fr_pipal_res50` weights; validation metric memory (CUDA, fp32, through `build_metric`, peak
   above the resident nets; 4K for DISTS/TOPIQ extrapolated, memory exactly linear in pixels at 1080p, 1440p, 1620p):

   | metric | 1080p | 4K |
   |---|---|---|
   | psnr | 141 MiB, 5 ms | 566 MiB, 7 ms |
   | ssim | 266 MiB, 0.55 s | 1067 MiB, 2.2 s |
   | lpips alex | 269 MiB, 19 ms | 1075 MiB, 61 ms |
   | dists | 2200 MiB, 0.23 s | ~8.6 GiB, ~1.1 s (est., 1061 MiB/Mpx) |
   | topiq pipal | 1531 MiB, 0.13 s | ~6.0 GiB, ~0.7 s (est., 738 MiB/Mpx) |

   Tiled validation does not lower these (the learned metrics get the full-size images). A 24 GB card fits 4K; on a
   12 GB card 4K DISTS would take the CPU fallback of `image._metric` (inferred) and TOPIQ is marginal. No cheap
   exact fix: DISTS needs whole feature maps per stage and is already streamed per stage (MT-13).

### Possible on the Windows machine (ordered by the compute it needs; least first)
The order of work is "Owner priorities" above; this list only says what fits this machine. Items 1 and 8
(OTF realism) are parked.
**No compute (reading, design, docs, TypeScript):**
1. **Phase B, OTF realism: design** (the owner's main goal: real-world photographs): **done as draft 2**,
   `stage3/otf/DESIGN.md`. Open: the owner's answers Q1-Q4 (above all 50+ real LQ samples for the holdout set R
   and the residual analysis O4b).
2. **Stage-2 audit, the open findings fixable here: done 2026-11-10** on the Linux machine (A-11, the MT-08/14/15
   sub-items, the A-07/W-09 doc mismatch); the arch-specific ones (A-12, A-17, W-16) are deferred by the owner's
   goal. T-11 rejected and ST-14 deferred by the owner (2026-11-11); D-16 waits with the parked OTF track
   (`stage2/CLOSURE.md`).
3. **Penalty-pass cost: done 2026-11-11** on the Linux machine (Needs the Linux machine, item 2).
4. **Studio, next steps** (Node only): arch constructor parameters in the surface contract and the real adapter
   are done (2026-11-11; real mode experimental, open owner question: fixed params shown disabled rather than
   hidden); left: ONNX capability badges from `no_onnx`/`onnx_static_only`, the review leftovers in
   `studio/server/API.md`.
5. **A/B harness into the repository: done 2026-11-11** (`scripts/ab/`).

**Seconds to minutes of CPU (targeted tests):**
6. **Test infrastructure: done 2026-11-11** on the Linux machine (stage-1 follow-up S.4): markers `cuda`/`slow`
   and the fast CPU subset (see "Consequences" above), the fixture paths from `NEOSR_NET_G_2000`/`NEOSR_VAL_GT`,
   `.github/workflows/tests.yml` (fast subset; Studio typecheck, unit tests, contract). The workflow has not run
   yet, and the subset's time on this machine is unmeasured.
7. **Windows portability of the tests: done 2026-11-10** (Linux-verified only; the next Windows run confirms):
   the gloo DDP test skips on win32 (gloo cannot resolve the Windows host name; whether DDP works on Windows at
   all is unverified), run-generated files are read and written as UTF-8 (`test_entry.py`, `test_s3_exports.py`,
   `test_s3_bench.py`), the symlink test skips without the privilege (WinError 1314), `test_path_list` keeps the
   undecodable-bytes name to POSIX, and the `extract_subimages` subprocess output is decoded as UTF-8.
8. **Phase B, OTF realism: code** steps O1-O5 of the design (resampler, probabilities and clean path, sharpening,
   `degrade.py`, schema/readme/Studio): opt-in, default-preserving, unit-tested on tiny tensors. Needs the owner's
   go-ahead (Q3: before or after the R3GAN A/B).
9. **Small core items: done in session 7** (see "Closed follow-ups"); the Studio half of S.11 done 2026-11-11
   (constructor parameters read by ast into the surface, the catalog checked against them).

### Waiting for the Linux machine (what the full suite does not cover)
The full suite at e0f387c (CPU + CUDA, 2026-11-10) covers the Windows-made changes of 2026-11-10: the Studio
surface script and test, `test_s3_hygiene`, OTF O1-O3 (the CUDA rows and the no-host-sync tests ran, none
skipped), the five registries (all per-arch files ran), `pad_reflect` in 14 archs (eager), `config.total_iter`,
safmn below 8 px and `color = "y"`. The ONNX sweep of the ten archs whose pad changed ran on 2026-11-11 (CPU):
all export and verify dynamically at `convert.py`'s three shapes; `srformer_medium` left `STATIC_ONLY` (48469bc2:
ORT matches eager within 6.3e-7 at six multi-window shapes). Still open:

| date | change | check there |
|---|---|---|
| 2026-11-10 | OTF O1-O3 | the cost of the new options (O6); a multi-worker loader run with `blur_prob` (O2) |

## Studio contract
`scripts/studio_surface.py` writes `studio/shared/catalog/neosr-surface.json` from the five registries (it loads
`neosr/utils/registry.py` by path, torch-free), `OPTIMIZERS`, `SCHEDULERS`, `config.SCHEMA`, the gan types and
`convert.py`'s export tables. Two tests hold the chain together:
- `tests/test_studio_surface.py` (pytest): the JSON is current; the script runs without torch; its read equals
  the imported registries and tables.
- `studio/src/app/core/catalog-contract.spec.ts` (`ng test`): the Studio catalog equals the JSON; `npm run contract`
  also validates every Studio template's TOML with the trainer's `load_config` + `validate_train`.

After adding or changing a registered type, a trainer table or `SCHEMA`: `python scripts/studio_surface.py`, commit the JSON, then fix what
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
| registration usability (T3 D3 tables) | session 7: the five registries, design T7 (ad84712) |
| one defaults table (S.5) | session 7: `config.total_iter` (fd30790); `max_skip_streak` and `keep_states` each have one reader default |
| per-arch capability metadata (S.11), registry half | session 7, owner decision: the decorator carries `scale` (and `choices` for metrics); ONNX gaps stay in `convert.py` |
| hma dynamic ONNX pads tiny inputs wrong | session 7: `arch_util.pad_reflect` for all 14 archs with that code (3e5887a) |
| dead `rep_rt4ksr` block in `convert.py` | session 7 (8bbdd04) |
| disposition of the 167 stage-2 findings | session 7: `stage2/CLOSURE.md` |
