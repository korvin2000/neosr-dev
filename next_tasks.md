# Next tasks (handoff for a new session)

Rewritten 2026-11-11 (session 9, Linux machine) after the owner's decisions on T-11 (rejected) and ST-14
(deferred) and a re-ranking for **one 4-hour session**: as many useful core changes and measurements as fit,
parallel lanes, no GPU run longer than ~20 minutes. Session 7's work is in `docs/refactor/stage3/PROGRESS.md`
(sessions 7-8); nothing of it is left except the ONNX sweep. `master` = `origin/master` at 8f2f6cc.

**The owner's focus for this session:** a tuned, optimised core: training quality, GAN modes, speed,
efficiency and stability, with the losses, metrics, strategies and optimisations around them. Studio may move
in parallel. Multi-hour runs are out of scope.

| rank | task | kind | GPU | section |
|---|---|---|---|---|
| 1 | Short GPU measurements: speed of each candidate option, per-term gradient shares, the R1/R2 penalty profile | measurement | ~25 min, alone | 3.1 (G1) |
| 2 | 8k A/Bs on the realplksr + metagan template: `grad_clip = "spike"` (T-02), no LDL (L-07), bf16 FDL backbone (L-03/GP-04), plus a seed replicate | quality A/B | ~75 min, two arms at a time | 3.1 (G2) |
| 3 | R1/R2 penalty pass: find and fix the ~614 ms per regularised update | GAN speed, code | short windows | 3.2 (P) |
| 4 | Test infrastructure: pytest markers, a fast CPU subset, fixture paths from the environment, a CI workflow | reliability, code | none | 3.2 (T) |
| 5 | ONNX sweep of the 10 archs that pad through `pad_reflect` | verification | none (CPU ORT) | 3.2 (O) |
| 6 | Studio: the real adapter (spawn `train.py`, parse its log into the simulator's events) | Studio, code | none | 3.2 (S), 5 |
| 7 | Apply what the owner decides at the A/B gate (template or default flips) | code | none | 3.2 (F) |
| 8 | Merge the lanes, full suite CPU + CUDA, docs | verification | ~12 min, alone | 3.3 |

Out of the 4-hour window (each needs hours of GPU): the R3GAN A/B R6, A-04 at >= 20k, Phase 0 E2/E4/E5 at
20k with two seeds, the 80k+ long run, the rpgan / R1-R2 acceptance run (>= 20k), DDP with NCCL (two GPUs).
Parked by the owner: OTF realism (`docs/refactor/stage3/otf/DESIGN.md`), all of stage-3 Phase B. Deferred by the owner
(2026-11-11): section 6 (pyvips) and L-16 (fp32 generator tail). D-16 waits with OTF. Run decisions are
delegated: the main session decides after consulting Fable (t5, high) and logs each decision (section 3.4).

## 0. Rules of the house (they caused friction once already)
- **The owner sets the priorities.** Do not promote a track because a document calls it a goal; when the next step
  is not named here, ask. (`AGENTS.md`, `docs/refactor/STATUS.md` "Owner priorities".)
- **Goal of the current phase:** a core that "works like a Swiss watch": stable, reliable, fast, good quality.
- **What this project is:** a reworked, optimised trainer of its own; the goal is the framework, not the
  existing archs (`docs/refactor/DECISIONS.md`, 2026-11-10). Measure of every change: clean, well-thought-out
  code, speed, a small footprint, good structure and compactness.
- **traiNNer-redux is a competitor**: ideas at most, nothing imitated or copied; "it has this" is no argument.
- **No hasty decisions.** Ask the owner on anything disputed or open; say what is verified and what is assumed.
- **Every code step:** implement -> targeted tests -> Codex review of the diff (read-only) -> fix -> one commit ->
  log in the track's `PROGRESS.md`, update `STATUS.md`. **Push** only on the owner's word.
- **Defaults do not change silently.** Exact rewrites ship as defaults behind parity tests; behaviour changes ship
  opt-in until an A/B on the owner's data decides (`docs/refactor/stage2/DECISIONS.md`, principles); the owner
  flips a default, not an agent.
- **The GPU is one resource.** Timing (bench, profiler, the full suite) runs alone on it; quality-only A/B arms
  may share it two at a time. `scripts/bench.py` refuses nothing by itself: check `nvidia-smi` first.
- The token in the `origin` URL is intentional: do not report it, never print the URL (mask command output).

## 1. Orientation (5 minutes)
| what | where |
|---|---|
| single status page, machines, open work | `docs/refactor/STATUS.md` |
| owner and consensus decisions | `docs/refactor/DECISIONS.md` |
| stage logs (history, numbers) | `docs/refactor/PROGRESS.md` (stage 1), `stage2/PROGRESS.md`, `stage3/PROGRESS.md` (also sessions 7-9) |
| the 167 audit findings | `docs/refactor/stage2/CLOSURE.md` (131 done, 16 opt-in, 7 rejected, 9 deferred, 4 open) |
| the 8k A/B protocol and the template decisions | `docs/refactor/stage2/DECISIONS.md` ("Template objective", "Kept opt-in pending an A/B") |
| GPU profile of the templates (launch-bound step, clip at 100 %, FDL cost) | `docs/refactor/stage2/analysis/gpu_profile.md` section 0 |
| R3GAN track, R4 numbers, harness | `docs/refactor/stage3/PROGRESS.md` "PAUSED", `stage3/r3gan/RESULTS.md`; runs and harness in the gitignored `.worktrees/refactor/experiments/r3gan_ab/` |
| architecture designs | `docs/refactor/design/T1.md` (loss pipeline, training step), `T2.md` (model, checkpoints), `T3.md` (config), `T5_aether_qat.md`, `T7_registration.md` (registries) |
| user docs | `readme.md`, `docs/training_guide.md` |
| Studio | `studio/` (`README.md`, `DEVELOPING.md`, `server/API.md`); contract `scripts/studio_surface.py` -> `studio/shared/catalog/neosr-surface.json` |

### Commands
```bash
.venv/bin/python -m pytest tests/<file>.py -q                 # Linux; the full suite: tests/ (~12 min CPU + CUDA)
.venv/bin/python scripts/bench.py -opt options/train_realplksr.toml          # ms/update, peak VRAM; GPU alone
.venv/bin/python scripts/term_grads.py -opt options/train_realplksr.toml --batches 8   # per-term G gradient share
.venv/bin/python scripts/studio_surface.py --check            # after a registry, table or SCHEMA change
cd studio && npm run typecheck && npx ng test --watch=false && npm run contract
```
On the Windows machine: `.venv/Scripts/python.exe -m pytest tests/<file>.py -q -p no:cacheprovider`, single files
only; never the full suite, sweeps, benchmarks or training (`AGENTS.md`). pytest-xdist is not installed.

### Traps
- **Shell heredocs mangle scripts** (Git Bash): write helper scripts into the scratchpad with the Write tool.
- **Windows: the Edit tool rewrites a CRLF file with LF**; keep line endings consistent there.
- `ruff format` (0.17.1) reformats existing lines of older files (line length 88): format only new files;
  `*_arch.py` is outside ruff.
- **Never `git stash` while a background test run or other agents read the tree.**
- **Parallel agents:** code lanes work in their own git worktrees; two lanes never edit the same file. A CPU
  pytest next to a GPU A/B: cap its threads (`OMP_NUM_THREADS=8`) so the A/B's data loaders keep up.
- Windows-only test failures unrelated to a change: listed in `STATUS.md`, portability item.
- Codex: `Agent` with `subagent_type: codex:codex-rescue`; the brief starts with `--model gpt-6.1-sol --effort high`
  and "read-only: advise only, do not edit files"; give it the commit hash and what was already verified.

---

## 2. Done since the previous handoff
| session | what | where |
|---|---|---|
| 7 (Windows) | the five registries (T7), `pad_reflect` for 14 archs, `config.total_iter`, the closure matrix | `stage3/PROGRESS.md` "Session 7" |
| 8 (Linux) | full suite 2583 passed (CPU + CUDA); A-11, MT-08/14/15; Windows test portability; the owner's "framework, not archs" | `stage3/PROGRESS.md` "Session 8" |
| 9 (Linux) | T-11 rejected, ST-14 deferred; docs refreshed; this plan | `DECISIONS.md` 2026-11-11 |

## 3. Session 9: the 4-hour plan (Linux, RTX 4090, 32 threads, 62 GB)

### 3.1 GPU lane (main thread; one owner of the GPU)
**G1, short measurements (T+0:10 to T+0:35, GPU alone).** Each answers a decision of G2 or lane P in minutes.
| measurement | command / method | answers |
|---|---|---|
| ms/update and peak VRAM: template; `fdl_opt.fp32_backbone = false`; without `[train.ldl_opt]`; `compile = true` | `scripts/bench.py` on temporary copies of `options/train_realplksr.toml` (scratch) | the speed half of L-03/GP-04 and L-07; today's baseline (stage-2 numbers predate GP-01 and stage 3) |
| per-term G gradient share (LDL, FDL, GAN, L1, ...) at a fresh G and at the R3GAN pretrain (`P0_pretrain`, EMA at 20k) | `scripts/term_grads.py --batches 8`, with and without `--net_g` | L-07 (does LDL do anything at today's weights), A-04 (the adversarial share), GP-04 |
| one regularised D update under `torch.profiler` (CUDA kernels, double backward) with the R3GAN recipe (`options/train_realplksr_r3gan.toml`, `reg_every = 1`), and the same penalty on metagan | a scratch script; trace and top-kernel table to the scratchpad | lane P: is the 614 ms in the r3gan D's grouped 3x3 / depthwise FIR double backward, or in the penalty path itself |

**G2, 8k A/Bs (T+0:35 to ~T+1:50).** Stage-2 protocol (`stage2/DECISIONS.md`): the owner's paired x4 data,
67 held-out pairs, the shipped realplksr + metagan template, 8000 updates, seeded, EMA weights, PSNR/SSIM/DISTS
every 1000. Configs and runs in the gitignored `experiments/` (scratch TOMLs), launched by a queue script, two
arms at a time (quality only; nobody reads their ms/update). Estimate ~15 min per arm alone (A0 eager was
97.6 ms/update in R4), ~25 min per pair: an assumption until the first pair is timed.
| arm | change vs template | finding | decides |
|---|---|---|---|
| B0 | none (current code) | reference | stage-2's 29.36 dB / 0.1787 predate stage 3 |
| B0s2 | seed 2 | noise floor | the tie margin: max(0.05 dB, \|B0 - B0s2\|) for PSNR, max(0.002, spread) for DISTS |
| B1 | `grad_clip = "spike"` | T-02 / GP finding 3 (the 1.0 clip acts on 100 % of updates; ST-13 called it harmless) | default clip of the templates (or of `config.py:449`) |
| B2 | no `[train.ldl_opt]` | L-07 / Phase 0 E3 | if it ties: delete the section from 85 templates (cost and simplicity) |
| B3 | `fdl_opt.fp32_backbone = false` | L-03 / GP-04 | if it ties: flip the default (`fdl_loss.py:327`, readme) for the measured speed-up |
| B5 (optional) | the winners together | interaction | only if two or more win |

**Owner gate (~T+1:50):** a table per arm (final PSNR/SSIM/DISTS, the curve every 1000, the margin, G1's speed);
the owner flips or keeps each. 8k is one architecture: a flip of a GAN default stays provisional until the
long run (STATUS item 6), and the doc entry says so.

**GPU windows after G2 (T+1:50 to T+2:45):** lane P's timing and CUDA tests; if time is left, a second seed of
an arm the owner wants to flip.

### 3.2 Code lanes (background agents, each in its own worktree; Codex review per lane; merged by the main thread)
| lane | goal and done-criterion | tier | files (no overlap) |
|---|---|---|---|
| **P** R1/R2 penalty pass | ms per regularised D update before/after (bench in a GPU window), the cause named from G1's profile; an exact rewrite (e.g. the depthwise FIR or the grouped 3x3 in a form whose double backward is cheap) behind a parity test on CPU and CUDA; fallback if no exact fix: the measured `reg_every` 4/16 costs recorded for the owner | t4 | `neosr/archs/r3gan_arch.py`, `neosr/losses/gan_loss.py` (penalty path), their tests |
| **T** test infrastructure | markers `cuda` and `slow` registered in `pyproject.toml`; a fast CPU subset (`-m "not slow and not cuda"`, target <= 3 min on Linux, picked from a `--durations` run); the three fixture paths (`tests/test_checkpoint.py:19`, `tests/test_entry.py:27`, `tests/test_metrics.py:17`) from one environment variable with today's values as defaults; `.github/workflows/` running the fast subset and the Studio checks (follow-up S.4); `STATUS.md` commands updated | t3 | `pyproject.toml`, `tests/conftest.py`, the three test files, `.github/workflows/` |
| **O** ONNX sweep | the 10 archs (dctlsa, ditn, drct, grformer, hat, hma, mfghmoe, microsr, srformer, swinir) export and verify at `convert.py`'s three shapes on CPU ORT as in stage 3 A8; `STATIC_ONLY` re-checked for `srformer_medium`; a failure is reported, not fixed in this lane | main thread, background script | none (results into `STATUS.md`) |
| **S** Studio adapter | `studio/server` spawns `train.py` behind a switch (the simulator stays the default) and turns its log lines into the simulator's event shapes (`server/API.md`, `shared/run.model.ts`); parser unit-tested against a recorded log excerpt (P0_pretrain's or B0's); `npm run typecheck`, `npx ng test`, `npm run contract` green | t3 | `studio/server/**`, a log fixture under `studio/` |
| **F** owner's flips (after the gate) | exactly what the owner approved: template sections, defaults, readme lines; `scripts/studio_surface.py` and the Studio contract if the surface changes; `tests/test_config.py`, `tests/test_studio_surface.py` | t2 | `options/*.toml`, `neosr/config.py` or `neosr/losses/fdl_loss.py`, `readme.md`, surface JSON (after S is merged) |

Optional, if a lane finishes early: the A/B harness into `scripts/ab/` (it reads frozen code today; point it at
the checkout), Studio's OTF keys in `studio/shared/catalog/degradations.ts`, the arch constructor parameters in
the surface contract (follow-up S.11), GP-05's missing peak-memory test, GP-07's readme note on `compile` for
transformer templates.

Why background agents and not one Workflow: the lanes differ in kind, two depend on GPU windows, and F waits for
the owner's gate; the Codex reviews of P, T and S can run as one parallel step when they are ready.

### 3.3 Timeline (T = start)
| time | GPU | lanes |
|---|---|---|
| 0:00-0:10 | idle | launch P, T, S, O |
| 0:10-0:35 | G1 (alone) | P reads the penalty path; T measures durations (CPU, capped threads); S builds the parser |
| 0:35-1:50 | G2 (pairs of arms) | P works from the profile; T, S implement; O runs and reports |
| 1:50-2:00 | - | owner gate on G2 |
| 2:00-2:45 | P timing, CUDA tests; optional second seed | F applies the flips; Codex reviews P, T, S in parallel; fixes |
| 2:45-3:05 | - | merge P, T, S, F into `master` (one commit per lane) |
| 3:05-3:20 | full suite CPU + CUDA (alone) | - |
| 3:20-3:50 | - | docs: `STATUS.md`, `CLOSURE.md`, `stage2/DECISIONS.md`, `stage3/PROGRESS.md`, this file; push on the owner's word |
| 3:50-4:00 | buffer | |

### 3.4 Flexible pool and swap rules (session 9 started 13:30)
Run decisions are the main session's, after a Fable consult (t5, effort high) at each checkpoint; each decision
is logged in `stage3/PROGRESS.md` with its evidence. **Checkpoints:** T+0:45 (bench numbers, first A/B pair
timed, the discriminator research shortlist), T+1:50 (A/B gate), T+2:45 (what still fits before the suite).

**Swap rules.**
1. Hard stops: the full suite starts by T+3:10, docs by T+3:25. A GPU job starts only if it ends before the
   next hard stop with 10 minutes to spare.
2. GPU priority when windows compete: lane P's timing > the A/B arms in the order B0 = B0s2 > B1 > B2 > B3 >
   B5 > pool items. An arm that cannot finish in time is dropped, not shortened (a shorter arm is not comparable).
3. A measurement (class S) that decides a question beats an A/B arm (class A8) that only narrows it.
4. A lane stuck on its approach after its budget (P: no exact fix by T+2:00) stops coding and records what it
   learned (P: profile, cause, `reg_every` 1/4/16 costs); its GPU window goes back to the pool.
5. A lane that finishes early pulls the next class-0 item; idle GPU time pulls the next class-S item.

**Pool** (filled further from `docs/refactor/stage3/discriminator/RESEARCH_REVIEW.md` when it lands):
| item | GPU class | value | needs |
|---|---|---|---|
| D cost: `bench.py` with `network_d = unet` (and the research candidates) vs metagan | S (~3 min each) | the speed half of A-04: metagan is 27 % of the step | - |
| `term_grads.py` on B0's 8k weights vs the pretrain: G's adversarial share after training | S | A-04's gradient half without a 20k run | B0 done |
| 8k arms `network_d = unet` at 0.1 and metagan at 0.03 (A-04 screen; the >= 20k run stays the decision) | A8 x2 | early sign for the template D | queue slack |
| B5: the winners of B1-B3 together | A8 | interaction | two or more winners |
| a second seed of an arm to be flipped | A8 | robustness of a flip | the gate |
| the A/B harness into `scripts/ab/` (pointed at the checkout, 8k and 20k budgets) | 0 | every later A/B | - |
| Studio: OTF keys in `degradations.ts`; arch constructor parameters in the surface (S.11) | 0 | Studio catalog complete | lane S merged |
| GP-05 peak-memory test; GP-07 readme note on `compile` for transformer templates | 0 | closes two sub-items | - |

---

## 4. The open audit findings (`docs/refactor/stage2/CLOSURE.md`)
Decided 2026-11-11: **T-11 rejected** (GP-09 measured no gain), **ST-14 deferred** (trigger: a run past
`wavelet_init` or R6 shows a dip), **L-16 deferred** (owner). 4 open:
- **needs a run:** A-04 the template discriminator (P1; >= 20k, the R6 arms A0/A1; this session only measures
  the adversarial share with `term_grads.py`), L-07 LDL (this session: G1 + arm B2), L-15 FDL grid (only if artifacts
  show);
- **waits with OTF:** D-16.
Seven arch-specific findings are deferred by the owner's goal (A-12, A-17, W-09, W-12, W-16, MX-15, MX-19).

---

## 5. Studio (owner priority 2)
State: prototype; recipe model, TOML compiler and validator are real, **training is simulated** (`studio/server/sim/`),
no adapter to `train.py`. The catalog matches the trainer as of 2026-11-10 except the OTF keys added later the same
day (`resize_mode*`, `noise_prob*`, `jpeg_prob*`, `blur_prob`, `clean_*`, `sharpen_*`): they are in the surface JSON
and in `config.py`, not yet in `studio/shared/catalog/degradations.ts` (the contract test only checks that Studio
emits no key the trainer does not know, so it is green).
Steps in the order they unblock each other:
1. The real adapter (session 9, lane S).
2. Arch constructor parameters in the surface contract (today hand-checked): the registry scan knows each type's
   module and name, so an `ast` read of the signatures there is a small step (also the rest of follow-up S.11).
3. Deferred UI items listed in `design/NeoSR_Studio_UI_Concept_v2.md` section 7 (light theme, folder browser,
   metric parameters in the Validation node, ...), ONNX capability badges from the surface's `no_onnx` /
   `onnx_static_only`.
Checks: `npm run typecheck`, `npx ng test --watch=false` (39 cases), `npm run contract` (13 TOMLs through the
trainer's validator), `npx ng build --configuration development --output-path dist/check-<name>`.

---

## 6. Ideas deferred by the owner (not tasks; do not start)

### 6.1 Replace OpenCV decoding by pyvips for lazy (region) reads
**The suggestion as the owner received it** (Russian, with typos, 2026-11-10): "maybe replace cv with pyvips:
lazy reading is still faster than reading the whole image into Python and cropping afterwards".
**What it means.** Today a training sample is produced by decoding the **whole** image file with OpenCV
(`cv2.imdecode` in `neosr/utils/img_util.py:186`, called from `read_img` in `neosr/data/data_util.py:54`) into a
numpy array and then cutting the random crop out of it. libvips (Python binding: pyvips) is demand-driven: it
opens the file, and pixels are decoded only when a region is asked for, so "open -> crop -> fetch" can skip
work and memory for the part of the image that is thrown away.

**What is already known in this repository (earlier sessions looked at it twice):**
| source | verdict |
|---|---|
| `docs/refactor/analysis/competitor/H2.md:43` (stage 1) | the competitor traiNNer-redux reads this way; "partial win for large GT only", a new system dependency, its implementation reads buffers as uint8 unconditionally (16-bit PNG breaks), ICC behaviour unverified. Fable: optional later behind `io_backend="vips"` after a measured I/O bottleneck; Codex: do not port |
| `docs/refactor/stage2/analysis/data.md` D-06 and line 414 (stage 2) | measured: decoding dominates (73 ms for a 4.6 MiB 2K PNG); "do not port pyvips just for crops, pre-tiling gives more for less": a 512 px tile decodes in ~7 ms, about 10x loader throughput |
| what stage 2 did instead | D-06 (a): crop in uint8 first, convert only the patch to float (-25-30 % CPU per sample on 2K images); D-06 (b): `scripts/extract_subimages.py` for pre-tiling |

**Technical points a decision needs (assessment of session 6, nothing measured for pyvips here):**
- The gain depends on the **file format and on image size vs crop size**. A PNG is one deflate stream: to reach a
  crop near the bottom every row above it must still be decompressed, so lazy reading saves memory and the copy,
  not most of the decode time. Formats with real random access (tiled TIFF, some JPEG paths) gain more.
- On **pre-tiled data** (the owner's set is 256 px tiles) the crop is almost the whole file: nothing to gain.
  The idea pays only when training reads **large untiled images** directly.
- Costs: a native dependency (libvips; a binary wheel exists for pyvips), its own thread pool inside DataLoader
  workers, a second decode path to keep exact (16-bit files, alpha, grey, ICC profiles), the LMDB path (decode
  from a buffer), and JPEG pixels that may differ from OpenCV's decoder by a level (training data would change
  slightly; PNG is lossless either way). The exact-crop tests of stage 2 (`tests/test_s2_data.py`, D-06) would
  need a twin.
- It fits the project's goals only as a small, optional backend with a parity test, not as a second pipeline.

**If the owner wants it examined:** first a measurement, no code in `neosr/`: per-sample time and peak RSS of
"cv2 full decode + crop" vs "pyvips region read" for the formats and sizes actually trained on (256 px tiles; 2K
and 4K PNG 8/16 bit; JPEG), inside a DataLoader with the usual worker count, plus a pixel-equality check per
format. CPU-only, so it can run on either machine; the numbers that matter are the training machine's. Then the
owner decides between: nothing (pre-tiling stays the answer), an optional `io_backend`, or a replacement.

**Open question for the owner:** which data should this serve: the current pre-tiled sets, or large original
photographs read directly?

## 7. What needs hours of GPU (after this session)
The standing list lives in `STATUS.md` ("Needs the Linux machine"): the R1/R2 penalty fix feeds R6 (R3GAN A/B,
rounds 1-2, seed 2, confirmation set), which also answers A-04; Phase 0 A/Bs; the 80k+ long run (crosses
`wavelet_init`, ST-14's trigger); the opt-in switches still waiting for an A/B.
