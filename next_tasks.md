# Next tasks (handoff for a new session)

Rewritten 2026-11-11 at the end of session 9 (Linux machine, 13:30-17:30). Session 9 ran as parallel lanes with
the owner's run decisions delegated to the main session after Fable consults; what it did is in section 2 and in
`docs/refactor/stage3/PROGRESS.md` "Session 9". Its commits are local on `master` (pushed only on the owner's
word).

**Owner's focus (unchanged):** a tuned, optimised core: training quality, GAN modes, speed, efficiency and
stability, with the losses, metrics, strategies and optimisations around them; Studio in parallel.

| rank | task | kind | GPU | section |
|---|---|---|---|---|
| 1 | Owner: confirm or revert the session's provisional and delegated decisions (LDL kept: its flip failed the seed replicate and waits on branch `held/ldl-flip`; no default changed by the screens; `compile = true` in the CNN templates (8k: no detectable quality difference; -21 % ms/update); the Studio real mode as experimental; the fixed-param display in Studio) | decisions | none | 2, 4 |
| 2 | R3GAN A/B R6, now affordable: the penalty fix made a regularised update 122.5 ms (was 732); round 1 at 20k (7 arms: A0, A1, R1-R5) is ~4-5 GPU-h in total instead of ~21 h | quality A/B | hours | 3.1 |
| 3 | A-04: at 8k no-D and metagan@0.03 lost to the template; unet@0.1 has an LPIPS lead that replicated on a second seed (WIN vs B0s2), but its D is clipped in 100 % of the updates. First the unet D's gradient scale (no code until the owner picks), then >= 20k with the GAN health log | quality A/B | hours | 3.1 |
| 4 | The unet D's gradient scale (A-04 prerequisite): why its grad norm sits at ~30-45 at weight 0.1, so the D clip is on in every update; read the arch and the health log, a short bench if needed; no default change | core, analysis | short | 3.2 |
| 5 | Metric memory at 4K: DISTS needs ~8.6 GiB, TOPIQ ~6 GiB at 3840x2160 (fp32); 12 GB cards cannot validate DISTS at 4K. Decide: document a limit, or a chunked DISTS that is exact | core, design | short | 3.3 |
| 6 | Studio: one real GPU training through the real mode (it ran only on CPU), then the deferred UI items | Studio | short | 5 |
| 7 | Housekeeping: the agent worktrees under `.claude/worktrees/` and their branches (all work is cherry-picked onto master); the old `.worktrees/refactor` keeps the R3GAN experiments | repo | none | 2 |

Parked by the owner: OTF realism (`docs/refactor/stage3/otf/DESIGN.md`), all of stage-3 Phase B. Deferred:
section 6 (pyvips), L-16, ST-14 (trigger: a run past `wavelet_init`), D-16 (with OTF).

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
| R3GAN track, R4 numbers | `docs/refactor/stage3/PROGRESS.md` "PAUSED", `stage3/r3gan/RESULTS.md`; the R5 split, the pretrain and old runs in the gitignored `.worktrees/refactor/experiments/r3gan_ab/` |
| A/B tooling and session 9 runs | `scripts/ab/README.md`; runs, configs and `NOTES.md` of session 9 in the gitignored `experiments/s9_ab/` |
| discriminator research | `docs/refactor/stage3/discriminator/RESEARCH_REVIEW.md` |
| architecture designs | `docs/refactor/design/T1.md` (loss pipeline, training step), `T2.md` (model, checkpoints), `T3.md` (config), `T5_aether_qat.md`, `T7_registration.md` (registries) |
| user docs | `readme.md`, `docs/training_guide.md` |
| Studio | `studio/` (`README.md`, `DEVELOPING.md`, `server/API.md`); contract `scripts/studio_surface.py` -> `studio/shared/catalog/neosr-surface.json` |

### Commands
```bash
.venv/bin/python -m pytest -m "not slow and not cuda" -q        # fast CPU subset (~100 s); the full suite: tests/ (~10 min CPU + CUDA)
.venv/bin/python -m pytest tests/<file>.py -q                    # one file
.venv/bin/python scripts/bench.py -opt <toml>                    # ms/update, peak VRAM; GPU alone
.venv/bin/python scripts/term_grads.py -opt <toml> --batches 8   # per-term G gradient share
.venv/bin/python scripts/ab/analyze.py --runs <runs> --data <data> --seed-ref <arm> --iter 8000 <ref> <arms...>
scripts/ab/queue.sh                                              # A/B queue (README in scripts/ab/)
.venv/bin/python scripts/studio_surface.py --check               # after a registry, table or SCHEMA change
cd studio && npm run typecheck && npx ng test --watch=false && npm run test:server && npm run contract
```
On the Windows machine: the fast subset (`.venv/Scripts/python.exe -m pytest -m "not slow and not cuda" -q -p
no:cacheprovider`) or single files; never the full suite, sweeps, benchmarks or training (`AGENTS.md`). pytest-xdist
is not installed.

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

---

## 2. What session 9 did (2026-11-11; commits local on `master` after 8f2f6cc)
Every code commit went through a review workflow (Codex + an independent reviewer, each finding then checked by
two skeptics); about 80 confirmed findings were fixed before or right after merging.

| area | result | commit(s) |
|---|---|---|
| owner decisions | T-11 rejected, ST-14 / L-16 / pyvips deferred, run decisions delegated (Fable consults) | b622aff, ee4dcfa |
| tests | markers `cuda`/`slow`, fast CPU subset (`pytest -m "not slow and not cuda"`, ~1850 tests in ~100 s), env-configurable fixtures, CI workflow (unrun); caplog tests order-independent | fd4c0e4, a172e06, 7633ec6 |
| GAN speed | r3gan D: R1/R2 double backward without per-group convs (exact); regularised update 732 -> 122.5 ms on the template recipe; R4 cost gate now passes | 94c36bd |
| GAN safety | R1/R2 refused at model build for a D without a usable double backward (BatchNorm, DySample) | cd3b0ad |
| GAN diagnostics | GAN health in the log: `grad_sr_gan`, `grad_sr_ratio`, `out_d_acc`, `out_d_abs` (hooks in the one backward, bitwise-neutral); `lr_d`; `[ mem: U / T GiB ]` | 59a6fbc, 6f3ca7e, 4b66e66 |
| DDP | two-rank gloo tests: r3gan + R1/R2 + health log in sync; save_best only on rank 0, barriers aligned, resume restores the record | c10b89c, 1951d9f |
| export | ONNX sweep of the 10 `pad_reflect` archs: all verify; srformer_medium leaves STATIC_ONLY | 48469bc |
| docs | FDL `fp32_backbone=false`: no measurable speed gain, saves ~160 MiB; compile note (-21 % realplksr; DAT stage-2 number) | 343e202, b94e7f5 |
| A/B tooling | `scripts/ab/` (run_ab, analyze with the gate rule, queue with deadline), tests | c03c44f, f2f7d91 |
| Studio | experimental real-trainer mode (`STUDIO_TRAINER=real`, `npm run api:real`), S.11 arch constructor params in the surface contract, real `/api/environment` | e97902b, 176990c, 2ede3c2, 587cbdb |
| measurements | bench, term_grads, 8k A/Bs, metric memory: `STATUS.md`, `stage2/DECISIONS.md`, `stage3/r3gan/RESULTS.md` | 100a588 + the final docs commit |

Gate results at 8k (219 dev tiles of the R5 split, from the 20k pretrain; floors from the seed pair B0/B0s2): B1
`grad_clip = "spike"` TIE (default unchanged; the fixed clip is near idle in this regime); B2 without LDL TIE (its
seed replicate B2s2 failed: LOSS vs B0s2 on LPIPS, a `grad_norm_g` spike of 248; LDL stays, the flip waits on branch
`held/ldl-flip`). Screens vs B0: R3GAN LOSS (DISTS, marginal; PSNR -0.81; TIE vs no-GAN), no-GAN LOSS (the template D
helps all three perceptual metrics), unet@0.1 TIE (LPIPS -0.0080, its D clipped 100 %; seed 2 WIN vs B0s2: LPIPS
-0.0079, DISTS -0.0030: the lead replicates), metagan@0.03 LOSS, `wavelet_init = 0` LOSS; `compile = true` TIE (no
detectable quality difference at 8k). No default changed (`stage2/DECISIONS.md` 15, 17, 18).

## 3. Next session: what to run
### 3.1 Hours of GPU (one per session or overnight; use `scripts/ab/`)
1. **R6, R3GAN A/B** (`stage3/PROGRESS.md` "PAUSED" has the arm list): with the penalty fix the R arms cost
   +11 % over the template; run round 1 at 20k (arms A0, A1, R1-R5), then round 2 on the best R, seed 2, the
   confirmation set. The harness now lives in `scripts/ab/`; point it at the checkout, not a frozen copy.
2. **A-04 at >= 20k**: metagan@0.3 vs unet@0.1 (its 8k LPIPS lead replicated on seed 2), with the health log as the
   mechanism read-out. At 8k unet's D was clipped in 100 % of the updates (`grad_norm_d` ~45, 31 late G-gradient spike
   windows): understand its gradient scale first (a D-side cause, not more damping). metagan@0.03 and no-D lost at 8k
   (`stage2/DECISIONS.md` 17).
3. **LDL at >= 20k**, two seeds (8k: seed 7 TIE, seed 2 LOSS on LPIPS with a `grad_norm_g` spike of 248; the
   flip waits on branch `held/ldl-flip`); **spike clip** from a fresh G (where the fixed
   clip is active), 20k, two seeds (T-02's own test).
4. Phase 0 (E2, E4, E5) and the 80k+ long run (crosses `wavelet_init`: ST-14's trigger).
### 3.2 Short GPU items (15-30 min each)
- The unet D's gradient scale (rank 4): at weight 0.1 its grad norm is ~30-45 and the D clip is on in 100 % of the
  updates (both seeds); find the cause before the A-04 round at >= 20k.
- No 8k screen is pending: compile ran (TIE), the moved arms got their second seed (B2s2, U_unet01s2).
### 3.3 No GPU
- Metric memory at 4K (table in `STATUS.md`): decide between a documented limit and an exact chunked DISTS.
- Research candidates still open (`stage3/discriminator/RESEARCH_REVIEW.md`): DR-06 (hinge arm), DR-09 (R1 on
  the template D), DR-10 (joint RGB+HF view) wait for the screens; DR-08 (metagan internals) is arch work.

## 4. The open audit findings (`docs/refactor/stage2/CLOSURE.md`)
131 done, 16 opt-in, 7 rejected, 9 deferred, 4 open. Open: A-04 (>= 20k, section 3.1), L-07 (>= 20k, section 3.1),
L-15 (only if FDL grid artifacts show), D-16 (with OTF). Deferred: the seven arch-specific findings, ST-14, L-16.

## 5. Studio (owner priority 2)
State: recipe model, compiler, validator; an **experimental real mode** (`STUDIO_TRAINER=real`, launched with
`npm run api:real`; the simulator stays the default) that runs `train.py` per phase in its own
`<name>_<run id>` folders with a pid lock, graceful stop/resume from the run's own newest state, live log events
(progress, scalars incl. GAN health, `lr_d`, `[ mem ]` as system events, validation, checkpoints). Verified end to
end on CPU only. The catalog is checked against the trainer's constructor signatures (S.11), preset-bound
parameters are read-only and blocked by the validator.
Next: one GPU run through the real mode; the deferred UI items of `design/NeoSR_Studio_UI_Concept_v2.md`
section 7; ONNX capability badges from `no_onnx` / `onnx_static_only`. The OTF keys stay out (track parked).
Checks: `npm run typecheck`, `npx ng test --watch=false`, `npm run test:server`, `npm run contract`.

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

## 7. What needs hours of GPU
Section 3.1; the standing list is `STATUS.md` "Needs the Linux machine".
