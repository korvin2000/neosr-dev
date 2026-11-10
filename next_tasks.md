# Next tasks (handoff for a new session)

Written 2026-11-10 at the end of session 7 (Windows machine). Tasks 1-3 of the previous handoff are **done**
(section 2); what is left is below, in the order the owner's priorities give. Nothing of session 7 is pushed until
the owner says so (local commits on `master`).

| # | task | kind | where |
|---|---|---|---|
| 1 | Run on the Linux machine what session 7 could not: the full suite (CPU + CUDA), the per-arch files, the ONNX sweep | verification | section 3 |
| 2 | Remaining open findings of the stage-2 audit (15 of 167) | small fixes here, A/Bs there, three owner decisions | section 4, `docs/refactor/stage2/CLOSURE.md` |
| 3 | NeoSR Studio: finish the UI | later, ask the owner before starting | section 5 |

Parked by the owner: OTF realism (steps O4, O5, the A/B), `docs/refactor/stage3/otf/DESIGN.md`. Do not continue it.
Ideas the owner has not decided on are in section 6 (pyvips lazy reads): do not start them.

## 0. Rules of the house (they caused friction once already)
- **The owner sets the priorities.** Do not promote a track because a document calls it a goal; when the next step
  is not named here, ask. (`AGENTS.md`, `docs/refactor/STATUS.md` "Owner priorities".)
- **Goal of the current phase:** a core that "works like a Swiss watch": stable, reliable, fast, good quality. New
  features wait.
- **What this project is:** a reworked, optimised trainer of its own. The measure of every change is clean,
  well-thought-out code, speed, a small footprint, good structure and compactness.
- **traiNNer-redux (https://github.com/the-database/traiNNer-redux) is a competitor**, often with bloated code
  and god classes. It may be read for ideas and examples; nothing is imitated or copied, and "it has this" is
  no argument (`docs/refactor/DECISIONS.md`, owner clarification of 2026-11-10).
- **No hasty decisions.** Ask the owner on anything disputed or open instead of settling it; say what is verified
  and what is assumed.
- **This machine (Windows, Ryzen 5 5500U, no GPU):** analysis, docs, small core changes, Studio. Run single test
  files only; never the full suite, per-arch sweeps, ONNX sweeps, benchmarks or training. Everything GPU-bound and
  the full suite belong to the Linux 4090 machine; list such leftovers in `STATUS.md` ("Waiting for the Linux
  machine").
- **Every code step:** implement -> targeted tests -> Codex review of the diff (read-only) -> fix -> one commit ->
  log in the track's `PROGRESS.md`, update `STATUS.md`. **Push:** the owner chose "commit locally, push at the end
  of the session on my word" (session 7); ask before pushing.
- **Defaults do not change silently.** Exact rewrites ship as defaults behind parity tests; behaviour changes ship
  opt-in until an A/B decides (`docs/refactor/stage2/DECISIONS.md`, principles).
- The token in the `origin` URL is intentional: do not report it, never print the URL (mask command output).

## 1. Orientation (5 minutes)
| what | where |
|---|---|
| single status page, machines, open work | `docs/refactor/STATUS.md` |
| owner and consensus decisions | `docs/refactor/DECISIONS.md` |
| stage logs (history, numbers) | `docs/refactor/PROGRESS.md` (stage 1), `stage2/PROGRESS.md`, `stage3/PROGRESS.md` |
| what became of the 167 audit findings | `docs/refactor/stage2/CLOSURE.md` |
| architecture designs | `docs/refactor/design/T1.md` (loss pipeline, training step), `T2.md` (model, checkpoints), `T3.md` (config, construction, imports), `T5_aether_qat.md`, **`T7_registration.md` (the registries)** |
| user docs | `readme.md` ("adding an architecture", "adding a loss"), `docs/training_guide.md` |
| trainer | `neosr/` (`archs/` 54 arch files, `losses/`, `metrics/`, `data/`, `models/`, `optimizers/`, `utils/registry.py`, `utils/factory.py`, `config.py`, `runtime.py`), `train.py`, `test.py`, `convert.py` |
| Studio (UI prototype) | `studio/` (`README.md`, `DEVELOPING.md`, `server/API.md`), concept `design/NeoSR_Studio_UI_Concept_v2.md` |
| Studio <-> trainer contract | `scripts/studio_surface.py` -> `studio/shared/catalog/neosr-surface.json`; `tests/test_studio_surface.py`; `studio/src/app/core/catalog-contract.spec.ts` |

### Commands on this machine
```bash
.venv/Scripts/python.exe -m pytest tests/<file>.py -q -p no:cacheprovider      # one file; -k to narrow
.venv/Scripts/python.exe -m ruff check <files> --output-format concise
python scripts/studio_surface.py            # regenerate the surface JSON after adding or changing a registered type or the schema
python scripts/studio_surface.py --check    # is it current?
python scripts/studio_surface.py --validate options/train_*.toml   # load_config + validate_train, no torch
cd studio && npm run typecheck && npx ng test --watch=false && npm run contract
```
The venv is hand-built (CPU torch 2.9.1, not `uv sync`): recipe in `STATUS.md`. Timings here: `import torch` 3.7 s;
test_registry 10 s; test_config + test_registry + test_imports + test_hygiene + test_studio_surface ~2 min;
`tests/test_s2_exportfix.py` 27 s (includes one hma ONNX export).

### Traps
- **Shell heredocs mangle scripts**: Git Bash breaks on apostrophes, and `\\n` inside a heredoc arrives as a real
  newline. Write helper scripts and edit specs with the Write tool into the scratchpad and run them.
- **The Edit tool rewrites a CRLF file with LF.** The working copy is CRLF (`autocrlf`), the index LF, so git does
  not care, but keep the files consistent: use a script that preserves the line ending, or convert back.
- `ruff format` (0.17.1 in the venv) writes LF and reformats **existing** lines of older files (line length 88).
  Format only new files; `*_arch.py` is outside ruff altogether.
- **Never `git stash` while a background test run or other agents read the tree**: they see the old files.
- Tests that fail on Windows for reasons unrelated to any change (do not chase them inside another task):
  `tests/test_runtime.py::test_two_rank_gloo_shares_rank0_seed` (gloo device), `tests/test_s2_data.py::
  test_scandir_skips_hidden_and_non_images` (symlink privilege), `tests/test_s2_data.py::test_path_list` (subprocess
  output decoding). Listed in `STATUS.md`, portability item.
- Codex: `Agent` with `subagent_type: codex:codex-rescue`, the brief starts with `--model gpt-6.1-sol --effort high`
  and "read-only: advise only, do not edit files". It reviews a commit well when given the hash and told what was
  already verified.

---

## 2. What session 7 did (local commits on `master` after 44ff761)
| commit | content |
|---|---|
| 8bbdd04 | `convert.py`: the dead `rep_rt4ksr` string literal deleted |
| fd30790 | `config.total_iter(cfg)`: the `logger.total_iter` default in one place (follow-up S.5); six readers use it |
| ad84712 | **the five registries** (`neosr/utils/registry.py`, design T7, mechanism A1 chosen by the owner): `@X_REGISTRY.register()` is a declaration, a source scan fills the registry lazily; the tables `ARCHS`, `LOSSES`, `METRICS`, `CHOICES`, `DATASETS`, `MODELS` and the test list `NON_SR_ARCHS` are gone; the Studio surface JSON is unchanged |
| be60252 | design note T7, `DECISIONS.md`, readme ("adding an architecture", "adding a loss") |
| f442724 | registry: fixes from the Codex review of ad84712 (import alias, missing folder, stricter test oracle) |
| 3e5887a | `arch_util.pad_reflect`: one trace-invariant pad-to-window for 14 archs (the hma item of the old Task 3, which turned out to be a family; the owner chose the shared helper) |
| aebbf83 | audit leftovers MX-17, L-13, L-11, L-12, L-14 |
| (last) | `docs/refactor/stage2/CLOSURE.md` (the closure matrix), `STATUS.md`, this file |

How a type is registered now: `readme.md` "adding an architecture"; the rules of the scan and why: `T7_registration.md`
sections 4-5. After adding or renaming a type run `python scripts/studio_surface.py` and commit the JSON.

Decided by the owner in session 7 (`DECISIONS.md`): mechanism A1; `scale=` required on archs, `choices=` on
metrics; ONNX gaps (`NO_ONNX`, `STATIC_ONLY`) stay in `convert.py`; the shared pad helper for all archs.
Implemented as proposed without a separate question, open to revision: no aliases, no plugin folders,
`OPTIMIZERS` / `SCHEDULERS` stay tables.

## 3. Task 1: what the Linux machine must confirm for session 7
Everything below changed code that the Windows machine checked only with the tests named in `STATUS.md`
("Waiting for the Linux machine"):
1. **Full suite, CPU and CUDA**, on the head. It covers the registry migration (all per-arch files read the
   registries: `tests/test_archs*.py`, `tests/test_s2_cnn.py`, `tests/test_losses_pipeline.py` were only
   *collected* here: 953 cases), `pad_reflect` in 14 archs, `safmn`, `config.total_iter`.
2. **ONNX sweep** of the exportable archs that now pad through `pad_reflect` (dctlsa, ditn, drct, grformer, hat,
   hma, mfghmoe, microsr, srformer, swinir; the others are in `NO_ONNX`): `convert.py` verification at its three
   shapes, as in stage 3 A8. On Windows only hma and a toy module were exported. The graphs of these archs
   change (index gathers instead of a `Pad` node), so re-check `STATIC_ONLY` for `srformer_medium`.
3. The older list in `STATUS.md` ("Needs the Linux machine": R3GAN penalty-pass profile and A/B, Phase 0 A/Bs, a
   long run) is unchanged.

## 4. Task 2: the open audit findings
`docs/refactor/stage2/CLOSURE.md` has the three ranked lists with the files and the size of each item:
- **fixable on Windows** (6 entries: A-11 `convert.py --prune`, A-17, W-16, A-12, three optional sub-items, two
  doc/code mismatches in `stage2/DECISIONS.md`): small, exact; W-16 touches the checkpoint format, ask first;
- **needs the Linux machine** (7: A-04 the template discriminator, L-07 LDL, W-09, W-12, MX-19, L-16, L-15);
- **the owner decides** (4: MX-15 microsr default windows, T-11, ST-14, D-16).
Per item: smallest coherent change, a test, a Codex review of the diff, one commit; anything that changes
training dynamics is opt-in until an A/B. Also open from the old Task 3: the TODO at
`neosr/data/otf_dataset.py` (kernel size range 7..21 hard-coded) belongs to the parked OTF track; the rest of
follow-up S.11 (window sizes and scales hand-kept in `studio/shared/catalog/archs.ts`) belongs to Studio step 2.

---

## 5. Task 3: Studio (after 1-2)
State: prototype; recipe model, TOML compiler and validator are real, **training is simulated** (`studio/server/sim/`),
no adapter to `train.py`. The catalog matches the trainer as of 2026-11-10 except the OTF keys added later the same
day (`resize_mode*`, `noise_prob*`, `jpeg_prob*`, `blur_prob`, `clean_*`, `sharpen_*`): they are in the surface JSON
and in `config.py`, not yet in `studio/shared/catalog/degradations.ts` (the contract test only checks that Studio
emits no key the trainer does not know, so it is green).
Next steps in the order they unblock each other (ask the owner before starting):
1. The real adapter: `studio/server/` spawns `train.py`, parses the log lines and TensorBoard scalars into the
   event shapes the simulator emits (`studio/server/API.md`, `studio/shared/run.model.ts`); testable against a
   recorded log, no GPU needed for the parser.
2. Arch constructor parameters in the surface contract (today hand-checked): the registry scan knows each type's
   module and name, so an `ast` read of the signatures there is a small step (also the rest of follow-up S.11).
3. Deferred UI items listed in `design/NeoSR_Studio_UI_Concept_v2.md` section 7 (light theme, folder browser,
   metric parameters in the Validation node, ...), ONNX capability badges from the surface's `no_onnx` /
   `onnx_static_only`.
Checks: `npm run typecheck`, `npx ng test --watch=false` (39 cases), `npm run contract` (13 TOMLs through the
trainer's validator), `npx ng build --configuration development --output-path dist/check-<name>`.

---

## 6. Ideas waiting for the owner's decision (not tasks; do not start)

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

## 7. What waits for the Linux machine (do not attempt here)
Section 3 above for session 7; the standing list lives in `STATUS.md`: "Needs the Linux machine" and "Waiting
for the Linux machine".
