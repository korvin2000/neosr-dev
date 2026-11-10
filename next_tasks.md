# Next tasks (handoff for a new session)

Written 2026-11-10 at the end of session 6 (Windows machine). The owner fixed the order below; **nothing in this
file has been started**. Read sections 0-1 first, then work on Task 1.

| # | task | kind | deliverable |
|---|---|---|---|
| 1 | Decorator-style registration (`@ARCH_REGISTRY.register()` ...) without the old cost | **design first**, the owner decides, then code | `docs/refactor/design/T7_registration.md`, then the implementation |
| 2 | Closure matrix of the stage-2 audit (167 findings), then fix what is open | reading + verification, then small fixes | `docs/refactor/stage2/CLOSURE.md`, then commits |
| 3 | Small core items (4 known) | small fixes | commits |
| 4 | NeoSR Studio: finish the UI | later, after 1-3 | see section 5 |

Parked by the owner: OTF realism (steps O4, O5, the A/B), `docs/refactor/stage3/otf/DESIGN.md`. Do not continue it.

## 0. Rules of the house (they caused friction once already)
- **The owner sets the priorities.** Do not promote a track because a document calls it a goal; when the next step
  is not named here, ask. (`AGENTS.md`, `docs/refactor/STATUS.md` "Owner priorities".)
- **Goal of the current phase:** a core that "works like a Swiss watch": stable, reliable, fast, good quality. New
  features wait.
- **Reuse before writing:** check what upstream traiNNer-redux (the-database,
  https://github.com/the-database/traiNNer-redux) and the owner's fork already have; adapt, never copy as is
  (`docs/refactor/DECISIONS.md`, competitor policy).
- **This machine (Windows, Ryzen 5 5500U, no GPU):** analysis, docs, small core changes, Studio. Run single test
  files only; never the full suite, per-arch sweeps, ONNX sweeps, benchmarks or training. Everything GPU-bound and
  the full suite belong to the Linux 4090 machine; list such leftovers in `STATUS.md` ("Waiting for the Linux
  machine").
- **Every code step:** implement -> targeted tests -> Codex review of the diff (read-only) -> fix -> one commit ->
  push to `master` -> log in the track's `PROGRESS.md`, update `STATUS.md`.
- **Defaults do not change silently.** Exact rewrites ship as defaults behind parity tests; behaviour changes ship
  opt-in until an A/B decides (`docs/refactor/stage2/DECISIONS.md`, principles).
- The token in the `origin` URL is intentional: do not report it, never print the URL (mask command output).

## 1. Orientation (5 minutes)
| what | where |
|---|---|
| single status page, machines, open work | `docs/refactor/STATUS.md` |
| owner and consensus decisions | `docs/refactor/DECISIONS.md` |
| stage logs (history, numbers) | `docs/refactor/PROGRESS.md` (stage 1), `stage2/PROGRESS.md`, `stage3/PROGRESS.md` |
| architecture designs | `docs/refactor/design/T1.md` (loss pipeline, training step), `T2.md` (model, checkpoints), `T3.md` (config, construction, imports: **the registry decision**), `T5_aether_qat.md` |
| user docs | `readme.md` ("adding an architecture", "adding a loss"), `docs/training_guide.md` |
| trainer | `neosr/` (`archs/` 54 arch files, `losses/`, `metrics/`, `data/`, `models/`, `optimizers/`, `utils/factory.py`, `config.py`, `runtime.py`), `train.py`, `test.py`, `convert.py` |
| Studio (UI prototype) | `studio/` (`README.md`, `DEVELOPING.md`, `server/API.md`), concept `design/NeoSR_Studio_UI_Concept_v2.md` |
| Studio <-> trainer contract | `scripts/studio_surface.py` -> `studio/shared/catalog/neosr-surface.json`; `tests/test_studio_surface.py`; `studio/src/app/core/catalog-contract.spec.ts` |

### Commands on this machine
```bash
.venv/Scripts/python.exe -m pytest tests/<file>.py -q -p no:cacheprovider      # one file; -k to narrow
.venv/Scripts/python.exe -m ruff check <files> --output-format concise
python scripts/studio_surface.py            # regenerate the surface JSON after a table/schema change
python scripts/studio_surface.py --check    # is it current?
python scripts/studio_surface.py --validate options/train_*.toml   # load_config + validate_train, no torch
cd studio && npm run typecheck && npx ng test --watch=false && npm run contract
```
The venv is hand-built (CPU torch 2.9.1, not `uv sync`): recipe in `STATUS.md`. Timings here: `import torch` 3.7 s;
test_config + test_registry + test_imports + test_hygiene + test_runtime ~2 min; `tests/test_s3_otf.py` 15 s.

### Traps found in session 6
- Git Bash heredocs break on apostrophes inside the script text. Write helper scripts with the Write tool into the
  scratchpad and run them; do not inline them.
- `ruff format` (0.17.1 in the venv) reformats **existing** lines of older files (line length 88). Format only new
  files; after formatting an old file check `git diff` for unrelated hunks and undo them.
- The working copy is CRLF (`autocrlf`); scripted edits must keep the file's line endings.
- Tests that fail on Windows for reasons unrelated to any change (do not chase them inside another task):
  `tests/test_runtime.py::test_two_rank_gloo_shares_rank0_seed` (gloo device), `tests/test_s2_data.py::
  test_scandir_skips_hidden_and_non_images` (symlink privilege), `tests/test_s2_data.py::test_path_list` (subprocess
  output decoding). Listed in `STATUS.md`, portability item.
- Codex: `Agent` with `subagent_type: codex:codex-rescue`, the brief starts with `--model gpt-6.1-sol --effort high`
  and "read-only: advise only, do not edit files". Its forwarder sometimes tries a second hand-back that is refused:
  ask it once for "only what is new"; the answer has been "nothing new" every time.

---

## 2. Task 1: decorator-style registration, designed for the user and still lazy

### 2.1 What the owner asked for and why
Stage 1 (design T3, decision D3) removed `@ARCH_REGISTRY.register()`, `@LOSS_REGISTRY.register()` etc. and put every
type into a table in the package `__init__.py`. That was right for start-up cost and import hygiene, but it lost
the convenience for the person who adds an arch or a loss: write the class, put a decorator on it, done. Today one
must also edit a table in another file (and for a scale-free net a list in a test). The owner wants the decorator
style back, **more efficient than the old one**.

### 2.2 How it was (removed in 8da8ae0, 2026-10-21; old code: `git show 8da8ae0^:neosr/utils/registry.py`)
- `neosr/utils/registry.py`: `class Registry` with `register()` (decorator or call), `get(name)`; registries
  `ARCH_REGISTRY`, `LOSS_REGISTRY`, `METRIC_REGISTRY`, `DATASET_REGISTRY`, `MODEL_REGISTRY`.
- `neosr/archs/__init__.py` scanned the folder and **imported every `*_arch.py`** at the first `build_network`
  (`git show 43db06a^:neosr/archs/__init__.py`); each arch file ran `upscale, __ = net_opt()` at import, i.e. it
  parsed argv and the TOML to learn the scale (`git show 43db06a^:neosr/archs/span_arch.py`, lines 7-10, 195).
- Why T3 removed it (`docs/refactor/design/T3.md` section 0 points 1-2, D3 at line 26, recipes R1/R4 at 414-428):
  importing ~54 arch modules for one network, work and argv/TOML reads at import time, the scale as a module
  global, `suffix=` unused, star-imports.

### 2.3 How it is now (read these, they are short)
| piece | file | shape |
|---|---|---|
| arch table | `neosr/archs/__init__.py` | `ARCHS: name -> (module under neosr.archs, scale kwarg or None)`, 105 rows: 95 generators, 9 discriminators (incl. the three `r3gan*`), `VGGFeatureExtractor` |
| loss table | `neosr/losses/__init__.py` | `LOSSES: type -> (module, None)`, 16 rows; `build_losses` owns weights, targets, schedules |
| metric table | `neosr/metrics/__init__.py` | `METRICS` (5 rows, all in `calculate`), `CHOICES`; the callable is the `<type>_pt` twin |
| datasets, models | `neosr/data/__init__.py`, `neosr/models/__init__.py` | `DATASETS` (3), `MODELS` (2) |
| optimizers, schedulers | `neosr/models/base.py:33-44` | `OPTIMIZERS`, `SCHEDULERS`: name -> class, mostly torch's own classes (cannot be decorated) |
| lookup and construction | `neosr/utils/factory.py` | `resolve(table, package, section, name)` imports **only** the row's module; `inject_scale` passes the run scale under the arch's own kwarg name; `construct` turns a bad kwarg into `ConfigError`; `build` |
| export tables keyed by arch name | `convert.py:321-341` | `NO_ONNX`, `STATIC_ONLY` |

The type name is the Python name of the class or factory function in its module (`resolve` does `getattr(module,
name)`). Factories (`def hat_s(**kwargs)`) are rows too.

### 2.4 Invariants a new design must keep (each has a test)
| invariant | guarded by |
|---|---|
| importing `neosr`, `neosr.archs`, `neosr.losses`, ... has no side effects and does **not** import the modules the tables name | `tests/test_imports.py` (`MODULES`, `TABLES`, a subprocess probe per module) |
| no `net_opt`, `parse_options`, `set_default_device`, `sys.argv`, `sys.exit` under `neosr/`; **the pattern `_REGISTRY\b` is banned today** | `tests/test_hygiene.py` (`PATTERNS["registry"]`): the design must change this rule on purpose |
| every row resolves; the scale kwarg reaches the constructor (reviewed exception: `omnisr` reads it from `**kwargs`); scale-free nets are exactly `NON_SR_ARCHS` + `VGGFeatureExtractor` | `tests/test_registry.py` |
| unknown type -> `ConfigError` with a did-you-mean | `tests/test_registry.py`, `tests/test_config.py` |
| arch forward/backward per row; fixture equivalence with the pre-refactor weights | `tests/test_archs.py` (+ `_a1/_a2/_a3`): slow, Linux |
| type names, `state_dict` keys and export metadata (`neosr.network_g.type`) do not change | `tests/test_s3_exports.py`, `tests/test_checkpoint.py` |
| the Studio surface is generated from the tables **statically (ast)** | `scripts/studio_surface.py` (`_table`, `_keys`), `tests/test_studio_surface.py`: must be adapted to whatever replaces the tables |
| every shipped template validates and builds | `tests/test_config.py`, `tests/test_losses_pipeline.py` (slow) |

### 2.5 Candidate designs (to be compared in the design note; none is decided)
| | idea | adding an arch | start-up cost | risks |
|---|---|---|---|---|
| A | **Decorator + static scan.** `@ARCH_REGISTRY.register(scale="upscale")` in the arch file; the registry builds `name -> module` by scanning the package's source text for decorated definitions on first use, then imports only the requested module (where the decorator runs for real and can be cross-checked against the scan). No table | one file | measured here on all 54 arch files (929 KiB): regex 7 ms + read 6 ms, `ast.parse` 233 ms; `import torch` is 3.7 s | the scan must understand the decorator syntax it allows (decide: regex on a strict form, or `ast`); source must be readable (zip / pyc-only installs); two files registering one name must fail at scan time |
| B | **Decorator + generated index.** As A, but a script writes the index (the present tables) and a test fails when it is stale (the pattern of `scripts/studio_surface.py`) | one file + run a script (or a pre-commit hook) | none at run time | the extra step is exactly the inconvenience the owner wants gone, unless the scan of A is the fallback when the index is stale |
| C | **Decorator as declaration, table kept.** The decorator only marks and validates; the table stays the lookup | two places, as today | none | does not solve the request |
| D | **Entry points / plugin folders** for third-party modules | n/a for in-tree | n/a | scope creep; mention as a later extension of A (a user folder added to the scan) |

Preliminary recommendation of session 6 (not reviewed): **A**, with the scan result cached in memory per process, the
decorator carrying what the table's second column carries today, and `scripts/studio_surface.py` reusing the same
scan function so the Studio contract keeps working without importing torch.

### 2.6 Questions the design must answer
1. Decorator signature: how is the scale kwarg declared (`register(scale="upscale")`, `scale=None` for scale-free
   nets) or inferred from the constructor signature (then `omnisr` and factories that forward `**kwargs` need a
   rule)? Should 1x-only (`nafnet`) and ONNX capability (`NO_ONNX`, `STATIC_ONLY` in `convert.py`) move into the
   decorator too (this is follow-up S.11, per-arch capability metadata: one place instead of three)?
2. Factories: `def hat_s(**kwargs)` rows are functions; the decorator must work on functions and classes.
3. Which packages get decorators: archs and losses certainly; metrics (function + `_pt` twin + `CHOICES`), datasets
   (3 rows), models (2 rows)? `OPTIMIZERS` / `SCHEDULERS` hold torch classes and stay tables.
4. Scan mechanics: exact accepted decorator forms; multi-line decorators; a decorated name inside `if TYPE_CHECKING`
   or a nested scope; what the error says when the scan and the imported module disagree.
5. Duplicate names, name vs Python identifier (aliases: yes or no), did-you-mean over the scanned names.
6. Import hygiene: the registry object lives in a module that imports nothing heavy; arch files import it at
   module level (cheap); `tests/test_hygiene.py`'s `_REGISTRY` ban becomes a rule about **what a registry may do**
   (no folder import, no argv, no work at import).
7. Workers and DDP ranks: every process scans once (~13 ms); is that acceptable, or pass the map down?
8. User modules outside `neosr/` (a `--plugin-dir` or a config key): now, later, or never.
9. Migration: mechanical edit of 54 arch files + the loss modules (one decorator per row; a script can generate the
   edit from the present tables, then the tables are deleted); order of packages; how `tests/test_registry.py`,
   `tests/test_imports.py` (`TABLES`), `scripts/studio_surface.py`, `readme.md` sections change.
10. What "more efficient than before" is measured by: no module import besides the requested one (existing probe in
    `tests/test_imports.py`), scan time bound, zero work at import.

### 2.7 Files the implementation will touch (for scoping the design)
`neosr/utils/factory.py` (or a new `neosr/utils/registry.py`), `neosr/archs/__init__.py` + 54 `*_arch.py`,
`neosr/losses/__init__.py` + loss files, optionally `neosr/metrics/`, `neosr/data/`, `neosr/models/`,
`convert.py` (if capabilities move), `scripts/studio_surface.py`, `tests/test_registry.py`, `tests/test_imports.py`,
`tests/test_hygiene.py`, `tests/test_studio_surface.py`, `readme.md` ("adding an architecture", "adding a loss",
"changes since the refactor"), `docs/refactor/DECISIONS.md` (supersedes T3 D3), `studio/shared/catalog/
neosr-surface.json` (must come out **identical** after the migration: that is the cheapest end-to-end proof).

### 2.8 Steps
1. Read `T3.md` (sections 0, 1 D3, 3.1-3.3, 4 R1/R4, 7), `neosr/utils/factory.py`, the four `__init__.py` tables,
   `tests/test_registry.py`, `tests/test_imports.py`, `tests/test_hygiene.py`, `scripts/studio_surface.py`.
2. Look at how upstream does it today (traiNNer-redux keeps the BasicSR registry: look for `utils/registry.py` and
   `archs/__init__.py` under `traiNNer/`; paths not verified from this machine; spandrel's arch registry is a
   second reference) for ideas only.
3. Write `docs/refactor/design/T7_registration.md`: the comparison of 2.5 with numbers, answers to 2.6, the
   migration plan, the tests. Keep it short (2-3 pages).
4. Codex consult on the draft (this is an architecture decision: effort `xhigh`), apply what holds.
5. **Stop and ask the owner to choose** (present A/B/... in a few lines). Implement only after the answer.
6. Implementation in steps with a Codex diff review each: registry + scan + tests -> archs migration (scripted) ->
   losses -> others -> docs, readme, Studio surface check. Verify on Windows with `tests/test_registry.py`,
   `tests/test_imports.py`, `tests/test_hygiene.py`, `tests/test_config.py`, `tests/test_studio_surface.py`,
   `python scripts/studio_surface.py --check`; the per-arch tests and the full suite go to the Linux list.

---

## 3. Task 2: closure matrix of the stage-2 audit, then the fixes

### 3.1 Why
The stage-2 audit produced **167 findings**. `docs/refactor/stage2/PLAN.md` assigns about 96 ids to implementation
lanes; `stage2/DECISIONS.md` lists what stayed opt-in; for the rest nobody recorded whether it was fixed, rejected
or forgotten. A stable core needs that list closed: each finding gets a verdict with evidence, the open ones get
fixed or explicitly deferred.

### 3.2 Inputs
| report (`docs/refactor/stage2/analysis/`) | id prefix in the file | id prefix in PLAN/tests | count |
|---|---|---|---|
| `trainer.md` | `T-` | `T-` | 17 |
| `optimizers.md` | `O-` | `O-` | 13 |
| `losses.md` | `L-` | `L-` | 18 |
| `data.md` | `D-` | `D-` | 20 |
| `archs_window.md` | `W-` | `W-` | 16 |
| `archs_mixed.md` | **`M-`** | **`MX-`** | 19 |
| `archs_cnn.md` | `A-` | `A-` | 18 |
| `metrics.md` | **`M-`** | **`MT-`** | 17 |
| `stability.md` | **`S-`** | **`ST-`** | 15 |
| `gpu_profile.md` | `GP-` | `GP-` | 14 |
| | | total | **167** |

Mind the two `M-` files and the renamed prefixes. Each report opens with a summary table (id, priority, one line) and
then has one section per finding with evidence, fix, expected effect, compatibility.
Other evidence: `stage2/PLAN.md` (lane table: which ids each lane owned), `stage2/DECISIONS.md` (decisions 1-13 and
"Kept opt-in pending an A/B"), `stage2/PROGRESS.md` (what merged, the 11 cross-lane review findings), the tests
`tests/test_s2_*.py` (they name the ids they cover: about 113 distinct ids appear in `tests/`), code comments with
ids (about 29 in `neosr/` and the entry points), `analyze.md` in the repo root (the earlier Codex audit, findings
F01-F25, which several lanes map to their own ids; `stability.md` section 1 has such a table), the stage-2
commits (`git log --oneline --since=2026-10-27 --until=2026-10-29`).

### 3.3 Deliverable
`docs/refactor/stage2/CLOSURE.md`: one table per report,

    | id | P | finding (one line) | verdict | evidence | next action |

with verdict one of `done` (code + test or commit named), `opt-in` (the key that enables it; its A/B is the Linux
machine's), `rejected` (where and why), `superseded` (by which later change), `needs-GPU` (a measurement claim that
cannot be checked here), `open`. Then a ranked list "open, fixable on Windows" and "open, needs the Linux machine",
copied into `STATUS.md`.

### 3.4 Method
- A verdict `done` needs something read: the code at a path:line or a test that fails without the fix. A statement
  in PROGRESS is a pointer, not evidence.
- One worker per report (ten reports, 30-48 KiB each, plus code reading): tier 2-3 agents with a brief that gives
  the report path, the prefix mapping, the evidence sources of 3.2 and the row format; the main thread spot-checks
  a sample of `done` rows and all `open` rows. Do not read the ten reports into the main context.
- Performance and memory claims (most of `gpu_profile.md`, parts of the arch reports) cannot be verified here:
  mark the code part and leave the number as `needs-GPU`.
- Known already (do not rediscover): the stage-1 follow-ups and their status are in `docs/refactor/PLAN.md`
  ("Follow-ups", status line of 2026-11-10) and `STATUS.md` ("Closed follow-ups").

### 3.5 After the matrix
Fix the open items in priority order, smallest coherent change each, with a test, Codex review, one commit.
Anything that changes training dynamics or an objective is opt-in until an A/B (owner's principle 2).

---

## 4. Task 3: small core items (known today)
| item | where | what | note |
|---|---|---|---|
| hma dynamic ONNX pads wrong for tiny inputs | `neosr/archs/hma_arch.py:1370-1377` (`check_image_size`) | `mode = "reflect" if ph < h and pw < w else "replicate"` is decided at trace time, so a dynamic export keeps the trace's mode; eager falls back to replicate when the pad reaches the size (a side under 8 px in `hma`, 16 px in `hma_medium`, e.g. 3x5, 5x7) | found by Codex in stage 3 A8 (`stage3/PROGRESS.md`, "A8 hygiene"); options: one mode for both cases, or refuse such inputs in `convert.py`. ONNX verification runs on CPU (`tests/test_s2_exportfix.py`), slow but possible here for one arch |
| dead code in `convert.py` | `convert.py:32-83` | a module-level string literal holding an old `rep_rt4ksr` under "TODO" | delete; no behaviour change |
| one defaults table (follow-up S.5) | `total_iter` default `1000000` at `neosr/config.py:541,595`, `neosr/models/image.py:227`, `neosr/utils/logger.py:104`, `train.py:141,185`; also `max_skip_streak`, `keep_states` | one constant or table instead of repeated literals | T3 rule: `load_config` fills no defaults (readers keep their `.get`), so the design choice is "a constants module the readers import", not "fill the config" |
| per-arch capability metadata (follow-up S.11) | today in three places: `convert.py:321-341` (`NO_ONNX`, `STATIC_ONLY`), `studio/shared/catalog/archs.ts` (window sizes, scales), scattered constructor checks | one source per arch | **decide inside Task 1** (decorator arguments), do not build it twice |
| open TODO | `neosr/data/otf_dataset.py:168` | kernel size range 7..21 is hard-coded | upstream makes it configurable (lane item C10); low priority |

Line numbers move; `grep -n` the quoted text.

---

## 5. Task 4: Studio (after 1-3)
State: prototype; recipe model, TOML compiler and validator are real, **training is simulated** (`studio/server/sim/`),
no adapter to `train.py`. The catalog matches the trainer as of 2026-11-10 except the OTF keys added later the same
day (`resize_mode*`, `noise_prob*`, `jpeg_prob*`, `blur_prob`, `clean_*`, `sharpen_*`): they are in the surface JSON
and in `config.py`, not yet in `studio/shared/catalog/degradations.ts` (the contract test only checks that Studio
emits no key the trainer does not know, so it is green).
Next steps in the order they unblock each other (ask the owner before starting):
1. The real adapter: `studio/server/` spawns `train.py`, parses the log lines and TensorBoard scalars into the
   event shapes the simulator emits (`studio/server/API.md`, `studio/shared/run.model.ts`); testable against a
   recorded log, no GPU needed for the parser.
2. Arch constructor parameters in the surface contract (today hand-checked): comes almost for free if Task 1 makes
   the scan read signatures.
3. Deferred UI items listed in `design/NeoSR_Studio_UI_Concept_v2.md` section 7 (light theme, folder browser,
   metric parameters in the Validation node, ...), ONNX capability badges from the surface's `no_onnx` /
   `onnx_static_only`.
Checks: `npm run typecheck`, `npx ng test --watch=false` (39 cases), `npm run contract` (13 TOMLs through the
trainer's validator), `npx ng build --configuration development --output-path dist/check-<name>`.

---

## 6. What waits for the Linux machine (do not attempt here)
The list lives in `STATUS.md`: "Needs the Linux machine" (full suite on the current head, the R3GAN penalty-pass
profile and A/B, Phase 0 A/Bs, a long run) and "Waiting for the Linux machine" (changes of session 6 not yet covered
by a full run: the Studio surface script, OTF O1-O3 incl. the CUDA-only no-host-sync test).

## 7. What session 6 did (commits `05954e0..8776bbd` on `master`)
| commit | content |
|---|---|
| 0fb201d, ca244a4 | `AGENTS.md` project facts; `STATUS.md` as the single status page, follow-ups closed |
| 6822a1b, 46dd2cc | `scripts/studio_surface.py` + JSON; Studio catalog, templates, validator, importer synced to the trainer; contract tests |
| 92081c6 | Windows CPU venv recipe, compute-ordered work list, `test_s3_hygiene.py` reads UTF-8 |
| e37dec7 | OTF realism design (parked) |
| 7f7cb78, d2f89ce, a7cb108 | OTF O1-O3 (opt-in, defaults bitwise unchanged): `neosr/data/resample.py`, keys in `otf_degradation.py` / `otf_dataset.py` / `config.py`, `tests/test_s3_otf.py` |
| 8776bbd | owner priorities recorded, OTF parked |
