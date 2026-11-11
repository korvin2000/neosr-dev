# NeoSR Studio (prototype)

A web UI for configuring, launching and watching [NeoSR](https://github.com/neosr-project/neosr) training runs. One canonical **recipe** is edited through three synchronized views — a guided **Wizard**, a ComfyUI/chaiNNer-style **node Graph**, and the generated **TOML** — and launched into a run cockpit that replaces TensorBoard for the common case.

> **Prototype status.** The TOML compiler, the validator and the recipe model are real and produce NeoSR-ready configs. Training is **simulated** by default by the bundled Node backend: no `neosr-train` is launched, no dataset is scanned, no file is written into the NeoSR repository. Every simulated number is labelled as such in the UI. `STUDIO_TRAINER=real` switches new runs to the actual `train.py` (see `server/API.md`, "Real trainer"). The concept and its review are in `../design/NeoSR_Studio_UI_Concept.md`.

## Run it

```bash
npm install
npm start          # web on http://localhost:4300 (proxies /api) + simulated backend on :4310
```

Or separately: `npm run web` and `npm run api`. Smoke-test the compiler with `npx tsx tools/smoke-compile.ts`; check the catalog against the trainer with `npm run contract` (see below).

## What it does (v2, "ask less, show what the decision needs")

| Screen | Default state | One click away |
|---|---|---|
| **Start** | Three outcome cards — Clean · Balanced · Detailed — each one line and an "≈ hours on this GPU" estimate; Continue / Running rows | All templates, Import TOML, environment popover |
| **Setup · Data** | Source switch (have low-res / generate on the fly), folders, validation choice; the data check runs by itself and reports "2,888 pairs · ×4 detected · 1 issue ›" | Issues popover, scale change, OTF scale control |
| **Setup · Review** | One editable sentence: *Train **SPAN** at ×4 … with **clean losses** for **150k updates**, **from scratch**, validating every 1k*; GPU-fit line; **Launch** | Token popovers (2–4 choices + "More ›"), Changed from preset, n to check, Open full recipe, ⋯ |
| **Build · List** | Find-setting box and seven rows — Data · Model · Losses · Training plan · Optimizer · Validation & saving · Runtime — each a one-line summary with "n changed" / "n to fix" | A section's Basics (≤ 5 fields), then a remembered "More options"; the loss × phase table; the GAN switch; the architecture picker |
| **Build · Graph** | Summary cards (title, 2–3 values, ports, status icon only when something is wrong), **+** to add | Click → inspector drawer (same section editor, connection pickers, ⋯); phase strip only with > 1 phase; display options in ⋯; phone: outline list |
| **Build · TOML** | Code, Copy · Download · Copy command · Issues n | "Edit" on section headers opens the inspector over the code; Import, Diff, Unmapped settings in ⋯ |
| **Run** | Status chip, phase-segmented progress bar with checkpoint ticks, one state-dependent action (Stop & save / Resume / Weights…), the **live training flow** (Run → Data → Generator → Losses → Discriminator → Optimizer → Checkpoint / Validation, GPU; animated edges and a beat that walks one update through the pipeline, live rates, loss bars, grad-norm and speed sparklines, memory gauge), one aggregated warning row, validation chart and latest-vs-best image (hold to blink) | Images · Diagnostics · Activity tabs; Config, Fork, Force stop in ⋯; phone: the flow as a vertical list |
| **Runs / Compare / Library** | Rows with status, progress, last activity, ETA and a visible ⋯; Compare from a selection; Library with search, facets and one-line cards | Row actions, config and image comparison, card details |

Rules behind it (see `../design/NeoSR_Studio_UI_Concept_v2.md`): presets are complete configurations, so the user supplies data and optionally weights; help is pulled (ⓘ toggletips, tooltips for icon names), never pushed; every field shows a "changed from preset" bar with reset; blockers appear under the field and on the primary button ("Launch · 1 to fix"); every hover or drag gesture has a visible twin; the whole app works in a single column on a phone (bottom tab bar, sheets instead of drawers).

## Layout

```text
shared/     recipe model · NeoSR catalog (95 archs, 9 discriminators, 16 losses, 6 optimizers, 5 metrics…) · compiler/validator/importer  (browser + server)
server/     Express API + deterministic trainer simulator (see server/API.md)
src/app/    Angular 22, standalone + signals + zoneless: core stores, ui primitives, feature pages
tools/      smoke-compile.ts, contract.ts (compiled TOMLs → the trainer's own validation)
```

See `DEVELOPING.md` for the store/command API and conventions, and `server/API.md` for the backend contract.

## Quality gates run on this prototype

- `npx ng build` (production) and `ng test` (39 compiler / importer / validator / sections / catalog-contract cases) pass, and so does `npm run contract`.
- Independent reviews by a second model: one of the UX concept before building (12 findings, folded into `../design/NeoSR_Studio_UI_Concept.md` §15a), one of the v1 code (20 findings; 18 fixed, 2 mitigated), and two design critiques of the v2 rework (Opus 5.5 and Codex) folded into `../design/NeoSR_Studio_UI_Concept_v2.md`.
- Browser walkthrough of every page against the simulated backend, including launch → live cockpit → graceful stop → resume.
- Visual QA pass with pixel captures at 1440×900 and 390×844 (`tools/shot.mjs`, Chrome DevTools protocol so phone widths are exact) of every screen and every Build section; captures live in `../design/screenshots/qa/` and `qa-phone/`.

Known limitations left open on purpose: the simulator keeps full-resolution series in memory until a run ends (snapshots are downsampled), and the TOML importer models metrics by name plus their fixed choices (TOPIQ `dataset`, LPIPS `net`); other metric parameters (`crop_border`, `test_y_channel`) are preserved as extras rather than edited in the Validation node.

## Facts the UI is built on

The catalog and the validation rules were extracted from this repository's `neosr/` package (`../design/neosr_config_surface.md`) and synced to the trainer state of 2026-11-10 (stage 3 Phase A + the R3GAN code): required keys, the GAN triple coupling (`network_d` ⇔ `optim_d` ⇔ `gan_opt`), the OTF bundle (`model_type = otf` + `[degradations]`), platform gates (`compile` not on Windows, `bfloat16` needs `use_amp`), per-arch scale limits (`nafnet` is ×1 only) and window sizes, the R3GAN scalar heads' fixed GT crop (128 / 256 px), and the `validate_train` rules (`val.save_best` names a metric, `train.clamp` / `train.grad_clip` value domains, `train.scheduler.nets` names existing optimizers, no R1/R2 with `dunet`).

The templates follow the shipped `options/train_*.toml`: G `adamw` lr 2e-4, D `adan_sf` lr 1e-4 with warm-up 600, L1 + MS-SSIM + consistency + LDL + FDL (DINOv2, 0.02) + GAN 0.3, EMA 0.999 with `ema_power` 0.75, no batch augmentation.

Phases are sequential runs: an objective change, a new discriminator or a separate fine-tune is a phase boundary, and a weight edited per phase holds for that whole run. Within one run NeoSR itself can switch or ramp a term: every loss row takes `start_iter`, `ramp_iters` and `end_iter` (row parameters, counted in the run's native updates).

## Staying in sync with the trainer

`shared/catalog/neosr-surface.json` is generated from the trainer's tables by `../scripts/studio_surface.py`; never edit it by hand.

| Step | Command |
|---|---|
| Regenerate the surface after a trainer change | `npm run surface` |
| Fail when the surface is stale | `npm run surface:check` |
| Catalog vs surface (types, arch constructor parameters and defaults, losses, metrics, `gan_type`, schema keys) | `npx ng test --watch=false` (`src/app/core/catalog-contract.spec.ts`) |
| Compile every template, an all-options recipe and the shipped realplksr recipes, then run the trainer's `load_config` + `validate_train` on them | `npm run contract` |

`--validate` covers the schema and the cross-key rules; constructor kwargs (`network_*`, `train.*_opt`, optimizers, scheduler, metrics) are opaque to it. The arch constructors are read with `ast` into the surface (`arch_params`: names, literal defaults, preset-bound `fixed` keys); the other constructors still come from reading the Python code.
