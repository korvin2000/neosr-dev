# NeoSR Studio (prototype)

A web UI for configuring, launching and watching [NeoSR](https://github.com/neosr-project/neosr) training runs. One canonical **recipe** is edited through three synchronized views — a guided **Wizard**, a ComfyUI/chaiNNer-style **node Graph**, and the generated **TOML** — and launched into a run cockpit that replaces TensorBoard for the common case.

> **Prototype status.** The TOML compiler, the validator and the recipe model are real and produce NeoSR-ready configs. Training is **simulated** by the bundled Node backend: no `neosr-train` is launched, no dataset is scanned, no file is written into the NeoSR repository. Every simulated number is labelled as such in the UI. The concept and its review are in `../design/NeoSR_Studio_UI_Concept.md`.

## Run it

```bash
npm install
npm start          # web on http://localhost:4300 (proxies /api) + simulated backend on :4310
```

Or separately: `npm run web` and `npm run api`. Smoke-test the compiler with `npx tsx tools/smoke-compile.ts`.

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
shared/     recipe model · NeoSR catalog (93 archs, 15 losses, 6 optimizers…) · compiler/validator/importer  (browser + server)
server/     Express API + deterministic trainer simulator (see server/API.md)
src/app/    Angular 22, standalone + signals + zoneless: core stores, ui primitives, feature pages
tools/      smoke-compile.ts
```

See `DEVELOPING.md` for the store/command API and conventions, and `server/API.md` for the backend contract.

## Quality gates run on this prototype

- `npx ng build` (production) and `ng test` (29 compiler / importer / validator / sections cases) pass.
- Independent reviews by a second model: one of the UX concept before building (12 findings, folded into `../design/NeoSR_Studio_UI_Concept.md` §15a), one of the v1 code (20 findings; 18 fixed, 2 mitigated), and two design critiques of the v2 rework (Opus 5.5 and Codex) folded into `../design/NeoSR_Studio_UI_Concept_v2.md`.
- Browser walkthrough of every page against the simulated backend, including launch → live cockpit → graceful stop → resume.
- Visual QA pass with pixel captures at 1440×900 and 390×844 (`tools/shot.mjs`, Chrome DevTools protocol so phone widths are exact) of every screen and every Build section; captures live in `../design/screenshots/qa/` and `qa-phone/`.

Known limitations left open on purpose: the simulator keeps full-resolution series in memory until a run ends (snapshots are downsampled), and the TOML importer models metrics by name (parameters are preserved as extras rather than edited in the Validation node).

## Facts the UI is built on

The catalog and the validation rules were extracted from this repository's `neosr/` package (`../design/neosr_config_surface.md`): required keys, the GAN triple coupling (`network_d` ⇔ `optim_d` ⇔ `gan_opt`), the OTF bundle (`model_type = otf` + `[degradations]`), platform gates (`compile` not on Windows, `bfloat16` needs `use_amp`), per-arch scale limits and window sizes, and the fact that NeoSR keeps loss weights constant within a run — which is why a changed weight is a phase boundary, not a ramp.
