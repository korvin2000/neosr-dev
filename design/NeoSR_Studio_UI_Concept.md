# NeoSR Studio — Unified UI Concept

> **Superseded in part.** The screen designs in this document are v1. After a density review, the screens were redesigned in `NeoSR_Studio_UI_Concept_v2.md` (ask less, progressive disclosure, phone-capable). The recipe model (§4), node catalog (§5), phase semantics (§6, §15a) and architecture (§14) remain the reference.

**Status:** Design concept for the prototype in `studio/` (Angular 22 + Node). Synthesizes `SISR_Training_Studio_Design_Proposal.md` (guided workflow, schema-driven forms, preflight, run cockpit) and `SR_Training_Studio_Proposal.md` (node workbench, one canonical spec with several views, phases, support levels), plus the node-UI reference (`screenshots/`, UNBI-Engine frontend).
**Scope:** NeoSR only. No live trainer integration in the prototype; the backend simulates runs. Everything the UI produces is a plain NeoSR TOML plus the `neosr-train` command, so the real integration later is an adapter, not a redesign.

---

## 1. Decisions in one table

| Question | Decision | Taken from |
|---|---|---|
| Is the graph the config, or a picture of it? | **One canonical Recipe** (JSON). Wizard, Graph and TOML are three synchronized views of it; editing any view edits the same store. | Doc 2 §13.1, user brief |
| How does the Wizard relate to nodes? | Each wizard step **owns a set of nodes**. Completing a step materializes those nodes, wired and auto-laid out. Re-opening the wizard shows the current node values. | User brief, Doc 1 §4.2 |
| How many nodes? | **6–12 meaningful blocks**, one per NeoSR TOML section family. Scalars are fields in the inspector, never nodes. | Doc 2 §1, §5.4 |
| Losses that change over training | NeoSR keeps loss weights constant within a run (verified in `neosr_config_surface.md` §9). So a **Phase timeline** owned by the Training Plan node; the Loss Recipe node holds per-phase weights. Compiles to one TOML per phase chained by `pretrain_network_g` / `resume_state` (orchestrated), while the native iteration knobs (`gan_opt.wavelet_init`, SF-optimizer `warmup_steps`, scheduler milestones, `val_freq`, `save_checkpoint_freq`) are drawn on the same timeline (native). | Doc 1 §7.4, Doc 2 §6.4, §12.4 |
| Beginner vs expert | Three disclosure levels in every form: **Essentials / Advanced / Native**. Wizard = essentials with suggestions; Graph inspector = all levels; TOML view = native. | Both docs |
| Presets | Two kinds: **Recipe templates** (whole graph) and **Node presets** (values of one node). Loading a template creates the nodes; the wizard can start from a template. | User brief, UNBI |
| Import | Existing NeoSR TOML can be **imported into nodes**; unknown keys are preserved in a "native extras" bag and re-emitted. | Doc 1 §5, Doc 2 §13.3 |
| Monitoring | A **Run cockpit** (not an embedded TensorBoard): progress, losses, LR, validation metrics, system, validation images with a checkpoint scrubber, events, logs. Native `tb_logger` output stays available. | Doc 1 §9, Doc 2 §9 |
| Runs vs recipes | The recipe is editable; **a launched run is immutable** and keeps a frozen copy of the recipe + TOML. Fork a run → new recipe. | Doc 2 §19 |
| Stack | Angular 22 (standalone, signals, zoneless) + `@foblex/flow` canvas + ECharts; Node/Express backend with SSE for run events; shared TypeScript model and compiler used by both. | User brief, UNBI |

---

## 2. Personas and the two paths they take

| Persona | Need | Path |
|---|---|---|
| First-time trainer | One successful run without reading the walkthrough | **Wizard** → review → launch. Never has to touch the canvas; sees it fill up on the side. |
| Experienced trainer | Fast iteration on a known recipe; exact control | **Template → Graph**, edit 2–3 nodes in the inspector, check TOML, launch. Or **Import TOML**. |
| Researcher | Phases, loss schedules, forks, comparisons | Graph + Phase timeline + Runs compare + Fork. |

Both paths end in the same place: **Review & Preflight** → immutable Run.

---

## 3. Application shell and navigation

```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ ◉ NeoSR Studio   Recipe: "4x_anime_span" ▾  ● draft   [Wizard] [Graph] [TOML]   GPU: RTX 4090 · 2 runs ●  ⚙ │
├────────┬─────────────────────────────────────────────────────────────┬───────────────┤
│ Home   │                                                             │ INSPECTOR     │
│ Build ●│   (Wizard | Graph canvas | TOML)                            │ (Graph only)  │
│ Runs   │                                                             │ Essentials    │
│ Compare│                                                             │ Advanced      │
│ Library│                                                             │ Native        │
│ Settings                                                             │               │
├────────┴─────────────────────────────────────────────────────────────┴───────────────┤
│ Problems: 0 blockers · 2 warnings │ Phases: 2 │ LQ 64 → GT 256 │ eff. batch 16 │ [Preflight] [Launch] │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

| Page | Purpose | Key content |
|---|---|---|
| **Home** | Orientation | Active runs, recent recipes, "New recipe" (Wizard / Template / Blank / Import TOML), GPU status |
| **Build** | Author the recipe | View switch Wizard · Graph · TOML; status bar with derived numbers and Problems; Preflight; Launch |
| **Runs** | Watch and control | Run list; Run cockpit per run |
| **Compare** | Decide | 2–3 runs: metric overlays, config diff, image compare |
| **Library** | Reuse | Recipe templates, my recipes, node presets, models (checkpoints), datasets |
| **Settings** | Environment | NeoSR root, experiments folder, GPU, theme, units |

The left nav never changes with the graph; a beginner finds Runs without understanding nodes (Doc 2 §4.1).

---

## 4. The Recipe model (canonical spec)

```ts
interface Recipe {
  id: string; name: string; version: 1;
  nodes: RecipeNode[];      // { id, type, title?, values: Record<string, unknown>, position, collapsed, disabled }
  edges: RecipeEdge[];      // { id, sourceNode, sourcePort, targetNode, targetPort }
  phases: Phase[];          // owned by the Training Plan; see §6
  nativeExtras: Record<string, unknown>; // TOML keys the UI does not model, preserved verbatim
  meta: { createdAt, updatedAt, templateId?, tags[] };
}
```

- The **semantic hash** covers nodes' types+values, edges, phases and extras; not positions or collapsed state (Doc 2 §13.1).
- Wizard answers are not stored separately: the wizard reads and writes node values through the same store, so the graph is always the truth.
- Undo/redo is one history for all views (command pattern over an immutable document, as in UNBI).

---

## 5. Node catalog (NeoSR-aligned)

Port types are semantic; a port only accepts its own type (colour = type everywhere).

| Node | Family / accent | Writes TOML | In ports | Out port | Essentials (card shows 2–4) |
|---|---|---|---|---|---|
| **Train Dataset** | Data · emerald | `[datasets.train]` (+ top-level `model_type` = image/otf derived from `type`) | `degradation?` (Degradation, required when type = otf) | `TrainData` | type paired/otf, GT path, LQ path (hidden for otf), patch_size (+GT crop readout), batch_size, accumulate (+effective batch), augmentation, workers |
| **OTF Degradation** | Data · emerald | `[degradations]` (two-stage Real-ESRGAN pipeline: blur kernels, resize, noise, JPEG, sinc) | — | `Degradation` | stage 1 / stage 2 groups with probabilities and ranges; "NeoSR default" preset = the table shipped in all `*_otf.toml` templates |
| **Validation** | Data · emerald | `[datasets.val]`, `[val]`, `[val.metrics.*]` | — | `ValSpec` | val GT/LQ paths, val_freq, metrics (psnr/ssim/dists/topiq), save_img, tile |
| **Generator** | Model · violet | `[network_g]`, top-level `scale` | — | `Model` | arch (card picker), scale, arch params (advanced, schema per arch) |
| **Discriminator** | Model · violet | `[network_d]` | — | `DModel` | arch (metagan, …), params |
| **Initialization** | Model · violet | `[path]` | `checkpoint?` (Checkpoint) | `Init` | mode scratch / pretrain / resume, pretrain_network_g, pretrain_network_d, resume_state, strict_load_g |
| **Loss Recipe** | Objective · rose | one `[train.<slot>_opt]` per row | — | `Objective` | loss rows: slot, type, weight, per-phase weight, params, `target` gt/lq for colour losses; `gan_loss` row carries `gan_type` and `wavelet_init` and requires DModel + OptimD |
| **Optimizer G** | Optimization · amber | `[train.optim_g]` | — | `OptimG` | type (adam/adamw/nadam/adan/adamw_sf/adan_sf), lr, betas, weight_decay, warmup_steps (SF only) |
| **Optimizer D** | Optimization · amber | `[train.optim_d]` | — | `OptimD` | same as G |
| **LR Schedule** | Optimization · amber | `[train.scheduler]` (one table, applies to G and D) | — | `Schedule` | type multisteplr/cosineannealing, milestones/gamma or T_max/eta_min; warned as redundant with schedule-free optimizers |
| **Runtime** | Run · sky | top-level `use_amp`, `bfloat16`, `fast_matmul`, `compile`, `compile_mode`, `manual_seed`, `num_gpu` | — | `Runtime` | precision preset (fp32 / amp fp16 / amp bf16), fast_matmul, compile (Linux only), seed |
| **Training Plan** | Run · sky | top-level `name`; `[train]` ema, ema_power, grad_clip, clamp, max_skip_streak; `[logger]` total_iter, save_checkpoint_freq, keep_states, print_freq, use_tb_logger, save_tb_img; **phases** | `data` TrainData, `val?` ValSpec, `model` Model, `discriminator?` DModel, `init?` Init, `losses` Objective, `optimG` OptimG, `optimD?` OptimD, `schedule?` Schedule, `runtime` Runtime | `Checkpoint`, `Run` | name, total iterations (= sum of phases), phases strip, save freq, EMA |
| **Export** | Deploy · teal | `neosr-convert` args | `checkpoint` Checkpoint | — | format (onnx/safetensors), fp16, opset, dynamic shapes |

Rules that produce **Problems** (computed, node-linked; the NeoSR rules come from `neosr_config_surface.md` §10):
- Training Plan missing `data`, `model`, `losses`, `optimG` or `runtime` → blocker. No generator loss with non-zero weight → blocker.
- GAN triple coupling: `gan_loss` row ⇔ Discriminator ⇔ Optimizer D, all present or all absent → blocker naming the missing part. `wavelet_init` without a GAN row → blocker.
- Generator scale outside the arch's supported scales (e.g. `cugan` ×1, `swinir_large` only ×4) → blocker; `patch_size` not a multiple of the arch window size → blocker; arch scale kwarg must never be emitted.
- `bfloat16` without `use_amp` → blocker; `compile` on Windows → blocker; `num_gpu > 1` → info "needs `--launcher`".
- OTF bundle: dataset `otf` ⇔ Degradation connected ⇔ `model_type = otf`; `dataroot_lq` on otf → blocker; Degradation edge on a paired dataset → warning (ignored). `queue_size ≥ batch_size`.
- `augmentation` / `aug_prob` length mismatch → blocker; non-cutblur augmentation with `batch_size = 1` → blocker.
- `val.tile > 2 × tile_overlap ≥ 0` → blocker; DISTS metric without `better = "lower"` → auto-set; val dataset without `name` → auto-set.
- `save_checkpoint_freq` missing → blocker; `val_freq` < `save_checkpoint_freq`/4 → warning (cost); `use_tb_logger = false` → warning ("Studio charts need tb logger or hooks").
- Schedule-free optimizer + LR Schedule node → warning (redundant); `adamw_sf` must keep `schedule_free = true`.
- Phase ranges zero-length, or a `continue` phase after a phase that changed arch / optimizer type → blocker (resume is strict).
- Two enabled nodes owning the same TOML section (e.g. two Runtime nodes) → blocker naming both owners (Doc 2 §13.2).

A simple pixel recipe is 8 nodes: Train Dataset, Validation, Generator, Initialization, Loss Recipe, Optimizer G, Runtime, Training Plan. A GAN recipe adds Discriminator, Optimizer D (and optionally LR Schedule, OTF Degradation, Export).

---

## 6. Phases and the Loss Timeline (the "losses vary with epochs" feature)

NeoSR trains one TOML per run. The UI therefore models a **curriculum** explicitly and honestly:

```text
PHASES   │ 1 · Pixel pretrain          │ 2 · Perceptual                │ 3 · GAN finish      │
iters    0                        100k                           250k                     300k
─────────┼─────────────────────────────┼───────────────────────────────┼─────────────────────┤
mssim    │ ████████████ 1.0            │ ████████████ 1.0              │ ████████ 1.0        │
fdl      │                             │ ░░░▓▓▓▓▓▓▓▓▓ 0 → 0.75 (ramp)  │ ████████ 0.75       │
gan      │                             │                               │ ▒▒▒▒▒▒▒▒ 0.3 · wavelet view from +80k (wavelet_init, native) │
LR (G)   │ 1e-3 ───────╲ cosine        │ 5e-4 ───────╲                 │ 1e-4 ───            │
markers  │ ▲ckpt every 10k             │ ▲ wavelet_init 80k (native)   │                     │
init     │ scratch                     │ ← fine-tune from P1 best      │ ← continue state    │
```

| Element | Semantics | Support level shown |
|---|---|---|
| Phase | iteration range, name, init policy (`fine-tune from weights` / `continue state`), per-phase overrides of loss weights, LR, patch/batch | **orchestrated** (N sequential `neosr-train` runs, chained by `pretrain_network_g` / `resume_state`) |
| Weight ramp inside a phase | linear ramp from a to b | **orchestrated** (prototype shows it; marked "requires hook") |
| `gan_opt.wavelet_init`, SF `warmup_steps`, scheduler milestones, `ema_power` ramp, `val_freq` / `save_checkpoint_freq` ticks | the only native iteration-dependent behaviour in NeoSR | **native** |

The timeline is editable (drag phase boundaries, type weights per cell) and is the same data the Loss Recipe inspector shows as rows. The wizard's objective preset (fidelity / balanced / perceptual) seeds the phases; "single phase" is always available.

Compile output for a 3-phase recipe: `options/<name>_p1.toml`, `_p2.toml`, `_p3.toml` and a `run_<name>.ps1/.sh` listing the three `neosr-train` commands with the checkpoint hand-off. Each TOML is complete and valid on its own.

---

## 7. Wizard (guided path)

Six steps, each with a **suggestion rail** (why, typical values, NeoSR key) and a **live impact panel** (LQ→GT crop, effective batch, phases, estimated VRAM class and time, node count, problems). Every step has "Show in graph" and "Show TOML".

| Step | Asks | Creates / updates nodes | Suggestions shown |
|---|---|---|---|
| 1 Goal | name, scale ×1/×2/×4, domain, objective (fidelity / balanced / perceptual), start (scratch / fine-tune), size tier (fast / balanced / quality) | Training Plan, Runtime | "Fidelity = pixel-only losses, best PSNR; Perceptual = adds FDL/DISTS + GAN later" |
| 2 Data | train type paired / OTF, GT and LQ folders (scan → counts, pairing check, resolution), patch_size, batch_size, accumulate, augmentation; validation folders, val_freq, metrics | Train Dataset, (OTF Degradation), Validation | "patch 64 / batch 8 is the usual ×4 start; GT crop will be 256"; "val every 1000 iters costs ~N s" |
| 3 Model | architecture cards (filtered by scale; badges: params, speed class, GAN-ready, has pretrain); arch options (advanced); discriminator (auto when objective ≠ fidelity); pretrain file | Generator, Discriminator, Initialization | "SPAN/compact: fast iteration; RealPLKSR/MoSR: balanced; DAT/HAT: quality, slow" |
| 4 Recipe | phases (preset by objective, editable), loss rows per phase, optimizer type/lr, schedule, total iterations | Loss Recipe, Optimizer G/D, LR Schedule, phases | "Add GAN only after a pixel phase"; "adan_sf is schedule-free: no LR schedule needed" |
| 5 Runtime & logging | AMP / bf16 / compile / seed / workers; save_checkpoint_freq, print_freq, tb logger, save_tb_img | Runtime, Training Plan (logger) | "bf16 needs Ampere+"; "checkpoints every 1k × 300k iters = N GB" |
| 6 Review | summary cards per node, Problems, TOML per phase, graph thumbnail | — | Actions: Save recipe · Save as template · Open graph · Launch |

Wizard behaviour rules:
- Changing an upstream answer never silently resets downstream edits; it offers "apply suggested defaults" with a diff.
- The wizard can be re-entered from the graph at any step; fields show the node's current values (one store).
- Keyboard-first: Enter/→ advance, all fields labelled, tooltips on focus.

---

## 8. Graph view

Layout (matches the reference screenshots): palette (left, Nodes / Presets tabs, search, grouped by family) · canvas · inspector (right, dock/float) · bottom dock (Problems · TOML · Timeline).

Node anatomy: tinted header (family accent, icon, title, status dot) → key rows with inline widgets (number, select, toggle, path) → ports on the border, colour = type → "Advanced (n)" fold → footer (✓ or ⚠ n, support badge). Collapsed = header + ports.

Canvas features: drag from palette or double-click, drag-to-connect with type checking (incompatible targets dimmed), reassign, multi-select, undo/redo, duplicate, disable (node + downstream muted), rename, auto-layout (layered left→right by family), fit, zoom controls, minimap, context menu, phase strip above the canvas (select phase → nodes show that phase's values; edges to phase-irrelevant nodes muted).

Inspector: Essentials · Advanced · Native tabs; field search; "modified only"; every field has label, unit, help, NeoSR key, reset-to-default; per-phase override toggle on phase-aware fields.

---

## 9. TOML view

- One tab per phase; syntax highlighted; **source map**: hovering a key highlights its node; clicking opens the inspector on that field.
- Diff against last saved compile.
- Copy · Download · Save into `options/` (prototype: download) · copy launch command(s).
- Native extras (imported unknown keys) are listed in their own block and editable as raw TOML with validation.

---

## 10. Preflight and Launch

Preflight is the same Problems engine plus mock environment checks (NeoSR found, GPU, free disk vs checkpoint volume, dataset reachable). Output: blockers (cannot launch), warnings (must acknowledge), info. Launch freezes recipe + TOML + command into an immutable Run and opens the cockpit. Never auto-fixes; each suggestion is "Apply to recipe" with a visible diff (Doc 1 §8).

---

## 11. Run cockpit

```text
RUN 4x_anime_span · phase 2/3 Perceptual · RUNNING          [Stop safely] [Force] [Fork → recipe]
Iter 142,300 / 300,000   3.9 it/s · ETA 11h 12m   VRAM 14.1/24 GB   last ckpt 142k (2 min)   ⚠ 1
┌ Losses (raw + smoothed, per loss, log) ─────────┐ ┌ Validation metrics (PSNR↑ SSIM↑ DISTS↓) ──┐
│                                                 │ │                                            │
├ Learning rate ──────────────┐ ┌ System ─────────┤ ├ Validation images  ◀ 130k ▶  [LQ|SR|GT] ──┤
│                             │ │ GPU util/mem    │ │   slider compare · zoom · pick image       │
└─────────────────────────────┘ └─────────────────┘ └────────────────────────────────────────────┘
Events: phase change 100k · ckpt 140k · warn: val cost 8% ...     Logs ▾     Frozen recipe ▾  TOML ▾
```

- Charts share x (iteration) and show checkpoint/phase markers; raw with smoothing overlay; missing segments are gaps, not lines (Doc 2 §9.2).
- Validation gallery: checkpoint scrubber keeps image, zoom and crop fixed ("checkpoint time machine").
- Controls honour the state machine: draft → queued → running → stopping → stopped / completed / failed; resume from state; fork to recipe.
- Data source is the Node mock trainer streaming SSE events (`progress`, `scalar`, `validation`, `checkpoint`, `phase`, `warning`, `log`, `status`). The event schema is what a real adapter would emit from NeoSR logs / tb_logger.

---

## 12. Compare and Library

- **Compare:** choose 2–3 runs → metric overlays, per-run KPIs, TOML/recipe diff, image compare at the same checkpoint index and crop.
- **Library:** templates (built-in: ×4 anime SPAN fidelity · ×4 photo RealPLKSR balanced · ×4 OTF GAN · ×2 compact fast · ×1 JPEG restoration), my recipes, node presets, models (mock checkpoints with arch/scale/iter, used by Initialization), datasets (registered roots with mock scan).

---

## 13. Visual language

Dark-first (training tools run for hours next to terminals), tokens as CSS custom properties, chaiNNer-sampled greys (as in the reference), Inter, 13 px in canvas, 14 px in forms. Family accents: Data emerald · Model violet · Objective rose · Optimization amber · Run sky · Deploy teal. Status: ok green · warn amber · danger red · running blue. Badges: `native` (grey) · `orchestrated` (blue outline) · `estimate` (dashed).

---

## 14. Architecture of the prototype

```text
studio/
  shared/        recipe model · node catalog (specs, archs, losses, optimizers, schedulers, metrics)
                 · compiler (recipe → TOML per phase, validate, import TOML → recipe) · run events
  server/        Express: /api/catalog /api/templates /api/recipes /api/compile /api/import
                 /api/runs (+ SSE events) /api/library/* · trainer simulator (seeded)
  src/app/
    core/        RecipeStore (immutable doc + command history) · CatalogService · ApiService · RunStore
    shell/       topbar · nav · status bar
    build/       wizard/ · graph/ (canvas, node, palette, inspector, phase strip, problems) · toml/ · timeline/
    runs/        list · cockpit (charts, gallery, events, logs)
    compare/  library/  home/  settings/
    ui/          icon · widgets (number, select, toggle, path, chips) · tooltip · dialog · tabs · kpi
```

The compiler lives in `shared/` so the browser renders TOML instantly and the server emits byte-identical files.

---

## 15a. Revisions after the independent concept review (Codex, verified against `neosr/`)

| Finding | Decision |
|---|---|
| `continue` cannot add a GAN or change optimizer type/SF mode: NeoSR resume is strict and restores optimizer/scheduler state | Every phase boundary has an explicit **transition contract**: `finetune` (new optimizers from the previous phase's saved weights) or `continue` (resume `.state`, only when the network/optimizer inventory is unchanged; the validator forces `finetune` otherwise). Under `continue` the phase LR override is shown as "not effective: restored from state". |
| Curriculum position vs native update counter differ per transition | Phases store a duration; the compiler emits `total_iter` = cumulative end for `continue`, phase length for `finetune`. The timeline shows both clocks. `wavelet_init` is drawn as "D input switches to wavelet view", expressed in the native clock of its phase. |
| Loss ramps need trainer hooks | Removed. Loss weights are piecewise-constant per phase. |
| `Checkpoint` port conflates G weights, D weights and `.state` | Port removed. Initialization selects the artifact explicitly (weights or state). Phase hand-off uses the previous phase's `net_g_latest.pth` / `net_d_latest.pth` (last durable save), never an unsaved "best". |
| Export node over-promises | Deferred; the run's artifacts panel shows the `neosr-convert` command for the saved generator. |
| Validation under-modelled | One Validation node with **named dataset rows** (paired with metrics, or `single` visual-only) plus shared `[val]` settings. |
| Validation event ≠ saved checkpoint | The cockpit scrubs validation events with badges: images · G weights saved · state available. |
| Some blockers stricter than NeoSR | Window divisibility is a warning (SwinIR pads); the tile rule applies only when `tile > 0`; `num_gpu > 1` emits the `torchrun` command; defaults (DISTS `better = "lower"`, val `name`) are applied visibly when a row is created, never silently on import. |
| Cockpit needs recovery semantics | Events carry sequence ids and run/phase identity; statuses include `stopping`, `saving`, `stopped`, `completed`, `failed`, `disconnected`; unsmoothed values are labelled "logged window means"; G/D losses, grad norms and skipped updates are shown; missing telemetry is a gap, not zero. |
| Wizard advice stated as rules | Single-phase by default; the two-phase curriculum is an explicit choice; suggestions are labelled heuristics with provenance (template / wiki / community). |
| Dataset inspection lost in the merge | Mock scan panel in the Data step: pair counts, unmatched files, minimum size vs patch, channel warnings, train/val overlap examples (fixtures). |
| Scope | Wizard · editable graph · TOML view with transactional import · curated templates · fixture-driven cockpit and compare · a two-phase weights-based hand-off demo. Deferred: ramps, export, remote workers, real scanning. |

## 15. Out of scope for the prototype (kept as explicit hooks)

Real `neosr-train` launch, real dataset scanning, TensorBoard file ingestion, remote workers, auth, multi-engine adapters, VRAM probe. Each has a mock with the final interface shape.

## 16. Implementation status (prototype in `studio/`)

| Concept section | Built | Notes |
|---|---|---|
| §4 Recipe model, one store, undo | ✓ | `shared/recipe.model.ts`, `src/app/core/recipe-store.ts`, command pattern, semantic hash, localStorage draft |
| §5 Node catalog | ✓ | 12 node types, typed ports, 93 archs / 6 discriminators / 15 losses / 6 optimizers / 2 schedulers / 4 metrics / OTF degradation table extracted from `neosr/` |
| §5 Problems engine | ✓ | `shared/compile/validate.ts`: 40+ rules tagged native / arch / heuristic / studio; node + field + phase focus |
| §6 Phases and loss timeline | ✓ | Phase strip, per-phase overrides (loss weights, LR, patch, batch), transition contract (scratch / fine-tune / continue) with resume-strictness checks, loss timeline SVG in the wizard, one TOML per phase + run script |
| §7 Wizard | ✓ | Six steps, suggestions with provenance, live impact rail, simulated dataset scan, template / import entry, review with TOML tabs |
| §8 Graph | ✓ | Foblex canvas, palette (nodes / presets / templates), inspector (Essentials / Advanced / Native, search, modified-only, phase mode), context menus, minimap, auto-layout |
| §9 TOML view | ✓ | Source map hover/click → node field, download per phase + `.ps1`/`.sh`, pin-and-diff, native extras, transactional import |
| §10 Preflight / Launch | ✓ (static) | Problems popover gates Launch; backend re-validates and rejects blockers. Environment probes are mocked |
| §11 Run cockpit | ✓ (simulated) | SSE-backed live charts (losses, validation, LR, grad norm, system), validation scrubber with swipe / blink / zoom, events, logs, frozen recipe + TOML, stop / force / resume / fork, chaos demo |
| §12 Compare, Library | ✓ | KPIs, metric overlays aligned by updates, recipe + TOML diff, image compare; templates, recipes, node presets, mock models and datasets |
| §14 Architecture | ✓ | `shared/` compiler used by browser and server; Express simulator (`server/API.md`) |
| Deferred | — | Real `neosr-train` launch, real scanning, TensorBoard ingestion, remote workers, ramps, export node |
