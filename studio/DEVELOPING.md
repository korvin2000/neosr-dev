# NeoSR Studio — developer guide

A web UI prototype for the NeoSR super-resolution trainer: Angular 22 (standalone components, signals, zoneless, OnPush) + a Node/Express mock backend. Everything the UI produces is a plain NeoSR TOML; the backend simulates runs.

Read `design/NeoSR_Studio_UI_Concept.md` (one level up, in `../design/`) for the UX concept and `design/neosr_config_surface.md` for the NeoSR facts. Section numbers below refer to the concept.

## Commands

| What | Command |
|---|---|
| Install | `npm install` (already done) |
| Web dev server (proxy `/api` → 4310) | `npm run web` → http://localhost:4300 |
| Backend | `npm run api` |
| Both | `npm start` |
| Typecheck (fast, no templates) | `npx tsc -p tsconfig.app.json --noEmit` |
| Full build incl. templates (**use a private output path when others may build concurrently**) | `npx ng build --configuration development --output-path dist/check-<yourname>` |
| Compiler smoke test | `npx tsx tools/smoke-compile.ts` |

Template errors (NGxxxx) only show up in `ng build`, so always finish with a build. If a build fails with an `.angular/cache` / ENOENT error, retry once.

## Layout

```text
shared/                      TypeScript shared by browser and server (import as `@shared/...` in src/, relative in server/)
  recipe.model.ts            Recipe, RecipeNode, RecipeEdge, Phase, LossRow, newId, phaseRanges, totalIters, semanticHash
  recipe-builder.ts          applyDraft (wizard/template → nodes), autoLayout, lossRowsFor(objective), newLossRow, newPhase
  templates.ts               TEMPLATE_INFOS, buildTemplate(id)
  run.model.ts               RunSummary/RunDetail/RunEvent/RunSnapshot, METRIC_INFO, EnvironmentInfo
  catalog/types.ts           NodeSpec, FieldSpec (widget kinds, level essentials|advanced|native, showWhen, phaseAware, suggestions), PortType, FAMILY_ACCENT
  catalog/node-catalog.ts    NODE_CATALOG, NODE_TYPES, nodeSpec(id), OBJECTIVES, objectiveLosses
  catalog/archs.ts …         ARCHS (93), DISCRIMINATORS, LOSSES, OPTIMIZERS, SCHEDULERS, METRIC_TYPES, AUGMENTATIONS, DEGRADATION_FIELDS
  compile/compile.ts         compileRecipe(recipe, {platform}) → { phases: CompiledPhase[] (lines, text, command, runName, nativeStart/End, hasGan), commands, problems, context }
  compile/validate.ts        the Problems rules (severity blocker|warning|info, source native|arch|heuristic|studio, nodeId/fieldKey/phaseId)
  compile/graph.ts           nodeValues(node), phaseNodeValues(node, phaseId), planContext(recipe), lossRows, activeLossRows, phaseHasGan, valDatasets, scaleOf, unreachableNodes
  compile/import-toml.ts     importToml(text, emptyRecipe(name)) → { recipe, warnings, preserved }
  compile/toml-writer.ts     TomlLine { text, kind, origin?: { nodeId, fieldKey } } (source map)
server/                      Express backend (see server/API.md) — do not edit unless you own it
src/app/
  core/recipe-store.ts       RecipeStore (the one editable recipe) — see below
  core/commands.ts           every edit is a Command; see list below
  core/api.service.ts        ApiService (typed HTTP), RecipeSummary, ModelFile, DatasetInfo, ScanReport, LaunchRejected
  core/run-store.ts          RunStore (list, refresh, active) and LiveRun (SSE-backed signals for one run)
  core/settings.service.ts   SettingsService.settings() { platform, neosrRoot, experimentsRoot, optionsDir, simSpeed, showNativeFields, inspectorDocked }
  core/environment.service.ts EnvironmentService.info()
  core/preset.service.ts     PresetService: node presets (built-in + localStorage): presets(), byType(type), save(name, node), remove(id)
  ui/icon.ts                 <app-icon name size weight> — names listed in ICON_NAMES (add new ones there)
  ui/field-widget.ts         <app-field-widget [field] [value] [compact] [overridden] [readout] (valueChange) (patch)> renders any FieldSpec except `custom`
  ui/editors/loss-editor.ts  <app-loss-editor [rows] [phases] [activePhaseId] [hasDiscriminator] [compact] (rowsChange) (applyObjective)>
  ui/editors/val-datasets-editor.ts <app-val-datasets-editor [rows] (rowsChange)>
  ui/editors/arch-params-editor.ts  <app-arch-params-editor [params] [value] (valueChange)>  (params from ARCH_BY_TYPE.get(type).params or DISCRIMINATORS)
  ui/toml-code.ts            <app-toml-code [lines] [highlightNodeId] [wrap] (originHover) (originClick)>
  ui/toast.service.ts        ToastService.show(kind, text, action?)
  app.ts/.html/.scss         shell: top bar (recipe name, Wizard/Graph/TOML switch), left nav, toasts
  build/build-page.*         hosts the three build views; status bar (derived numbers, Problems popover), undo/redo, Save, Launch
  build/wizard/  build/graph/  build/toml/   runs/  compare/  library/  home/  settings/   ← feature folders (stubs exist)
src/styles/_tokens.scss      CSS custom properties (colours, geometry, motion), `.accent-<family>` and `.port-<PortType>` classes
src/styles.scss              reset, `.btn .btn--primary/--ghost/--sm/--icon/--danger`, `.input .select .textarea`, `.badge .badge--ok/--warn/--danger/--running/--accent/--native/--orchestrated/--estimate`, `.chip .chip--active`, `.card`, `.row`, `.spacer`, `.muted`, `.faint`, `.mono`, `.kbd`, Foblex connection styles
```

## The store (read this twice)

```ts
const store = inject(RecipeStore);
store.recipe()                 // Recipe signal — the only source of truth
store.dispatch(cmd)            // apply a Command (undoable). Never mutate the recipe.
store.setValue(nodeId, key, value)            // shortcut: commands.setValues
store.setPhaseValue(nodeId, phaseId, key, v)  // per-phase override (undefined removes it)
store.undo() / store.redo() / canUndo() / canRedo() / undoLabel()
store.load(recipe, markSaved?) / newBlank() / loadTemplate(id) / markSaved()
store.selection() / select(ids) / selectOne(id) / selectedNode() / selectedEdge()
store.activePhaseId()          // null = shared values; a phase id = show that phase's effective values
store.compiled()               // CompileResult (recomputed on every change — cheap)
store.problems() / problemCounts() / problemsByNode() / launchable()
store.context()                // PlanContext: { plan, data, degradation, val, model, dmodel, init, losses, optimG, optimD, schedule, runtime } (nodes wired into the plan)
store.derived()                // { scale, patch, gtCrop, batch, accumulate, effectiveBatch, totalIter, phases, arch, hasGan, vramClass }
store.dirty() / hash()
```

Commands (`import * as commands from '../core/commands'`): `addNode(type, position, values?)`, `addNodeWired(type, position, values?)` (also wires it into the Training Plan if the port is free), `removeNodes(ids)`, `moveNodes([{id, position}])`, `setValues(nodeId, patch, label?)`, `setPhaseValues(nodeId, phaseId, patch)`, `setNodeFlags(nodeId, {collapsed|disabled|title})`, `connect({sourceNode, sourcePort, targetNode, targetPort})`, `removeEdges(ids)`, `disconnectNode(id)`, `duplicateNode(id)`, `setName`, `setPhases(phases)`, `updatePhase(id, patch)`, `addPhase(phase, afterId?)`, `removePhase(id)`, `setExtras`, `applyDraftCommand(draft, label)`, `autoLayout()`, `replaceRecipe(recipe, label)`, `clearCanvas()`.

Reading values: `nodeValues(node)` fills catalog defaults; `phaseNodeValues(node, phaseId)` adds that phase's overrides. A field is "overridden" in a phase when `node.phaseValues[phaseId]?.[key] !== undefined`. Readout widgets (`widget.kind === 'readout'`) are computed by the host: `gt_crop` = patch × scale, `effective_batch` = batch × accumulate, `total_iter` = totalIters(recipe.phases).

Visibility: `fieldVisible(field, values)` / `portVisible(port, values)` from `@shared/catalog/types`. Levels: show `essentials` by default, `advanced` folded, `native` only when `SettingsService.settings().showNativeFields` or an explicit "Native" tab.

## Conventions

- Standalone components, `ChangeDetectionStrategy.OnPush`, `input()`/`output()`/`computed()`/`signal()`; no `ngModel`, no NgRx, no Material. Use `@if/@for/@switch` control flow. Templates in `.html` files for anything over ~30 lines, SCSS in `.scss` next to the component.
- Styling: only the tokens and primitives in `src/styles`; no hard-coded colours except via `color-mix` on tokens. Dark theme only. Density: 13 px base, 12 px in dense tables/nodes. Everything keyboard-reachable; buttons have `title`/`aria-label`.
- Family accent for a node: `class="accent-{{ FAMILY_ACCENT[spec.family] }}"` then use `var(--accent-node)`, `var(--accent-node-header)`, `var(--accent-node-dim)`. Port colour: `class="port-{{ port.type }}"` then `var(--port-colour)`.
- Never write backslash-containing literals inside Angular templates (Windows paths): build them in TS (`['C:', 'datasets'].join('\\')`) and bind.
- Labels and copy: short, factual, NeoSR terms (`patch_size`, updates, `net_g_latest.pth`). Suggestions are heuristics — say where they come from (template / wiki / studio).
- Support badges: native vs orchestrated vs estimate (`.badge--native` etc.) wherever the UI does something NeoSR itself does not.
- No `any` (use `$any()` only in templates for event targets), no unused code, no console noise. Comments only for non-obvious reasons.
- Do not edit files outside your feature folder except where your brief says so; do not touch `shared/`, `core/`, `ui/` — if you need a change there, implement a local workaround and report it.

---

# v2 — calm UI (read this before touching any screen)

The design is `../design/NeoSR_Studio_UI_Concept_v2.md` (final; §1 rules, §2 screens, §3 disclosure mechanics). v1 screens are being replaced; v1 files under `build/wizard/` are dead (routes redirect to `/setup`) and may be deleted by whoever owns Setup.

## New primitives (use these, do not reinvent)

| Component / service | Use |
|---|---|
| `ui/page-header.ts` `<app-page-header heading subtitle back>` with slots `[title]`, `[status]`, `[actions]`, `[tabs]` | Every page's first row. One primary action; the rest in `<app-menu>`. |
| `ui/menu.ts` `<app-menu [items] (picked) icon align>` | The ⋯ overflow. `MenuItem { id, label, icon?, danger?, disabled?, divider?, hint? }` |
| `ui/tooltip.ts` `[appTooltip]="text"` | Names and short definitions only (hover 500 ms, focus, long-press). Never `title=` for help. |
| `ui/help-popover.ts` `<app-help [field]>` or `[help] [tomlKey]` | The ⓘ toggletip: explanation, typical values, NeoSR key. |
| `ui/disclosure.ts` `<app-disclosure title summary badge [open] (openChange)>` | Collapsible section with a one-line summary. Remember state with `DisclosureState` (`core/disclosure-state.service.ts`: `isOpen(key)`, `set(key, open)`). |
| `ui/drawer.ts` `<app-drawer [open] title width (closed)>` | Right overlay; full-screen sheet on phones (`UiModeService.isPhone()`). Dock it yourself at `isWide()` if the screen reserves space. |
| `ui/field-row.ts` `<app-field-row [field] [value] [changed] [baseline] [problems] [suggestion] [showKey] [phase] (valueChange) (patch) (reset) (overrideToggle)>` | One setting with its changed bar, ↺, inline problem and one contextual suggestion. |
| `ui/section-editor.ts` `<app-section-editor [section]>` or `[nodeId]` plus `[focusField]` | **The one inspector.** Renders a section from `shared/sections.ts` (Basics + remembered "More options", custom editors, GAN switch, phase chips when phases > 1) or a single node. Writes through the store. |
| `ui/editors/phases-editor.ts` | Phase list (name, updates, transition). Used inside the Training plan section. |
| `shared/sections.ts` | `SECTIONS` (data · model · losses · plan · optimizer · validation · runtime), `sectionOf(id)`, `partNode`, `partFields`, `recipeChanges(recipe)` (changed-from-preset list), `isChanged`, `baselineValue`, `sectionProblems`, `findSetting(query, recipe)`, `sectionForNodeType`. |
| `shared/recipe.model.ts` `snapshotPreset(recipe, label)` / `meta.presetSnapshot`, `meta.presetLabel` | Templates snapshot their values; "changed" compares against it. Call `snapshotPreset` after applying a preset in Setup. |
| `core/ui-mode.service.ts` `UiModeService.isPhone() / isNarrow() / isWide()` | Breakpoints 720 / 1400 px. |
| `core/settings.service.ts` `showKeys`, `openSections` | Preferences replacing the Expert switch. |

## Rules that are now mandatory
- No permanent hint paragraphs, no suggestion chip rows, no uppercase micro-labels, no mono TOML keys next to headings (keys appear in ⓘ and when `showKeys`). Badges only for status and the global "Simulated" marker. "≈" marks estimates.
- Default state of a screen shows the summary tier only; details open on click; "More options" is remembered per section.
- Every hover/right-click/drag gesture has a visible twin (⋯ menus, ⓘ, + buttons, pickers).
- Validate on commit; blockers inline under the field; the primary button reads "Launch · n to fix" and jumps to the first one (`BuildPage.focusProblem` navigates to `/build/list?section=&node=&field=` or `/build/graph?node=&field=`).
- Phone (≤ 720 px): single column, drawers become sheets, inputs 16 px, targets ≥ 44 px, nothing hidden behind hover. Test at 390 × 844.
- Keep the v1 visual tokens (dark, chaiNNer greys) but use colour for status only; node family colour only as a 3 px stripe / port colour.

## Running several dev servers
`ng serve` instances share `.angular/cache/**/vite/deps`; two at once keep invalidating each other ("504 Outdated Optimize Dep"). Run one dev server at a time, or give the second one its own cache: `NG_PERSISTENT_BUILD_CACHE=0` is not enough — use `npx ng serve --port 4302` only after the other has finished its first build, and prefer `ng build --output-path dist/check-<name>` for verification.
