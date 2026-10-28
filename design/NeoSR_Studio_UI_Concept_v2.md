# NeoSR Studio — Concept v2: ask less, show what the decision needs

**Status:** final for implementation. Supersedes the *screen designs* of v1; v1's recipe model, node catalog, compiler, validator and phase semantics (incl. §15a) stay unchanged. Reviewed by two independent models (Opus 5.5 design critique, Codex gpt-6-astra UX review); their converging recommendations are folded in, disagreements resolved in §6.

**Why v2.** v1 (`screenshots/studio-v1/`) is complete but reads like a cockpit: three columns, permanent hints, suggestion chips under every field, eight charts, every node expanded with inline inputs. Calm does not come from folding the same amount away; it comes from **asking fewer questions** (the data and the preset already answer most) and from showing, at each step, only what that decision needs.

---

## 1. Rules

| Rule | Concretely |
|---|---|
| Ask less | A preset is a complete, compatible training configuration (architecture, losses, optimizer, duration, patch/batch, precision, checkpoint and validation cadence, the full GAN/OTF bundle when applicable). The user supplies data folders and, optionally, weights. Scale is detected from the data; the run name is generated. |
| Three tiers, local | *Summary* (always) → *Basics* (≤ 5 fields, on open) → *More options* (sticky per section, remembered). No global Expert mode: a "Show NeoSR keys" preference, a **Find setting** search (⌘K / search box) that matches labels and TOML keys and opens the section, and prototype/debug controls under Settings › Prototype. |
| Help is pulled | Tooltips (hover 500 ms, focus) only for names and short definitions. A ⓘ toggletip (click/tap) for the explanation, typical values, NeoSR key and source. Inline text only for a consequence at the decision point ("Without validation you only see losses"). |
| Changes, not values | Each loaded preset is snapshotted. A changed field shows a 2 px accent bar and ↺ ("Preset: 64"); collapsed sections say "2 changed"; Review lists the changes. A suggested value appears only when context makes the preset wrong ("Use 32 — HAT-S window"), never as a chip row. |
| Issues where you act | A blocker: red icon + one sentence under the field; the primary button becomes "Launch · 1 to fix" and jumps to it. Warnings: inline when the field is visible, else counted in the readiness pill. Info: pill popover only. Mechanical fixes get a button ("Add discriminator + optimizer"). Validate on commit, not per keystroke. |
| Visible twins | Every hover/right-click/drag gesture has a visible equivalent: ⋯ on selected nodes and rows, ⓘ for help, connection pickers in the inspector, a + button for adding nodes. No double-click-to-expand. |
| Execution semantics stay visible | "Simulated" chip in the top bar; "2 sequential runs" and the hand-off policy on Review; the native clock on demand; never hidden behind a mode. |
| One anatomy | Top bar: title · status · primary action · ⋯. Content, single column that gains side panels on wide screens (drawers overlay below 1400 px, dock above). Nav: labelled rail (72 px) on desktop, 4-tab bottom bar on phones. |
| Visual calm | Body 15 px (16 on phone, inputs ≥ 16 px), weights 400/600 only, no uppercase micro-labels, no mono keys next to headings. Neutral greys; one accent for primary/selection/focus; red/amber/green/blue for status only, always with icon + text. Family colour only as a 3 px stripe on cards and on ports. Badges only for status and "Simulated"; "≈" marks estimates. |

---

## 2. Screens

### 2.1 Start
Always: three outcome cards — **Clean** (pixel losses, best PSNR), **Balanced** (template losses without GAN), **Detailed** (full GAN recipe, "may invent texture") — each with one meta line ("Small model · ≈ 2 h on this GPU"); **Continue "name"** row when a draft exists; **Running** row when a run is active; links *All templates*, *Import TOML*. One click away: environment popover from the GPU chip; template list. Settings via the gear.

### 2.2 Setup (two screens, progress "Data · 1 of 2")
**Data** — source switch "I have low-res images / Generate low-res on the fly"; High-res folder; Low-res folder (hidden for OTF); Validation: *paired folder* / *input images only* / *skip* (consequence shown); the check runs automatically when a folder is set and its result sits under the folders: "2,888 pairs · ×4 detected · 2 issues ›" (popover: unmatched, too small, overlap, example). Scale: "×4 detected" with a change popover; a segmented ×1 ×2 ×4 (more…) control for OTF or when detection fails. Next.
**Review** — editable title; one sentence with tokens: *Train **SPAN** at ×4 on 2,888 pairs with **clean losses** for **150k updates**, **from scratch**, checking **10 images every 1k**; checkpoints every 1k.* Each token opens a popover with 2–4 choices and "More ›" (→ Build list, that section). A line "Fits this GPU ≈ 64 px patches · batch 8 — suggested, not measured". "Changed from preset (n) ›" and "n to check ›" only when non-empty. **Launch** (or "Launch · 1 to fix"); *Open full recipe*; ⋯ (Show TOML, Copy command, Save, Save as template). For a curriculum: "2 sequential runs, phase 2 fine-tunes from phase 1 weights".

### 2.3 Build (one screen, switch **List · Graph · TOML**, one inspector)
Sections, each with a summary line and "n changed", several may be open (remembered): **Data** · **Model** (architecture picker: search + size/speed facets; discriminator only when a GAN row exists; "Start from") · **Losses** (table; phase columns only when phases > 1; GAN as a switch that adds/removes D + Optimizer D + gan row together) · **Training plan** (duration; phases list; timeline on demand) · **Optimizer** · **Validation & saving** · **Runtime**. Basics ≤ 5 fields; "More options" for the rest (advanced, then "Rarely needed" = native). Find setting box at the top. Launch in the top bar with the readiness pill ("ready" / "2 issues ›").
- **Graph**: summary cards (title, 2–3 values, 3 px family stripe, ports, status icon only when something is wrong). Click → the same section editor in a drawer (docked ≥ 1400 px). Selection highlights the node's connections. **+** → sheet (Nodes · Presets · Templates). ⋯ on the selected node (rename, duplicate, disable, disconnect, delete, save as preset). Inputs also editable as **connection pickers** in the drawer. Phase strip only when phases > 1. Display options (⋯): minimap, zoom %, auto-layout, docked palette. Phone: an **outline list** in flow order instead of the canvas; "View graph" still available read-only.
- **TOML**: code (phase tabs when > 1); toolbar Copy · Download (current / all + run script) · Copy command · Issues n (when > 0) · ⋯ (Import, Diff vs saved, Unmapped settings n). Clicking code selects text; section headers carry a muted "Edit ← Node" link that opens the drawer over the code. Issue markers in the gutter.

### 2.4 Run
Header: ← name · status chip ("Running · phase 2/3 GAN fine-tune" / "1 warning") · **one state-dependent primary action** (Running → *Stop & save*; stopped with state → *Resume*; completed → *Weights…*; failed → *Logs*) · ⋯ (Force stop with consequence, Fork to recipe, Download TOML; Chaos only from Settings › Prototype). Progress bar split into phase segments with checkpoint ticks; ETA as secondary text.
**Overview**: three KPIs — **Progress** (updates done / target, ETA), **Validation quality** (preset metric: PSNR for Clean, DISTS ↓ for Detailed; dataset, at update, ↑/↓ trend; "visual only" when no paired set), **Recovery point** (last resumable state, age; "weights only" when applicable) — two charts (validation metric with best point; generator loss components, log, phase bands, gaps kept) — one image: latest validation vs best, tap to blink — one aggregated warning row only when there is one ("skipped update ×24 ›").
Tabs: **Overview · Images · Diagnostics · Activity** (events + logs with filters). Config (commands, TOML, artifacts) in ⋯.

### 2.5 Runs · Compare · Library · Settings
- **Runs**: status · name · phase + progress · last activity · ETA; row ⋯ always visible; *Select* → checkboxes → *Compare (2)*.
- **Compare**: context line first ("val · PSNR ↑ · aligned by updates"), metric overlay, "Config differences (n)", Images tab.
- **Library**: search + facets (Scale · Look · Speed) with active-filter count; tabs Templates · My recipes · Models · Datasets; cards: title, one line, *Use*; details in ⋯.
- **Settings**: Environment (platform, paths) · Preferences (show NeoSR keys, open sections by default, theme) · **Prototype** (simulation speed, chaos, reset, hash) clearly labelled.

---

## 3. Disclosure mechanics

| Pattern | Used for | Desktop | Keyboard | Touch |
|---|---|---|---|---|
| Tooltip | icon names, truncated text, preset value of a changed field | hover 500 ms, stays while hovered | focus; Esc | — (icons get labels) |
| Toggletip ⓘ | field help: what, typical, NeoSR key, source | click | Enter; Esc | tap |
| Popover | one decision: review token, issues, readiness, suggestion | click; outside closes | arrows; Esc returns focus | bottom sheet |
| Expander | More options, sections | click | Enter | tap |
| Drawer | inspector beside canvas/TOML | select; docked ≥ 1400 px | Enter on node; Esc | full sheet |
| Sheet | + nodes, import, launch preflight | — | focus trap | swipe down + Close |
| Modal | irreversible only (force stop, discard) | | | |

---

## 4. Phone (≤ 720 px)
Bottom tabs Start · Build · Runs · Library. First-class: Start → Setup → Run Overview, Runs list, Stop/Resume. Build opens the **List** (sections) by default; Graph is the outline list; TOML read-only with Copy. Drawers become sheets; charts stack; inputs 16 px; targets ≥ 44 px; forms reflow at 320 px.

---

## 5. Density after v2 (default state)

| Screen | v1 | v2 |
|---|---|---|
| Start | 3 cards + 3 panels + env card | 3 cards + 1–2 rows + 2 links |
| Setup step 1 | 14 controls, rail, impact panel, hint boxes | 5 controls, 1 result line |
| Build | palette + 10 expanded nodes (~60 inputs) + inspector + strip + minimap + 5-stat bar | 7 summary rows (List) or summary cards (Graph); inspector on selection |
| Run | 7 KPIs, 8 charts, gallery, events, logs, config | 3 KPIs, 2 charts, 1 image, 1 optional warning row |
| TOML | code + 4 side panels + list | code + 4 toolbar buttons |

---

## 6. Review disagreements and decisions
- *Goal screen* (Codex) vs *Data first* (Opus): the Start card **is** the goal; Setup starts with Data; look, model, length and initialization are tokens on Review.
- *Scale inferred* (Opus) vs *asked* (Codex): inferred from the paired check, shown as "×4 detected" with a change control; asked for OTF.
- *Accordion one-open* (draft) vs *free* (both): several sections may be open; state remembered.
- *Expert switch* (draft): dropped, replaced by local More options + Find setting + the keys preference.
- *Inline node expansion* (draft): dropped.
- *Compare in nav* (draft): opened from Runs selection; nav is Start · Build · Runs · Library.

---

## 7. Implementation status (`studio/`, screenshots in `screenshots/studio-v2/`)

| Area | Built |
|---|---|
| Rules §1 | Preset snapshot + "changed from preset" (`shared/sections.ts recipeChanges`), inline blockers + "Launch · n to fix", ⓘ toggletips and 500 ms tooltips, visible twins (⋯ menus, + button, connection pickers), "Simulated" marker, find-setting search, "Show NeoSR keys" preference |
| Start, Setup Data → Review | ✓ incl. auto data check, scale detection, token popovers, GPU-fit line, change list |
| Build List · Graph · TOML with one inspector | ✓ (`ui/section-editor`), summary cards, docked/overlay/sheet drawer, phase strip on demand, architecture picker sheet, TOML "Edit" links |
| Run overview, Images, Diagnostics, Activity; Runs; Compare | ✓ live training flow (`runs/live-flow`: Run · Data · Generator · Losses · Discriminator · Optimizer · Checkpoint · Validation · GPU nodes with animated edges, a beat per update, rates, loss bars, sparklines, memory gauge; vertical list on a phone), validation chart, hold-to-blink image, aggregated warning row, state-dependent primary action |
| Phone | ✓ bottom tab bar, single column, sheets, outline list instead of the canvas, 16 px inputs |
| Visual QA | ✓ pixel captures of every screen at 1440×900 (`screenshots/qa/`) and 390×844 (`screenshots/qa-phone/`); fixed: loss table rows forced to table layout, TOML "Edit" inline after the header, build subtitle never repeats the title, flow sub-lines wrap instead of truncating, activity log breaks at spaces |
| Deferred | Light theme; real folder browser; metric parameters editable in the Validation node (preserved as extras instead); TOML issue links opening the drawer in place; scale-vs-data warning (needs a real data probe) |
