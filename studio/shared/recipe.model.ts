/**
 * The canonical Recipe: one document that the Wizard, the Graph and the TOML view all edit.
 *
 * Positions, collapsed state and titles are presentation; everything else is semantics and takes
 * part in `semanticHash()`.
 */

export interface Point {
  readonly x: number;
  readonly y: number;
}

export interface RecipeNode {
  readonly id: string;
  /** NodeSpec id from the catalog, e.g. `train-dataset`. */
  readonly type: string;
  /** User-given title; empty means "use the spec label". */
  readonly title: string;
  readonly position: Point;
  /** Field values keyed by field key. Missing keys fall back to the spec default. */
  readonly values: Readonly<Record<string, unknown>>;
  /**
   * Per-phase overrides for phase-aware fields: phaseId → fieldKey → value.
   * Only the Loss Recipe, Optimizer and Train Dataset nodes use this.
   */
  readonly phaseValues: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  readonly collapsed: boolean;
  readonly disabled: boolean;
}

export interface RecipeEdge {
  readonly id: string;
  readonly sourceNode: string;
  readonly sourcePort: string;
  readonly targetNode: string;
  readonly targetPort: string;
}

/**
 * The transition contract at a phase boundary.
 * - `scratch`: first phase only; weights come from the Initialization node.
 * - `finetune`: new run initialised from the previous phase's last saved weights
 *   (`net_g_latest.pth`, plus `net_d_latest.pth` when both phases have a discriminator);
 *   optimizers, EMA and warm-up restart. Required whenever the network or optimizer inventory
 *   changes (NeoSR resume is strict).
 * - `continue`: resume the previous phase's last `.state`; native counters are cumulative and the
 *   optimizer/scheduler state is restored, so a per-phase LR override is not effective.
 */
export type PhaseInit = 'scratch' | 'finetune' | 'continue';

export interface Phase {
  readonly id: string;
  readonly name: string;
  /** Duration in optimizer updates; phases are contiguous, starting at 0. */
  readonly iters: number;
  readonly init: PhaseInit;
  /** Free-text note shown in the timeline, e.g. "add GAN". */
  readonly note: string;
}

/** One row of the Loss Recipe node. Stored in `values.losses` of that node. */
export interface LossRow {
  readonly id: string;
  /** Loss type from the catalog, e.g. `mssim_loss`. */
  readonly type: string;
  readonly weight: number;
  readonly enabled: boolean;
  /** Extra loss parameters, e.g. `{ criterion: 'huber' }`. */
  readonly params: Readonly<Record<string, unknown>>;
  /**
   * Per-phase overrides. A phase without an entry uses `enabled` and `weight`.
   * Weights are piecewise-constant: NeoSR has no native loss schedule, so a change of weight is
   * a phase boundary (a new run).
   */
  readonly schedule: Readonly<Record<string, LossPhaseSchedule>>;
}

export interface LossPhaseSchedule {
  readonly active: boolean;
  readonly weight: number;
}

export interface RecipeMeta {
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly templateId?: string;
  readonly tags: readonly string[];
  readonly description: string;
  /** Human name of the preset this recipe started from ("Balanced"), shown on Review. */
  readonly presetLabel?: string;
  /**
   * Node values at the moment the preset was applied, keyed by node type. "Changed from preset"
   * compares against this; recipes without a snapshot compare against the catalog defaults.
   */
  readonly presetSnapshot?: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  /** Phases at the moment the preset was applied (duration and count are user decisions too). */
  readonly presetPhases?: readonly Phase[];
  /** Per-phase overrides at the moment the preset was applied, keyed by node type. */
  readonly presetOverrides?: Readonly<Record<string, Readonly<Record<string, Readonly<Record<string, unknown>>>>>>;
}

export interface Recipe {
  readonly format: 'neosr-studio-recipe';
  readonly version: 1;
  readonly id: string;
  readonly name: string;
  readonly nodes: readonly RecipeNode[];
  readonly edges: readonly RecipeEdge[];
  readonly phases: readonly Phase[];
  /** TOML content the UI does not model, preserved verbatim and re-emitted. */
  readonly nativeExtras: Readonly<Record<string, unknown>>;
  readonly meta: RecipeMeta;
}

export function newId(prefix = 'n'): string {
  const rnd =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10);
  return `${prefix}_${rnd}`;
}

export function emptyRecipe(name = 'untitled'): Recipe {
  const now = new Date().toISOString();
  return {
    format: 'neosr-studio-recipe',
    version: 1,
    id: newId('r'),
    name,
    nodes: [],
    edges: [],
    phases: [{ id: newId('p'), name: 'Train', iters: 100_000, init: 'scratch', note: '' }],
    nativeExtras: {},
    meta: { createdAt: now, updatedAt: now, tags: [], description: '' },
  };
}

export function newNode(type: string, position: Point, values: Record<string, unknown> = {}): RecipeNode {
  return { id: newId(), type, title: '', position, values, phaseValues: {}, collapsed: false, disabled: false };
}

/** Phase start/end in absolute iterations. */
export function phaseRanges(phases: readonly Phase[]): { phase: Phase; start: number; end: number }[] {
  let start = 0;
  return phases.map((phase) => {
    const range = { phase, start, end: start + phase.iters };
    start += phase.iters;
    return range;
  });
}

export function totalIters(phases: readonly Phase[]): number {
  return phases.reduce((sum, p) => sum + p.iters, 0);
}

/** Stable JSON of the semantic parts only; cheap FNV-1a hash for display and change detection. */
export function semanticHash(recipe: Recipe): string {
  const semantic = {
    nodes: [...recipe.nodes]
      .map((n) => ({ id: n.id, type: n.type, values: sortKeys(n.values), phaseValues: sortKeys(n.phaseValues), disabled: n.disabled }))
      .sort((a, b) => a.id.localeCompare(b.id)),
    edges: [...recipe.edges].map((e) => [e.sourceNode, e.sourcePort, e.targetNode, e.targetPort]).sort(),
    phases: recipe.phases,
    extras: sortKeys(recipe.nativeExtras),
  };
  const text = JSON.stringify(semantic);
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortKeys);
  }
  if (value && typeof value === 'object') {
    return Object.keys(value as Record<string, unknown>)
      .sort()
      .reduce<Record<string, unknown>>((acc, key) => {
        acc[key] = sortKeys((value as Record<string, unknown>)[key]);
        return acc;
      }, {});
  }
  return value;
}

/** Capture the current node values as the preset baseline (one node per type is enough: singletons). */
export function snapshotPreset(recipe: Recipe, presetLabel?: string): Recipe {
  const presetSnapshot: Record<string, Record<string, unknown>> = {};
  const presetOverrides: Record<string, Record<string, Record<string, unknown>>> = {};
  for (const node of recipe.nodes) {
    if (!(node.type in presetSnapshot)) {
      presetSnapshot[node.type] = { ...node.values };
      presetOverrides[node.type] = Object.fromEntries(Object.entries(node.phaseValues).map(([id, v]) => [id, { ...v }]));
    }
  }
  return { ...recipe, meta: { ...recipe.meta, presetSnapshot, presetOverrides, presetPhases: recipe.phases, presetLabel: presetLabel ?? recipe.meta.presetLabel } };
}
