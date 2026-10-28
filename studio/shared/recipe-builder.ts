/**
 * Builds and updates recipes from a high-level draft. The Wizard, the templates and the TOML
 * importer all go through `applyDraft`, which is idempotent: each wizard-owned node is upserted by
 * type, only the provided keys are written, edges to the Training Plan are (re)created, nodes set to
 * `null` are removed, and new nodes are laid out in columns.
 */
import { LOSSES } from './catalog/losses';
import { NODE_TYPES, nodeSpec, objectiveLosses, type ObjectiveId } from './catalog/node-catalog';
import { defaultValues } from './catalog/types';
import { planContext } from './compile/graph';
import { newId, newNode, type LossRow, type Phase, type Point, type Recipe, type RecipeEdge, type RecipeNode } from './recipe.model';

type Values = Record<string, unknown>;

export interface RecipeDraft {
  readonly dataset?: Values;
  /** `null` removes the node. */
  readonly degradation?: Values | null;
  readonly validation?: Values | null;
  readonly generator?: Values;
  readonly discriminator?: Values | null;
  readonly init?: Values;
  readonly losses?: readonly LossRow[];
  readonly optimG?: Values;
  readonly optimD?: Values | null;
  readonly schedule?: Values | null;
  readonly runtime?: Values;
  readonly plan?: Values;
  readonly phases?: readonly Phase[];
}

/** Which Training Plan port each draft node feeds. */
const PLAN_PORT: Record<string, string> = {
  [NODE_TYPES.trainDataset]: 'data',
  [NODE_TYPES.validation]: 'val',
  [NODE_TYPES.generator]: 'model',
  [NODE_TYPES.discriminator]: 'discriminator',
  [NODE_TYPES.initialization]: 'init',
  [NODE_TYPES.losses]: 'losses',
  [NODE_TYPES.optimizerG]: 'optimG',
  [NODE_TYPES.optimizerD]: 'optimD',
  [NODE_TYPES.schedule]: 'schedule',
  [NODE_TYPES.runtime]: 'runtime',
};

/** Column layout, left to right; rows stack within a column. */
const COLUMNS: readonly (readonly string[])[] = [
  [NODE_TYPES.degradation],
  [NODE_TYPES.trainDataset, NODE_TYPES.validation, NODE_TYPES.initialization, NODE_TYPES.schedule],
  [NODE_TYPES.generator, NODE_TYPES.discriminator, NODE_TYPES.losses, NODE_TYPES.optimizerG, NODE_TYPES.optimizerD, NODE_TYPES.runtime],
  [NODE_TYPES.plan],
];
const COLUMN_X = [40, 380, 720, 1080];
const NODE_GAP = 28;

/** Approximate card height per type, enough for a tidy auto-layout without measuring the DOM. */
const APPROX_HEIGHT: Record<string, number> = {
  [NODE_TYPES.degradation]: 330,
  [NODE_TYPES.trainDataset]: 320,
  [NODE_TYPES.validation]: 230,
  [NODE_TYPES.initialization]: 150,
  [NODE_TYPES.schedule]: 170,
  [NODE_TYPES.generator]: 150,
  [NODE_TYPES.discriminator]: 110,
  [NODE_TYPES.losses]: 240,
  [NODE_TYPES.optimizerG]: 150,
  [NODE_TYPES.optimizerD]: 150,
  [NODE_TYPES.runtime]: 150,
  [NODE_TYPES.plan]: 420,
};

function firstOfType(recipe: Recipe, type: string): RecipeNode | undefined {
  return recipe.nodes.find((n) => n.type === type);
}

export function applyDraft(base: Recipe, draft: RecipeDraft): Recipe {
  let nodes = [...base.nodes];
  let edges = [...base.edges];
  const created = new Set<string>();

  // The node a draft edits is the one wired into the plan; a disconnected duplicate never wins.
  const wiredOf = (type: string): RecipeNode | undefined => {
    const ctx = planContext({ ...base, nodes, edges });
    return [ctx.plan, ctx.data, ctx.degradation, ctx.val, ctx.model, ctx.dmodel, ctx.init, ctx.losses, ctx.optimG, ctx.optimD, ctx.schedule, ctx.runtime].find((n) => n?.type === type);
  };
  const upsert = (type: string, values: Values | undefined): RecipeNode | undefined => {
    if (values === undefined) {
      return wiredOf(type) ?? nodes.find((n) => n.type === type);
    }
    const existing = wiredOf(type) ?? nodes.find((n) => n.type === type);
    if (existing) {
      const updated: RecipeNode = { ...existing, values: { ...existing.values, ...values } };
      nodes = nodes.map((n) => (n.id === existing.id ? updated : n));
      return updated;
    }
    const node = newNode(type, { x: 0, y: 0 }, { ...defaultValues(nodeSpec(type)), ...values });
    nodes.push(node);
    created.add(node.id);
    return node;
  };
  const remove = (type: string) => {
    const victims = nodes.filter((n) => n.type === type).map((n) => n.id);
    nodes = nodes.filter((n) => !victims.includes(n.id));
    edges = edges.filter((e) => !victims.includes(e.sourceNode) && !victims.includes(e.targetNode));
  };

  // The plan first, so every other node has something to connect to.
  const plan = upsert(NODE_TYPES.plan, draft.plan ?? (firstOfType(base, NODE_TYPES.plan) ? undefined : {})) as RecipeNode;

  const nullable: [keyof RecipeDraft, string][] = [
    ['degradation', NODE_TYPES.degradation],
    ['validation', NODE_TYPES.validation],
    ['discriminator', NODE_TYPES.discriminator],
    ['optimD', NODE_TYPES.optimizerD],
    ['schedule', NODE_TYPES.schedule],
  ];
  for (const [key, type] of nullable) {
    if (draft[key] === null) {
      remove(type);
    }
  }

  const data = upsert(NODE_TYPES.trainDataset, draft.dataset);
  const degradation = draft.degradation === null ? undefined : upsert(NODE_TYPES.degradation, draft.degradation ?? undefined);
  upsert(NODE_TYPES.validation, draft.validation ?? undefined);
  upsert(NODE_TYPES.generator, draft.generator);
  upsert(NODE_TYPES.discriminator, draft.discriminator ?? undefined);
  upsert(NODE_TYPES.initialization, draft.init);
  upsert(NODE_TYPES.losses, draft.losses ? { losses: draft.losses } : undefined);
  upsert(NODE_TYPES.optimizerG, draft.optimG);
  upsert(NODE_TYPES.optimizerD, draft.optimD ?? undefined);
  upsert(NODE_TYPES.schedule, draft.schedule ?? undefined);
  upsert(NODE_TYPES.runtime, draft.runtime);

  // Wire every singleton into the plan (once) and the degradations into the dataset.
  const connect = (source: RecipeNode, sourcePort: string, target: RecipeNode, targetPort: string) => {
    if (edges.some((e) => e.targetNode === target.id && e.targetPort === targetPort)) {
      return;
    }
    const edge: RecipeEdge = { id: newId('e'), sourceNode: source.id, sourcePort, targetNode: target.id, targetPort };
    edges.push(edge);
  };
  for (const node of nodes) {
    const port = PLAN_PORT[node.type];
    if (port && node.id !== plan.id) {
      const out = nodeSpec(node.type).outputs[0];
      if (out) {
        connect(node, out.key, plan, port);
      }
    }
  }
  if (data && degradation) {
    connect(degradation, 'degradation', data, 'degradation');
  }

  let recipe: Recipe = {
    ...base,
    nodes,
    edges,
    phases: draft.phases ?? base.phases,
    meta: { ...base.meta, updatedAt: new Date().toISOString() },
  };
  if (created.size > 0) {
    recipe = autoLayout(recipe, base.nodes.length === 0 ? undefined : created);
  }
  return recipe;
}

/**
 * Column layout. With `only`, just those nodes are placed (into free rows of their column) so a
 * wizard step that adds a node does not shuffle what the user already arranged.
 */
export function autoLayout(recipe: Recipe, only?: ReadonlySet<string>): Recipe {
  const positions = new Map<string, Point>();
  COLUMNS.forEach((types, col) => {
    let y = 40;
    const column = recipe.nodes.filter((n) => types.includes(n.type)).sort((a, b) => types.indexOf(a.type) - types.indexOf(b.type));
    if (only) {
      // Keep existing nodes where they are; start new ones below the lowest existing one.
      const existing = column.filter((n) => !only.has(n.id));
      for (const n of existing) {
        y = Math.max(y, n.position.y + (APPROX_HEIGHT[n.type] ?? 160) + NODE_GAP);
      }
    }
    for (const node of column) {
      if (only && !only.has(node.id)) {
        continue;
      }
      positions.set(node.id, { x: COLUMN_X[col], y });
      y += (APPROX_HEIGHT[node.type] ?? 160) + NODE_GAP;
    }
  });
  // Centre the plan vertically against the tallest column.
  const plan = recipe.nodes.find((n) => n.type === NODE_TYPES.plan);
  if (plan && (!only || only.has(plan.id))) {
    const maxY = Math.max(40, ...recipe.nodes.filter((n) => n.id !== plan.id).map((n) => (positions.get(n.id)?.y ?? n.position.y) + (APPROX_HEIGHT[n.type] ?? 160)));
    positions.set(plan.id, { x: COLUMN_X[3], y: Math.max(40, (maxY - (APPROX_HEIGHT[NODE_TYPES.plan] ?? 400)) / 2) });
  }
  return {
    ...recipe,
    nodes: recipe.nodes.map((n) => (positions.has(n.id) ? { ...n, position: positions.get(n.id)! } : n)),
  };
}

/** Loss rows for an objective preset, with the template parameters filled in. */
export function lossRowsFor(objective: ObjectiveId): LossRow[] {
  return objectiveLosses(objective).map(({ type, weight }) => {
    const info = LOSSES.find((l) => l.type === type);
    const params: Record<string, unknown> = {};
    if (type === 'fdl_loss') {
      params['model'] = 'dinov2';
    }
    if (type === 'gan_loss') {
      params['gan_type'] = 'bce';
      params['wavelet_init'] = 80000;
    }
    void info;
    return { id: newId('l'), type, weight, enabled: true, params, schedule: {} };
  });
}

/**
 * A new loss row of a given type with its reference default weight. When another row already uses
 * the type's default slot, the new row gets a numbered slot so both can compile.
 */
export function newLossRow(type: string, existing: readonly LossRow[] = []): LossRow {
  const info = LOSSES.find((l) => l.type === type);
  const defaultSlot = info?.slot ?? type;
  const used = new Set(existing.map((r) => (r.params['slot'] ? String(r.params['slot']) : LOSSES.find((l) => l.type === r.type)?.slot ?? r.type)));
  const params: Record<string, unknown> = {};
  if (used.has(defaultSlot)) {
    let n = 2;
    while (used.has(`${defaultSlot}_${n}`)) {
      n++;
    }
    params['slot'] = `${defaultSlot}_${n}`;
  }
  return { id: newId('l'), type, weight: info?.templateWeight ?? info?.defaultWeight ?? 1, enabled: true, params, schedule: {} };
}

export function newPhase(name: string, iters: number, init: Phase['init'] = 'finetune', note = ''): Phase {
  return { id: newId('p'), name, iters, init, note };
}
