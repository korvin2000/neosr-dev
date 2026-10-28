/**
 * Read helpers over a Recipe: values with defaults, phase overrides, and the nodes wired into the
 * Training Plan. Pure functions; the store, the compiler and the validator all use them.
 */
import { NODE_TYPES, nodeSpec } from '../catalog/node-catalog';
import { defaultValues, type NodeSpec } from '../catalog/types';
import type { LossRow, Recipe, RecipeEdge, RecipeNode } from '../recipe.model';

export interface ValDatasetRow {
  readonly id: string;
  readonly name: string;
  readonly type: 'paired' | 'single';
  readonly dataroot_gt: string;
  readonly dataroot_lq: string;
  /** Original TOML table (`datasets.val_2`), kept on import so preserved extras stay with their set. */
  readonly section?: string;
}

/** TOML table of the i-th validation set. */
export function valSection(row: ValDatasetRow, index: number): string {
  return row.section ?? (index === 0 ? 'datasets.val' : `datasets.val_${index}`);
}

/** Field values with catalog defaults filled in. */
export function nodeValues(node: RecipeNode, spec: NodeSpec = nodeSpec(node.type)): Record<string, unknown> {
  return { ...defaultValues(spec), ...node.values };
}

/** Field values for one phase: defaults, then shared values, then that phase's overrides. */
export function phaseNodeValues(node: RecipeNode, phaseId: string, spec: NodeSpec = nodeSpec(node.type)): Record<string, unknown> {
  return { ...nodeValues(node, spec), ...(node.phaseValues[phaseId] ?? {}) };
}

export function findNodes(recipe: Recipe, type: string): RecipeNode[] {
  return recipe.nodes.filter((n) => n.type === type);
}

export function findNode(recipe: Recipe, id: string): RecipeNode | undefined {
  return recipe.nodes.find((n) => n.id === id);
}

export function incomingEdge(recipe: Recipe, nodeId: string, portKey: string): RecipeEdge | undefined {
  return recipe.edges.find((e) => e.targetNode === nodeId && e.targetPort === portKey);
}

/** The enabled node feeding `portKey` of `nodeId`, if any. */
export function upstreamNode(recipe: Recipe, nodeId: string, portKey: string): RecipeNode | undefined {
  const edge = incomingEdge(recipe, nodeId, portKey);
  if (!edge) {
    return undefined;
  }
  const source = findNode(recipe, edge.sourceNode);
  return source && !source.disabled ? source : undefined;
}

/** Everything wired into the Training Plan, resolved once. */
export interface PlanContext {
  readonly plan?: RecipeNode;
  readonly data?: RecipeNode;
  readonly degradation?: RecipeNode;
  readonly val?: RecipeNode;
  readonly model?: RecipeNode;
  readonly dmodel?: RecipeNode;
  readonly init?: RecipeNode;
  readonly losses?: RecipeNode;
  readonly optimG?: RecipeNode;
  readonly optimD?: RecipeNode;
  readonly schedule?: RecipeNode;
  readonly runtime?: RecipeNode;
}

export function planContext(recipe: Recipe): PlanContext {
  const plan = findNodes(recipe, NODE_TYPES.plan).find((n) => !n.disabled);
  if (!plan) {
    return {};
  }
  const up = (port: string) => upstreamNode(recipe, plan.id, port);
  const data = up('data');
  return {
    plan,
    data,
    degradation: data ? upstreamNode(recipe, data.id, 'degradation') : undefined,
    val: up('val'),
    model: up('model'),
    dmodel: up('discriminator'),
    init: up('init'),
    losses: up('losses'),
    optimG: up('optimG'),
    optimD: up('optimD'),
    schedule: up('schedule'),
    runtime: up('runtime'),
  };
}

export function lossRows(node: RecipeNode | undefined): LossRow[] {
  const rows = node ? (nodeValues(node)['losses'] as LossRow[] | undefined) : undefined;
  return Array.isArray(rows) ? rows : [];
}

/**
 * Rows active in a phase with their effective weight. Zero-weight generator terms are dropped
 * (NeoSR drops them too), but a zero-weight gan_loss stays: NeoSR keeps its discriminator term.
 */
export function activeLossRows(rows: readonly LossRow[], phaseId: string): { row: LossRow; weight: number }[] {
  return rows
    .map((row) => {
      const s = row.schedule[phaseId];
      const active = s ? s.active : row.enabled;
      const weight = s ? s.weight : row.weight;
      return { row, weight, active };
    })
    .filter((r) => r.active && (r.weight > 0 || r.row.type === 'gan_loss'))
    .map(({ row, weight }) => ({ row, weight }));
}

/** Effective `[train.<slot>_opt]` slot of a row. */
export function lossSlot(row: LossRow, defaultSlot: string): string {
  return row.params['slot'] ? String(row.params['slot']) : defaultSlot;
}

/** Update number encoded in a `.state` file name, e.g. `training_states/50000.state` -> 50000. */
export function stateIter(path: string): number | null {
  const m = /(\d+)\.state$/.exec(path.trim());
  return m ? Number(m[1]) : null;
}

export function phaseHasGan(rows: readonly LossRow[], phaseId: string): boolean {
  return activeLossRows(rows, phaseId).some((r) => r.row.type === 'gan_loss');
}

export function valDatasets(node: RecipeNode | undefined): ValDatasetRow[] {
  const rows = node ? (nodeValues(node)['datasets'] as ValDatasetRow[] | undefined) : undefined;
  return Array.isArray(rows) ? rows : [];
}

export function scaleOf(model: RecipeNode | undefined): number {
  const raw = model ? nodeValues(model)['scale'] : undefined;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : 4;
}

/** Nodes that are enabled but do not reach the Training Plan through any edge chain. */
export function unreachableNodes(recipe: Recipe): RecipeNode[] {
  const plan = findNodes(recipe, NODE_TYPES.plan).find((n) => !n.disabled);
  if (!plan) {
    return recipe.nodes.filter((n) => !n.disabled);
  }
  const reached = new Set<string>([plan.id]);
  const queue = [plan.id];
  while (queue.length) {
    const id = queue.pop() as string;
    for (const e of recipe.edges) {
      if (e.targetNode === id && !reached.has(e.sourceNode)) {
        reached.add(e.sourceNode);
        queue.push(e.sourceNode);
      }
    }
  }
  return recipe.nodes.filter((n) => !n.disabled && !reached.has(n.id));
}

export function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 24) || 'phase';
}
