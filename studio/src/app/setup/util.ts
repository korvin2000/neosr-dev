import { nodeSpec } from '@shared/catalog/node-catalog';
import type { FieldSpec } from '@shared/catalog/types';
import type { PlanContext } from '@shared/compile/graph';
import type { Problem } from '@shared/compile/problems';
import type { Recipe, RecipeNode } from '@shared/recipe.model';

/** The node of a type the Training Plan uses; a disconnected duplicate never wins. */
export function nodeOf(ctx: PlanContext, recipe: Recipe, type: string): RecipeNode | undefined {
  const wired = [ctx.plan, ctx.data, ctx.degradation, ctx.val, ctx.model, ctx.dmodel, ctx.init, ctx.losses, ctx.optimG, ctx.optimD, ctx.schedule, ctx.runtime].find((n) => n?.type === type);
  return wired ?? recipe.nodes.find((n) => n.type === type && !n.disabled);
}

export function fieldOf(type: string, key: string, overrides: Partial<FieldSpec> = {}): FieldSpec {
  const field = nodeSpec(type).fields.find((f) => f.key === key);
  if (!field) {
    throw new Error(`Unknown field ${type}.${key}`);
  }
  return { ...field, ...overrides };
}

export function problemsOf(problems: readonly Problem[], nodeId: string | undefined, key: string): Problem[] {
  return nodeId ? problems.filter((p) => p.nodeId === nodeId && p.fieldKey === key) : [];
}

/** A value as the "changed" list shows it: short, never a JSON dump. */
export function fmtValue(v: unknown): string {
  if (v === undefined || v === null || v === '') {
    return 'empty';
  }
  if (Array.isArray(v)) {
    return v.every((x) => typeof x !== 'object') ? v.join(', ') : `${v.length} rows`;
  }
  if (typeof v === 'object') {
    return Object.keys(v as object).length ? 'edited' : 'defaults';
  }
  if (typeof v === 'boolean') {
    return v ? 'on' : 'off';
  }
  return String(v);
}
