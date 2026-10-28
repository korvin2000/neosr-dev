/** Recipe and TOML diffs for Compare. */
import { NODE_BY_ID } from '@shared/catalog/node-catalog';
import { nodeValues } from '@shared/compile/graph';
import type { Recipe, RecipeNode } from '@shared/recipe.model';

export interface ConfigDiffRow {
  readonly group: string;
  readonly key: string;
  readonly title: string;
  /** One display value per recipe, in the order given. */
  readonly values: readonly string[];
}

export interface DiffLine {
  readonly kind: 'same' | 'add' | 'del' | 'change';
  readonly a: string | null;
  readonly b: string | null;
  readonly aNo: number | null;
  readonly bNo: number | null;
}

const ABSENT = '— (no node)';

/** Differing phase settings and node values, nodes matched by type (and order within a type). */
export function configDiff(recipes: readonly Recipe[]): ConfigDiffRow[] {
  const rows: ConfigDiffRow[] = [];
  const phaseCount = Math.max(...recipes.map((r) => r.phases.length));
  for (let i = 0; i < phaseCount; i++) {
    const values = recipes.map((r) => {
      const p = r.phases[i];
      return p ? `${p.name} · ${p.iters.toLocaleString('en-US')} · ${p.init}` : '— (no phase)';
    });
    pushIfDiffers(rows, { group: 'Phases', key: `phase ${i + 1}`, title: 'name · updates · transition', values });
  }

  const types: string[] = [];
  for (const r of recipes) {
    for (const n of r.nodes) {
      if (!types.includes(n.type)) {
        types.push(n.type);
      }
    }
  }
  for (const type of types) {
    const spec = NODE_BY_ID.get(type);
    const perRecipe = recipes.map((r) => r.nodes.filter((n) => n.type === type));
    const count = Math.max(...perRecipe.map((n) => n.length));
    for (let i = 0; i < count; i++) {
      const group = `${spec?.label ?? type}${count > 1 ? ` #${i + 1}` : ''}`;
      const nodes = perRecipe.map((list) => list[i]);
      const maps = nodes.map((n) => (n ? effectiveValues(n) : null));
      if (nodes.some((n) => !n)) {
        pushIfDiffers(rows, { group, key: '(node)', title: 'node present in the recipe', values: nodes.map((n) => (n ? (n.disabled ? 'disabled' : 'present') : ABSENT)) });
      } else {
        pushIfDiffers(rows, { group, key: '(disabled)', title: 'node disabled', values: nodes.map((n) => String(!!n?.disabled)) });
      }
      const keys: string[] = [];
      for (const m of maps) {
        for (const k of Object.keys(m ?? {})) {
          if (!keys.includes(k)) {
            keys.push(k);
          }
        }
      }
      // A missing node is already one row; its keys only differ where two present nodes disagree.
      for (const key of keys) {
        const label = spec?.fields.find((f) => f.key === key)?.label ?? key;
        const canon = maps.map((m) => (m ? canonical(m[key]) : ABSENT));
        if (new Set(canon.filter((v) => v !== ABSENT)).size > 1) {
          const shown = maps.map((m) => (m ? formatValue(m[key]) : ABSENT));
          // Display forms can coincide (same loss types, different parameters): fall back to the canonical text.
          rows.push({ group, key, title: label, values: new Set(shown).size > 1 ? shown : canon });
        }
      }
      const overrides = nodes.map((n) => (n ? canonical(n.phaseValues) : ABSENT));
      if (new Set(overrides.filter((v) => v !== ABSENT)).size > 1) {
        rows.push({ group, key: '(phase overrides)', title: 'per-phase overrides', values: overrides.map((v) => (v === '{}' ? '—' : v)) });
      }
    }
  }
  return rows;
}

/** Stable JSON without generated ids: the comparison key. */
function canonical(value: unknown): string {
  return JSON.stringify(sortKeys(value), (k, v: unknown) => (k === 'id' ? undefined : v)) ?? '';
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

function effectiveValues(node: RecipeNode): Record<string, unknown> {
  const spec = NODE_BY_ID.get(node.type);
  return spec ? nodeValues(node, spec) : { ...node.values };
}

function pushIfDiffers(rows: ConfigDiffRow[], row: ConfigDiffRow): void {
  if (new Set(row.values).size > 1) {
    rows.push(row);
  }
}

/** Display form; ids are dropped so equal rows with different generated ids compare equal. */
export function formatValue(value: unknown): string {
  if (value === undefined || value === null || value === '') {
    return '—';
  }
  if (typeof value !== 'object') {
    return String(value);
  }
  if (Array.isArray(value)) {
    if (value.every((v) => typeof v !== 'object' || v === null)) {
      return `[${value.join(', ')}]`;
    }
    if (value.every((v) => typeof v === 'object' && v !== null && 'type' in v)) {
      return value
        .map((v: Record<string, unknown>) => `${String(v['type'])}${v['weight'] !== undefined ? ` ×${String(v['weight'])}` : ''}${v['enabled'] === false ? ' (off)' : ''}`)
        .join(', ');
    }
  }
  return JSON.stringify(value, (k, v: unknown) => (k === 'id' ? undefined : v));
}

/** Line diff by longest common subsequence; adjacent delete/add runs pair up as changes. */
export function diffLines(a: readonly string[], b: readonly string[]): DiffLine[] {
  const n = a.length;
  const m = b.length;
  const dp = new Uint32Array((n + 1) * (m + 1));
  const at = (i: number, j: number) => dp[i * (m + 1) + j] ?? 0;
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i * (m + 1) + j] = a[i] === b[j] ? at(i + 1, j + 1) + 1 : Math.max(at(i + 1, j), at(i, j + 1));
    }
  }
  const ops: { kind: 'same' | 'add' | 'del'; i: number; j: number }[] = [];
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) {
      ops.push({ kind: 'same', i: i++, j: j++ });
    } else if (j < m && (i >= n || at(i, j + 1) >= at(i + 1, j))) {
      ops.push({ kind: 'add', i, j: j++ });
    } else {
      ops.push({ kind: 'del', i: i++, j });
    }
  }
  const out: DiffLine[] = [];
  for (let k = 0; k < ops.length; ) {
    const op = ops[k] as (typeof ops)[number];
    if (op.kind === 'same') {
      out.push({ kind: 'same', a: a[op.i] ?? '', b: b[op.j] ?? '', aNo: op.i + 1, bNo: op.j + 1 });
      k++;
      continue;
    }
    const dels: number[] = [];
    const adds: number[] = [];
    while (k < ops.length && ops[k]?.kind !== 'same') {
      const o = ops[k] as (typeof ops)[number];
      if (o.kind === 'del') {
        dels.push(o.i);
      } else {
        adds.push(o.j);
      }
      k++;
    }
    const pairs = Math.max(dels.length, adds.length);
    for (let p = 0; p < pairs; p++) {
      const di = dels[p];
      const aj = adds[p];
      out.push({
        kind: di !== undefined && aj !== undefined ? 'change' : di !== undefined ? 'del' : 'add',
        a: di !== undefined ? (a[di] ?? '') : null,
        b: aj !== undefined ? (b[aj] ?? '') : null,
        aNo: di !== undefined ? di + 1 : null,
        bNo: aj !== undefined ? aj + 1 : null,
      });
    }
  }
  return out;
}
