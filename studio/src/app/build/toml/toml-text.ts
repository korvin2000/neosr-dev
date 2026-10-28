/** Pure text helpers for the TOML view: line diff, native-extras paths, run scripts. */
import type { CompiledPhase } from '@shared/compile/compile';

export interface DiffRow {
  readonly kind: 'add' | 'del' | 'chg';
  readonly text: string;
  /** Previous text of a changed line. */
  readonly old?: string;
  /** 1-based line number in the current text (add/chg) or the old text (del). */
  readonly line: number;
}

/** LCS line diff; a run of removals followed by additions is paired up as changes. */
export function lineDiff(before: readonly string[], after: readonly string[]): DiffRow[] {
  let head = 0;
  while (head < before.length && head < after.length && before[head] === after[head]) {
    head++;
  }
  let tail = 0;
  while (tail < before.length - head && tail < after.length - head && before[before.length - 1 - tail] === after[after.length - 1 - tail]) {
    tail++;
  }
  const a = before.slice(head, before.length - tail);
  const b = after.slice(head, after.length - tail);
  const n = a.length;
  const m = b.length;
  const lcs = new Int32Array((n + 1) * (m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i * (m + 1) + j] = a[i] === b[j] ? lcs[(i + 1) * (m + 1) + j + 1] + 1 : Math.max(lcs[(i + 1) * (m + 1) + j], lcs[i * (m + 1) + j + 1]);
    }
  }
  const rows: DiffRow[] = [];
  let dels: DiffRow[] = [];
  let adds: DiffRow[] = [];
  const flush = () => {
    const paired = Math.min(dels.length, adds.length);
    for (let k = 0; k < paired; k++) {
      rows.push({ kind: 'chg', text: adds[k].text, old: dels[k].text, line: adds[k].line });
    }
    rows.push(...dels.slice(paired), ...adds.slice(paired));
    dels = [];
    adds = [];
  };
  let i = 0;
  let j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) {
      flush();
      i++;
      j++;
    } else if (j < m && (i >= n || lcs[i * (m + 1) + j + 1] >= lcs[(i + 1) * (m + 1) + j])) {
      adds.push({ kind: 'add', text: b[j], line: head + j + 1 });
      j++;
    } else {
      dels.push({ kind: 'del', text: a[i], line: head + i + 1 });
      i++;
    }
  }
  flush();
  return rows;
}

export interface ExtraEntry {
  readonly path: readonly string[];
  readonly value: unknown;
}

function isTable(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Leaves of the extras tree; arrays and scalars are leaves. */
export function flattenExtras(tree: Readonly<Record<string, unknown>>, prefix: readonly string[] = []): ExtraEntry[] {
  return Object.entries(tree).flatMap(([key, value]) => (isTable(value) && Object.keys(value).length > 0 ? flattenExtras(value, [...prefix, key]) : [{ path: [...prefix, key], value }]));
}

/** A copy of the tree without `path`; tables left empty are removed too. */
export function removeExtra(tree: Readonly<Record<string, unknown>>, path: readonly string[]): Record<string, unknown> {
  const [key, ...rest] = path;
  const next: Record<string, unknown> = { ...tree };
  const child = next[key];
  if (rest.length === 0 || !isTable(child)) {
    delete next[key];
    return next;
  }
  const pruned = removeExtra(child, rest);
  if (Object.keys(pruned).length === 0) {
    delete next[key];
  } else {
    next[key] = pruned;
  }
  return next;
}

export interface ScriptOptions {
  readonly platform: 'windows' | 'linux';
  readonly recipeName: string;
  readonly neosrRoot: string;
  readonly optionsDir: string;
}

/** A launch script that runs every phase in order and stops at the first failure. */
export function runScript(phases: readonly CompiledPhase[], opts: ScriptOptions): { fileName: string; text: string } {
  const base = `run_${opts.recipeName.trim().replace(/[^A-Za-z0-9_.-]+/g, '_') || 'recipe'}`;
  const header = (c: string) => [
    `${c} NeoSR Studio run script: ${opts.recipeName} (${phases.length} phase${phases.length === 1 ? '' : 's'}), generated ${new Date().toISOString()}.`,
    `${c} Put ${phases.map((p) => p.fileName).join(', ')} into ${opts.optionsDir} first; the commands run from the NeoSR root.`,
  ];
  const phaseComment = (c: string, p: CompiledPhase) => `${c} Phase ${p.index + 1}/${phases.length}: ${p.phase.name} (${p.phase.init}, native ${p.nativeStart}..${p.nativeEnd})`;
  if (opts.platform === 'windows') {
    const quote = (s: string) => `'${s.replace(/'/g, "''")}'`;
    const lines = [...header('#'), "$ErrorActionPreference = 'Stop'", `Set-Location -LiteralPath ${quote(opts.neosrRoot)}`];
    for (const p of phases) {
      lines.push('', phaseComment('#', p), p.command, 'if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }');
    }
    return { fileName: `${base}.ps1`, text: lines.join('\r\n') + '\r\n' };
  }
  const quote = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;
  const lines = ['#!/usr/bin/env bash', ...header('#'), 'set -euo pipefail', `cd ${quote(opts.neosrRoot)}`];
  for (const p of phases) {
    lines.push('', phaseComment('#', p), p.command);
  }
  return { fileName: `${base}.sh`, text: lines.join('\n') + '\n' };
}
