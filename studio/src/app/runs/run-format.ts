/** Formatting and derivation helpers shared by the runs list, the run screen and Compare. */
import type { Recipe } from '@shared/recipe.model';
import { METRIC_INFO, type RunPhaseInfo, type RunStatus, type ScalarPoint } from '@shared/run.model';

/**
 * Categorical series hues (derived from the token set). Order chosen with the dataviz palette
 * validator so adjacent slots stay apart for normal vision; legends always name each series.
 */
export const SERIES_PALETTE: readonly string[] = ['#8fbdff', '#e3a86d', '#6cc5a0', '#c9a6ff', '#c9b458', '#f2b8c6'];

export type StatusTone = 'running' | 'ok' | 'warn' | 'danger' | 'neutral';

export interface StatusView {
  readonly label: string;
  readonly icon: string;
  readonly tone: StatusTone;
}

export function statusView(status: RunStatus): StatusView {
  switch (status) {
    case 'queued':
      return { label: 'Queued', icon: 'clock', tone: 'neutral' };
    case 'starting':
      return { label: 'Starting…', icon: 'play', tone: 'running' };
    case 'running':
      return { label: 'Running', icon: 'activity', tone: 'running' };
    case 'stopping':
    case 'saving':
      return { label: 'Saving…', icon: 'save', tone: 'running' };
    case 'stopped':
      return { label: 'Stopped', icon: 'pause', tone: 'neutral' };
    case 'disconnected':
      return { label: 'Disconnected', icon: 'alert-triangle', tone: 'warn' };
    case 'completed':
      return { label: 'Completed', icon: 'check', tone: 'ok' };
    case 'failed':
      return { label: 'Failed', icon: 'alert-triangle', tone: 'danger' };
  }
}

export function isResumable(status: RunStatus): boolean {
  return status === 'stopped' || status === 'disconnected';
}

/** Coarse remaining time for an estimate: `11 h`, `1 h 20 min`, `14 min`, `<1 min`. */
export function approxDuration(sec: number): string {
  if (!Number.isFinite(sec) || sec <= 0) {
    return '—';
  }
  if (sec >= 36_000) {
    return `${Math.round(sec / 3600)} h`;
  }
  const minutes = Math.round(sec / 60);
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  if (h > 0) {
    return m > 0 ? `${h} h ${m} min` : `${h} h`;
  }
  return m > 0 ? `${m} min` : '<1 min';
}

export function formatInt(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}

/** `142.3k`, `1.2M`, `950`. */
export function formatCompact(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1e6) {
    return `${trim(n / 1e6)}M`;
  }
  if (abs >= 1e3) {
    return `${trim(n / 1e3)}k`;
  }
  return trim(n);
}

function trim(n: number): string {
  return Number(n.toFixed(1)).toString();
}

/** Axis/tooltip value: scientific for tiny learning rates, 4 significant digits otherwise. */
export function formatValue(v: number): string {
  if (!Number.isFinite(v)) {
    return '—';
  }
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) {
    return v.toExponential(2);
  }
  return Number(v.toPrecision(4)).toString();
}

export function timeAgo(iso: string | undefined, now = Date.now()): string {
  if (!iso) {
    return '';
  }
  const sec = Math.max(0, (now - Date.parse(iso)) / 1000);
  return agoText(sec);
}

export function agoText(sec: number): string {
  if (sec < 60) {
    return 'just now';
  }
  if (sec < 3600) {
    return `${Math.floor(sec / 60)} min ago`;
  }
  if (sec < 86_400) {
    return `${Math.floor(sec / 3600)} h ago`;
  }
  return `${Math.floor(sec / 86_400)} d ago`;
}

export function metricLabel(key: string): string {
  const info = METRIC_INFO[key];
  return info ? info.label : key.toUpperCase();
}

export function metricArrow(key: string): string {
  return METRIC_INFO[key]?.better === 'lower' ? '↓' : '↑';
}

export function metricBetter(key: string): 'higher' | 'lower' {
  return METRIC_INFO[key]?.better ?? 'higher';
}

export function formatMetric(key: string, value: number): string {
  return key === 'psnr' ? value.toFixed(2) : value.toFixed(4);
}

/** `30.34 dB`, `0.0950`. */
export function metricText(key: string, value: number): string {
  const unit = METRIC_INFO[key]?.unit;
  return unit ? `${formatMetric(key, value)} ${unit}` : formatMetric(key, value);
}

/**
 * The metric a run is judged by: DISTS when it trains a GAN (pixel fidelity drops by design),
 * PSNR otherwise; falls back to the first available metric in METRIC_INFO order.
 */
export function primaryMetric(available: readonly string[], hasGan: boolean): string | null {
  const preferred = hasGan ? 'dists' : 'psnr';
  if (available.includes(preferred)) {
    return preferred;
  }
  const order = Object.keys(METRIC_INFO);
  return order.find((k) => available.includes(k)) ?? available[0] ?? null;
}

const WARNING_LABELS: Readonly<Record<string, string>> = {
  NONFINITE: 'skipped update',
  VAL_COST: 'validation cost',
  GAN_START: 'discriminator enabled',
};

export function warningLabel(code: string): string {
  return WARNING_LABELS[code] ?? code.toLowerCase().replace(/_/g, ' ');
}

export interface PhaseClock {
  readonly index: number;
  readonly phase: RunPhaseInfo | undefined;
  /** NeoSR's own update counter inside the phase run. */
  readonly native: number;
  readonly nativeTotal: number;
}

/**
 * Native counter of the phase containing `iter`: `finetune` restarts at 0, `continue` resumes the
 * previous phase's `.state`, so its counter keeps counting from the last restart.
 */
export function phaseClock(phases: readonly RunPhaseInfo[], recipe: Recipe | undefined, iter: number): PhaseClock {
  if (!phases.length) {
    return { index: 0, phase: undefined, native: iter, nativeTotal: iter };
  }
  let index = phases.findIndex((p) => iter >= p.start && iter < p.end);
  if (index < 0) {
    index = iter <= 0 ? 0 : phases.length - 1;
  }
  let base = index;
  while (base > 0 && initOf(phases[base], recipe, base) === 'continue') {
    base--;
  }
  const start = phases[base].start;
  return { index, phase: phases[index], native: iter - start, nativeTotal: phases[index].end - start };
}

function initOf(phase: RunPhaseInfo, recipe: Recipe | undefined, index: number): string {
  const byId = recipe?.phases.find((p) => p.id === phase.id);
  return (byId ?? recipe?.phases[index])?.init ?? 'finetune';
}

/** `phase 2/3 GAN fine-tune`, or just the name for a single-phase run. */
export function phaseLabel(phases: readonly RunPhaseInfo[], index: number): string {
  const phase = phases[Math.min(index, phases.length - 1)];
  if (!phase) {
    return '';
  }
  return phases.length > 1 ? `phase ${Math.min(index, phases.length - 1) + 1}/${phases.length} ${phase.name}` : phase.name;
}

/** Validation metric series `val/<dataset>/<metric>` grouped by metric, metrics in METRIC_INFO order. */
export function valSeriesByMetric(series: Readonly<Record<string, readonly ScalarPoint[]>>): { metric: string; series: Record<string, readonly ScalarPoint[]> }[] {
  const groups = new Map<string, Record<string, readonly ScalarPoint[]>>();
  for (const [key, points] of Object.entries(series)) {
    const m = /^val\/(.+)\/([^/]+)$/.exec(key);
    if (!m) {
      continue;
    }
    const [, dataset, metric] = m as unknown as [string, string, string];
    const group = groups.get(metric) ?? {};
    group[dataset] = points;
    groups.set(metric, group);
  }
  const order = Object.keys(METRIC_INFO);
  const rank = (k: string) => (order.includes(k) ? order.indexOf(k) : order.length);
  return [...groups.entries()].sort((a, b) => rank(a[0]) - rank(b[0]) || a[0].localeCompare(b[0])).map(([metric, s]) => ({ metric, series: s }));
}

/** Platform-correct relative path; built in TS because templates must not hold backslashes. */
export function joinPath(platform: 'windows' | 'linux', ...parts: string[]): string {
  return parts.join(platform === 'windows' ? '\\' : '/');
}

/** Generator weights of the last phase a run reached, and the export command for them. */
export function weightsOf(run: { phases: readonly RunPhaseInfo[]; phaseIndex: number; arch: string; scale: number }, platform: 'windows' | 'linux'): { path: string; convert: string } | null {
  const phase = run.phases[Math.min(run.phaseIndex, run.phases.length - 1)];
  if (!phase) {
    return null;
  }
  const path = joinPath(platform, 'experiments', phase.runName, 'models', 'net_g_latest.pth');
  return { path, convert: `neosr-convert --input ${path} -net ${run.arch} -s ${run.scale} -onnx` };
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function downloadText(fileName: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/toml' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
