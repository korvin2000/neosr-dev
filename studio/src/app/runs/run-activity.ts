import { ChangeDetectionStrategy, Component, computed, input, model, signal } from '@angular/core';
import type { RunEvent, RunPhaseInfo } from '@shared/run.model';
import { Icon } from '../ui/icon';
import { formatCompact, formatMetric, metricArrow, metricLabel, warningLabel } from './run-format';

export type ActivityFilter = 'all' | 'warning' | 'checkpoint' | 'validation' | 'phase' | 'log';

export interface LogLine {
  readonly iter: number;
  readonly level: 'info' | 'warn' | 'error';
  readonly line: string;
  readonly at: string;
}

interface Row {
  readonly kind: 'event' | 'log';
  readonly filter: ActivityFilter | 'status';
  /** Warn/error log lines also show under Warnings. */
  readonly warning: boolean;
  readonly icon: string;
  readonly tone: string;
  readonly iter: string;
  readonly time: string;
  readonly at: number;
  readonly title: string;
  readonly detail: string;
  readonly search: string;
}

const FILTERS: readonly { id: ActivityFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'warning', label: 'Warnings' },
  { id: 'checkpoint', label: 'Checkpoints' },
  { id: 'validation', label: 'Validation' },
  { id: 'phase', label: 'Phases' },
  { id: 'log', label: 'Log' },
];

const PAGE = 300;

/** Events and log lines merged into one timeline, newest first, with type filters and search. */
@Component({
  selector: 'app-run-activity',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  templateUrl: './run-activity.html',
  styleUrl: './run-activity.scss',
})
export class RunActivity {
  readonly events = input.required<readonly RunEvent[]>();
  readonly logs = input.required<readonly LogLine[]>();
  readonly phases = input<readonly RunPhaseInfo[]>([]);
  readonly filter = model<ActivityFilter>('all');

  protected readonly filters = FILTERS;
  protected readonly query = signal('');
  protected readonly limit = signal(PAGE);
  /** Rows are cached per source object so a new line does not rebuild (or re-render) the rest. */
  private readonly cache = new WeakMap<object, Row | null>();

  private readonly all = computed<Row[]>(() => {
    const events = this.events();
    const logs = this.logs();
    const phases = this.phases();
    const out: Row[] = [];
    let e = events.length - 1;
    let l = logs.length - 1;
    while (e >= 0 || l >= 0) {
      const ev = e >= 0 ? this.row(events[e] as RunEvent, () => describe(events[e] as RunEvent, phases)) : null;
      const lg = l >= 0 ? this.row(logs[l] as LogLine, () => logRow(logs[l] as LogLine)) : null;
      if (e >= 0 && !ev) {
        e--;
        continue;
      }
      if (ev && (!lg || ev.at >= lg.at)) {
        out.push(ev);
        e--;
      } else if (lg) {
        out.push(lg);
        l--;
      }
    }
    return out;
  });

  protected readonly counts = computed(() => {
    const counts: Record<string, number> = { all: this.all().length };
    for (const r of this.all()) {
      counts[r.filter] = (counts[r.filter] ?? 0) + 1;
      if (r.warning && r.filter !== 'warning') {
        counts['warning'] = (counts['warning'] ?? 0) + 1;
      }
    }
    return counts;
  });

  private readonly matching = computed(() => {
    const filter = this.filter();
    const q = this.query().trim().toLowerCase();
    return this.all().filter((r) => (filter === 'all' || r.filter === filter || (filter === 'warning' && r.warning)) && (!q || r.search.includes(q)));
  });

  protected readonly rows = computed(() => this.matching().slice(0, this.limit()));
  protected readonly hidden = computed(() => this.matching().length - this.rows().length);

  protected setFilter(id: ActivityFilter): void {
    this.filter.set(id);
    this.limit.set(PAGE);
  }

  protected more(): void {
    this.limit.update((n) => n + PAGE * 2);
  }

  private row(source: object, build: () => Row | null): Row | null {
    let row = this.cache.get(source);
    if (row === undefined) {
      row = build();
      this.cache.set(source, row);
    }
    return row;
  }
}

function base(iter: number, at: string, text: string): Pick<Row, 'iter' | 'time' | 'at' | 'search'> {
  return { iter: formatCompact(iter), time: new Date(at).toLocaleTimeString('en-US', { hour12: false }), at: Date.parse(at), search: text.toLowerCase() };
}

function logRow(l: LogLine): Row {
  return { kind: 'log', filter: 'log', warning: l.level !== 'info', icon: 'terminal', tone: l.level === 'info' ? 'log' : l.level === 'warn' ? 'warn' : 'danger', title: '', detail: l.line, ...base(l.iter, l.at, l.line) };
}

function describe(e: RunEvent, phases: readonly RunPhaseInfo[]): Row | null {
  const make = (filter: Row['filter'], icon: string, tone: string, title: string, detail: string, extra = ''): Row => ({ kind: 'event', filter, warning: filter === 'warning', icon, tone, title, detail, ...base(e.iter, e.at, `${title} ${detail} ${extra}`) });
  switch (e.type) {
    case 'phase':
      return make('phase', 'layers', 'accent', `Phase ${e.phaseIndex + 1}/${phases.length || e.phaseIndex + 1} · ${e.name}`, e.runName);
    case 'checkpoint':
      return make('checkpoint', 'save', 'muted', 'Checkpoint', `${e.files.join(', ')} · ${e.sizeMb} MB${e.state ? '' : ' · weights only'}`);
    case 'validation': {
      const metrics = Object.entries(e.metrics).map(([k, v]) => `${metricLabel(k)} ${metricArrow(k)} ${formatMetric(k, v)}`);
      return make('validation', 'target', 'muted', `Validation · ${e.dataset}`, metrics.length ? metrics.join(' · ') : `${e.images.length} images, visual only`);
    }
    case 'warning':
      return make('warning', 'alert-triangle', 'warn', warningLabel(e.code), e.message, e.code);
    case 'status':
      return make('status', 'activity', e.status === 'failed' ? 'danger' : e.status === 'completed' ? 'ok' : 'running', `Status · ${e.status}`, e.message ?? '');
    default:
      return null;
  }
}
