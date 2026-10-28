import { ChangeDetectionStrategy, Component, ElementRef, computed, inject, input, signal } from '@angular/core';
import { METRIC_INFO, type ScalarPoint } from '@shared/run.model';
import type { LiveRun } from '../core/run-store';
import { UiModeService } from '../core/ui-mode.service';
import { Icon } from '../ui/icon';
import { Tooltip } from '../ui/tooltip';
import { SERIES_PALETTE, metricArrow, metricBetter, metricLabel, valSeriesByMetric } from './run-format';
import { type ChartBand, type ChartMarker, ScalarChart, type XRange } from './scalar-chart';

type SeriesMap = Readonly<Record<string, readonly ScalarPoint[]>>;
type ChartId = 'losses' | 'validation' | 'lr' | 'grad' | 'system';

const CHARTS: readonly { id: ChartId; label: string }[] = [
  { id: 'losses', label: 'Losses' },
  { id: 'validation', label: 'Validation' },
  { id: 'lr', label: 'Learning rate' },
  { id: 'grad', label: 'Gradient norm' },
  { id: 'system', label: 'System' },
];

const LABELS: Readonly<Record<string, string>> = {
  'lr/g': 'G',
  'lr/d': 'D',
  'grad_norm/g': 'G',
  'sys/gpu_util': 'GPU util',
  'sys/data_wait': 'data wait',
  'sys/vram': 'VRAM',
};

/** Ordered subset of the series map; the order fixes each key's colour. */
function pick(series: SeriesMap, keys: readonly string[]): Record<string, readonly ScalarPoint[]> {
  const out: Record<string, readonly ScalarPoint[]> = {};
  for (const key of keys) {
    const points = series[key];
    if (points) {
      out[key] = points;
    }
  }
  return out;
}

function legend(series: SeriesMap, labels: Readonly<Record<string, string>>): { key: string; label: string; color: string }[] {
  return Object.keys(series).map((key, i) => ({ key, label: labels[key] ?? key, color: SERIES_PALETTE[i % SERIES_PALETTE.length] ?? '' }));
}

/** Every chart the run logs, sharing one x range; chips choose which charts are shown. */
@Component({
  selector: 'app-run-diagnostics',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, ScalarChart, Tooltip],
  templateUrl: './run-diagnostics.html',
  styleUrl: './run-diagnostics.scss',
  host: { '(document:click)': 'onDocumentClick($event)', '(document:keydown.escape)': 'optionsOpen.set(false)' },
})
export class RunDiagnostics {
  readonly live = input.required<LiveRun>();

  protected readonly mode = inject(UiModeService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  protected readonly charts = CHARTS;
  protected readonly labels = LABELS;
  protected readonly shown = signal<ReadonlySet<ChartId>>(new Set(CHARTS.map((c) => c.id)));
  protected readonly smoothing = signal(0.6);
  protected readonly lossLog = signal(true);
  protected readonly hiddenLosses = signal<ReadonlySet<string>>(new Set());
  protected readonly xRange = signal<XRange | null>(null);
  protected readonly optionsOpen = signal(false);

  private readonly run = computed(() => this.live().run());
  private readonly series = computed<SeriesMap>(() => this.live().series());

  protected readonly xDomain = computed<XRange>(() => {
    let max = this.run()?.iter ?? 0;
    for (const points of Object.values(this.series())) {
      max = Math.max(max, points.at(-1)?.iter ?? 0);
    }
    return [0, Math.max(max, 1)];
  });

  protected readonly bands = computed<ChartBand[]>(() => {
    const phases = this.run()?.phases ?? [];
    return phases.length > 1 ? phases.map((p) => ({ from: p.start, to: p.end, label: p.name })) : [];
  });

  protected readonly losses = computed(() => {
    const keys = Object.keys(this.series()).filter((k) => k.startsWith('loss/g/') || k === 'loss/d/total');
    const rank = (k: string) => (k === 'loss/g/total' ? 0 : k.startsWith('loss/g/') ? 1 : 2);
    keys.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
    const series = pick(this.series(), keys);
    const labels: Record<string, string> = Object.fromEntries(keys.map((k) => [k, k === 'loss/d/total' ? 'D total' : k === 'loss/g/total' ? 'G total' : k.slice('loss/g/'.length)]));
    return { series, labels, legend: legend(series, labels), visible: keys.filter((k) => !this.hiddenLosses().has(k)) };
  });

  protected readonly validation = computed(() =>
    valSeriesByMetric(this.series()).map((g) => ({
      metric: g.metric,
      series: g.series,
      title: `${metricLabel(g.metric)} ${metricArrow(g.metric)}`,
      unit: METRIC_INFO[g.metric]?.unit ?? '',
      direction: metricBetter(g.metric),
      sparse: Object.values(g.series).every((p) => p.length <= 60),
    })),
  );

  protected readonly lr = computed(() => {
    const series = pick(this.series(), ['lr/g', 'lr/d']);
    return { series, legend: legend(series, LABELS) };
  });

  protected readonly grad = computed(() => {
    const skipped = (this.series()['skipped/g'] ?? []).filter((p) => p.value > 0);
    return {
      series: pick(this.series(), ['grad_norm/g']),
      skipped: skipped.length,
      markers: skipped.map<ChartMarker>((p) => ({ kind: 'warning', iter: p.iter })),
    };
  });

  protected readonly system = computed(() => {
    const pct = pick(this.series(), ['sys/gpu_util', 'sys/data_wait']);
    return { pct, legend: legend(pct, LABELS), vram: pick(this.series(), ['sys/vram']), vramTotal: this.live().system()?.vramTotalGb ?? null };
  });

  protected toggleChart(id: ChartId): void {
    this.shown.update((s) => toggled(s, id));
  }

  protected toggleLoss(key: string): void {
    this.hiddenLosses.update((s) => toggled(s, key));
  }

  protected onSmoothing(event: Event): void {
    this.smoothing.set(Number((event.target as HTMLInputElement).value));
  }

  protected onDocumentClick(event: Event): void {
    if (this.optionsOpen() && !this.host.nativeElement.querySelector('.options')?.contains(event.target as Node)) {
      this.optionsOpen.set(false);
    }
  }
}

function toggled<T>(set: ReadonlySet<T>, value: T): ReadonlySet<T> {
  const next = new Set(set);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}
