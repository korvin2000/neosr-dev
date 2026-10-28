import { ChangeDetectionStrategy, Component, computed, inject, input, output } from '@angular/core';
import { Router } from '@angular/router';
import type { ScalarPoint } from '@shared/run.model';
import type { LiveRun, ValidationPoint } from '../core/run-store';
import { Icon } from '../ui/icon';
import { LiveFlow } from './live-flow';
import { SERIES_PALETTE, formatCompact, metricArrow, metricBetter, metricLabel, primaryMetric, warningLabel } from './run-format';
import { type ChartBand, ScalarChart, type XRange } from './scalar-chart';
import { ZoomView } from './zoom-view';

/**
 * The run at a glance: the live training flow (progress, quality and recovery live in its nodes), the
 * validation metric and the generator loss components, the latest validation image (hold to blink
 * against the best one), and one warning row if needed.
 */
@Component({
  selector: 'app-run-overview',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, LiveFlow, ScalarChart, ZoomView],
  templateUrl: './run-overview.html',
  styleUrl: './run-overview.scss',
})
export class RunOverview {
  readonly live = input.required<LiveRun>();
  readonly active = input(false);
  /** Ticks every few seconds so "2 min ago" stays true. */
  readonly now = input(Date.now());
  readonly showWarnings = output<void>();

  private readonly router = inject(Router);

  protected readonly run = computed(() => this.live().run());
  private readonly series = computed(() => this.live().series());
  private readonly hasGan = computed(() => (this.run()?.phases ?? []).some((p) => p.hasGan));

  /** Validations of the first dataset: the one every KPI and chart on this tab talks about. */
  private readonly vals = computed<readonly ValidationPoint[]>(() => {
    const all = this.live().validations();
    const dataset = all[0]?.dataset;
    return all.filter((v) => v.dataset === dataset);
  });
  protected readonly dataset = computed(() => this.vals()[0]?.dataset ?? null);
  protected readonly metric = computed(() => primaryMetric(Object.keys(this.vals().at(-1)?.metrics ?? {}), this.hasGan()));

  // --- charts ----------------------------------------------------------------------------------
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

  protected readonly metricChart = computed(() => {
    const metric = this.metric();
    const dataset = this.dataset();
    const points = metric && dataset ? this.series()[`val/${dataset}/${metric}`] : undefined;
    return {
      title: metric ? `Validation ${metricLabel(metric)} ${metricArrow(metric)}` : 'Validation',
      series: points ? { [dataset as string]: points } : {},
      labels: { [dataset ?? '']: metric ? metricLabel(metric) : '' },
      direction: metric ? metricBetter(metric) : null,
      sparse: (points?.length ?? 0) <= 60,
    };
  });

  protected readonly lossChart = computed(() => {
    const series = this.series();
    const keys = Object.keys(series)
      .filter((k) => k.startsWith('loss/g/') && k !== 'loss/g/total')
      .sort();
    if (this.hasGan() && series['loss/d/total']) {
      keys.push('loss/d/total');
    }
    const picked: Record<string, readonly ScalarPoint[]> = {};
    const labels: Record<string, string> = {};
    for (const key of keys) {
      picked[key] = series[key] ?? [];
      labels[key] = key === 'loss/d/total' ? 'D total' : key.slice('loss/g/'.length);
    }
    return { series: picked, labels, legend: keys.map((k, i) => ({ key: k, label: labels[k] ?? k, color: SERIES_PALETTE[i % SERIES_PALETTE.length] ?? '' })) };
  });

  // --- image -----------------------------------------------------------------------------------
  protected readonly image = computed(() => {
    const vals = this.vals();
    const latest = vals.at(-1);
    const first = latest?.images[0];
    if (!latest || !first) {
      return null;
    }
    const metric = this.metric();
    let best: ValidationPoint | null = null;
    if (metric) {
      const higher = metricBetter(metric) === 'higher';
      for (const v of vals) {
        const value = v.metrics[metric];
        const current = best?.metrics[metric];
        if (value !== undefined && (current === undefined || (higher ? value > current : value < current))) {
          best = v;
        }
      }
    }
    const bestImage = best && best.iter !== latest.iter ? best.images.find((i) => i.id === first.id) : undefined;
    const caption = [`latest ${formatCompact(latest.iter)}`];
    if (best) {
      caption.push(best.iter === latest.iter ? 'latest is the best' : `best ${formatCompact(best.iter)}`);
    }
    return {
      name: first.name,
      latest: first.srUrl,
      best: bestImage?.srUrl ?? null,
      latestLabel: `latest ${formatCompact(latest.iter)}`,
      bestLabel: best ? `best ${formatCompact(best.iter)}` : '',
      caption: caption.join(' · '),
    };
  });

  // --- warnings --------------------------------------------------------------------------------
  protected readonly warnings = computed(() => {
    const counts = new Map<string, number>();
    for (const e of this.live().events()) {
      if (e.type === 'warning') {
        counts.set(e.code, (counts.get(e.code) ?? 0) + 1);
      }
    }
    const parts = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([code, n]) => `${warningLabel(code)} ×${n}`);
    const earlier = (this.run()?.warnings ?? 0) - [...counts.values()].reduce((a, b) => a + b, 0);
    if (earlier > 0) {
      parts.push(`${earlier} earlier`);
    }
    return parts.join(' · ');
  });

  /** The cockpit reads `?tab=` into its tab input, so a flow node opens a tab by navigating. */
  protected openTab(tab: 'images' | 'diagnostics'): void {
    void this.router.navigate([], { queryParams: { tab }, queryParamsHandling: 'merge', replaceUrl: true });
  }
}
