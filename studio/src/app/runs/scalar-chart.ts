import { ChangeDetectionStrategy, Component, DestroyRef, ElementRef, afterNextRender, computed, effect, inject, input, output, signal, viewChild } from '@angular/core';
import type { ScalarPoint } from '@shared/run.model';
import { CustomChart, LineChart, type CustomSeriesOption, type LineSeriesOption } from 'echarts/charts';
import { DataZoomComponent, GridComponent, MarkAreaComponent, MarkLineComponent, MarkPointComponent, TooltipComponent, type DataZoomComponentOption, type GridComponentOption, type TooltipComponentOption } from 'echarts/components';
import * as echarts from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { SERIES_PALETTE, formatCompact, formatInt, formatValue } from './run-format';

echarts.use([LineChart, CustomChart, GridComponent, TooltipComponent, DataZoomComponent, MarkAreaComponent, MarkLineComponent, MarkPointComponent, CanvasRenderer]);

type ChartOption = echarts.ComposeOption<LineSeriesOption | CustomSeriesOption | GridComponentOption | TooltipComponentOption | DataZoomComponentOption>;
type Datum = [number, number | null];
type RenderItem = NonNullable<CustomSeriesOption['renderItem']>;

/** `phase` and `cursor` draw vertical lines; the others draw ticks on the x axis. */
export interface ChartMarker {
  readonly kind: 'phase' | 'cursor' | 'checkpoint' | 'validation' | 'warning';
  readonly iter: number;
  readonly label?: string;
}

export type XRange = readonly [number, number];

/** A shaded x interval, e.g. a training phase; alternate bands are tinted so neighbours stay apart. */
export interface ChartBand {
  readonly from: number;
  readonly to: number;
  readonly label?: string;
}

interface Prepared {
  readonly key: string;
  readonly label: string;
  readonly color: string;
  readonly points: readonly ScalarPoint[];
  readonly raw: Datum[];
  readonly smooth: Datum[] | null;
  readonly smoothValues: readonly number[] | null;
}

interface Theme {
  readonly text: string;
  readonly muted: string;
  readonly faint: string;
  readonly border: string;
  readonly grid: string;
  readonly surface: string;
  readonly tooltip: string;
  readonly tooltipBorder: string;
  readonly accent: string;
  readonly warn: string;
  readonly font: string;
  readonly mono: string;
}

const TICK_HEIGHT: Readonly<Record<string, number>> = { checkpoint: 5, validation: 3, warning: 9 };

/**
 * One scalar chart: raw logged values (35 % opacity when smoothed) plus a debiased EMA overlay,
 * phase bands or boundaries, checkpoint/validation ticks, an optional best point, and an x range
 * that the host shares between charts (`xRange` in, `xRangeChange` out) so every chart zooms
 * together. Zoom/pan is opt-in (`zoomable`) so a chart never captures page scrolling on touch.
 */
@Component({
  selector: 'app-scalar-chart',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { '[style.height.px]': 'height()' },
  template: `
    <div #plot class="plot"></div>
    @if (empty()) {
      <div class="empty">{{ emptyText() }}</div>
    }
  `,
  styles: `
    :host { display: block; position: relative; min-width: 0; }
    .plot { position: absolute; inset: 0; }
    .empty { position: absolute; inset: 0; display: grid; place-items: center; color: var(--text-faint); font-size: var(--fs-sm); pointer-events: none; }
  `,
})
export class ScalarChart {
  /** Series in display order; a key's colour is its position here, so toggling never repaints. */
  readonly series = input.required<Readonly<Record<string, readonly ScalarPoint[]>>>();
  /** Keys to draw; `null` draws all. */
  readonly selected = input<readonly string[] | null>(null);
  readonly labels = input<Readonly<Record<string, string>>>({});
  readonly colors = input<Readonly<Record<string, string>> | null>(null);
  /** EMA weight 0–0.95; 0 shows the raw values only. */
  readonly smoothing = input(0);
  readonly markers = input<readonly ChartMarker[]>([]);
  readonly bands = input<readonly ChartBand[]>([]);
  readonly logScale = input(false);
  readonly yLabel = input('');
  /** Marks the best point of the first drawn series. */
  readonly direction = input<'higher' | 'lower' | null>(null);
  readonly xDomain = input<XRange | null>(null);
  readonly xRange = input<XRange | null>(null);
  readonly height = input(200);
  readonly showSymbols = input(false);
  readonly emptyText = input('No data yet');
  /** Ctrl + wheel zoom and drag to pan. */
  readonly zoomable = input(false);
  readonly xRangeChange = output<XRange | null>();

  private readonly plot = viewChild.required<ElementRef<HTMLDivElement>>('plot');
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly chart = signal<echarts.EChartsType | null>(null);
  private theme: Theme | null = null;
  private current: readonly Prepared[] = [];
  private domainNow: XRange = [0, 1];
  private appliedRange: XRange | null = null;

  protected readonly prepared = computed<Prepared[]>(() => {
    const all = this.series();
    const selected = this.selected();
    const labels = this.labels();
    const colors = this.colors();
    const w = Math.min(Math.max(this.smoothing(), 0), 0.95);
    const log = this.logScale();
    return Object.keys(all)
      .map((key, i) => ({ key, i }))
      .filter(({ key }) => !selected || selected.includes(key))
      .map(({ key, i }) => {
        const points = all[key] ?? [];
        const smoothValues = w > 0 ? ema(points, w) : null;
        return {
          key,
          label: labels[key] ?? key,
          color: colors?.[key] ?? SERIES_PALETTE[i % SERIES_PALETTE.length] ?? '#8fbdff',
          points,
          raw: withGaps(points, points.map((p) => p.value), log),
          smooth: smoothValues ? withGaps(points, smoothValues, log) : null,
          smoothValues,
        };
      });
  });

  protected readonly empty = computed(() => this.prepared().every((s) => s.points.length === 0));

  constructor() {
    const destroyRef = inject(DestroyRef);
    afterNextRender(() => {
      const el = this.plot().nativeElement;
      this.theme = readTheme(this.host.nativeElement);
      const chart = echarts.init(el, undefined, { renderer: 'canvas' });
      chart.setOption(this.baseOption());
      chart.on('datazoom', (e) => this.onZoom(e));
      const observer = new ResizeObserver(() => chart.resize());
      observer.observe(el);
      destroyRef.onDestroy(() => {
        observer.disconnect();
        chart.dispose();
      });
      this.chart.set(chart);
    });

    effect(() => {
      const chart = this.chart();
      const prepared = this.prepared();
      const markers = this.markers();
      const bands = this.bands();
      const log = this.logScale();
      const yLabel = this.yLabel();
      const direction = this.direction();
      const domain = this.xDomain() ?? dataDomain(prepared);
      const symbols = this.showSymbols();
      if (!chart) {
        return;
      }
      this.current = prepared;
      this.domainNow = domain;
      chart.setOption(this.dataOption(prepared, markers, bands, log, yLabel, direction, domain, symbols), { replaceMerge: ['series'] });
      if (this.appliedRange) {
        chart.dispatchAction({ type: 'dataZoom', dataZoomIndex: 0, startValue: this.appliedRange[0], endValue: this.appliedRange[1] }, { silent: true });
      }
    });

    effect(() => {
      const chart = this.chart();
      const disabled = !this.zoomable();
      chart?.setOption({ dataZoom: [{ disabled }] });
    });

    effect(() => {
      const chart = this.chart();
      const range = this.xRange();
      if (!chart || sameRange(range, this.appliedRange)) {
        return;
      }
      this.appliedRange = range;
      const payload = range ? { startValue: range[0], endValue: range[1] } : { start: 0, end: 100 };
      chart.dispatchAction({ type: 'dataZoom', dataZoomIndex: 0, ...payload }, { silent: true });
    });
  }

  private onZoom(event: unknown): void {
    const e = event as { start?: number; end?: number; batch?: readonly { start?: number; end?: number }[] };
    const b = e.batch?.[0] ?? e;
    if (b.start === undefined || b.end === undefined) {
      return;
    }
    const [d0, d1] = this.domainNow;
    const range: XRange | null = b.start <= 0 && b.end >= 100 ? null : [d0 + ((d1 - d0) * b.start) / 100, d0 + ((d1 - d0) * b.end) / 100];
    this.appliedRange = range;
    this.xRangeChange.emit(range);
  }

  private baseOption(): ChartOption {
    const t = this.theme as Theme;
    return {
      animation: false,
      textStyle: { fontFamily: t.font },
      grid: { left: 50, right: 14, top: 20, bottom: 22 },
      tooltip: {
        trigger: 'axis',
        confine: true,
        transitionDuration: 0,
        backgroundColor: t.tooltip,
        borderColor: t.tooltipBorder,
        borderWidth: 1,
        padding: [6, 9],
        textStyle: { color: t.text, fontSize: 11, fontFamily: t.font },
        axisPointer: { type: 'line', animation: false, lineStyle: { color: t.faint, width: 1 } },
        formatter: (params: unknown) => this.tooltipHtml(params),
      },
      dataZoom: [{ type: 'inside', xAxisIndex: 0, filterMode: 'filter', zoomOnMouseWheel: 'ctrl', moveOnMouseMove: true, moveOnMouseWheel: false, disabled: !this.zoomable() }],
    };
  }

  private dataOption(prepared: readonly Prepared[], markers: readonly ChartMarker[], bands: readonly ChartBand[], log: boolean, yLabel: string, direction: 'higher' | 'lower' | null, domain: XRange, symbols: boolean): ChartOption {
    const t = this.theme as Theme;
    const lines = markers.filter((m) => m.kind === 'phase' || m.kind === 'cursor');
    const ticks = markers.filter((m) => m.kind !== 'phase' && m.kind !== 'cursor');
    const shaded = bands.map((b) => ({ ...b, from: Math.max(b.from, domain[0]), to: Math.min(b.to, domain[1]) })).filter((b) => b.to > b.from);
    const series: (LineSeriesOption | CustomSeriesOption)[] = [];
    prepared.forEach((s, i) => {
      const smoothed = !!s.smooth;
      const best = i === 0 && direction ? bestOf(s.points, direction, log) : null;
      series.push({
        type: 'line',
        id: `${s.key}::raw`,
        name: s.label,
        data: s.raw,
        showSymbol: symbols,
        symbol: 'circle',
        symbolSize: 4,
        sampling: symbols ? undefined : 'lttb',
        connectNulls: false,
        lineStyle: { width: smoothed ? 1 : 1.5, color: s.color, opacity: smoothed ? 0.35 : 1 },
        itemStyle: { color: s.color, opacity: smoothed ? 0.35 : 1 },
        emphasis: { disabled: true },
        z: 2,
        markLine:
          i === 0 && lines.length
            ? {
                silent: true,
                symbol: ['none', 'none'],
                animation: false,
                label: { color: t.faint, fontSize: 10, position: 'end', distance: 3 },
                lineStyle: { color: t.faint, type: [4, 3], width: 1 },
                data: lines.map((m) =>
                  m.kind === 'cursor'
                    ? { xAxis: m.iter, label: { show: false }, lineStyle: { color: t.accent, type: 'solid' as const, width: 1, opacity: 0.7 } }
                    : { xAxis: m.iter, label: { show: !!m.label, formatter: m.label ?? '' } },
                ),
              }
            : undefined,
        markArea:
          i === 0 && shaded.length > 1
            ? {
                silent: true,
                animation: false,
                label: { color: t.faint, fontSize: 10, position: 'insideTopLeft', distance: 6 },
                data: shaded.map((b, k) => [
                  { xAxis: b.from, name: b.label ?? '', itemStyle: { color: t.text, opacity: k % 2 ? 0.045 : 0 } },
                  { xAxis: b.to },
                ]),
              }
            : undefined,
        markPoint: best
          ? {
              silent: true,
              animation: false,
              symbol: 'circle',
              symbolSize: 8,
              itemStyle: { color: s.color, borderColor: t.surface, borderWidth: 2 },
              // Beside the point, facing the plot: never off the canvas edge, never over the phase labels on top.
              label: { show: true, position: best.iter > (domain[0] + domain[1]) / 2 ? 'left' : 'right', distance: 5, color: t.text, fontSize: 10, formatter: `best ${formatValue(best.value)}` },
              data: [{ name: 'best', coord: [best.iter, best.value] }],
            }
          : undefined,
      });
      if (s.smooth) {
        series.push({
          type: 'line',
          id: `${s.key}::ema`,
          name: `${s.label} (smoothed)`,
          data: s.smooth,
          showSymbol: false,
          sampling: 'lttb',
          connectNulls: false,
          lineStyle: { width: 1.5, color: s.color },
          itemStyle: { color: s.color },
          emphasis: { disabled: true },
          z: 3,
        });
      }
    });
    if (ticks.length && prepared.length) {
      const colours: Readonly<Record<string, string>> = { checkpoint: t.muted, validation: t.accent, warning: t.warn };
      const renderTick: RenderItem = (params, api) => {
        const kind = ticks[params.dataIndex]?.kind ?? 'checkpoint';
        const sys = params.coordSys as unknown as { y: number; height: number };
        const x = api.coord([api.value(0), log ? 1 : 0])[0] ?? 0;
        const bottom = sys.y + sys.height;
        return {
          type: 'line',
          silent: true,
          shape: { x1: x, y1: bottom, x2: x, y2: bottom - (TICK_HEIGHT[kind] ?? 5) },
          style: { stroke: colours[kind] ?? t.muted, lineWidth: 1 },
        };
      };
      series.push({ type: 'custom', id: '__ticks', silent: true, clip: true, z: 1, encode: { x: 0 }, data: ticks.map((m) => [m.iter]), renderItem: renderTick, tooltip: { show: false } });
    }
    return {
      xAxis: {
        type: 'value',
        min: domain[0],
        max: domain[1],
        axisLabel: { color: t.faint, fontSize: 10, fontFamily: t.mono, hideOverlap: true, formatter: (v: number) => formatCompact(v) },
        axisLine: { lineStyle: { color: t.border } },
        axisTick: { show: false },
        splitLine: { show: false },
      },
      yAxis: {
        type: log ? 'log' : 'value',
        scale: true,
        name: yLabel,
        nameGap: 8,
        nameTextStyle: { color: t.faint, fontSize: 10, align: 'left', padding: [0, 0, 0, -44] },
        axisLabel: { color: t.faint, fontSize: 10, fontFamily: t.mono, formatter: (v: number) => formatValue(v) },
        axisLine: { show: false },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: t.grid } },
      },
      series,
    };
  }

  private tooltipHtml(params: unknown): string {
    const t = this.theme as Theme;
    const first = (Array.isArray(params) ? params[0] : params) as { axisValue?: number | string } | undefined;
    const x = Number(first?.axisValue);
    if (!Number.isFinite(x)) {
      return '';
    }
    const rows = this.current
      .map((s) => {
        const i = nearestIndex(s.points, x);
        if (i < 0) {
          return '';
        }
        const raw = s.points[i]?.value ?? NaN;
        const smooth = s.smoothValues?.[i];
        const main = smooth !== undefined ? formatValue(smooth) : formatValue(raw);
        const extra = smooth !== undefined ? `<span style="color:${t.faint};font-family:${t.mono}">${formatValue(raw)}</span>` : '';
        return `<div style="display:flex;align-items:center;gap:7px;line-height:17px"><span style="width:10px;height:2px;border-radius:1px;background:${s.color}"></span><span style="color:${t.muted}">${escapeHtml(s.label)}</span><span style="margin-left:auto;padding-left:12px;font-family:${t.mono};color:${t.text}">${main}</span>${extra}</div>`;
      })
      .join('');
    const head = `<div style="color:${t.faint};font-family:${t.mono};margin-bottom:3px">iter ${formatInt(x)}${this.current.some((s) => s.smoothValues) ? ' · smoothed / raw' : ''}</div>`;
    return head + rows;
  }
}

/** TensorBoard-style debiased EMA, so the first points are not pulled towards zero. */
function ema(points: readonly ScalarPoint[], w: number): number[] {
  let last = 0;
  let n = 0;
  return points.map((p) => {
    if (!Number.isFinite(p.value)) {
      return NaN;
    }
    last = last * w + (1 - w) * p.value;
    n++;
    return last / (1 - w ** n);
  });
}

/** Missing telemetry is a gap, not a line: a step much longer than the usual one breaks the line. */
function withGaps(points: readonly ScalarPoint[], values: readonly number[], log: boolean): Datum[] {
  const steps: number[] = [];
  for (let i = 1; i < points.length; i++) {
    steps.push((points[i] as ScalarPoint).iter - (points[i - 1] as ScalarPoint).iter);
  }
  const median = steps.length ? ([...steps].sort((a, b) => a - b)[Math.floor(steps.length / 2)] ?? 0) : 0;
  const out: Datum[] = [];
  points.forEach((p, i) => {
    if (i > 0 && median > 0 && (steps[i - 1] ?? 0) > median * 4) {
      out.push([(p.iter + (points[i - 1] as ScalarPoint).iter) / 2, null]);
    }
    const v = values[i] ?? NaN;
    out.push([p.iter, Number.isFinite(v) && (!log || v > 0) ? v : null]);
  });
  return out;
}

function bestOf(points: readonly ScalarPoint[], direction: 'higher' | 'lower', log: boolean): ScalarPoint | null {
  let best: ScalarPoint | null = null;
  for (const p of points) {
    if (Number.isFinite(p.value) && (!log || p.value > 0) && (!best || (direction === 'lower' ? p.value < best.value : p.value > best.value))) {
      best = p;
    }
  }
  return best;
}

function nearestIndex(points: readonly ScalarPoint[], x: number): number {
  if (!points.length) {
    return -1;
  }
  let lo = 0;
  let hi = points.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if ((points[mid] as ScalarPoint).iter < x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo > 0 && Math.abs((points[lo - 1] as ScalarPoint).iter - x) <= Math.abs((points[lo] as ScalarPoint).iter - x)) {
    lo--;
  }
  return lo;
}

function dataDomain(prepared: readonly Prepared[]): XRange {
  let min = Infinity;
  let max = -Infinity;
  for (const s of prepared) {
    if (s.points.length) {
      min = Math.min(min, (s.points[0] as ScalarPoint).iter);
      max = Math.max(max, (s.points[s.points.length - 1] as ScalarPoint).iter);
    }
  }
  return Number.isFinite(min) && max > min ? [min, max] : [0, Math.max(1, Number.isFinite(max) ? max : 1)];
}

function sameRange(a: XRange | null, b: XRange | null): boolean {
  if (!a || !b) {
    return a === b;
  }
  return Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6;
}

function readTheme(el: HTMLElement): Theme {
  const css = getComputedStyle(el);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return {
    text: v('--text', '#eaeded'),
    muted: v('--text-muted', '#a9b2c1'),
    faint: v('--text-faint', '#7b8494'),
    border: v('--border', '#2a323f'),
    grid: v('--border-soft', '#232a36'),
    surface: v('--panel', '#1a202b'),
    tooltip: v('--panel-raised', '#262d3b'),
    tooltipBorder: v('--border-strong', '#3a4352'),
    accent: v('--accent-text', '#8fbdff'),
    warn: v('--warn', '#ecc94b'),
    font: v('--font', 'sans-serif'),
    mono: v('--font-mono', 'monospace'),
  };
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] ?? c);
}
