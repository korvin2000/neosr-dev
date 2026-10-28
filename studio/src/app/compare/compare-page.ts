import { ChangeDetectionStrategy, Component, ElementRef, computed, effect, inject, input, linkedSignal, signal, untracked } from '@angular/core';
import { Router } from '@angular/router';
import { METRIC_INFO, type RunSnapshot, type ScalarPoint, type ValidationImage } from '@shared/run.model';
import { ApiService } from '../core/api.service';
import { RunStore } from '../core/run-store';
import { UiModeService } from '../core/ui-mode.service';
import { SERIES_PALETTE, formatCompact, formatMetric, metricArrow, metricLabel, valSeriesByMetric } from '../runs/run-format';
import { ScalarChart, type XRange } from '../runs/scalar-chart';
import { ZoomView, clampCrop, type CropCenter, type ZoomLevel } from '../runs/zoom-view';
import { Disclosure } from '../ui/disclosure';
import { Icon } from '../ui/icon';
import { PageHeader } from '../ui/page-header';
import { Tooltip } from '../ui/tooltip';
import { configDiff, diffLines } from './compare-diff';

const MAX_RUNS = 3;

type Tab = 'metrics' | 'config' | 'images';

const TABS: readonly { id: Tab; label: string }[] = [
  { id: 'metrics', label: 'Metrics' },
  { id: 'config', label: 'Config' },
  { id: 'images', label: 'Images' },
];

interface RunView {
  readonly id: string;
  readonly name: string;
  readonly color: string;
  readonly snapshot: RunSnapshot;
  /** Validations of the first dataset, in iteration order. */
  readonly validations: readonly { iter: number; images: readonly ValidationImage[] }[];
  readonly dataset: string | null;
}

/**
 * Compare two or three runs from static snapshots, aligned by curriculum updates: metric and
 * loss overlays, config differences, and validation images side by side.
 */
@Component({
  selector: 'app-compare-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, PageHeader, Disclosure, ScalarChart, Tooltip, ZoomView],
  templateUrl: './compare-page.html',
  styleUrl: './compare-page.scss',
  host: { '(document:click)': 'onDocumentClick($event)', '(document:keydown.escape)': 'pickerOpen.set(false)' },
})
export class ComparePage {
  /** `?runs=a,b` */
  readonly runs = input<string | undefined>(undefined);

  protected readonly store = inject(RunStore);
  private readonly api = inject(ApiService);
  private readonly router = inject(Router);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  protected readonly mode = inject(UiModeService);

  protected readonly tabs = TABS;
  protected readonly maxRuns = MAX_RUNS;
  protected readonly selected = linkedSignal<readonly string[]>(() => [...new Set((this.runs() ?? '').split(',').filter(Boolean))].slice(0, MAX_RUNS));
  protected readonly snapshots = signal<Readonly<Record<string, RunSnapshot>>>({});
  protected readonly failed = signal<ReadonlySet<string>>(new Set());
  protected readonly tab = signal<Tab>('metrics');
  protected readonly pickerOpen = signal(false);
  protected readonly xRange = signal<XRange | null>(null);
  protected readonly diffOther = signal<string | null>(null);
  protected readonly changesOnly = signal(true);
  protected readonly imageIter = signal<number | null>(null);
  protected readonly imageId = signal<string | null>(null);
  protected readonly zoom = signal<ZoomLevel>(1);
  protected readonly crop = signal<CropCenter>({ x: 0.5, y: 0.5 });
  protected readonly zooms: readonly ZoomLevel[] = [1, 2, 4];
  protected readonly formatCompact = formatCompact;
  private readonly loadingIds = new Set<string>();

  protected readonly views = computed<RunView[]>(() => {
    const snaps = this.snapshots();
    return this.selected()
      .map((id) => snaps[id])
      .filter((s): s is RunSnapshot => !!s)
      .map((snapshot) => {
        const i = this.selected().indexOf(snapshot.run.id);
        const vals = snapshot.events.filter((e) => e.type === 'validation');
        const dataset = vals[0]?.type === 'validation' ? vals[0].dataset : null;
        return {
          id: snapshot.run.id,
          name: snapshot.run.name,
          color: SERIES_PALETTE[i % SERIES_PALETTE.length] ?? '',
          snapshot,
          dataset,
          validations: vals.flatMap((e) => (e.type === 'validation' && e.dataset === dataset ? [{ iter: e.iter, images: e.images }] : [])),
        };
      });
  });

  /** Chips in selection order, including runs whose snapshot is still loading. */
  protected readonly chips = computed(() => {
    const byId = new Map(this.store.runs().map((r) => [r.id, r.name]));
    return this.selected().map((id, i) => ({
      id,
      name: this.snapshots()[id]?.run.name ?? byId.get(id) ?? id,
      color: SERIES_PALETTE[i % SERIES_PALETTE.length] ?? '',
      state: this.snapshots()[id] ? 'ok' : this.failed().has(id) ? 'failed' : 'loading',
    }));
  });

  protected readonly addable = computed(() => this.store.runs().filter((r) => !this.selected().includes(r.id)));

  protected readonly xDomain = computed<XRange>(() => [0, Math.max(1, ...this.views().map((v) => Math.max(v.snapshot.run.iter, v.snapshot.run.totalIter)))]);
  protected readonly colors = computed(() => Object.fromEntries(this.views().map((v) => [v.id, v.color])));
  protected readonly labels = computed(() => Object.fromEntries(this.views().map((v) => [v.id, v.name])));

  /** One overlay per metric: series keyed by run id, first validation dataset of each run. */
  protected readonly metricCharts = computed(() => {
    const byMetric = new Map<string, Record<string, readonly ScalarPoint[]>>();
    for (const v of this.views()) {
      for (const g of valSeriesByMetric(v.snapshot.series)) {
        const points = (v.dataset ? g.series[v.dataset] : undefined) ?? Object.values(g.series)[0];
        if (points) {
          const map = byMetric.get(g.metric) ?? {};
          map[v.id] = points;
          byMetric.set(g.metric, map);
        }
      }
    }
    return [...byMetric.entries()].map(([metric, series]) => ({ metric, series, title: `${metricLabel(metric)} ${metricArrow(metric)}`, unit: METRIC_INFO[metric]?.unit ?? '' }));
  });

  protected readonly context = computed(() => {
    const dataset = this.views().find((v) => v.dataset)?.dataset;
    const metric = this.metricCharts()[0]?.metric;
    const parts = [dataset ?? 'no validation', metric ? `${metricLabel(metric)} ${metricArrow(metric)}` : 'visual only', 'aligned by curriculum updates, not wall time'];
    return parts.join(' · ');
  });

  protected readonly lossSeries = computed(() => Object.fromEntries(this.views().flatMap((v) => (v.snapshot.series['loss/g/total'] ? [[v.id, v.snapshot.series['loss/g/total']]] : []))));

  // --- config -------------------------------------------------------------------------------
  protected readonly configRows = computed(() => (this.views().length > 1 ? configDiff(this.views().map((v) => v.snapshot.run.recipe)) : []));

  protected readonly diffPair = computed(() => {
    const views = this.views();
    const base = views[0];
    const other = views.find((v) => v.id === this.diffOther() && v !== base) ?? views[1];
    return base && other ? { base, other } : null;
  });

  protected readonly tomlDiff = computed(() => {
    const pair = this.diffPair();
    if (!pair) {
      return null;
    }
    const a = pair.base.snapshot.run.tomls[0];
    const b = pair.other.snapshot.run.tomls[0];
    const lines = diffLines((a?.text ?? '').split('\n'), (b?.text ?? '').split('\n'));
    return {
      aName: a?.name ?? '—',
      bName: b?.name ?? '—',
      lines: this.changesOnly() ? lines.filter((l) => l.kind !== 'same') : lines,
      changed: lines.filter((l) => l.kind !== 'same').length,
    };
  });

  // --- images -------------------------------------------------------------------------------
  protected readonly imageIters = computed(() => [...new Set(this.views().flatMap((v) => v.validations.map((p) => p.iter)))].sort((a, b) => a - b));
  protected readonly imageIndex = computed(() => {
    const iters = this.imageIters();
    const chosen = this.imageIter();
    if (!iters.length) {
      return -1;
    }
    if (chosen === null) {
      return iters.length - 1;
    }
    const idx = iters.findIndex((i) => i >= chosen);
    return idx < 0 ? iters.length - 1 : idx;
  });
  protected readonly imageAt = computed<number | null>(() => this.imageIters()[this.imageIndex()] ?? null);
  protected readonly imageIds = computed(() => this.views().find((v) => v.validations.length)?.validations[0]?.images.map((i) => i.id) ?? []);
  protected readonly activeImageId = computed(() => {
    const id = this.imageId();
    const ids = this.imageIds();
    return id && ids.includes(id) ? id : (ids[0] ?? null);
  });

  /** Per run: the nearest validation at or before the chosen iteration. */
  protected readonly imageColumns = computed(() => {
    const at = this.imageAt();
    const id = this.activeImageId();
    if (at === null || !id) {
      return [];
    }
    return this.views().map((v) => {
      let point: { iter: number; images: readonly ValidationImage[] } | undefined;
      for (const p of v.validations) {
        if (p.iter <= at) {
          point = p;
        }
      }
      const image = point?.images.find((i) => i.id === id);
      return {
        id: v.id,
        name: v.name,
        color: v.color,
        iter: point?.iter ?? null,
        image: image ?? null,
        metrics: image ? Object.entries(image.metrics).map(([k, val]) => ({ key: k, text: `${metricLabel(k)} ${formatMetric(k, val)}` })) : [],
      };
    });
  });
  protected readonly gtUrl = computed(() => {
    const id = this.activeImageId();
    for (const v of this.views()) {
      const url = v.validations[0]?.images.find((i) => i.id === id)?.gtUrl;
      if (url) {
        return url;
      }
    }
    return null;
  });

  constructor() {
    void this.store.refresh().catch(() => undefined);
    effect(() => {
      const ids = this.selected();
      untracked(() => {
        for (const id of ids) {
          if (!this.snapshots()[id] && !this.failed().has(id)) {
            void this.load(id);
          }
        }
      });
    });
  }

  protected add(id: string): void {
    this.pickerOpen.set(false);
    this.setSelected([...this.selected(), id].slice(0, MAX_RUNS));
  }

  protected remove(id: string): void {
    this.setSelected(this.selected().filter((x) => x !== id));
  }

  protected onImageSlider(event: Event): void {
    const iter = this.imageIters()[Number((event.target as HTMLInputElement).value)];
    this.imageIter.set(iter ?? null);
  }

  protected setZoom(z: ZoomLevel): void {
    this.zoom.set(z);
    this.crop.update((c) => clampCrop(c, z));
  }

  protected onDocumentClick(event: Event): void {
    if (this.pickerOpen() && !this.host.nativeElement.querySelector('.add')?.contains(event.target as Node)) {
      this.pickerOpen.set(false);
    }
  }

  private setSelected(ids: readonly string[]): void {
    this.selected.set(ids);
    void this.router.navigate([], { queryParams: { runs: ids.length ? ids.join(',') : null }, replaceUrl: true });
  }

  private async load(id: string): Promise<void> {
    if (this.loadingIds.has(id)) {
      return;
    }
    this.loadingIds.add(id);
    try {
      const snapshot = await this.api.getRun(id);
      this.snapshots.update((s) => ({ ...s, [id]: snapshot }));
    } catch {
      this.failed.update((f) => new Set(f).add(id));
    } finally {
      this.loadingIds.delete(id);
    }
  }
}
