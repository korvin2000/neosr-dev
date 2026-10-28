import { ChangeDetectionStrategy, Component, computed, input, output, signal } from '@angular/core';
import type { Recipe } from '@shared/recipe.model';
import type { RunPhaseInfo, ValidationImage } from '@shared/run.model';
import type { CheckpointPoint, ValidationPoint } from '../core/run-store';
import { Icon } from '../ui/icon';
import { Tooltip } from '../ui/tooltip';
import { formatInt, formatMetric, metricArrow, metricBetter, metricLabel, phaseClock, phaseLabel } from './run-format';
import { ZoomView, clampCrop, type CropCenter, type ZoomLevel } from './zoom-view';

type CompareMode = 'side' | 'swipe' | 'blink';

interface MetricView {
  readonly key: string;
  readonly label: string;
  readonly arrow: string;
  readonly value: string;
  readonly delta: string;
  readonly improved: boolean | null;
}

/**
 * Validation gallery and "checkpoint time machine": a scrubber over validation events with
 * saved-artifact badges, and an image comparer whose image, mode, zoom and crop stay fixed while
 * scrubbing, so only the SR image changes.
 */
@Component({
  selector: 'app-validation-gallery',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, Tooltip, ZoomView],
  templateUrl: './validation-gallery.html',
  styleUrl: './validation-gallery.scss',
})
export class ValidationGallery {
  readonly validations = input.required<readonly ValidationPoint[]>();
  readonly checkpoints = input.required<readonly CheckpointPoint[]>();
  readonly phases = input<readonly RunPhaseInfo[]>([]);
  readonly recipe = input<Recipe | undefined>(undefined);
  /** `null` follows the latest validation. */
  readonly selectedIter = input<number | null>(null);
  readonly live = input(false);
  readonly selectedIterChange = output<number | null>();

  protected readonly dataset = signal<string | null>(null);
  protected readonly imageId = signal<string | null>(null);
  protected readonly mode = signal<CompareMode>('side');
  protected readonly zoom = signal<ZoomLevel>(1);
  protected readonly crop = signal<CropCenter>({ x: 0.5, y: 0.5 });
  protected readonly zooms: readonly ZoomLevel[] = [1, 2, 4];
  protected readonly modes: readonly { id: CompareMode; label: string; title: string }[] = [
    { id: 'side', label: 'LQ | SR | GT', title: 'Side by side' },
    { id: 'swipe', label: 'Swipe', title: 'SR over GT with a draggable divider' },
    { id: 'blink', label: 'Blink', title: 'SR; press and hold to show GT' },
  ];
  protected readonly formatInt = formatInt;

  protected readonly datasets = computed(() => [...new Set(this.validations().map((v) => v.dataset))]);
  protected readonly activeDataset = computed(() => {
    const chosen = this.dataset();
    const all = this.datasets();
    return chosen && all.includes(chosen) ? chosen : (all[0] ?? null);
  });
  protected readonly points = computed(() => this.validations().filter((v) => v.dataset === this.activeDataset()));

  protected readonly index = computed(() => {
    const points = this.points();
    const iter = this.selectedIter();
    if (!points.length) {
      return -1;
    }
    if (iter === null) {
      return points.length - 1;
    }
    let best = 0;
    points.forEach((p, i) => {
      if (Math.abs(p.iter - iter) < Math.abs((points[best] as ValidationPoint).iter - iter)) {
        best = i;
      }
    });
    return best;
  });

  protected readonly point = computed(() => this.points()[this.index()] ?? null);
  protected readonly following = computed(() => this.selectedIter() === null);

  protected readonly clock = computed(() => {
    const p = this.point();
    return p ? phaseClock(this.phases(), this.recipe(), p.iter) : null;
  });
  protected readonly phaseText = computed(() => {
    const c = this.clock();
    return c ? phaseLabel(this.phases(), c.index) : '';
  });

  protected readonly saved = computed(() => {
    const p = this.point();
    const ckpt = p ? this.checkpoints().find((c) => c.iter === p.iter) : undefined;
    return {
      images: !!p && p.images.length > 0,
      weights: !!p && (p.checkpoint || !!ckpt) && (!ckpt || ckpt.files.some((f) => f.startsWith('net_g'))),
      state: !!ckpt?.state,
      files: ckpt ? ckpt.files.join(', ') : '',
    };
  });

  protected readonly image = computed<ValidationImage | null>(() => {
    const images = this.point()?.images ?? [];
    const id = this.imageId();
    return images.find((i) => i.id === id) ?? images[0] ?? null;
  });

  /** LQ and GT do not change between validations; keep the first URLs so only SR reloads. */
  protected readonly reference = computed(() => {
    const image = this.image();
    const first = this.points()[0]?.images.find((i) => i.id === image?.id);
    return { lq: first?.lqUrl ?? image?.lqUrl ?? '', gt: first?.gtUrl ?? image?.gtUrl ?? null };
  });

  protected readonly summary = computed(() => {
    const p = this.point();
    return p ? Object.entries(p.metrics).map(([key, value]) => ({ key, label: metricLabel(key), arrow: metricArrow(key), value: formatMetric(key, value) })) : [];
  });

  protected readonly imageMetrics = computed<MetricView[]>(() => {
    const image = this.image();
    const prevPoint = this.points()[this.index() - 1];
    const prev = prevPoint?.images.find((i) => i.id === image?.id);
    if (!image) {
      return [];
    }
    return Object.entries(image.metrics).map(([key, value]) => {
      const before = prev?.metrics[key];
      const diff = before === undefined ? null : value - before;
      const better = metricBetter(key);
      return {
        key,
        label: metricLabel(key),
        arrow: metricArrow(key),
        value: formatMetric(key, value),
        delta: diff === null ? '' : `${diff >= 0 ? '+' : ''}${formatMetric(key, diff)}`,
        improved: diff === null || diff === 0 ? null : better === 'higher' ? diff > 0 : diff < 0,
      };
    });
  });

  protected select(index: number): void {
    const points = this.points();
    const clamped = Math.max(0, Math.min(points.length - 1, index));
    const p = points[clamped];
    if (p) {
      this.selectedIterChange.emit(p.iter);
    }
  }

  protected onSlider(event: Event): void {
    this.select(Number((event.target as HTMLInputElement).value));
  }

  protected setZoom(z: ZoomLevel): void {
    this.zoom.set(z);
    this.crop.update((c) => clampCrop(c, z));
  }

  protected setDataset(event: Event): void {
    this.dataset.set((event.target as HTMLSelectElement).value);
  }
}
