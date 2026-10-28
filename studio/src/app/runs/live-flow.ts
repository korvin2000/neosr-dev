import { ChangeDetectionStrategy, Component, DestroyRef, ElementRef, type WritableSignal, afterNextRender, computed, effect, inject, input, output, signal, untracked } from '@angular/core';
import { ARCH_BY_TYPE } from '@shared/catalog/archs';
import { OPTIMIZERS } from '@shared/catalog/optimizers';
import { SCHEDULERS } from '@shared/catalog/schedulers';
import { nodeValues, phaseNodeValues, planContext, scaleOf } from '@shared/compile/graph';
import type { ScalarPoint } from '@shared/run.model';
import { ACTIVE_STATUSES, type LiveRun, type ValidationPoint } from '../core/run-store';
import { UiModeService } from '../core/ui-mode.service';
import { Icon } from '../ui/icon';
import { Tooltip } from '../ui/tooltip';
import { SERIES_PALETTE, approxDuration, formatCompact, formatInt, formatMetric, metricArrow, metricBetter, metricLabel, metricText, phaseClock, phaseLabel, primaryMetric, statusView, timeAgo } from './run-format';

type NodeId = 'run' | 'data' | 'gen' | 'losses' | 'disc' | 'optim' | 'ckpt' | 'val' | 'gpu';
type Side = 'left' | 'right' | 'top' | 'bottom';

interface EdgeSpec {
  readonly id: string;
  readonly from: NodeId;
  readonly to: NodeId;
  readonly out: Side;
  readonly in: Side;
  /** Family accent of the source node. */
  readonly family: string;
  readonly gan?: boolean;
  /** Routed along the row gap instead of a direct curve (Generator → Validation passes behind row 2 otherwise). */
  readonly bus?: boolean;
}

interface EdgePath {
  readonly id: string;
  readonly d: string;
  readonly family: string;
  readonly off: boolean;
}

interface RingSegment {
  readonly id: string;
  readonly dash: string;
  readonly fillDash: string;
  readonly rotate: number;
  readonly active: boolean;
  readonly done: boolean;
}

interface LossBar {
  readonly key: string;
  readonly label: string;
  readonly value: string;
  readonly pct: number;
  readonly color: string;
}

const EDGES: readonly EdgeSpec[] = [
  { id: 'run-data', from: 'run', to: 'data', out: 'right', in: 'left', family: 'sky' },
  { id: 'data-gen', from: 'data', to: 'gen', out: 'right', in: 'left', family: 'emerald' },
  { id: 'gen-losses', from: 'gen', to: 'losses', out: 'right', in: 'left', family: 'violet' },
  { id: 'losses-optim', from: 'losses', to: 'optim', out: 'right', in: 'left', family: 'rose' },
  { id: 'optim-ckpt', from: 'optim', to: 'ckpt', out: 'right', in: 'left', family: 'amber' },
  { id: 'losses-disc', from: 'losses', to: 'disc', out: 'bottom', in: 'left', family: 'rose', gan: true },
  { id: 'disc-optim', from: 'disc', to: 'optim', out: 'right', in: 'bottom', family: 'violet', gan: true },
  { id: 'gen-val', from: 'gen', to: 'val', out: 'bottom', in: 'top', family: 'violet', bus: true },
];

const RING_R = 36;
const RING_C = 2 * Math.PI * RING_R;
/** Gap between phase segments, in degrees. */
const RING_GAP = 6;
const SPARK_LEN = 60;
/** Four rows fit the node; the Diagnostics tab has the rest. */
const MAX_LOSS_BARS = 4;
/** Below this change a validation metric counts as flat. */
const FLAT: Readonly<Record<string, number>> = { psnr: 0.01 };
const FLAT_DEFAULT = 0.0005;
const FLASH_MS = 650;

export const FLOW_TIPS: Readonly<Record<NodeId, string>> = {
  run: 'Curriculum progress over all phases; the ring is split per phase.',
  data: 'The loader feeding LQ/GT patches; wait is the share of time the GPU idles for data.',
  gen: 'The generator being trained; updates in NeoSR’s own counter for this phase.',
  losses: 'Logged window means of each generator term; D is the discriminator’s own loss.',
  disc: 'The discriminator: its loss and mean outputs on real and generated patches.',
  optim: 'Generator optimizer: current learning rate, warm-up and gradient norm; skipped counts non-finite updates.',
  val: 'Validation on the first dataset: the preset metric and its trend since the previous run.',
  ckpt: 'Last saved checkpoint; resumable when a .state file was written with it.',
  gpu: 'Trainer GPU: utilisation and VRAM in use.',
};

/**
 * The trainer as a flow: one row of stage nodes with live numbers, SVG edges that carry a travelling
 * dash while the run is active, and a "beat" that walks one training step along the chain. Phones
 * get the same nodes as a vertical list.
 */
@Component({
  selector: 'app-live-flow',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, Tooltip],
  templateUrl: './live-flow.html',
  styleUrl: './live-flow.scss',
  host: {
    '[class.flow--phone]': 'mode.isPhone()',
    '[class.flow--live]': 'motion()',
    '[class.flow--paused]': 'reconnecting()',
    '[style.--flow-dur]': 'flowDuration()',
  },
})
export class LiveFlow {
  readonly live = input.required<LiveRun>();
  /** Ticks every few seconds so "2 min ago" stays true (the host owns the one interval). */
  readonly now = input(Date.now());
  readonly openTab = output<'images' | 'diagnostics'>();

  protected readonly mode = inject(UiModeService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly destroyRef = inject(DestroyRef);

  protected readonly tips = FLOW_TIPS;
  protected readonly ringR = RING_R;
  protected readonly ringC = RING_C;
  protected readonly formatInt = formatInt;

  protected readonly run = computed(() => this.live().run());
  protected readonly status = computed(() => this.live().status());
  private readonly series = computed(() => this.live().series());
  private readonly system = computed(() => this.live().system());
  private readonly connected = computed(() => this.live().connected());
  private readonly everConnected = signal(false);

  protected readonly active = computed(() => ACTIVE_STATUSES.includes(this.status()));
  /** Motion runs only while the trainer is producing updates. */
  protected readonly motion = computed(() => ['queued', 'starting', 'running'].includes(this.status()) && this.connected());
  protected readonly reconnecting = computed(() => this.active() && this.everConnected() && !this.connected());

  private readonly ctx = computed(() => {
    const run = this.run();
    return run ? planContext(run.recipe) : {};
  });
  protected readonly hasGanAny = computed(() => (this.run()?.phases ?? []).some((p) => p.hasGan));
  protected readonly phase = computed(() => {
    const run = this.run();
    return run ? run.phases[Math.min(run.phaseIndex, run.phases.length - 1)] : undefined;
  });
  protected readonly ganNow = computed(() => this.phase()?.hasGan ?? false);
  protected readonly native = computed(() => {
    const run = this.run();
    return run ? (this.live().nativeIter() ?? phaseClock(run.phases, run.recipe, run.iter).native) : null;
  });
  /** Last reported rate; kept after a stop so the nodes hold their final numbers. */
  protected readonly itps = computed(() => {
    const run = this.run();
    return run && run.itersPerSec > 0 ? run.itersPerSec : null;
  });
  protected readonly flowDuration = computed(() => `${Math.min(2, Math.max(0.4, 2 / (this.itps() ?? 1))).toFixed(2)}s`);

  // --- Run -------------------------------------------------------------------------------------
  protected readonly statusWord = computed(() => {
    if (this.reconnecting()) {
      return 'Reconnecting…';
    }
    const status = this.status();
    return status === 'stopping' ? 'Stopping' : status === 'saving' ? 'Saving' : statusView(status).label.replace('…', '');
  });
  protected readonly pct = computed(() => {
    const run = this.run();
    return run && run.totalIter ? Math.min(100, Math.floor((run.iter / run.totalIter) * 100)) : 0;
  });
  protected readonly runBig = computed(() => {
    const run = this.run();
    return run ? `${formatInt(run.iter)} / ${formatInt(run.totalIter)}` : '—';
  });
  protected readonly runLine = computed(() => {
    const run = this.run();
    if (!run) {
      return '';
    }
    const parts = [phaseLabel(run.phases, run.phaseIndex)];
    if (this.status() === 'running' && run.etaSec > 0) {
      parts.push(`ETA ≈ ${approxDuration(run.etaSec)}`);
    } else if (run.status === 'completed') {
      parts.push(`finished ${timeAgo(run.finishedAt, this.now())}`);
    }
    return parts.filter(Boolean).join(' · ');
  });

  protected readonly ring = computed<RingSegment[]>(() => {
    const run = this.run();
    if (!run) {
      return [];
    }
    const total = Math.max(1, run.totalIter);
    const phases = run.phases.length ? run.phases : [{ id: 'all', start: 0, end: total }];
    const finished = run.status === 'completed';
    const gap = phases.length > 1 ? (RING_GAP / 360) * RING_C : 0;
    let offset = 0;
    return phases.map((p, i) => {
      const span = Math.max(1, p.end - p.start);
      const len = (span / total) * RING_C - gap;
      const fill = Math.min(1, Math.max(0, (run.iter - p.start) / span));
      const seg: RingSegment = {
        id: p.id,
        dash: `${Math.max(0, len)} ${RING_C}`,
        fillDash: `${Math.max(0, len * fill)} ${RING_C}`,
        rotate: (offset / RING_C) * 360 - 90,
        active: !finished && i === run.phaseIndex,
        done: finished || i < run.phaseIndex,
      };
      offset += len + gap;
      return seg;
    });
  });

  // --- Data ------------------------------------------------------------------------------------
  private readonly dataValues = computed(() => {
    const data = this.ctx().data;
    const phaseId = this.phase()?.id;
    return data ? (phaseId ? phaseNodeValues(data, phaseId) : nodeValues(data)) : null;
  });
  protected readonly batch = computed(() => Number(this.dataValues()?.['batch_size']) || null);
  protected readonly dataBig = computed(() => {
    const itps = this.itps();
    const batch = this.batch();
    return itps && batch ? `${Math.round(itps * batch)} img/s` : '—';
  });
  protected readonly dataRate = computed(() => {
    const itps = this.itps();
    const batch = this.batch();
    return itps ? `${itps.toFixed(1)} it/s × batch ${batch ?? '—'}` : batch ? `batch ${batch}` : '';
  });
  protected readonly dataLine = computed(() => {
    const v = this.dataValues();
    if (!v) {
      return '—';
    }
    const patch = Number(v['patch_size']) || 0;
    const scale = scaleOf(this.ctx().model);
    const kind = v['type'] === 'otf' ? 'on-the-fly degradations' : 'paired';
    return patch ? `${kind} · ${patch} px → ${patch * scale}` : kind;
  });
  protected readonly dataWait = computed(() => this.system()?.dataWaitPct ?? null);

  // --- Generator -------------------------------------------------------------------------------
  protected readonly genBig = computed(() => {
    const itps = this.itps();
    return itps ? `${itps.toFixed(1)} it/s` : '—';
  });
  protected readonly genLine = computed(() => {
    const run = this.run();
    if (!run) {
      return '';
    }
    const arch = ARCH_BY_TYPE.get(run.arch)?.label ?? run.arch;
    const native = this.native();
    return [arch, `×${run.scale}`, native !== null ? `update ${formatInt(native)} (native)` : ''].filter(Boolean).join(' · ');
  });
  private readonly itpsBuffer = signal<number[]>([]);
  protected readonly itpsSpark = computed(() => sparkPath(this.itpsBuffer()));

  // --- Losses ----------------------------------------------------------------------------------
  protected readonly lossTotal = computed(() => {
    const v = last(this.series()['loss/g/total']);
    return v === null ? '—' : `total ${formatLoss(v)}`;
  });
  protected readonly lossBars = computed<LossBar[]>(() => {
    const series = this.series();
    const keys = Object.keys(series)
      .filter((k) => k.startsWith('loss/g/') && k !== 'loss/g/total')
      .sort();
    const values = keys.map((k) => ({ key: k, value: last(series[k]) ?? 0 }));
    const max = Math.max(1e-9, ...values.map((v) => v.value));
    return values.slice(0, MAX_LOSS_BARS).map((v, i) => ({
      key: v.key,
      label: v.key.slice('loss/g/'.length),
      value: formatLoss(v.value),
      pct: Math.max(2, (v.value / max) * 100),
      color: SERIES_PALETTE[i % SERIES_PALETTE.length] ?? '',
    }));
  });
  protected readonly lossLine = computed(() => {
    const n = Object.keys(this.series()).filter((k) => k.startsWith('loss/g/') && k !== 'loss/g/total').length;
    if (!n) {
      return 'no terms logged yet';
    }
    return `${n} term${n === 1 ? '' : 's'}${this.ganNow() ? ' · GAN' : ''}`;
  });

  // --- Discriminator ---------------------------------------------------------------------------
  protected readonly discBig = computed(() => {
    const v = last(this.series()['loss/d/total']);
    return v === null ? '—' : `D ${v.toFixed(2)}`;
  });
  protected readonly discLine = computed(() => {
    if (!this.ganNow()) {
      return 'off this phase';
    }
    const real = last(this.series()['out/d/real']);
    const fake = last(this.series()['out/d/fake']);
    return real === null || fake === null ? 'no outputs yet' : `real ${real.toFixed(2)} · fake ${fake.toFixed(2)}`;
  });

  // --- Optimizer -------------------------------------------------------------------------------
  protected readonly lrBig = computed(() => {
    const v = last(this.series()['lr/g']);
    return v === null ? '—' : `lr ${v.toExponential(1).replace('e-0', 'e-').replace('e+0', 'e')}`;
  });
  protected readonly optimLine = computed(() => {
    const ctx = this.ctx();
    if (!ctx.optimG) {
      return '—';
    }
    const values = nodeValues(ctx.optimG);
    const type = String(values['type'] ?? '');
    const info = OPTIMIZERS.find((o) => o.type === type);
    if (info?.scheduleFree) {
      const warmup = Number(values['warmup_steps']) || 0;
      const native = this.native() ?? 0;
      return warmup > 0 && native < warmup ? `${type} · warming up · ${formatInt(native)} / ${formatInt(warmup)}` : `${type} · warm-up done`;
    }
    const schedule = ctx.schedule ? String(nodeValues(ctx.schedule)['type'] ?? '') : '';
    const label = SCHEDULERS.find((s) => s.type === schedule)?.label ?? schedule;
    return label ? `${type} · ${label}` : type;
  });
  protected readonly gradSpark = computed(() => sparkPath((this.series()['grad_norm/g'] ?? []).slice(-SPARK_LEN).map((p) => p.value)));
  protected readonly skipped = computed(() => (this.series()['skipped/g'] ?? []).reduce((sum, p) => sum + p.value, 0));

  // --- Validation ------------------------------------------------------------------------------
  private readonly vals = computed<readonly ValidationPoint[]>(() => {
    const all = this.live().validations();
    const dataset = all[0]?.dataset;
    return all.filter((v) => v.dataset === dataset);
  });
  private readonly metric = computed(() => primaryMetric(Object.keys(this.vals().at(-1)?.metrics ?? {}), this.hasGanAny()));
  protected readonly valView = computed(() => {
    const vals = this.vals();
    const latest = vals.at(-1);
    const metric = this.metric();
    if (!latest) {
      return { big: 'none yet', empty: true, trend: null, label: '' };
    }
    const value = metric ? latest.metrics[metric] : undefined;
    if (!metric || value === undefined) {
      return { big: 'visual only', empty: true, trend: null, label: latest.dataset };
    }
    const prev = vals.at(-2)?.metrics[metric];
    let trend: { arrow: string; text: string; tone: 'better' | 'worse' | 'flat' } | null = null;
    if (prev !== undefined) {
      const diff = value - prev;
      const flat = Math.abs(diff) < (FLAT[metric] ?? FLAT_DEFAULT);
      const better = metricBetter(metric) === 'higher' ? diff > 0 : diff < 0;
      trend = { arrow: flat ? '→' : diff > 0 ? '↑' : '↓', text: flat ? 'flat' : `${diff > 0 ? '+' : ''}${formatMetric(metric, diff)}`, tone: flat ? 'flat' : better ? 'better' : 'worse' };
    }
    return { big: metricText(metric, value), empty: false, trend, label: `${metricLabel(metric)} ${metricArrow(metric)}` };
  });
  private readonly valFreq = computed(() => {
    const val = this.ctx().val;
    return val ? Number(nodeValues(val)['val_freq']) || 0 : 0;
  });
  protected readonly valLine = computed(() => {
    const freq = this.valFreq();
    const parts = [this.valView().label || 'val'];
    if (freq) {
      parts.push(`every ${formatCompact(freq)}`);
      const next = this.nextIn(freq);
      if (next !== null) {
        parts.push(`next in ${formatInt(next)}`);
      }
    } else if (!this.ctx().val) {
      parts.push('no validation set');
    }
    return parts.join(' · ');
  });
  protected readonly valFlash = signal(false);

  // --- Checkpoint ------------------------------------------------------------------------------
  private readonly lastCheckpoint = computed(() => this.live().checkpoints().at(-1) ?? null);
  protected readonly ckptBig = computed(() => {
    const last = this.lastCheckpoint();
    return last ? formatCompact(last.iter) : 'none yet';
  });
  protected readonly ckptLine = computed(() => {
    const last = this.lastCheckpoint();
    const parts: string[] = [];
    if (last) {
      const events = this.live().events();
      let at: string | undefined;
      for (let i = events.length - 1; i >= 0; i--) {
        const e = events[i];
        if (e?.type === 'checkpoint' && e.iter === last.iter) {
          at = e.at;
          break;
        }
      }
      parts.push(timeAgo(at, this.now()), last.state ? 'resumable' : 'weights only');
    }
    const plan = this.ctx().plan;
    const freq = plan ? Number(nodeValues(plan)['save_checkpoint_freq']) || 0 : 0;
    if (freq) {
      parts.push(`every ${formatCompact(freq)}`);
      const next = this.nextIn(freq);
      if (next !== null) {
        parts.push(`next in ${formatInt(next)}`);
      }
    }
    return parts.filter(Boolean).join(' · ');
  });
  protected readonly ckptFlash = signal(false);

  // --- GPU -------------------------------------------------------------------------------------
  protected readonly gpuUtil = computed(() => this.system()?.gpuUtil ?? null);
  protected readonly gpuUtilDash = computed(() => {
    const c = 2 * Math.PI * 22;
    const util = this.gpuUtil() ?? 0;
    return `${(Math.min(100, Math.max(0, util)) / 100) * c} ${c}`;
  });
  protected readonly vram = computed(() => {
    const sys = this.system();
    if (!sys) {
      return null;
    }
    const pct = sys.vramTotalGb > 0 ? (sys.vramUsedGb / sys.vramTotalGb) * 100 : 0;
    return { text: `${sys.vramUsedGb.toFixed(1)} / ${Math.round(sys.vramTotalGb)} GB`, pct: Math.min(100, pct), high: pct > 90 };
  });
  protected readonly gpuName = computed(() => (this.run()?.gpu || '—').replace(/^NVIDIA (GeForce )?/, ''));

  // --- edges -----------------------------------------------------------------------------------
  private readonly rects = signal<Map<NodeId, DOMRect> | null>(null);
  protected readonly edges = computed<EdgePath[]>(() => {
    const rects = this.rects();
    if (!rects) {
      return [];
    }
    const ganAny = this.hasGanAny();
    const ganNow = this.ganNow();
    const out: EdgePath[] = [];
    for (const e of EDGES) {
      if (e.gan && !ganAny) {
        continue;
      }
      const a = rects.get(e.from);
      const b = rects.get(e.to);
      if (!a || !b) {
        continue;
      }
      out.push({ id: e.id, d: e.bus ? busPath(a, b) : curvePath(a, e.out, b, e.in), family: e.family, off: !!e.gan && !ganNow });
    }
    return out;
  });

  constructor() {
    effect(() => {
      if (this.connected()) {
        this.everConnected.set(true);
      }
    });
    // it/s history, sampled when the progress event changes it.
    effect(() => {
      const v = this.run()?.itersPerSec;
      if (v !== undefined && v > 0) {
        untracked(() => this.itpsBuffer.update((buf) => (buf.at(-1) === v ? buf : [...buf, v].slice(-SPARK_LEN))));
      }
    });
    this.flashOn(() => this.live().validations().length, this.valFlash);
    this.flashOn(() => this.live().checkpoints().length, this.ckptFlash);

    afterNextRender(() => {
      const grid = this.host.nativeElement.querySelector<HTMLElement>('.grid');
      if (!grid || typeof ResizeObserver === 'undefined') {
        return;
      }
      const observer = new ResizeObserver(() => this.measure(grid));
      observer.observe(grid);
      this.destroyRef.onDestroy(() => observer.disconnect());
    });
  }

  protected open(tab: 'images' | 'diagnostics'): void {
    this.openTab.emit(tab);
  }

  private nextIn(freq: number): number | null {
    const native = this.native();
    return native === null || !this.active() ? null : freq - (native % freq);
  }

  /** Flashes `flag` for a moment whenever `count` grows during the session (not on the initial replay). */
  private flashOn(count: () => number, flag: WritableSignal<boolean>): void {
    let seen: number | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    effect(() => {
      const n = count();
      if (seen !== null && n > seen && untracked(this.active)) {
        flag.set(true);
        if (timer) {
          clearTimeout(timer);
        }
        timer = setTimeout(() => flag.set(false), FLASH_MS);
      }
      seen = n;
    });
    this.destroyRef.onDestroy(() => {
      if (timer) {
        clearTimeout(timer);
      }
    });
  }

  private measure(grid: HTMLElement): void {
    const origin = grid.getBoundingClientRect();
    const map = new Map<NodeId, DOMRect>();
    for (const el of grid.querySelectorAll<HTMLElement>('[data-node]')) {
      const r = el.getBoundingClientRect();
      map.set(el.dataset['node'] as NodeId, new DOMRect(r.left - origin.left, r.top - origin.top, r.width, r.height));
    }
    this.rects.set(map);
  }
}

function last(points: readonly ScalarPoint[] | undefined): number | null {
  const v = points?.at(-1)?.value;
  return v === undefined || !Number.isFinite(v) ? null : v;
}

function formatLoss(v: number): string {
  return v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3);
}

/** A 60×20 polyline through the values, scaled to their own range; empty for fewer than two points. */
function sparkPath(values: readonly number[]): string {
  if (values.length < 2) {
    return '';
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const step = 60 / (values.length - 1);
  return values.map((v, i) => `${(i * step).toFixed(1)},${(18 - ((v - min) / span) * 16).toFixed(1)}`).join(' ');
}

function anchor(r: DOMRect, side: Side): [number, number] {
  switch (side) {
    case 'left':
      return [r.left, r.top + r.height / 2];
    case 'right':
      return [r.right, r.top + r.height / 2];
    case 'top':
      return [r.left + r.width / 2, r.top];
    case 'bottom':
      return [r.left + r.width / 2, r.bottom];
  }
}

function normal(side: Side, k: number): [number, number] {
  switch (side) {
    case 'left':
      return [-k, 0];
    case 'right':
      return [k, 0];
    case 'top':
      return [0, -k];
    case 'bottom':
      return [0, k];
  }
}

function curvePath(a: DOMRect, out: Side, b: DOMRect, into: Side): string {
  const [x0, y0] = anchor(a, out);
  const [x1, y1] = anchor(b, into);
  const k = Math.max(16, Math.hypot(x1 - x0, y1 - y0) * 0.45);
  const [dx0, dy0] = normal(out, k);
  const [dx1, dy1] = normal(into, k);
  return `M ${x0} ${y0} C ${x0 + dx0} ${y0 + dy0}, ${x1 + dx1} ${y1 + dy1}, ${x1} ${y1}`;
}

/** Down from `a`, along the row gap, down into `b`: rounded orthogonal corners. */
function busPath(a: DOMRect, b: DOMRect): string {
  const x0 = a.left + a.width / 2;
  const x1 = b.left + Math.min(28, b.width / 4);
  const midY = (a.bottom + b.top) / 2;
  const r = Math.min(12, (b.top - a.bottom) / 2, (x1 - x0) / 2);
  return `M ${x0} ${a.bottom} V ${midY - r} Q ${x0} ${midY} ${x0 + r} ${midY} H ${x1 - r} Q ${x1} ${midY} ${x1} ${midY + r} V ${b.top}`;
}
