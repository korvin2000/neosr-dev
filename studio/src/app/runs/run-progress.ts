import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import type { RunSummary } from '@shared/run.model';
import type { CheckpointPoint } from '../core/run-store';
import { Tooltip } from '../ui/tooltip';
import { approxDuration, formatInt } from './run-format';

/** Ticks closer than this (in % of the bar) would merge into a comb; keep every n-th and the last. */
const MAX_TICKS = 24;

/** Run progress: one bar split into named phase segments, checkpoint ticks, and one line of numbers. */
@Component({
  selector: 'app-run-progress',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Tooltip],
  template: `
    <div class="bar">
      @for (s of segments(); track s.id) {
        <div class="seg" [class.seg--active]="s.active" [class.seg--done]="s.done" [style.flex-grow]="s.share">
          @if (segments().length > 1) {
            <span class="seg__name">{{ s.name }}</span>
          }
          <span class="seg__track"><span class="seg__fill" [style.width.%]="s.fill"></span></span>
        </div>
      }
      <div class="ticks" aria-hidden="true">
        @for (t of ticks(); track t.iter) {
          <span class="tick" [class.tick--last]="t.last" [style.left.%]="t.pos"></span>
        }
      </div>
    </div>
    <p class="text">
      <span class="text__main">{{ formatInt(run().iter) }} / {{ formatInt(run().totalIter) }} updates</span>
      @if (active() && run().itersPerSec > 0) {
        <span> · {{ run().itersPerSec.toFixed(1) }} it/s</span>
      }
      @if (active() && run().etaSec > 0) {
        <span> · ETA ≈ {{ eta() }}</span>
      }
      @if (native() !== null && native() !== run().iter) {
        <span class="text__native" appTooltip="NeoSR's own counter in this phase (a fine-tune phase restarts at 0)">native {{ formatInt(native()!) }}</span>
      }
    </p>
  `,
  styles: `
    :host { display: block; }
    .bar { position: relative; display: flex; gap: 3px; padding-bottom: 6px; }
    .seg { display: flex; flex-direction: column; gap: 4px; flex-basis: 0; min-width: 0; }
    .seg__name { font-size: var(--fs-xs); color: var(--text-faint); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .seg--done .seg__name { color: var(--text-muted); }
    .seg--active .seg__name { color: var(--accent-text); font-weight: 600; }
    .seg__track { display: block; height: 6px; border-radius: 3px; background: var(--panel-raised); overflow: hidden; }
    .seg__fill { display: block; height: 100%; background: color-mix(in srgb, var(--accent) 55%, var(--panel-raised)); }
    .seg--active .seg__fill { background: var(--accent); }
    .ticks { position: absolute; left: 0; right: 0; bottom: 0; height: 5px; }
    .tick { position: absolute; top: 0; width: 1px; height: 4px; background: var(--text-faint); opacity: 0.6; }
    .tick--last { height: 5px; width: 2px; background: var(--accent-text); opacity: 1; }
    .text { margin-top: 4px; color: var(--text-muted); font-size: var(--fs-sm); }
    .text__main { color: var(--text); }
    .text__native { margin-left: 10px; color: var(--text-faint); font-size: var(--fs-xs); white-space: nowrap; }
  `,
})
export class RunProgress {
  readonly run = input.required<RunSummary>();
  readonly checkpoints = input<readonly CheckpointPoint[]>([]);
  readonly native = input<number | null>(null);
  readonly active = input(false);

  protected readonly formatInt = formatInt;
  protected readonly eta = computed(() => approxDuration(this.run().etaSec));

  protected readonly segments = computed(() => {
    const run = this.run();
    const total = Math.max(1, run.totalIter);
    const phases = run.phases.length ? run.phases : [{ id: 'all', name: '', start: 0, end: total }];
    const finished = run.status === 'completed';
    return phases.map((p, i) => {
      const span = Math.max(1, p.end - p.start);
      return {
        id: p.id,
        name: phases.length > 1 ? `${i + 1} · ${p.name}` : p.name,
        share: span / total,
        fill: Math.min(100, Math.max(0, ((run.iter - p.start) / span) * 100)),
        active: !finished && i === run.phaseIndex,
        done: finished || i < run.phaseIndex,
      };
    });
  });

  protected readonly ticks = computed(() => {
    const total = Math.max(1, this.run().totalIter);
    const all = this.checkpoints();
    const step = Math.ceil(all.length / MAX_TICKS) || 1;
    return all
      .filter((_, i) => i % step === 0 || i === all.length - 1)
      .map((c, i, list) => ({ iter: c.iter, pos: Math.min(100, (c.iter / total) * 100), last: i === list.length - 1 }));
  });
}
