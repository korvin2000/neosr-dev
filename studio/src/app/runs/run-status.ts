import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import type { RunStatus } from '@shared/run.model';
import { Icon } from '../ui/icon';
import { statusView } from './run-format';

/**
 * The one status chip: icon + text in the status colour. `detail` follows the label
 * ("Running · phase 2/3 GAN fine-tune"); `warnings` adds an amber " · n warnings" segment.
 * `reconnecting` overrides an active status while the event stream is down.
 */
@Component({
  selector: 'app-run-status',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  template: `
    <span class="chip" [class]="'chip chip--' + view().tone" [class.chip--md]="size() === 'md'">
      <app-icon [name]="view().icon" [size]="size() === 'md' ? 13 : 12" [weight]="2.2" />
      <span class="chip__text">{{ view().label }}{{ detail() ? ' · ' + detail() : '' }}</span>
      @if (warnings() > 0) {
        <span class="chip__warn">· <app-icon name="alert-triangle" [size]="12" [weight]="2.2" /> {{ warnings() }} warning{{ warnings() === 1 ? '' : 's' }}</span>
      }
    </span>
  `,
  styles: `
    :host { display: inline-flex; min-width: 0; max-width: 100%; }
    .chip { display: inline-flex; align-items: center; gap: 5px; min-width: 0; height: 22px; padding: 0 9px; border-radius: 999px; font-size: var(--fs-xs); font-weight: 600; white-space: nowrap; background: var(--panel-raised); color: var(--text-muted); }
    .chip--md { height: 26px; padding: 0 11px; font-size: var(--fs-sm); }
    .chip__text { overflow: hidden; text-overflow: ellipsis; min-width: 0; }
    .chip__warn { display: inline-flex; align-items: center; gap: 4px; color: var(--warn); flex: none; }
    .chip--running { background: var(--running-soft); color: var(--running); }
    .chip--ok { background: var(--ok-soft); color: var(--ok); }
    .chip--warn { background: var(--warn-soft); color: var(--warn); }
    .chip--danger { background: var(--danger-soft); color: var(--danger); }
  `,
})
export class RunStatusChip {
  readonly status = input.required<RunStatus>();
  readonly detail = input('');
  readonly warnings = input(0);
  readonly reconnecting = input(false);
  readonly size = input<'sm' | 'md'>('sm');

  protected readonly view = computed(() => (this.reconnecting() ? { label: 'Reconnecting…', icon: 'refresh', tone: 'warn' as const } : statusView(this.status())));
}
