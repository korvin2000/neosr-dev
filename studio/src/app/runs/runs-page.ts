import { NgTemplateOutlet } from '@angular/common';
import { ChangeDetectionStrategy, Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import type { RunSummary } from '@shared/run.model';
import { ApiService } from '../core/api.service';
import { ACTIVE_STATUSES, RunStore } from '../core/run-store';
import { Menu, type MenuItem } from '../ui/menu';
import { PageHeader } from '../ui/page-header';
import { ToastService } from '../ui/toast.service';
import { ConfirmDialog } from './confirm-dialog';
import { agoText, approxDuration, formatCompact, formatInt, isResumable, timeAgo } from './run-format';
import { RunStatusChip } from './run-status';

type RowAction = 'open' | 'stop' | 'force' | 'resume' | 'delete';

interface RowView {
  readonly run: RunSummary;
  readonly phase: string;
  readonly progress: number;
  readonly progressText: string;
  readonly activity: string;
  readonly eta: string;
  readonly menu: readonly MenuItem[];
}

const POLL_MS = 5000;
const MAX_COMPARE = 3;

/** Runs list: status, name, phase and progress, last activity, ETA; ⋯ per row; select two or three to compare. */
@Component({
  selector: 'app-runs-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [NgTemplateOutlet, RouterLink, PageHeader, Menu, ConfirmDialog, RunStatusChip],
  templateUrl: './runs-page.html',
  styleUrl: './runs-page.scss',
})
export class RunsPage {
  protected readonly store = inject(RunStore);
  private readonly api = inject(ApiService);
  private readonly router = inject(Router);
  private readonly toasts = inject(ToastService);

  protected readonly offline = signal(false);
  protected readonly loaded = signal(false);
  protected readonly selecting = signal(false);
  protected readonly selected = signal<readonly string[]>([]);
  protected readonly confirming = signal<{ run: RunSummary; action: 'force' | 'delete' } | null>(null);
  private readonly busy = signal<string | null>(null);
  private readonly now = signal(Date.now());

  protected readonly maxCompare = MAX_COMPARE;
  protected readonly headerMenu: readonly MenuItem[] = [
    { id: 'refresh', label: 'Refresh', icon: 'refresh' },
    { id: 'compare', label: 'Compare…', icon: 'compare' },
  ];

  protected readonly rows = computed<RowView[]>(() => {
    const now = this.now();
    const busy = this.busy();
    return this.store.runs().map((run) => ({
      run,
      phase: phaseText(run),
      progress: run.totalIter > 0 ? Math.min(100, (run.iter / run.totalIter) * 100) : 0,
      progressText: `${formatInt(run.iter)} / ${formatInt(run.totalIter)} updates`,
      activity: lastActivity(run, now),
      eta: isActive(run) && run.etaSec > 0 ? `≈ ${approxDuration(run.etaSec)}` : '',
      menu: rowMenu(run, busy === run.id),
    }));
  });

  protected readonly canCompare = computed(() => this.selected().length >= 2 && this.selected().length <= MAX_COMPARE);

  constructor() {
    void this.refresh();
    const timer = setInterval(() => {
      this.now.set(Date.now());
      if (this.store.active().length > 0) {
        void this.refresh();
      }
    }, POLL_MS);
    inject(DestroyRef).onDestroy(() => clearInterval(timer));
  }

  protected async refresh(): Promise<void> {
    try {
      await this.store.refresh();
      this.offline.set(false);
    } catch {
      this.offline.set(true);
    } finally {
      this.loaded.set(true);
    }
  }

  protected onHeaderMenu(id: string): void {
    if (id === 'refresh') {
      void this.refresh();
    } else if (id === 'compare') {
      this.selecting.set(true);
    }
  }

  protected cancelSelect(): void {
    this.selecting.set(false);
    this.selected.set([]);
  }

  protected toggle(id: string): void {
    this.selected.update((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : ids.length < MAX_COMPARE ? [...ids, id] : ids));
  }

  protected compare(): void {
    if (this.canCompare()) {
      void this.router.navigate(['/compare'], { queryParams: { runs: this.selected().join(',') } });
    }
  }

  protected onRowMenu(run: RunSummary, id: string): void {
    const action = id as RowAction;
    if (action === 'open') {
      void this.router.navigate(['/runs', run.id]);
    } else if (action === 'force' || action === 'delete') {
      this.confirming.set({ run, action });
    } else {
      void this.act(run, action);
    }
  }

  protected confirmText(c: { run: RunSummary; action: 'force' | 'delete' }): { heading: string; body: string; label: string } {
    if (c.action === 'delete') {
      return { heading: `Delete ${c.run.name}?`, body: 'Removes the run and its history from Studio. Files NeoSR wrote are not touched.', label: 'Delete' };
    }
    const last = c.run.lastCheckpointIter;
    const loss = last !== undefined ? `progress since ${formatCompact(last)} is lost` : 'all progress is lost';
    return { heading: `Force stop ${c.run.name}?`, body: `Stops now without saving a checkpoint; ${loss}.`, label: 'Force stop' };
  }

  protected async confirmed(): Promise<void> {
    const c = this.confirming();
    this.confirming.set(null);
    if (c) {
      await this.act(c.run, c.action);
    }
  }

  private async act(run: RunSummary, action: Exclude<RowAction, 'open'>): Promise<void> {
    this.busy.set(run.id);
    try {
      switch (action) {
        case 'stop':
          await this.api.stopRun(run.id, false);
          this.toasts.show('info', `${run.name}: saving a checkpoint, then stopping.`);
          break;
        case 'force':
          await this.api.stopRun(run.id, true);
          this.toasts.show('warn', `${run.name}: force-stopped without a checkpoint.`);
          break;
        case 'resume':
          await this.api.resumeRun(run.id);
          this.toasts.show('ok', `${run.name}: resuming from ${formatCompact(run.lastCheckpointIter ?? 0)}.`);
          break;
        case 'delete':
          await this.api.deleteRun(run.id);
          this.selected.update((ids) => ids.filter((x) => x !== run.id));
          this.toasts.show('info', `Deleted ${run.name}.`);
          break;
      }
    } catch {
      this.toasts.show('danger', `${run.name}: the backend rejected the request or is not reachable.`);
    } finally {
      this.busy.set(null);
      await this.refresh();
    }
  }
}

function isActive(run: RunSummary): boolean {
  return ACTIVE_STATUSES.includes(run.status);
}

function phaseText(run: RunSummary): string {
  const i = Math.min(run.phaseIndex, run.phases.length - 1);
  const phase = run.phases[i];
  if (!phase) {
    return '';
  }
  return run.phases.length > 1 ? `${i + 1}/${run.phases.length} ${phase.name}` : phase.name;
}

/** One line on what last happened; for a live run the validation age is derived from the update rate (≈). */
function lastActivity(run: RunSummary, now: number): string {
  switch (run.status) {
    case 'queued':
      return 'queued';
    case 'starting':
      return 'starting';
    case 'completed':
      return `finished ${timeAgo(run.finishedAt, now)}`;
    case 'failed':
      return `failed ${timeAgo(run.finishedAt, now)}`;
  }
  if (isActive(run) && run.lastValidationIter !== undefined && run.itersPerSec > 0) {
    const sec = (run.iter - run.lastValidationIter) / run.itersPerSec;
    return sec < 60 ? 'validated just now' : `validated ≈ ${agoText(sec)}`;
  }
  return run.lastCheckpointIter !== undefined ? `checkpoint ${formatCompact(run.lastCheckpointIter)}` : 'no checkpoint yet';
}

function rowMenu(run: RunSummary, busy: boolean): MenuItem[] {
  const active = isActive(run);
  const saving = run.status === 'stopping' || run.status === 'saving';
  const items: MenuItem[] = [{ id: 'open', label: 'Open', icon: 'arrow' }];
  if (active) {
    items.push({ id: 'stop', label: 'Stop & save', icon: 'stop', disabled: busy || saving }, { id: 'force', label: 'Force stop…', icon: 'power', danger: true, disabled: busy });
  }
  if (isResumable(run.status)) {
    items.push({ id: 'resume', label: 'Resume', icon: 'play', disabled: busy });
  }
  if (!active) {
    items.push({ id: 'delete', label: 'Delete…', icon: 'trash', danger: true, divider: true, disabled: busy });
  }
  return items;
}
