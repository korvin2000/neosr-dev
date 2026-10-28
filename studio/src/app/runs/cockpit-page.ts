import { NgTemplateOutlet } from '@angular/common';
import { ChangeDetectionStrategy, Component, DestroyRef, ElementRef, computed, effect, inject, input, linkedSignal, signal, untracked } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { newId } from '@shared/recipe.model';
import { RecipeStore } from '../core/recipe-store';
import { ACTIVE_STATUSES, type LiveRun, RunStore } from '../core/run-store';
import { SettingsService } from '../core/settings.service';
import { UiModeService } from '../core/ui-mode.service';
import { Drawer } from '../ui/drawer';
import { Icon } from '../ui/icon';
import { Menu, type MenuItem } from '../ui/menu';
import { PageHeader } from '../ui/page-header';
import { ToastService } from '../ui/toast.service';
import { Tooltip } from '../ui/tooltip';
import { ConfirmDialog } from './confirm-dialog';
import { type ActivityFilter, RunActivity } from './run-activity';
import { RunDiagnostics } from './run-diagnostics';
import { copyText, downloadText, formatCompact, phaseClock, phaseLabel, weightsOf } from './run-format';
import { RunOverview } from './run-overview';
import { RunProgress } from './run-progress';
import { RunRecipePanel } from './run-recipe-panel';
import { RunStatusChip } from './run-status';
import { ValidationGallery } from './validation-gallery';

type Tab = 'overview' | 'images' | 'diagnostics' | 'activity';

const TABS: readonly { id: Tab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'images', label: 'Images' },
  { id: 'diagnostics', label: 'Diagnostics' },
  { id: 'activity', label: 'Activity' },
];

type Primary = { kind: 'stop' | 'saving' | 'resume' | 'weights' | 'logs'; label: string; icon: string };

/** Warnings the user has looked at (Activity › Warnings), per run; the status chip counts the rest. */
const SEEN_KEY = 'neosr-studio.run-warnings-seen.v1';

function readSeen(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(SEEN_KEY) ?? '{}') as Record<string, number>;
  } catch {
    return {};
  }
}

/**
 * One run: header with the state-dependent primary action, phase-segmented progress, and four
 * tabs — Overview (the decision tier), Images, Diagnostics and Activity. Config lives in ⋯.
 */
@Component({
  selector: 'app-cockpit-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [NgTemplateOutlet, RouterLink, PageHeader, Menu, Drawer, Icon, Tooltip, ConfirmDialog, RunStatusChip, RunProgress, RunOverview, RunDiagnostics, RunActivity, RunRecipePanel, ValidationGallery],
  templateUrl: './cockpit-page.html',
  styleUrl: './cockpit-page.scss',
  host: { '(document:click)': 'onDocumentClick($event)', '(document:keydown.escape)': 'weightsOpen.set(false)' },
})
export class CockpitPage {
  readonly id = input.required<string>();
  /** `?tab=` keeps the tab across reloads and links (e.g. straight to Activity). */
  readonly tab = input<string | undefined>(undefined);

  private readonly runs = inject(RunStore);
  private readonly recipes = inject(RecipeStore);
  private readonly router = inject(Router);
  private readonly toasts = inject(ToastService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  protected readonly settings = inject(SettingsService);
  protected readonly mode = inject(UiModeService);

  protected readonly tabs = TABS;
  protected readonly live = signal<LiveRun | null>(null);
  protected readonly activeTab = linkedSignal<Tab>(() => TABS.find((t) => t.id === this.tab())?.id ?? 'overview');
  protected readonly activityFilter = signal<ActivityFilter>('all');
  /** `null` follows the newest validation in the gallery. */
  protected readonly selectedVal = signal<number | null>(null);
  protected readonly dialog = signal<'force' | 'fork' | null>(null);
  protected readonly configOpen = signal(false);
  protected readonly weightsOpen = signal(false);
  protected readonly busy = signal(false);
  protected readonly now = signal(Date.now());
  private readonly everConnected = signal(false);
  private readonly seen = signal(0);

  protected readonly run = computed(() => this.live()?.run() ?? null);
  protected readonly status = computed(() => this.run()?.status ?? 'queued');
  protected readonly active = computed(() => ACTIVE_STATUSES.includes(this.status()));
  protected readonly reconnecting = computed(() => this.active() && this.everConnected() && !this.live()?.connected());

  protected readonly statusDetail = computed(() => {
    const run = this.run();
    return run && (this.status() === 'running' || this.status() === 'stopping' || this.status() === 'saving') && run.phases.length > 1 ? phaseLabel(run.phases, run.phaseIndex) : '';
  });
  protected readonly unresolved = computed(() => Math.max(0, (this.run()?.warnings ?? 0) - this.seen()));

  protected readonly native = computed(() => {
    const run = this.run();
    return run ? (this.live()?.nativeIter() ?? phaseClock(run.phases, run.recipe, run.iter).native) : null;
  });

  protected readonly primary = computed<Primary>(() => {
    switch (this.status()) {
      case 'queued':
      case 'starting':
      case 'running':
        return { kind: 'stop', label: 'Stop & save', icon: 'stop' };
      case 'stopping':
      case 'saving':
        return { kind: 'saving', label: 'Saving…', icon: 'save' };
      case 'completed':
        return { kind: 'weights', label: 'Weights…', icon: 'package' };
      case 'failed':
        return { kind: 'logs', label: 'Logs', icon: 'terminal' };
      default:
        return { kind: 'resume', label: 'Resume', icon: 'play' };
    }
  });

  protected readonly weights = computed(() => {
    const run = this.run();
    return run ? weightsOf(run, this.settings.settings().platform) : null;
  });

  private readonly currentToml = computed(() => {
    const run = this.run();
    return run ? (run.tomls[Math.min(run.phaseIndex, run.tomls.length - 1)] ?? null) : null;
  });

  /** What a force stop throws away, said in the menu label itself. */
  protected readonly forceLoss = computed(() => {
    const run = this.run();
    const last = run?.lastCheckpointIter;
    return run && last !== undefined ? `loses ${formatCompact(Math.max(0, run.iter - last))} updates since ${formatCompact(last)}` : 'loses all progress';
  });

  protected readonly menu = computed<MenuItem[]>(() => {
    const run = this.run();
    const items: MenuItem[] = [
      { id: 'config', label: 'Config…', icon: 'code' },
      { id: 'fork', label: 'Fork to recipe', icon: 'fork' },
      { id: 'download', label: 'Download TOML', icon: 'download', hint: run && run.tomls.length > 1 ? `phase ${Math.min(run.phaseIndex, run.tomls.length - 1) + 1}` : undefined },
    ];
    const status = this.status();
    if (this.active() && status !== 'stopping' && status !== 'saving') {
      items.push({ id: 'force', label: `Force stop · ${this.forceLoss()}`, icon: 'power', danger: true, divider: true });
    }
    if (status === 'running') {
      items.push({ id: 'nan', label: 'Simulate non-finite loss', icon: 'zap', hint: 'prototype only', divider: true }, { id: 'disconnect', label: 'Simulate disconnect', icon: 'zap', hint: 'prototype only' });
    }
    return items;
  });

  constructor() {
    effect((onCleanup) => {
      const id = this.id();
      const live = this.runs.live(id);
      untracked(() => {
        this.live.set(live);
        this.selectedVal.set(null);
        this.everConnected.set(false);
        this.seen.set(readSeen()[id] ?? 0);
      });
      void live.open();
      onCleanup(() => live.close());
    });
    effect(() => {
      if (this.live()?.connected()) {
        this.everConnected.set(true);
      }
    });
    // Looking at the warnings resolves them for the status chip.
    effect(() => {
      const warnings = this.run()?.warnings ?? 0;
      if (this.activeTab() === 'activity' && this.activityFilter() === 'warning' && warnings > untracked(this.seen)) {
        this.markSeen(warnings);
      }
    });
    const timer = setInterval(() => this.now.set(Date.now()), 15_000);
    inject(DestroyRef).onDestroy(() => clearInterval(timer));
  }

  protected setTab(tab: Tab): void {
    this.activeTab.set(tab);
    void this.router.navigate([], { queryParams: { tab: tab === 'overview' ? null : tab }, queryParamsHandling: 'merge', replaceUrl: true });
  }

  protected showWarnings(): void {
    this.activityFilter.set('warning');
    this.setTab('activity');
  }

  protected onPrimary(): void {
    switch (this.primary().kind) {
      case 'stop':
        void this.control(async (live) => {
          await live.stop(false);
          this.toasts.show('info', 'Saving a checkpoint, then stopping.');
        });
        break;
      case 'resume':
        // Resume reopens the stream on purpose: not a reconnect.
        this.everConnected.set(false);
        void this.control(async (live) => {
          await live.resume();
          this.toasts.show('ok', `Resuming from ${formatCompact(this.run()?.lastCheckpointIter ?? 0)}.`);
        });
        break;
      case 'weights':
        this.weightsOpen.update((v) => !v);
        break;
      case 'logs':
        this.activityFilter.set('log');
        this.setTab('activity');
        break;
    }
  }

  protected onMenu(id: string): void {
    switch (id) {
      case 'config':
        this.configOpen.set(true);
        break;
      case 'fork':
        if (this.recipes.dirty()) {
          this.dialog.set('fork');
        } else {
          this.fork();
        }
        break;
      case 'download': {
        const toml = this.currentToml();
        if (toml) {
          downloadText(toml.name, toml.text);
        }
        break;
      }
      case 'force':
        this.dialog.set('force');
        break;
      case 'nan':
      case 'disconnect':
        void this.control(async (live) => {
          await live.chaos(id);
          this.toasts.show('info', id === 'disconnect' ? 'Simulated a trainer disconnect.' : 'Injected a non-finite loss.');
        });
        break;
    }
  }

  protected forceStop(): void {
    this.dialog.set(null);
    void this.control(async (live) => {
      await live.stop(true);
      this.toasts.show('warn', 'Force-stopped without a checkpoint.');
    });
  }

  protected fork(): void {
    const run = this.run();
    this.dialog.set(null);
    if (!run) {
      return;
    }
    const now = new Date().toISOString();
    this.recipes.load({ ...run.recipe, id: newId('r'), name: `${run.name}-fork`, meta: { ...run.recipe.meta, createdAt: now, updatedAt: now } });
    this.toasts.show('ok', `Forked ${run.name} into a new recipe.`);
    void this.router.navigate(['/build/list']);
  }

  protected async copy(text: string, what: string): Promise<void> {
    const ok = await copyText(text);
    this.toasts.show(ok ? 'ok' : 'warn', ok ? `Copied ${what}.` : 'Clipboard not available here.');
  }

  protected onDocumentClick(event: Event): void {
    if (this.weightsOpen() && !this.host.nativeElement.querySelector('.weights')?.contains(event.target as Node)) {
      this.weightsOpen.set(false);
    }
  }

  private markSeen(count: number): void {
    this.seen.set(count);
    try {
      localStorage.setItem(SEEN_KEY, JSON.stringify({ ...readSeen(), [this.id()]: count }));
    } catch {
      // Private mode: the chip simply counts again next time.
    }
  }

  private async control(action: (live: LiveRun) => Promise<void>): Promise<void> {
    const live = this.live();
    if (!live || this.busy()) {
      return;
    }
    this.busy.set(true);
    try {
      await action(live);
      void this.runs.refresh().catch(() => undefined);
    } catch {
      this.toasts.show('danger', 'The backend rejected the request or is not reachable.');
    } finally {
      this.busy.set(false);
    }
  }
}
