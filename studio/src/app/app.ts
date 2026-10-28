import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { NavigationError, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { EnvironmentService } from './core/environment.service';
import { RunStore } from './core/run-store';
import { UiModeService } from './core/ui-mode.service';
import { Icon } from './ui/icon';
import { ToastService } from './ui/toast.service';
import { Tooltip } from './ui/tooltip';

interface NavItem {
  readonly path: string;
  readonly label: string;
  readonly icon: string;
  readonly exact: boolean;
}

const NAV: readonly NavItem[] = [
  { path: '/', label: 'Start', icon: 'home', exact: true },
  { path: '/build', label: 'Build', icon: 'layers', exact: false },
  { path: '/runs', label: 'Runs', icon: 'activity', exact: false },
  { path: '/library', label: 'Library', icon: 'book', exact: false },
];

/**
 * Shell: a quiet top bar (brand, simulated marker, environment, settings), a labelled nav rail on
 * desktop or a bottom tab bar on phones, and the routed page. Pages own their own headers.
 */
@Component({
  selector: 'app-root',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterOutlet, RouterLink, RouterLinkActive, Icon, Tooltip],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  protected readonly env = inject(EnvironmentService);
  protected readonly runs = inject(RunStore);
  protected readonly toasts = inject(ToastService);
  protected readonly mode = inject(UiModeService);

  protected readonly nav = NAV;
  protected readonly envOpen = signal(false);
  protected readonly activeRuns = computed(() => this.runs.active().length);
  protected readonly gpuLabel = computed(() => {
    const info = this.env.info();
    if (!info) {
      return this.env.offline() ? 'Backend offline' : '…';
    }
    const gpu = info.gpus[0];
    return gpu ? gpu.name.replace('NVIDIA GeForce ', '') : 'No GPU';
  });

  private readonly router = inject(Router);

  constructor() {
    void this.env.load();
    void this.runs.refresh().catch(() => undefined);
    setInterval(() => void this.runs.refresh().catch(() => undefined), 10_000);
    // A dev server rebuilt under us (or a deploy changed hashes): the lazy chunk of a view no
    // longer exists. Instead of a dead tab, load the target URL fresh — once.
    this.router.events.subscribe((event) => {
      if (event instanceof NavigationError && isChunkFailure(event.error)) {
        const key = 'neosr-studio.reloaded-for-chunk';
        if (sessionStorage.getItem(key) !== event.url) {
          sessionStorage.setItem(key, event.url);
          this.toasts.show('info', 'The app was rebuilt; reloading this view.');
          window.location.assign(event.url);
        }
      }
    });
  }
}

function isChunkFailure(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '');
  return /dynamically imported module|Loading chunk|Outdated Optimize Dep|ChunkLoadError/i.test(message);
}
