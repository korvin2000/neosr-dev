import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, HostListener, computed, inject, signal } from '@angular/core';
import { NavigationEnd, Router, RouterLink, RouterOutlet } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter, map } from 'rxjs';
import type { Problem } from '@shared/compile/problems';
import { newId, type Recipe } from '@shared/recipe.model';
import { recipeChanges, sectionForField } from '@shared/sections';
import { ApiService, LaunchRejected } from '../core/api.service';
import { RecipeStore } from '../core/recipe-store';
import { RunStore } from '../core/run-store';
import { SettingsService } from '../core/settings.service';
import { UiModeService } from '../core/ui-mode.service';
import { Icon } from '../ui/icon';
import { Menu, type MenuItem } from '../ui/menu';
import { PageHeader } from '../ui/page-header';
import { ToastService } from '../ui/toast.service';
import { Tooltip } from '../ui/tooltip';

type View = 'list' | 'graph' | 'toml';

/**
 * Build: one screen, three views of the same recipe (List · Graph · TOML). The header carries the
 * name, the readiness pill and the single primary action; everything secondary is in ⋯.
 */
@Component({
  selector: 'app-build-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterOutlet, RouterLink, Icon, Menu, PageHeader, Tooltip, DecimalPipe],
  templateUrl: './build-page.html',
  styleUrl: './build-page.scss',
})
export class BuildPage {
  protected readonly store = inject(RecipeStore);
  protected readonly mode = inject(UiModeService);
  private readonly api = inject(ApiService);
  private readonly settings = inject(SettingsService);
  private readonly runs = inject(RunStore);
  private readonly toasts = inject(ToastService);
  private readonly router = inject(Router);

  protected readonly issuesOpen = signal(false);
  protected readonly launching = signal(false);
  protected readonly derived = this.store.derived;
  protected readonly counts = this.store.problemCounts;

  protected readonly view = toSignal(
    this.router.events.pipe(
      filter((e): e is NavigationEnd => e instanceof NavigationEnd),
      map((e) => viewOf(e.urlAfterRedirects)),
    ),
    { initialValue: viewOf(this.router.url) },
  );

  protected readonly problems = computed(() => [...this.store.problems()].sort((a, b) => rank(a) - rank(b)));
  protected readonly blockers = computed(() => this.problems().filter((p) => p.severity === 'blocker'));
  protected readonly warnings = computed(() => this.problems().filter((p) => p.severity === 'warning'));
  protected readonly changes = computed(() => recipeChanges(this.store.recipe(), this.store.context()));
  protected readonly presetLabel = computed(() => {
    const { name, meta } = this.store.recipe();
    // A loaded template names the recipe after itself; repeating that under the title says nothing.
    if (meta.presetLabel && meta.presetLabel !== name) {
      return meta.presetLabel;
    }
    return meta.templateId || meta.presetLabel ? 'from template' : 'custom';
  });
  protected readonly runsLabel = computed(() => {
    const n = this.store.compiled().phases.length;
    return n > 1 ? `${n} sequential runs` : '';
  });

  protected readonly launchLabel = computed(() => {
    const b = this.counts().blocker;
    if (b > 0) {
      return `Launch · ${b} to fix`;
    }
    const phases = this.store.compiled().phases.length;
    return phases > 1 ? `Launch ${phases} runs` : 'Launch';
  });

  protected readonly menu = computed<MenuItem[]>(() => [
    { id: 'undo', label: 'Undo', icon: 'undo', hint: 'Ctrl+Z', disabled: !this.store.canUndo() },
    { id: 'redo', label: 'Redo', icon: 'redo', hint: 'Ctrl+Y', disabled: !this.store.canRedo() },
    { id: 'save', label: 'Save to library', icon: 'save', hint: 'Ctrl+S', divider: true },
    { id: 'template', label: 'Save as template', icon: 'bookmark' },
    { id: 'rename', label: 'Rename…', icon: 'tag' },
    { id: 'new', label: 'New blank recipe', icon: 'file', divider: true },
  ]);

  @HostListener('window:keydown', ['$event'])
  protected onKey(event: KeyboardEvent): void {
    const target = event.target as HTMLElement | null;
    const typing = !!target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable);
    if (!(event.ctrlKey || event.metaKey)) {
      return;
    }
    const key = event.key.toLowerCase();
    if (key === 'z' && !event.shiftKey && !typing) {
      event.preventDefault();
      this.store.undo();
    } else if ((key === 'y' || (key === 'z' && event.shiftKey)) && !typing) {
      event.preventDefault();
      this.store.redo();
    } else if (key === 's') {
      event.preventDefault();
      void this.save();
    }
  }

  protected onMenu(id: string): void {
    switch (id) {
      case 'undo':
        this.store.undo();
        break;
      case 'redo':
        this.store.redo();
        break;
      case 'save':
        void this.save();
        break;
      case 'template':
        void this.saveAsTemplate();
        break;
      case 'rename': {
        const name = window.prompt('Recipe name', this.store.recipe().name);
        if (name?.trim()) {
          this.rename(name.trim());
        }
        break;
      }
      case 'new':
        if (!this.store.dirty() || window.confirm('Discard unsaved changes and start a blank recipe?')) {
          this.store.newBlank();
        }
        break;
    }
  }

  protected rename(name: string): void {
    if (name && name !== this.store.recipe().name) {
      this.store.dispatch({ label: 'Rename recipe', apply: (r) => ({ ...r, name }) });
    }
  }

  protected async save(): Promise<void> {
    this.store.saving.set(true);
    const hash = this.store.hash();
    try {
      await this.api.saveRecipe(this.store.recipe());
      this.store.markSaved(hash);
      this.toasts.show('ok', `Saved "${this.store.recipe().name}".`);
    } catch {
      this.toasts.show('danger', 'Could not save: is the backend running? (npm run api)');
    } finally {
      this.store.saving.set(false);
    }
  }

  private async saveAsTemplate(): Promise<void> {
    const name = window.prompt('Template name', `${this.store.recipe().name} template`);
    if (!name?.trim()) {
      return;
    }
    const base = this.store.recipe();
    const copy: Recipe = { ...base, id: newId('r'), name: name.trim(), meta: { ...base.meta, tags: [...new Set([...base.meta.tags, 'template'])] } };
    try {
      await this.api.saveRecipe(copy);
      this.toasts.show('ok', `Template "${copy.name}" saved to the library.`);
    } catch {
      this.toasts.show('danger', 'Could not save: is the backend running?');
    }
  }

  protected async launch(): Promise<void> {
    if (this.blockers().length) {
      this.focusProblem(this.blockers()[0]);
      return;
    }
    if (this.launching()) {
      return;
    }
    this.launching.set(true);
    try {
      const run = await this.api.launch(this.store.recipe(), this.settings.settings().platform, this.settings.settings().simSpeed);
      await this.runs.refresh().catch(() => undefined);
      await this.router.navigate(['/runs', run.id]);
    } catch (error) {
      if (error instanceof LaunchRejected) {
        this.toasts.show('danger', error.problems[0]?.message ?? 'The backend rejected the recipe.');
        this.issuesOpen.set(true);
      } else {
        this.toasts.show('danger', 'Could not launch: is the backend running? (npm run api)');
      }
    } finally {
      this.launching.set(false);
    }
  }

  /** Jump to the field that owns a problem, in whichever view the user is in. */
  protected focusProblem(problem: Problem): void {
    this.issuesOpen.set(false);
    // A shared-value problem must be edited in the shared view, not in whatever phase was active.
    this.store.activePhaseId.set(problem.phaseId ?? null);
    if (!problem.nodeId) {
      return;
    }
    const node = this.store.node(problem.nodeId);
    const ctx = this.store.context();
    const wired = node && Object.values(ctx).some((n) => n?.id === node.id);
    // The List edits the nodes wired into the plan; a disconnected duplicate is only reachable on the canvas.
    if (this.view() === 'graph' || !wired) {
      this.store.selectOne(problem.nodeId);
      void this.router.navigate(['/build/graph'], { queryParams: { node: problem.nodeId, field: problem.fieldKey ?? null } });
      return;
    }
    const section = node ? sectionForField(node.type, problem.fieldKey) : undefined;
    void this.router.navigate(['/build/list'], { queryParams: { section: section?.id ?? null, node: problem.nodeId, field: problem.fieldKey ?? null } });
  }
}

function viewOf(url: string): View {
  return url.includes('/build/graph') ? 'graph' : url.includes('/build/toml') ? 'toml' : 'list';
}

function rank(p: Problem): number {
  return p.severity === 'blocker' ? 0 : p.severity === 'warning' ? 1 : 2;
}
