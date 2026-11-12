import { ChangeDetectionStrategy, Component, computed, effect, inject, input, signal, untracked } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { nodeSpec } from '@shared/catalog/node-catalog';
import type { ImportResult } from '@shared/compile/import-toml';
import type { Problem } from '@shared/compile/problems';
import { formatValue, type TomlOrigin } from '@shared/compile/toml-writer';
import * as commands from '../../core/commands';
import { RecipeStore } from '../../core/recipe-store';
import { SettingsService } from '../../core/settings.service';
import { copyText, downloadText } from '../../library/util';
import { Drawer } from '../../ui/drawer';
import { Icon } from '../../ui/icon';
import { Menu, type MenuItem } from '../../ui/menu';
import { SectionEditor } from '../../ui/section-editor';
import { TomlCode } from '../../ui/toml-code';
import { ToastService } from '../../ui/toast.service';
import { Tooltip } from '../../ui/tooltip';
import { TomlImport } from './toml-import';
import { flattenExtras, lineDiff, removeExtra, runScript, type ExtraEntry } from './toml-text';

interface Snapshot {
  readonly recipeId: string;
  readonly at: string;
  readonly source: 'saved' | 'pinned';
  readonly texts: readonly string[];
}

type Panel = 'import' | 'diff' | 'extras';

/** Survives switching between List / Graph / TOML (the component is recreated each time). */
const snapshot = signal<Snapshot | null>(null);

const DIFF_ROWS = 200;

/**
 * Build › TOML: the compiled config full width, phase tabs only when there are several, and a
 * small toolbar. Section headers carry an "Edit" link that opens the section editor in a drawer
 * over the code, and so does an issue link, at the offending setting; everything else (import,
 * diff, unmapped settings) lives behind ⋯.
 */
@Component({
  selector: 'app-toml-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, Menu, Drawer, Tooltip, TomlCode, TomlImport, SectionEditor],
  templateUrl: './toml-page.html',
  styleUrl: './toml-page.scss',
})
export class TomlPage {
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);
  private readonly toasts = inject(ToastService);
  private readonly settings = inject(SettingsService);
  protected readonly store = inject(RecipeStore);

  /** Query params bound by the router (`withComponentInputBinding`): an issue link from the Build header. */
  readonly node = input<string>();
  readonly field = input<string>();

  protected readonly phases = computed(() => this.store.compiled().phases);
  private readonly selected = signal(0);
  protected readonly activeIndex = computed(() => Math.max(0, Math.min(this.selected(), this.phases().length - 1)));
  protected readonly active = computed(() => this.phases()[this.activeIndex()] ?? null);

  // --- source map ------------------------------------------------------------------
  protected readonly hover = signal<TomlOrigin | null>(null);
  protected readonly editNode = signal<string | null>(null);
  /** The setting the drawer scrolls to and flashes; set by an issue link. */
  protected readonly editField = signal<string | null>(null);
  protected readonly highlight = computed(() => this.editNode() ?? this.hover()?.nodeId ?? null);
  /** Origins of header lines, by reference: only those open the drawer; pair lines just select text. */
  private readonly headerOrigins = computed(() => new Set((this.active()?.lines ?? []).filter((l) => l.kind === 'header' && l.origin).map((l) => l.origin)));
  protected readonly editLabel = computed(() => {
    const node = this.editNode() ? this.store.node(this.editNode()!) : undefined;
    return node ? node.title || nodeSpec(node.type).label : '';
  });

  // --- issues ------------------------------------------------------------------------
  protected readonly issuesOpen = signal(false);
  protected readonly problems = computed<readonly Problem[]>(() => {
    const phaseId = this.active()?.phase.id;
    return this.store.problems().filter((p) => p.severity !== 'info' && (!p.phaseId || p.phaseId === phaseId));
  });

  // --- panels ----------------------------------------------------------------------------
  protected readonly panel = signal<Panel | null>(null);
  protected readonly snapshot = computed(() => {
    const s = snapshot();
    return s && s.recipeId === this.store.recipe().id ? s : null;
  });
  protected readonly diff = computed(() => {
    const snap = this.snapshot();
    const phase = this.active();
    if (!snap || !phase) {
      return null;
    }
    const rows = lineDiff((snap.texts[phase.index] ?? '').split('\n'), phase.text.split('\n'));
    return {
      rows: rows.slice(0, DIFF_ROWS),
      hidden: Math.max(0, rows.length - DIFF_ROWS),
      added: rows.filter((r) => r.kind === 'add').length,
      removed: rows.filter((r) => r.kind === 'del').length,
      changed: rows.filter((r) => r.kind === 'chg').length,
      newPhase: snap.texts[phase.index] === undefined,
    };
  });
  protected readonly extras = computed<readonly (ExtraEntry & { key: string; text: string })[]>(() =>
    flattenExtras(this.store.recipe().nativeExtras).map((e) => ({ ...e, key: e.path.join('.'), text: formatValue(e.value) })),
  );

  protected readonly downloadMenu = computed<MenuItem[]>(() => {
    const phase = this.active();
    const n = this.phases().length;
    return [
      { id: 'phase', label: phase ? `Download ${phase.fileName}` : 'Download', icon: 'file', disabled: !phase },
      { id: 'all', label: n > 1 ? `All ${n} phases + run script ${this.scriptExt()}` : `TOML + run script ${this.scriptExt()}`, icon: 'download', disabled: !phase },
    ];
  });

  protected readonly moreMenu = computed<MenuItem[]>(() => {
    const snap = this.snapshot();
    const extras = this.extras().length;
    return [
      { id: 'import', label: 'Import…', icon: 'import' },
      { id: 'pin', label: 'Pin snapshot', icon: 'bookmark', divider: true, disabled: !this.active() },
      { id: 'diff', label: snap ? `Diff vs ${snap.source === 'saved' ? 'saved' : 'pinned'} snapshot` : 'Diff vs snapshot', icon: 'compare', disabled: !snap || !this.active() },
      { id: 'extras', label: 'Unmapped settings', icon: 'list', hint: String(extras), divider: true },
    ];
  });

  constructor() {
    // Whenever the draft matches its last save, remember that compile as the diff base.
    effect(() => {
      if (!this.store.dirty()) {
        const texts = this.phases().map((p) => p.text);
        const recipeId = this.store.recipe().id;
        untracked(() => snapshot.set({ recipeId, at: new Date().toISOString(), source: 'saved', texts }));
      }
    });
    effect(() => {
      const nodeId = this.node();
      const field = this.field();
      if (!nodeId) {
        return;
      }
      untracked(() => {
        // The Build header has already chosen the phase to edit: show that phase's TOML under the drawer.
        const tab = this.phases().findIndex((p) => p.phase.id === this.store.activePhaseId());
        if (tab >= 0) {
          this.selected.set(tab);
        }
        this.inspect(nodeId, field ?? null);
        // Clear the params so the same problem can be focused again later.
        void this.router.navigate([], { relativeTo: this.route, queryParams: {}, replaceUrl: true });
      });
    });
  }

  protected select(index: number): void {
    this.selected.set(index);
  }

  protected onOriginClick(origin: TomlOrigin): void {
    if (this.headerOrigins().has(origin) && this.store.node(origin.nodeId)) {
      // The drawer must edit the phase whose TOML is on screen.
      this.store.activePhaseId.set(this.phases().length > 1 ? (this.active()?.phase.id ?? null) : null);
      this.inspect(origin.nodeId, null);
    }
  }

  /** An issue opens the inspector over the code, on the node that owns it and at its setting. */
  protected focusProblem(problem: Problem): void {
    this.issuesOpen.set(false);
    // A shared-value problem must be edited in the shared view, not in whatever phase was active.
    this.store.activePhaseId.set(problem.phaseId ?? null);
    if (problem.nodeId) {
      this.inspect(problem.nodeId, problem.fieldKey ?? null);
    }
  }

  protected closeInspector(): void {
    this.editNode.set(null);
    // Reset, so the same issue focuses its setting again next time.
    this.editField.set(null);
  }

  private inspect(nodeId: string, field: string | null): void {
    if (this.store.node(nodeId)) {
      this.editNode.set(nodeId);
      this.editField.set(field);
    }
  }

  // --- toolbar -------------------------------------------------------------------------

  protected copyToml(): void {
    const phase = this.active();
    if (phase) {
      void this.copy(phase.text, phase.fileName);
    }
  }

  protected copyCommand(): void {
    const commandsText = this.store.compiled().commands;
    void this.copy(commandsText.join('\n'), commandsText.length > 1 ? `${commandsText.length} commands` : 'the command');
  }

  protected onDownload(id: string): void {
    const phase = this.active();
    if (!phase) {
      return;
    }
    if (id === 'phase') {
      downloadText(phase.fileName, phase.text, 'application/toml');
      return;
    }
    for (const p of this.phases()) {
      downloadText(p.fileName, p.text, 'application/toml');
    }
    const s = this.settings.settings();
    const script = runScript(this.phases(), { platform: s.platform, recipeName: this.store.recipe().name, neosrRoot: s.neosrRoot, optionsDir: s.optionsDir });
    downloadText(script.fileName, script.text);
  }

  protected onMore(id: string): void {
    switch (id) {
      case 'import':
      case 'diff':
      case 'extras':
        this.panel.set(id);
        break;
      case 'pin':
        snapshot.set({ recipeId: this.store.recipe().id, at: new Date().toISOString(), source: 'pinned', texts: this.phases().map((p) => p.text) });
        this.toasts.show('info', 'Pinned the current TOML; later edits show as a diff.');
        break;
    }
  }

  protected scriptExt(): string {
    return this.settings.settings().platform === 'windows' ? '.ps1' : '.sh';
  }

  // --- panels ---------------------------------------------------------------------------

  protected removeExtra(path: readonly string[]): void {
    this.store.dispatch(commands.setExtras(removeExtra(this.store.recipe().nativeExtras, path)));
  }

  protected replace(result: ImportResult): void {
    if (this.store.dirty() && !confirm(`Replace “${this.store.recipe().name}” (unsaved changes) with the imported recipe?`)) {
      return;
    }
    this.store.load(result.recipe);
    this.selected.set(0);
    this.panel.set(null);
    this.toasts.show('ok', `Imported “${result.recipe.name}”: ${result.recipe.nodes.length} nodes, ${result.preserved.length} keys preserved.`);
  }

  protected time(iso: string): string {
    return new Date(iso).toLocaleTimeString();
  }

  private async copy(text: string, what: string): Promise<void> {
    const ok = await copyText(text);
    this.toasts.show(ok ? 'ok' : 'warn', ok ? `Copied ${what}.` : 'Clipboard not available in this context.');
  }
}
