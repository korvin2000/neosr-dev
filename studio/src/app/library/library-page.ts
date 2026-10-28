import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, effect, inject, signal, untracked } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { map } from 'rxjs';
import { ARCH_BY_TYPE } from '@shared/catalog/archs';
import { DEGRADATION_TEMPLATE } from '@shared/catalog/degradations';
import { NODE_TYPES } from '@shared/catalog/node-catalog';
import { slug, valDatasets, type ValDatasetRow } from '@shared/compile/graph';
import { newId, type Recipe, type RecipeNode } from '@shared/recipe.model';
import { sectionForNodeType } from '@shared/sections';
import { TEMPLATE_INFOS, type TemplateInfo } from '@shared/templates';
import { ApiService, type DatasetInfo, type ModelFile, type RecipeSummary } from '../core/api.service';
import * as commands from '../core/commands';
import { RecipeStore } from '../core/recipe-store';
import { LOOKS, fmtUpdates } from '../setup/looks';
import { ScanService, scanIssues } from '../setup/scan.service';
import { Icon } from '../ui/icon';
import { Menu, type MenuItem } from '../ui/menu';
import { PageHeader } from '../ui/page-header';
import { ToastService } from '../ui/toast.service';
import { RemoteList, apiErrorMessage, downloadText, readPickedFile, relativeTime, safeFileName } from './util';

type Tab = 'templates' | 'recipes' | 'models' | 'datasets';

const TABS: readonly { id: Tab; label: string }[] = [
  { id: 'templates', label: 'Templates' },
  { id: 'recipes', label: 'My recipes' },
  { id: 'models', label: 'Models' },
  { id: 'datasets', label: 'Datasets' },
];

const NEW_NODE_AT = { x: 100, y: 100 };

interface TemplateCard {
  readonly info: TemplateInfo;
  readonly look: string;
  readonly speed: string;
  readonly menu: readonly MenuItem[];
}

const TEMPLATE_MENU: readonly MenuItem[] = [
  { id: 'details', label: 'Details', icon: 'info' },
  { id: 'graph', label: 'Open in Graph', icon: 'grid' },
];
const RECIPE_MENU: readonly MenuItem[] = [
  { id: 'duplicate', label: 'Duplicate', icon: 'copy' },
  { id: 'export', label: 'Export JSON', icon: 'download' },
  { id: 'delete', label: 'Delete', icon: 'trash', danger: true, divider: true },
];
const DATASET_MENU: readonly MenuItem[] = [
  { id: 'train', label: 'Use for training', icon: 'database' },
  { id: 'val', label: 'Use for validation', icon: 'check-image' },
  { id: 'check', label: 'Check (simulated)', icon: 'search', divider: true },
];

/** Library: templates, saved recipes, checkpoints and datasets. Cards say one thing; details sit in ⋯. */
@Component({
  selector: 'app-library-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, Icon, Menu, PageHeader, DecimalPipe],
  templateUrl: './library-page.html',
  styleUrl: './library-page.scss',
})
export class LibraryPage {
  private readonly api = inject(ApiService);
  private readonly router = inject(Router);
  private readonly toasts = inject(ToastService);
  private readonly scans = inject(ScanService);
  protected readonly store = inject(RecipeStore);

  protected readonly tabs = TABS;
  protected readonly tab = toSignal(
    inject(ActivatedRoute).paramMap.pipe(map((p): Tab => (TABS.some((t) => t.id === p.get('tab')) ? (p.get('tab') as Tab) : 'templates'))),
    { initialValue: 'templates' as Tab },
  );
  protected readonly relativeTime = relativeTime;
  protected readonly fmtUpdates = fmtUpdates;

  // --- search and facets -----------------------------------------------------------------------
  protected readonly query = signal('');
  protected readonly scale = signal('');
  protected readonly look = signal('');
  protected readonly speed = signal('');
  protected readonly filterCount = computed(() => [this.scale(), this.look(), this.speed()].filter(Boolean).length);
  protected readonly scales = [...new Set(TEMPLATE_INFOS.map((t) => t.scale))].sort((a, b) => a - b);
  protected readonly looks = LOOKS;
  protected readonly speeds = ['fast', 'medium', 'slow'] as const;

  private readonly words = computed(() => this.query().toLowerCase().split(/\s+/).filter(Boolean));
  private readonly matches = (hay: string) => this.words().every((w) => hay.toLowerCase().includes(w));

  protected readonly templates = computed<TemplateCard[]>(() =>
    TEMPLATE_INFOS.map((info) => ({
      info,
      look: LOOKS.find((l) => l.id === info.objective)?.label ?? info.objective,
      speed: ARCH_BY_TYPE.get(info.arch)?.speed ?? 'medium',
      menu: TEMPLATE_MENU,
    })).filter((c) => {
      const hay = [c.info.label, c.info.description, c.info.arch, c.look, c.info.provenance, `x${c.info.scale}`, `×${c.info.scale}`, ...c.info.tags].join(' ');
      return this.matches(hay) && (!this.scale() || String(c.info.scale) === this.scale()) && (!this.look() || c.look === this.look()) && (!this.speed() || c.speed === this.speed());
    }),
  );
  protected readonly detailsOpen = signal<ReadonlySet<string>>(new Set());

  // --- remote lists ----------------------------------------------------------------------------
  protected readonly recipes = new RemoteList<RecipeSummary>(() => this.api.listRecipes());
  protected readonly models = new RemoteList<ModelFile>(() => this.api.listModels());
  protected readonly datasets = new RemoteList<DatasetInfo>(() => this.api.listDatasets());
  protected readonly busy = signal<string | null>(null);
  protected readonly recipeMenu = RECIPE_MENU;
  protected readonly datasetMenu = DATASET_MENU;

  protected readonly recipeRows = computed(() => this.recipes.data().filter((r) => this.matches([r.name, r.arch, r.description, `x${r.scale}`].join(' '))));
  protected readonly modelRows = computed(() => this.models.data().filter((m) => this.matches([m.file, m.arch, m.runName, m.variant, `x${m.scale}`].join(' '))));
  protected readonly datasetRows = computed(() => this.datasets.data().filter((d) => this.matches([d.name, d.gtPath, d.lqPath ?? '', d.note, d.format].join(' '))));
  protected readonly generator = computed(() => ({ arch: this.store.derived().arch, scale: this.store.derived().scale }));

  constructor() {
    effect(() => {
      const tab = this.tab();
      untracked(() => {
        if (tab === 'recipes') {
          this.recipes.ensure();
        } else if (tab === 'models') {
          this.models.ensure();
        } else if (tab === 'datasets') {
          this.datasets.ensure();
        }
      });
    });
  }

  protected clearFilters(): void {
    this.scale.set('');
    this.look.set('');
    this.speed.set('');
  }

  // --- templates ---------------------------------------------------------------------------------

  protected useTemplate(t: TemplateInfo, target: 'setup' | 'graph' = 'setup'): void {
    if (!this.confirmDiscard()) {
      return;
    }
    this.store.loadTemplate(t.id);
    void this.router.navigate([target === 'setup' ? '/setup/data' : '/build/graph']);
  }

  protected onTemplateMenu(card: TemplateCard, id: string): void {
    if (id === 'graph') {
      this.useTemplate(card.info, 'graph');
    } else if (id === 'details') {
      this.detailsOpen.update((set) => {
        const next = new Set(set);
        if (!next.delete(card.info.id)) {
          next.add(card.info.id);
        }
        return next;
      });
    }
  }

  // --- recipes ---------------------------------------------------------------------------------

  protected async openRecipe(r: RecipeSummary): Promise<void> {
    if (!this.confirmDiscard()) {
      return;
    }
    await this.withRecipe(r, (recipe) => {
      this.store.load(recipe, true);
      void this.router.navigate(['/build']);
    });
  }

  protected async onRecipeMenu(r: RecipeSummary, id: string): Promise<void> {
    switch (id) {
      case 'duplicate':
        await this.withRecipe(r, async (recipe) => {
          const now = new Date().toISOString();
          const copy: Recipe = { ...recipe, id: newId('r'), name: `${recipe.name} copy`, meta: { ...recipe.meta, createdAt: now, updatedAt: now } };
          await this.api.saveRecipe(copy);
          this.toasts.show('ok', `Saved "${copy.name}".`);
          await this.recipes.load();
        });
        break;
      case 'export':
        await this.withRecipe(r, (recipe) => downloadText(`${safeFileName(recipe.name)}.neosr-recipe.json`, JSON.stringify(recipe, null, 2), 'application/json'));
        break;
      case 'delete':
        await this.remove(r);
        break;
    }
  }

  private async remove(r: RecipeSummary): Promise<void> {
    if (!confirm(`Delete the recipe "${r.name}" from the library? Runs made from it are kept.`)) {
      return;
    }
    this.busy.set(r.id);
    try {
      await this.api.deleteRecipe(r.id);
      this.toasts.show('info', `Deleted "${r.name}".`);
      await this.recipes.load();
    } catch (e) {
      this.toasts.show('danger', `Delete failed: ${apiErrorMessage(e)}`);
    } finally {
      this.busy.set(null);
    }
  }

  protected async importJson(event: Event): Promise<void> {
    const file = await readPickedFile(event);
    if (!file) {
      return;
    }
    let recipe: Recipe;
    try {
      recipe = JSON.parse(file.text) as Recipe;
    } catch {
      this.toasts.show('danger', `${file.name} is not JSON.`);
      return;
    }
    if (recipe?.format !== 'neosr-studio-recipe' || !Array.isArray(recipe.nodes) || !Array.isArray(recipe.phases) || typeof recipe.id !== 'string') {
      this.toasts.show('danger', `${file.name} is not a NeoSR Studio recipe.`);
      return;
    }
    try {
      await this.api.saveRecipe(recipe);
      this.toasts.show('ok', `Imported "${recipe.name}" into the library.`);
      await this.recipes.load();
    } catch (e) {
      this.toasts.show('danger', `Import failed: ${apiErrorMessage(e)}`);
    }
  }

  private async withRecipe(r: RecipeSummary, run: (recipe: Recipe) => void | Promise<void>): Promise<void> {
    this.busy.set(r.id);
    try {
      await run(await this.api.getRecipe(r.id));
    } catch (e) {
      this.toasts.show('danger', `"${r.name}": ${apiErrorMessage(e)}`);
    } finally {
      this.busy.set(null);
    }
  }

  // --- models ------------------------------------------------------------------------------------

  protected modelMenu(m: ModelFile): MenuItem[] {
    const g = this.generator();
    const parts: string[] = [];
    if (g.arch && m.arch !== g.arch) {
      parts.push(`arch ${m.arch} ≠ ${g.arch}`);
    }
    if (m.scale !== g.scale) {
      parts.push(`×${m.scale} ≠ ×${g.scale}`);
    }
    const hint = parts.join(', ') || undefined;
    return m.variant === 'state' ? [{ id: 'resume', label: 'Use to resume', icon: 'play', hint }] : [{ id: 'pretrain', label: 'Use as pretrain', icon: 'import', hint }];
  }

  protected useModel(m: ModelFile): void {
    const resume = m.variant === 'state';
    const patch = resume ? { mode: 'resume', resume_state: m.path } : { mode: 'pretrain', pretrain_network_g: m.path };
    const init = this.store.context().init ?? this.store.recipe().nodes.find((n) => n.type === NODE_TYPES.initialization);
    let id = init?.id;
    if (init) {
      this.store.setValues(init.id, patch, resume ? 'Resume from state' : 'Use as pretrain');
    } else {
      this.store.dispatch(commands.addNodeWired(NODE_TYPES.initialization, NEW_NODE_AT, patch));
      id = this.lastNodeOf(NODE_TYPES.initialization)?.id;
    }
    this.toasts.show('ok', `Start from: ${m.file}${resume ? ' (resume)' : ''}.`, this.openBuildAction(NODE_TYPES.initialization, id, resume ? 'resume_state' : 'pretrain_network_g'));
  }

  // --- datasets ----------------------------------------------------------------------------------

  protected onDatasetMenu(d: DatasetInfo, id: string): void {
    if (id === 'train') {
      this.useForTraining(d);
    } else if (id === 'val') {
      this.useForValidation(d);
    } else if (id === 'check') {
      void this.scans.run(this.scanReq(d));
    }
  }

  protected scanReq(d: DatasetInfo) {
    return { gt: d.gtPath, lq: d.lqPath, scale: d.scale ?? this.store.derived().scale, patch: this.store.derived().patch };
  }

  protected scanLine(d: DatasetInfo): { text: string; issues: ReturnType<typeof scanIssues> } | null {
    const state = this.scans.state(this.scanReq(d));
    if (!state) {
      return null;
    }
    if (state.status === 'checking') {
      return { text: 'checking…', issues: [] };
    }
    if (state.status !== 'done') {
      return { text: 'Could not check (backend offline).', issues: [] };
    }
    const r = state.report!;
    const issues = scanIssues(r);
    const count = d.lqPath ? `${r.pairs.toLocaleString()} pairs` : `${r.gtCount.toLocaleString()} images`;
    return { text: `${count} · ${issues.length ? `${issues.length} issue${issues.length === 1 ? '' : 's'}` : 'no issues'} · simulated`, issues };
  }

  private useForTraining(d: DatasetInfo): void {
    const patch = { type: d.lqPath ? 'paired' : 'otf', dataroot_gt: d.gtPath, dataroot_lq: d.lqPath ?? '' };
    const node = this.store.context().data ?? this.store.recipe().nodes.find((n) => n.type === NODE_TYPES.trainDataset);
    let id = node?.id;
    if (node) {
      this.store.setValues(node.id, patch, `Use dataset ${d.name}`);
    } else {
      this.store.dispatch(commands.addNodeWired(NODE_TYPES.trainDataset, NEW_NODE_AT, patch));
      id = this.lastNodeOf(NODE_TYPES.trainDataset)?.id;
    }
    // A GT-only set means on-the-fly degradations, which NeoSR needs as a complete table.
    if (!d.lqPath && !this.store.context().degradation) {
      this.store.dispatch(commands.addNodeWired(NODE_TYPES.degradation, { x: NEW_NODE_AT.x - 320, y: NEW_NODE_AT.y }, { ...DEGRADATION_TEMPLATE }));
    }
    this.toasts.show('ok', `Training data: ${d.name} (${d.lqPath ? 'paired' : 'on the fly'}).`, this.openBuildAction(NODE_TYPES.trainDataset, id, 'dataroot_gt'));
  }

  private useForValidation(d: DatasetInfo): void {
    // A visual-only set reads its images from the LQ folder: for a GT-only dataset that is the GT folder.
    const row: ValDatasetRow = d.lqPath
      ? { id: newId('v'), name: slug(d.name), type: 'paired', dataroot_gt: d.gtPath, dataroot_lq: d.lqPath }
      : { id: newId('v'), name: slug(d.name), type: 'single', dataroot_gt: '', dataroot_lq: d.gtPath };
    const node = this.store.context().val ?? this.store.recipe().nodes.find((n) => n.type === NODE_TYPES.validation);
    let id = node?.id;
    if (node && valDatasets(node).some((r) => r.dataroot_gt === d.gtPath)) {
      this.toasts.show('info', `${d.name} is already a validation set.`);
      return;
    }
    if (node) {
      this.store.setValues(node.id, { datasets: [...valDatasets(node), row] }, `Add validation set ${d.name}`);
    } else {
      this.store.dispatch(commands.addNodeWired(NODE_TYPES.validation, NEW_NODE_AT, { datasets: [row] }));
      id = this.lastNodeOf(NODE_TYPES.validation)?.id;
    }
    this.toasts.show('ok', `Validation: added ${row.type} set "${row.name}".`, this.openBuildAction(NODE_TYPES.validation, id, 'datasets'));
  }

  // --- helpers -------------------------------------------------------------------------------------

  private lastNodeOf(type: string): RecipeNode | undefined {
    return this.store.recipe().nodes.filter((n) => n.type === type).at(-1);
  }

  private openBuildAction(nodeType: string, nodeId: string | undefined, field: string): { label: string; run: () => void } {
    const section = sectionForNodeType(nodeType)?.id ?? null;
    return { label: 'Open in Build', run: () => void this.router.navigate(['/build/list'], { queryParams: { section, node: nodeId ?? null, field } }) };
  }

  private confirmDiscard(): boolean {
    return !this.store.dirty() || confirm(`Replace the current draft "${this.store.recipe().name}"? Unsaved changes will be lost.`);
  }
}
