/**
 * The one editable Recipe, with history, selection and derived compilation. The Wizard, the Graph
 * and the TOML view all read `recipe()` and write through `dispatch()`.
 */
import { Injectable, computed, effect, inject, signal, untracked } from '@angular/core';
import { compileRecipe, type CompileResult } from '@shared/compile/compile';
import { countBy, type Problem } from '@shared/compile/problems';
import { nodeValues, phaseNodeValues, planContext, scaleOf } from '@shared/compile/graph';
import { nodeSpec } from '@shared/catalog/node-catalog';
import { emptyRecipe, semanticHash, totalIters, type Recipe, type RecipeNode } from '@shared/recipe.model';
import { buildTemplate } from '@shared/templates';
import type { Command } from './commands';
import * as commands from './commands';
import { SettingsService } from './settings.service';

const DRAFT_KEY = 'neosr-studio.draft.v1';
const HISTORY_LIMIT = 200;

export interface DerivedNumbers {
  readonly scale: number;
  readonly patch: number;
  readonly gtCrop: number;
  readonly batch: number;
  readonly accumulate: number;
  readonly effectiveBatch: number;
  readonly totalIter: number;
  readonly phases: number;
  readonly arch: string;
  readonly hasGan: boolean;
  /** Rough VRAM class from arch size, patch and batch: an estimate, labelled as such in the UI. */
  readonly vramClass: 'low' | 'medium' | 'high' | 'very high';
}

@Injectable({ providedIn: 'root' })
export class RecipeStore {
  private readonly settings = inject(SettingsService);

  private readonly restored = restoreDraft();
  readonly recipe = signal<Recipe>(this.restored?.recipe ?? buildTemplate('4x-span-fidelity'));
  private past: { recipe: Recipe; label: string }[] = [];
  private future: { recipe: Recipe; label: string }[] = [];
  readonly canUndo = signal(false);
  readonly canRedo = signal(false);
  readonly undoLabel = signal<string | null>(null);
  readonly redoLabel = signal<string | null>(null);

  /** Hash at the last save; `dirty` compares against it. */
  private readonly savedHash = signal<string | null>(null);
  readonly hash = computed(() => semanticHash(this.recipe()));
  readonly dirty = computed(() => this.savedHash() !== this.hash());
  readonly saving = signal(false);

  // Selection is presentation state and is not part of history.
  readonly selection = signal<ReadonlySet<string>>(new Set());
  readonly selectedEdge = signal<string | null>(null);
  /** `null` = shared values; a phase id = that phase's effective values on the canvas. */
  readonly activePhaseId = signal<string | null>(null);

  readonly compiled = computed<CompileResult>(() => compileRecipe(this.recipe(), { platform: this.settings.settings().platform }));
  readonly problems = computed(() => this.compiled().problems);
  readonly problemCounts = computed(() => countBy(this.problems()));
  readonly problemsByNode = computed(() => {
    const map = new Map<string, Problem[]>();
    for (const p of this.problems()) {
      if (p.nodeId) {
        map.set(p.nodeId, [...(map.get(p.nodeId) ?? []), p]);
      }
    }
    return map;
  });
  readonly launchable = computed(() => this.problemCounts().blocker === 0 && this.compiled().phases.length > 0);

  readonly context = computed(() => planContext(this.recipe()));

  readonly derived = computed<DerivedNumbers>(() => {
    const recipe = this.recipe();
    const ctx = this.context();
    const phaseId = this.activePhaseId() ?? recipe.phases[0]?.id ?? '';
    const data = ctx.data ? phaseNodeValues(ctx.data, phaseId) : {};
    const scale = scaleOf(ctx.model);
    const patch = Number(data['patch_size'] ?? 64);
    const batch = Number(data['batch_size'] ?? 8);
    const accumulate = Number(data['accumulate'] ?? 1);
    const arch = ctx.model ? String(nodeValues(ctx.model)['type']) : '';
    const hasGan = this.compiled().phases.some((p) => p.hasGan);
    return {
      scale,
      patch,
      gtCrop: patch * scale,
      batch,
      accumulate,
      effectiveBatch: batch * accumulate,
      totalIter: totalIters(recipe.phases),
      phases: recipe.phases.length,
      arch,
      hasGan,
      vramClass: vramClass(arch, patch, batch, hasGan),
    };
  });

  readonly selectedNode = computed<RecipeNode | undefined>(() => {
    const ids = [...this.selection()];
    return ids.length === 1 ? this.recipe().nodes.find((n) => n.id === ids[0]) : undefined;
  });

  constructor() {
    // A restored draft keeps the baseline it had; a freshly built template counts as saved.
    untracked(() => this.savedHash.set(this.restored ? this.restored.savedHash : this.hash()));
    effect(() => {
      const recipe = this.recipe();
      const savedHash = this.savedHash();
      try {
        localStorage.setItem(DRAFT_KEY, JSON.stringify({ recipe, savedHash }));
      } catch {
        // Draft autosave is best effort.
      }
    });
  }

  // --- history --------------------------------------------------------------------

  dispatch(command: Command): void {
    const before = this.recipe();
    const after = command.apply(before);
    if (after === before) {
      return;
    }
    this.past.push({ recipe: before, label: command.label });
    if (this.past.length > HISTORY_LIMIT) {
      this.past.shift();
    }
    this.future = [];
    this.recipe.set(after);
    this.pruneSelection(after);
    this.syncHistoryFlags();
  }

  undo(): void {
    const entry = this.past.pop();
    if (!entry) {
      return;
    }
    this.future.push({ recipe: this.recipe(), label: entry.label });
    this.recipe.set(entry.recipe);
    this.pruneSelection(entry.recipe);
    this.syncHistoryFlags();
  }

  redo(): void {
    const entry = this.future.pop();
    if (!entry) {
      return;
    }
    this.past.push({ recipe: this.recipe(), label: entry.label });
    this.recipe.set(entry.recipe);
    this.pruneSelection(entry.recipe);
    this.syncHistoryFlags();
  }

  private syncHistoryFlags(): void {
    this.canUndo.set(this.past.length > 0);
    this.canRedo.set(this.future.length > 0);
    this.undoLabel.set(this.past.at(-1)?.label ?? null);
    this.redoLabel.set(this.future.at(-1)?.label ?? null);
  }

  private pruneSelection(recipe: Recipe): void {
    const ids = new Set(recipe.nodes.map((n) => n.id));
    const current = this.selection();
    if ([...current].some((id) => !ids.has(id))) {
      this.selection.set(new Set([...current].filter((id) => ids.has(id))));
    }
    if (this.activePhaseId() && !recipe.phases.some((p) => p.id === this.activePhaseId())) {
      this.activePhaseId.set(null);
    }
  }

  // --- loading / saving ------------------------------------------------------------

  /** Replace the recipe and reset history (open, new, import, template). */
  load(recipe: Recipe, markSaved = false): void {
    this.past = [];
    this.future = [];
    this.recipe.set(recipe);
    this.selection.set(new Set());
    this.selectedEdge.set(null);
    this.activePhaseId.set(null);
    this.syncHistoryFlags();
    this.savedHash.set(markSaved ? semanticHash(recipe) : null);
  }

  newBlank(name = 'untitled'): void {
    this.load(emptyRecipe(name));
  }

  loadTemplate(id: string): void {
    this.load(buildTemplate(id));
  }

  /** Mark a version as saved: pass the hash captured before the save request so later edits stay dirty. */
  markSaved(hash: string = this.hash()): void {
    this.savedHash.set(hash);
  }

  // --- convenience edits -------------------------------------------------------------

  setValue(nodeId: string, key: string, value: unknown): void {
    const node = this.recipe().nodes.find((n) => n.id === nodeId);
    const label = node ? `${nodeSpec(node.type).label}: ${key}` : 'Edit';
    this.dispatch(commands.setValues(nodeId, { [key]: value }, label));
  }

  setValues(nodeId: string, patch: Record<string, unknown>, label?: string): void {
    this.dispatch(commands.setValues(nodeId, patch, label));
  }

  setPhaseValue(nodeId: string, phaseId: string, key: string, value: unknown): void {
    this.dispatch(commands.setPhaseValues(nodeId, phaseId, { [key]: value }));
  }

  select(ids: Iterable<string>): void {
    this.selection.set(new Set(ids));
    this.selectedEdge.set(null);
  }

  selectOne(id: string | null): void {
    this.select(id ? [id] : []);
  }

  node(id: string): RecipeNode | undefined {
    return this.recipe().nodes.find((n) => n.id === id);
  }
}

function restoreDraft(): { recipe: Recipe; savedHash: string | null } | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as { recipe?: Recipe; savedHash?: string | null } & Partial<Recipe>;
    if (parsed.recipe?.format === 'neosr-studio-recipe') {
      return { recipe: parsed.recipe, savedHash: parsed.savedHash ?? null };
    }
    // Older drafts stored the bare recipe; they are unsaved by definition.
    return parsed.format === 'neosr-studio-recipe' ? { recipe: parsed as Recipe, savedHash: null } : null;
  } catch {
    return null;
  }
}

function vramClass(arch: string, patch: number, batch: number, gan: boolean): DerivedNumbers['vramClass'] {
  const heavy = /^(hat|drct|atd|dat|rgt|hit_srf|hma|swinir_large|grformer|mfghmoe|microsr)/.test(arch) ? 3 : /^(realplksr|plksr|esc|man|rcan|esrgan|desrgan|moesr|mosr|flexnet|srformer|swinir)/.test(arch) ? 2 : 1;
  const pixels = (patch * patch * batch) / (64 * 64 * 8);
  const score = heavy * pixels * (gan ? 1.5 : 1);
  if (score < 1.2) {
    return 'low';
  }
  if (score < 2.5) {
    return 'medium';
  }
  if (score < 5) {
    return 'high';
  }
  return 'very high';
}
