import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import type { Phase, PhaseInit } from '@shared/recipe.model';
import * as commands from '../../core/commands';
import { RecipeStore } from '../../core/recipe-store';
import { Icon } from '../../ui/icon';
import { Tooltip } from '../../ui/tooltip';

const INIT_OPTIONS: readonly { value: PhaseInit; label: string; help: string }[] = [
  { value: 'finetune', label: 'fine-tune from the previous weights', help: 'A new run starts from the previous phase’s net_g_latest.pth; optimizer and warm-up restart.' },
  { value: 'continue', label: 'continue the previous state', help: 'Resumes the previous phase’s .state: counters and optimizer continue, so a new learning rate here would not apply.' },
];

/**
 * Chips for "Shared" and each phase; the active one decides which values the cards show. Only
 * rendered when the recipe has more than one phase. ✎ opens a small popover for the active phase.
 */
@Component({
  selector: 'app-phase-strip',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, Tooltip],
  templateUrl: './phase-strip.html',
  styleUrl: './phase-strip.scss',
  host: { '(document:keydown.escape)': 'editing.set(false)' },
})
export class PhaseStrip {
  protected readonly store = inject(RecipeStore);
  protected readonly editing = signal(false);
  protected readonly initOptions = INIT_OPTIONS;

  protected readonly phases = computed(() => this.store.recipe().phases);
  protected readonly activeIndex = computed(() => this.phases().findIndex((p) => p.id === this.store.activePhaseId()));
  protected readonly active = computed<Phase | null>(() => this.phases()[this.activeIndex()] ?? null);

  protected activate(id: string | null): void {
    this.store.activePhaseId.set(id);
    this.editing.set(false);
  }

  protected update(id: string, patch: Partial<Phase>): void {
    this.store.dispatch(commands.updatePhase(id, patch));
  }

  protected setIters(id: string, raw: string): void {
    const n = Number(raw);
    if (Number.isFinite(n) && n > 0) {
      this.update(id, { iters: Math.round(n) });
    }
  }

  protected remove(phase: Phase): void {
    if (window.confirm(`Delete phase "${phase.name}"? Its per-phase overrides are dropped.`)) {
      this.editing.set(false);
      this.store.dispatch(commands.removePhase(phase.id));
    }
  }

  protected k(n: number): string {
    return n >= 1000 ? `${Math.round(n / 100) / 10}k` : String(n);
  }
}
