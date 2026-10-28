import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { ARCH_BY_TYPE } from '@shared/catalog/archs';
import { TEMPLATE_INFOS } from '@shared/templates';
import { RecipeStore } from '../core/recipe-store';
import { RunStore } from '../core/run-store';
import { relativeTime } from '../library/util';
import { LOOKS, estimateDuration, sizeLabel, type Look } from '../setup/looks';
import { Icon } from '../ui/icon';
import { ImportSheet } from './import-sheet';

interface OutcomeCard {
  readonly look: Look;
  readonly meta: string;
}

/**
 * Start: the three outcomes, the current draft, what is running, and two links. The card is the
 * goal; Setup only asks for data.
 */
@Component({
  selector: 'app-start-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, Icon, ImportSheet],
  templateUrl: './start-page.html',
  styleUrl: './start-page.scss',
})
export class StartPage {
  private readonly store = inject(RecipeStore);
  private readonly runs = inject(RunStore);
  private readonly router = inject(Router);

  protected readonly importOpen = signal(false);

  protected readonly cards: readonly OutcomeCard[] = LOOKS.map((look) => {
    const t = TEMPLATE_INFOS.find((x) => x.id === look.templateId);
    return { look, meta: t ? `${ARCH_BY_TYPE.get(t.arch)?.label ?? t.arch} · ${sizeLabel(t.arch).toLowerCase()} · ${estimateDuration(t.arch, t.totalIter)} on this GPU` : '' };
  });

  protected readonly draft = computed(() => {
    const r = this.store.recipe();
    // A template that was only loaded is nothing to continue; edits or a save make it one.
    const touched = this.store.dirty() || Date.parse(r.meta.updatedAt) - Date.parse(r.meta.createdAt) > 5000;
    return r.nodes.length && touched ? { name: r.name, nodes: r.nodes.length, preset: r.meta.presetLabel ?? 'custom', edited: relativeTime(r.meta.updatedAt) } : null;
  });

  protected readonly active = computed(() => this.runs.active().map((r) => ({ ...r, pct: r.totalIter ? Math.min(100, (100 * r.iter) / r.totalIter) : 0, phase: r.phases[r.phaseIndex]?.name ?? '' })));

  protected pick(card: OutcomeCard): void {
    if (this.store.dirty() && !confirm(`Replace the current draft "${this.store.recipe().name}"? Unsaved changes will be lost.`)) {
      return;
    }
    this.store.loadTemplate(card.look.templateId);
    void this.router.navigate(['/setup/data']);
  }
}
