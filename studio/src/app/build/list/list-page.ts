import { ChangeDetectionStrategy, Component, computed, effect, inject, input, signal, untracked } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { SECTIONS, findSetting, recipeChanges, sectionProblems, type SectionDef, type SettingHit } from '@shared/sections';
import { DisclosureState } from '../../core/disclosure-state.service';
import { RecipeStore } from '../../core/recipe-store';
import { SettingsService } from '../../core/settings.service';
import { Disclosure } from '../../ui/disclosure';
import { Icon } from '../../ui/icon';
import { SectionEditor } from '../../ui/section-editor';

interface SectionRow {
  readonly section: SectionDef;
  readonly summary: string;
  readonly badge: string;
  readonly badgeKind: 'danger' | 'warn' | 'accent' | 'neutral';
}

interface Focus {
  readonly section: string;
  readonly field: string | null;
}

/**
 * Build › List: the seven sections as summary rows that open into the section editor. A "Find
 * setting" box at the top matches labels and TOML keys and lands on the field. Open state is
 * remembered per section; `?section=&field=` (from the readiness pill) opens and focuses.
 */
@Component({
  selector: 'app-list-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Disclosure, Icon, SectionEditor],
  templateUrl: './list-page.html',
  styleUrl: './list-page.scss',
})
export class ListPage {
  /** Query params bound by the router (`withComponentInputBinding`). */
  readonly section = input<string>();
  readonly field = input<string>();

  private readonly store = inject(RecipeStore);
  private readonly settings = inject(SettingsService);
  private readonly disclosure = inject(DisclosureState);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  protected readonly query = signal('');
  protected readonly focus = signal<Focus | null>(null);
  protected readonly showKeys = computed(() => this.settings.settings().showKeys);
  protected readonly hits = computed<SettingHit[]>(() => findSetting(this.query(), this.store.recipe()));

  protected readonly rows = computed<SectionRow[]>(() => {
    const recipe = this.store.recipe();
    const ctx = this.store.context();
    const problems = this.store.problems();
    const changes = recipeChanges(recipe, ctx);
    return SECTIONS.map((section) => {
      const own = sectionProblems(problems, recipe, section, ctx);
      const blockers = own.filter((p) => p.severity === 'blocker').length;
      const warnings = own.filter((p) => p.severity === 'warning').length;
      const changed = changes.filter((c) => c.section === section.id).length;
      const badge = blockers ? `${blockers} to fix` : warnings ? `${warnings} note${warnings === 1 ? '' : 's'}` : changed ? `${changed} changed` : '';
      return { section, summary: section.summary(recipe, ctx), badge, badgeKind: blockers ? 'danger' : warnings ? 'warn' : 'accent' };
    });
  });

  constructor() {
    effect(() => {
      const section = this.section();
      const field = this.field();
      if (!section) {
        return;
      }
      untracked(() => {
        this.open(section, field ?? null);
        // Clear the params so the same problem can be focused again later.
        void this.router.navigate([], { relativeTo: this.route, queryParams: {}, replaceUrl: true });
      });
    });
  }

  protected isOpen(id: string): boolean {
    return this.disclosure.isOpen(`section:${id}`, this.settings.settings().openSections);
  }

  protected setOpen(id: string, open: boolean): void {
    this.disclosure.set(`section:${id}`, open);
  }

  protected focusFor(id: string): string | null {
    const f = this.focus();
    return f && f.section === id ? f.field : null;
  }

  protected pick(hit: SettingHit): void {
    if (hit.inMore) {
      this.disclosure.set(`more:${hit.section.id}:${hit.part.nodeType}`, true);
    }
    this.open(hit.section.id, hit.field.key);
    this.query.set('');
  }

  protected onKey(event: KeyboardEvent): void {
    if (event.key === 'Enter' && this.hits().length) {
      event.preventDefault();
      this.pick(this.hits()[0]);
    } else if (event.key === 'Escape') {
      this.query.set('');
    }
  }

  private open(section: string, field: string | null): void {
    if (!SECTIONS.some((s) => s.id === section)) {
      return;
    }
    this.disclosure.set(`section:${section}`, true);
    this.focus.set({ section, field });
  }
}
