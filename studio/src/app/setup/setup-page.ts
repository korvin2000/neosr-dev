import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { filter, map } from 'rxjs';
import { RecipeStore } from '../core/recipe-store';
import { PageHeader } from '../ui/page-header';

const BALANCED = '4x-realplksr-balanced';

/** Setup: two screens (Data · Review) in one narrow column under a slim header with progress. */
@Component({
  selector: 'app-setup-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterOutlet, PageHeader],
  template: `
    <app-page-header heading="Set up training" [subtitle]="step() === 'review' ? 'Review · 2 of 2' : 'Data · 1 of 2'" [back]="step() === 'review' ? '/setup/data' : '/'" />
    <div class="content">
      <router-outlet />
    </div>
  `,
  styles: `
    :host { display: flex; flex-direction: column; flex: 1; min-height: 0; overflow: auto; }
    .content { width: 100%; max-width: 688px; margin: 0 auto; padding: 20px 24px 40px; }
    @media (max-width: 720px) { .content { padding: 12px 16px 32px; } }
  `,
})
export class SetupPage {
  private readonly router = inject(Router);
  private readonly store = inject(RecipeStore);

  private readonly url = toSignal(
    this.router.events.pipe(
      filter((e): e is NavigationEnd => e instanceof NavigationEnd),
      map((e) => e.urlAfterRedirects),
    ),
    { initialValue: this.router.url },
  );
  protected readonly step = computed(() => (this.url().includes('/setup/review') ? 'review' : 'data'));

  constructor() {
    // Setup edits a preset; a blank recipe has nothing to review, so it starts from Balanced.
    if (this.store.recipe().nodes.length === 0) {
      this.store.loadTemplate(BALANCED);
    }
  }
}
