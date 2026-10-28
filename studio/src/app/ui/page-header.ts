import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Icon } from './icon';

/**
 * Every screen's first row: back link · title · status · primary action · ⋯. Pages project the
 * status chip into `[status]`, buttons into `[actions]`, and an optional second row (tabs, view
 * switch) into `[tabs]`.
 */
@Component({
  selector: 'app-page-header',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, Icon],
  template: `
    <div class="row">
      @if (back()) {
        <a class="back" [routerLink]="back()" aria-label="Back"><app-icon name="arrow-left" [size]="18" /></a>
      }
      <div class="title">
        <ng-content select="[title]" />
        @if (heading()) {
          <h1 class="heading">{{ heading() }}</h1>
        }
        @if (subtitle()) {
          <p class="subtitle">{{ subtitle() }}</p>
        }
      </div>
      <ng-content select="[status]" />
      <span class="spacer"></span>
      <div class="actions">
        <ng-content select="[actions]" />
      </div>
    </div>
    <ng-content select="[tabs]" />
  `,
  styles: `
    :host { display: block; padding: 14px 24px 0; }
    .row { display: flex; align-items: center; gap: 12px; min-height: 40px; }
    .back { display: inline-flex; color: var(--text-muted); padding: 4px; border-radius: var(--radius); }
    .back:hover { color: var(--text); background: var(--panel-raised); text-decoration: none; }
    .title { display: flex; flex-direction: column; min-width: 0; }
    .heading { font-size: var(--fs-xl); font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .subtitle { color: var(--text-muted); font-size: var(--fs-sm); }
    .actions { display: flex; align-items: center; gap: 8px; }
    @media (max-width: 720px) { :host { padding: 10px 16px 0; } .row { flex-wrap: wrap; row-gap: 8px; } .title { flex: 1 1 100%; } .heading { font-size: var(--fs-lg); white-space: normal; } .actions { margin-left: auto; } .back { padding: 9px; } }
  `,
})
export class PageHeader {
  readonly heading = input('');
  readonly subtitle = input('');
  readonly back = input<string | null>(null);
}
