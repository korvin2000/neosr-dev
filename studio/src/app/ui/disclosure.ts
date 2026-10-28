import { ChangeDetectionStrategy, Component, input, model } from '@angular/core';
import { Icon } from './icon';

/**
 * A collapsible section with a one-line summary. Closed by default: the summary is the whole
 * story until the user asks for more. Several in a row make an accordion when the parent keeps
 * one `open` at a time.
 */
@Component({
  selector: 'app-disclosure',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  template: `
    <button type="button" class="head" [class.head--open]="open()" (click)="open.set(!open())" [attr.aria-expanded]="open()">
      <app-icon class="head__chevron" name="chevron-right" [size]="14" />
      <span class="head__title">{{ title() }}</span>
      @if (!open() && summary()) {
        <span class="head__summary">{{ summary() }}</span>
      }
      @if (badge()) {
        <span class="badge" [class.badge--danger]="badgeKind() === 'danger'" [class.badge--warn]="badgeKind() === 'warn'" [class.badge--accent]="badgeKind() === 'accent'">{{ badge() }}</span>
      }
    </button>
    @if (open()) {
      <div class="body">
        <ng-content />
      </div>
    }
  `,
  styles: `
    :host { display: block; border-bottom: 1px solid var(--border-soft); }
    .head { display: flex; align-items: center; gap: 8px; width: 100%; min-height: 44px; padding: 8px 4px; text-align: left; color: var(--text); border-radius: var(--radius); }
    .head:hover { background: var(--panel-raised); }
    .head__chevron { color: var(--text-faint); transition: transform var(--tempo) var(--ease); flex: none; }
    .head--open .head__chevron { transform: rotate(90deg); }
    .head__title { font-weight: 600; flex: none; }
    .head__summary { color: var(--text-muted); font-size: var(--fs-sm); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
    .badge { margin-left: auto; flex: none; }
    .body { padding: 4px 4px 16px 26px; }
    @media (max-width: 720px) { .body { padding-left: 4px; } .head { flex-wrap: wrap; row-gap: 2px; } .head__summary { flex-basis: 100%; padding-left: 22px; white-space: normal; } }
  `,
})
export class Disclosure {
  readonly title = input.required<string>();
  readonly summary = input('');
  readonly badge = input('');
  readonly badgeKind = input<'neutral' | 'danger' | 'warn' | 'accent'>('neutral');
  readonly open = model(false);
}
