import { ChangeDetectionStrategy, Component, ElementRef, HostListener, inject, input, output, signal } from '@angular/core';
import { Icon } from './icon';

export interface MenuItem {
  readonly id: string;
  readonly label: string;
  readonly icon?: string;
  readonly danger?: boolean;
  readonly disabled?: boolean;
  /** Draw a separator above this item. */
  readonly divider?: boolean;
  readonly hint?: string;
}

/** The ⋯ overflow menu: secondary actions live here, not on the screen. */
@Component({
  selector: 'app-menu',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  template: `
    <button type="button" class="btn btn--ghost btn--sm trigger" [class.btn--icon]="!text()" [class.trigger--open]="open()" (click)="toggle($event)" [attr.aria-label]="label()" [attr.aria-expanded]="open()">
      <app-icon [name]="icon()" [size]="16" />
      @if (text()) {
        <span>{{ text() }}</span>
      }
    </button>
    @if (open()) {
      <div class="menu" role="menu" [class.menu--left]="align() === 'left'">
        @for (item of items(); track item.id) {
          @if (item.divider) {
            <span class="menu__divider"></span>
          }
          <button type="button" role="menuitem" class="menu__item" [class.menu__item--danger]="item.danger" [disabled]="item.disabled" (click)="pick(item)">
            @if (item.icon) {
              <app-icon [name]="item.icon" [size]="14" />
            }
            <span>{{ item.label }}</span>
            @if (item.hint) {
              <span class="menu__hint">{{ item.hint }}</span>
            }
          </button>
        }
      </div>
    }
  `,
  styles: `
    :host { position: relative; display: inline-flex; }
    .trigger--open { background: var(--panel-raised); color: var(--text); }
    .menu { position: absolute; top: calc(100% + 4px); right: 0; z-index: 80; min-width: 200px; width: max-content; max-width: min(360px, 90vw); padding: 4px; border-radius: var(--radius-lg); background: var(--panel-raised); border: 1px solid var(--border-strong); box-shadow: var(--shadow-panel); }
    .menu--left { right: auto; left: 0; }
    .menu__item { display: flex; align-items: center; gap: 8px; width: 100%; min-height: 32px; padding: 6px 10px; border-radius: var(--radius); text-align: left; color: var(--text); font-size: var(--fs); white-space: nowrap; }
    @media (max-width: 720px) { .trigger { height: 40px; min-width: 40px; } .menu__item { min-height: 44px; } }
    .menu__item:hover:not(:disabled) { background: var(--panel); }
    .menu__item--danger { color: var(--danger); }
    .menu__hint { margin-left: auto; color: var(--text-faint); font-size: var(--fs-xs); }
    .menu__divider { display: block; height: 1px; margin: 4px 6px; background: var(--border); }
  `,
})
export class Menu {
  readonly items = input.required<readonly MenuItem[]>();
  readonly icon = input('more');
  readonly label = input('More actions');
  /** Visible text next to the icon (a labelled trigger such as "Download"). */
  readonly text = input('');
  readonly align = input<'left' | 'right'>('right');
  readonly picked = output<string>();
  readonly open = signal(false);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);

  protected toggle(event: Event): void {
    event.stopPropagation();
    this.open.update((v) => !v);
  }

  protected pick(item: MenuItem): void {
    this.open.set(false);
    this.picked.emit(item.id);
  }

  @HostListener('document:click', ['$event'])
  protected onDocumentClick(event: Event): void {
    if (this.open() && !this.host.nativeElement.contains(event.target as Node)) {
      this.open.set(false);
    }
  }

  @HostListener('document:keydown.escape')
  protected onEscape(): void {
    this.open.set(false);
  }
}
