import { ChangeDetectionStrategy, Component, HostListener, OnDestroy, effect, inject, input, output } from '@angular/core';
import { UiModeService } from '../core/ui-mode.service';
import { Icon } from './icon';

/**
 * A side drawer: overlays the content from the right on wide screens, becomes a full-screen
 * sheet on phones. Escape or the backdrop closes it. Content is projected.
 */
/** Open drawers, innermost last: Escape only closes the top one (an arch picker inside an inspector). */
const stack: Drawer[] = [];

@Component({
  selector: 'app-drawer',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  template: `
    @if (open()) {
      @if (backdrop() && !docked()) {
        <div class="backdrop" (click)="closed.emit()"></div>
      }
      <aside class="drawer" [class.drawer--sheet]="mode.isPhone()" [class.drawer--docked]="docked() && !mode.isPhone()" [style.width.px]="mode.isPhone() ? null : width()" role="dialog" [attr.aria-label]="title()">
        <header class="drawer__head">
          <ng-content select="[drawer-title]" />
          <h3 class="drawer__title">{{ title() }}</h3>
          <span class="spacer"></span>
          <ng-content select="[drawer-actions]" />
          <button type="button" class="btn btn--ghost btn--sm btn--icon" (click)="closed.emit()" aria-label="Close"><app-icon name="close" [size]="14" /></button>
        </header>
        <div class="drawer__body">
          <ng-content />
        </div>
      </aside>
    }
  `,
  styles: `
    :host { display: contents; }
    .backdrop { position: fixed; inset: 0; z-index: 95; background: transparent; }
    .drawer { position: fixed; top: var(--topbar-h); right: 0; bottom: 0; z-index: 96; display: flex; flex-direction: column; background: var(--panel); border-left: 1px solid var(--border-strong); box-shadow: var(--shadow-panel); animation: slide var(--tempo) var(--ease); }
    .drawer--sheet { left: 0; top: var(--topbar-h); width: auto; border-left: 0; }
    .drawer--docked { position: relative; top: auto; right: auto; bottom: auto; height: 100%; box-shadow: none; animation: none; }
    .drawer__head { display: flex; align-items: center; gap: 8px; height: 44px; padding: 0 8px 0 14px; border-bottom: 1px solid var(--border); flex: none; }
    .drawer__title { font-size: var(--fs-md); font-weight: 600; }
    .drawer__body { flex: 1; min-height: 0; overflow: auto; padding: 12px 14px 20px; }
    @keyframes slide { from { transform: translateX(16px); opacity: 0; } to { transform: none; opacity: 1; } }
    @media (prefers-reduced-motion: reduce) { .drawer { animation: none; } }
  `,
})
export class Drawer implements OnDestroy {
  readonly open = input(false);
  readonly title = input('');
  readonly width = input(380);
  /** Render inline as a column instead of an overlay (the host reserves the space). */
  readonly docked = input(false);
  /** Click-outside backdrop; off for non-modal side panels. */
  readonly backdrop = input(true);
  /** Escape closes the drawer. Hosts with inner popovers may turn this off. */
  readonly escapeCloses = input(true);
  readonly closed = output<void>();
  protected readonly mode = inject(UiModeService);

  constructor() {
    effect(() => {
      const index = stack.indexOf(this);
      if (this.open() && index < 0) {
        stack.push(this);
      } else if (!this.open() && index >= 0) {
        stack.splice(index, 1);
      }
    });
  }

  ngOnDestroy(): void {
    const index = stack.indexOf(this);
    if (index >= 0) {
      stack.splice(index, 1);
    }
  }

  @HostListener('document:keydown.escape')
  protected onEscape(): void {
    const innerOverlayOpen = !!document.querySelector('app-menu .menu, app-help .help__pop');
    if (this.open() && this.escapeCloses() && stack[stack.length - 1] === this && !innerOverlayOpen) {
      this.closed.emit();
    }
  }
}
