import { ChangeDetectionStrategy, Component, ElementRef, HostListener, inject, input, signal } from '@angular/core';
import { UiModeService } from '../core/ui-mode.service';
import { Icon } from '../ui/icon';

/**
 * One decision, anchored to its trigger: a small popover on desktop, a bottom sheet on phones.
 * Wrap the trigger in `[trigger]`; the body is projected. Outside click and Escape close it;
 * the parent calls `close()` after a choice.
 */
@Component({
  selector: 'app-popover',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  template: `
    <span class="trigger" (click)="toggle($event)"><ng-content select="[trigger]" /></span>@if (open()) {
      @if (mode.isPhone()) {
        <div class="backdrop" (click)="close()"></div>
      }
      <div class="pop" [class.pop--sheet]="mode.isPhone()" [class.pop--right]="align() === 'right'" [style.width.px]="mode.isPhone() ? null : width()" role="dialog" [attr.aria-label]="title()">
        @if (title() || mode.isPhone()) {
          <header class="pop__head">
            <span class="pop__title">{{ title() }}</span>
            <span class="spacer"></span>
            @if (mode.isPhone()) {
              <button type="button" class="btn btn--ghost btn--sm btn--icon" (click)="close()" aria-label="Close"><app-icon name="close" [size]="14" /></button>
            }
          </header>
        }
        <div class="pop__body"><ng-content /></div>
      </div>
    }
  `,
  styles: `
    :host { position: relative; display: inline-block; max-width: 100%; }
    .trigger { display: contents; }
    // Above the shell's bottom tab bar (89) and top bar (90), below toasts.
    .backdrop { position: fixed; inset: 0; z-index: 94; background: var(--overlay); }
    .pop { position: absolute; z-index: 71; top: calc(100% + 6px); left: 0; padding: 8px; border-radius: var(--radius-lg); background: var(--panel-raised); border: 1px solid var(--border-strong); box-shadow: var(--shadow-panel); font-size: var(--fs); animation: pop var(--tempo) var(--ease); }
    .pop--right { left: auto; right: 0; }
    .pop--sheet { position: fixed; z-index: 95; top: auto; left: 0; right: 0; bottom: 0; width: auto; max-height: 70vh; overflow: auto; border-radius: var(--radius-lg) var(--radius-lg) 0 0; padding: 4px 16px 24px; animation: sheet var(--tempo-slow) var(--ease); }
    .pop__head { display: flex; align-items: center; gap: 8px; min-height: 36px; padding: 0 4px 6px; }
    .pop__title { font-weight: 600; }
    .pop--sheet .pop__head { position: sticky; top: 0; background: var(--panel-raised); padding-top: 8px; }
    .pop__body { display: flex; flex-direction: column; gap: 2px; }
    @keyframes pop { from { transform: translateY(-4px); opacity: 0; } to { transform: none; opacity: 1; } }
    @keyframes sheet { from { transform: translateY(24px); opacity: 0; } to { transform: none; opacity: 1; } }
    @media (prefers-reduced-motion: reduce) { .pop { animation: none; } }
  `,
})
export class Popover {
  readonly title = input('');
  readonly width = input(280);
  readonly align = input<'left' | 'right'>('left');
  readonly open = signal(false);
  protected readonly mode = inject(UiModeService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);

  protected toggle(event: Event): void {
    event.stopPropagation();
    this.open.update((v) => !v);
  }

  close(): void {
    this.open.set(false);
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
