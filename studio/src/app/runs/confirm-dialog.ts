import { ChangeDetectionStrategy, Component, ElementRef, afterNextRender, inject, input, output } from '@angular/core';

/** Modal for irreversible actions only (force stop, delete, discard). Escape and the backdrop cancel. */
@Component({
  selector: 'app-confirm-dialog',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { '(document:keydown.escape)': 'cancelled.emit()' },
  template: `
    <div class="backdrop" (click)="cancelled.emit()"></div>
    <div class="dialog" role="alertdialog" aria-modal="true" [attr.aria-label]="heading()">
      <h2 class="dialog__title">{{ heading() }}</h2>
      <p class="dialog__body">{{ body() }}</p>
      <div class="dialog__actions">
        <button type="button" class="btn cancel" (click)="cancelled.emit()">Cancel</button>
        <button type="button" class="btn confirm" [class.btn--primary]="!danger()" [class.confirm--danger]="danger()" (click)="confirmed.emit()">{{ confirmLabel() }}</button>
      </div>
    </div>
  `,
  styles: `
    .backdrop { position: fixed; inset: 0; z-index: 90; background: var(--overlay); }
    .dialog { position: fixed; z-index: 91; top: 22vh; left: 50%; width: min(420px, calc(100vw - 32px)); transform: translateX(-50%); padding: 18px 20px 16px; border-radius: var(--radius-lg); background: var(--panel); border: 1px solid var(--border-strong); box-shadow: var(--shadow-dialog); }
    .dialog__title { font-size: var(--fs-lg); }
    .dialog__body { margin-top: 8px; color: var(--text-muted); }
    .dialog__actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 18px; }
    .confirm--danger { background: var(--danger); border-color: transparent; color: #fff; }
    .confirm--danger:hover:not(:disabled) { background: color-mix(in srgb, var(--danger) 85%, #fff); border-color: transparent; }
    @media (max-width: 720px) {
      .dialog { top: auto; bottom: 0; left: 0; width: 100%; transform: none; border-radius: var(--radius-lg) var(--radius-lg) 0 0; padding-bottom: 24px; }
      .dialog__actions .btn { flex: 1; justify-content: center; height: 44px; }
    }
  `,
})
export class ConfirmDialog {
  readonly heading = input.required<string>();
  readonly body = input('');
  readonly confirmLabel = input('Confirm');
  readonly danger = input(false);
  readonly confirmed = output<void>();
  readonly cancelled = output<void>();

  constructor() {
    const host = inject<ElementRef<HTMLElement>>(ElementRef);
    // A destructive dialog starts on Cancel so a stray Enter does not confirm it.
    afterNextRender(() => host.nativeElement.querySelector<HTMLButtonElement>(this.danger() ? '.cancel' : '.confirm')?.focus());
  }
}
