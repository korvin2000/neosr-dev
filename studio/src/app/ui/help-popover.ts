import { ChangeDetectionStrategy, Component, ElementRef, HostListener, computed, inject, input, signal } from '@angular/core';
import type { FieldSpec } from '@shared/catalog/types';
import { Icon } from './icon';

const POP_WIDTH = 300;

/**
 * The ⓘ next to a field label. Click (or Enter) opens a small popover with the explanation, the
 * NeoSR key and typical values; nothing is shown otherwise. Help is pulled, not pushed.
 * With a `label` the trigger is a quiet badge carrying that text instead of the ⓘ.
 */
@Component({
  selector: 'app-help',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  template: `
    <button type="button" [class]="label() ? 'badge badge--native help__badge' : 'help__btn'" [class.help__btn--open]="open()" (click)="toggle($event)" [attr.aria-expanded]="open()" [attr.aria-label]="label() ? null : 'Help'">
      @if (label()) {
        {{ label() }}
      } @else {
        <app-icon name="help" [size]="13" [weight]="1.6" />
      }
    </button>
    @if (open()) {
      <div class="help__pop" role="dialog" [style.left.px]="left()">
        <p class="help__text">{{ text() }}</p>
        @if (key()) {
          <p class="help__key"><span class="faint">NeoSR</span> <code>{{ key() }}</code></p>
        }
        @if (typical().length) {
          <p class="help__typical"><span class="faint">Typical</span> {{ typical().join(' · ') }}</p>
        }
      </div>
    }
  `,
  styles: `
    :host { position: relative; display: inline-flex; vertical-align: middle; }
    .help__btn { display: inline-flex; align-items: center; justify-content: center; min-width: 24px; min-height: 24px; color: var(--text-faint); border-radius: 50%; }
    @media (max-width: 720px) { .help__btn { min-width: 32px; min-height: 32px; } }
    .help__btn:hover, .help__btn--open { color: var(--accent-text); }
    .help__badge { position: relative; font-weight: 400; }
    .help__badge::after { content: ''; position: absolute; inset: -6px -2px; }
    .help__badge:hover, .help__badge.help__btn--open { color: var(--text); border-color: var(--node-border-hover); }
    .help__pop { position: absolute; z-index: 60; top: 22px; width: 300px; max-width: calc(100vw - 32px); padding: 10px 12px; border-radius: var(--radius-lg); background: var(--panel-raised); border: 1px solid var(--border-strong); box-shadow: var(--shadow-panel); font-size: var(--fs-sm); line-height: 1.5; }
    .help__text { margin: 0; color: var(--text); }
    .help__key, .help__typical { margin: 6px 0 0; color: var(--text-muted); }
    code { font-family: var(--font-mono); font-size: var(--fs-xs); color: var(--accent-text); }
  `,
  host: { '[class.open]': 'open()' },
})
export class HelpPopover {
  readonly field = input<FieldSpec | null>(null);
  readonly help = input<string>('');
  readonly tomlKey = input<string>('');
  /** Badge text; empty renders the ⓘ. */
  readonly label = input<string>('');
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  readonly open = signal(false);
  protected readonly left = signal(-8);

  protected readonly text = computed(() => this.field()?.help ?? this.help());
  protected readonly key = computed(() => this.field()?.tomlKey ?? this.tomlKey());
  protected readonly typical = computed(() => (this.field()?.suggestions ?? []).map((s) => s.label));

  protected toggle(event: Event): void {
    event.stopPropagation();
    // A trigger near the right edge (a drawer) shifts the popover left so it stays on screen.
    this.left.set(Math.min(-8, window.innerWidth - 16 - POP_WIDTH - this.host.nativeElement.getBoundingClientRect().left));
    this.open.update((v) => !v);
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
