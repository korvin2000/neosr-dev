import { Directive, ElementRef, OnDestroy, inject, input } from '@angular/core';

/**
 * `[appTooltip]="text"`: a positioned tooltip on hover (after a short delay), keyboard focus and
 * touch long-press. One shared element for the whole app; nothing is rendered until needed.
 * Use it for help; never `title` (inconsistent delay, no touch, no styling).
 */
let shared: HTMLDivElement | null = null;

function element(): HTMLDivElement {
  if (!shared) {
    shared = document.createElement('div');
    shared.className = 'app-tooltip';
    shared.setAttribute('role', 'tooltip');
    document.body.appendChild(shared);
  }
  return shared;
}

@Directive({
  selector: '[appTooltip]',
  host: {
    '(mouseenter)': 'scheduleShow()',
    '(mouseleave)': 'hide()',
    '(focusin)': 'show()',
    '(focusout)': 'hide()',
    '(touchstart)': 'onTouchStart()',
    '(touchend)': 'onTouchEnd()',
    '(keydown.escape)': 'hide()',
  },
})
export class Tooltip implements OnDestroy {
  readonly appTooltip = input<string | null | undefined>('');
  /** Preferred side; flips when there is no room. */
  readonly tooltipSide = input<'top' | 'bottom'>('top');
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private timer: ReturnType<typeof setTimeout> | null = null;
  private visible = false;

  protected scheduleShow(): void {
    this.clear();
    this.timer = setTimeout(() => this.show(), 500);
  }

  protected onTouchStart(): void {
    this.clear();
    this.timer = setTimeout(() => this.show(), 500);
  }

  protected onTouchEnd(): void {
    this.clear();
    setTimeout(() => this.hide(), 1500);
  }

  protected show(): void {
    this.clear();
    const text = (this.appTooltip() ?? '').trim();
    if (!text) {
      return;
    }
    const tip = element();
    tip.textContent = text;
    tip.classList.add('app-tooltip--visible');
    this.visible = true;
    const rect = this.host.nativeElement.getBoundingClientRect();
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let left = rect.left + rect.width / 2 - w / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
    const above = this.tooltipSide() === 'top' ? rect.top - h - 8 >= 4 : rect.bottom + h + 8 > window.innerHeight;
    const top = above ? rect.top - h - 8 : rect.bottom + 8;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }

  protected hide(): void {
    this.clear();
    if (this.visible && shared) {
      shared.classList.remove('app-tooltip--visible');
      this.visible = false;
    }
  }

  private clear(): void {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  ngOnDestroy(): void {
    this.hide();
  }
}
