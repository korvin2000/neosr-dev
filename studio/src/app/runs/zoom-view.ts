import { ChangeDetectionStrategy, Component, computed, input, output, signal } from '@angular/core';

export interface CropCenter {
  readonly x: number;
  readonly y: number;
}

export type ZoomLevel = 1 | 2 | 4;

/**
 * A square image viewport with a shared zoom and crop centre (drag to pan). `swipe` lays `overlay`
 * over `src` with a draggable divider; `blink` shows `src` and flips to `overlay` while pressed
 * (mouse, touch, or Space/Enter when focused). Hosts keep the crop in one signal so several views
 * stay on the same detail.
 */
@Component({
  selector: 'app-zoom-view',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div
      class="vp"
      [class.vp--pan]="zoom() > 1"
      [class.vp--blink]="mode() === 'blink'"
      [attr.tabindex]="mode() === 'blink' ? 0 : null"
      [attr.role]="mode() === 'blink' ? 'button' : null"
      [attr.aria-label]="mode() === 'blink' ? 'Hold to show ' + overlayLabel() : null"
      (pointerdown)="down($event)"
      (pointermove)="move($event)"
      (pointerup)="up()"
      (pointercancel)="up()"
      (pointerleave)="held.set(false)"
      (contextmenu)="mode() === 'blink' && $event.preventDefault()"
      (keydown)="key($event, true)"
      (keyup)="key($event, false)"
      (dblclick)="center.emit({ x: 0.5, y: 0.5 })"
    >
      <img class="img" [src]="shown()" [style.width.%]="zoom() * 100" [style.left.%]="pos().left" [style.top.%]="pos().top" draggable="false" [alt]="held() ? overlayLabel() : label()" />
      @if (mode() === 'swipe' && overlay()) {
        <div class="clip" [style.clip-path]="'inset(0 ' + (100 - divider() * 100) + '% 0 0)'">
          <img class="img" [src]="overlay()" [style.width.%]="zoom() * 100" [style.left.%]="pos().left" [style.top.%]="pos().top" draggable="false" [alt]="overlayLabel()" />
        </div>
        <div class="divider" [style.left.%]="divider() * 100" (pointerdown)="grabDivider($event)"><span class="divider__handle"></span></div>
        <span class="tag tag--left">{{ overlayLabel() }}</span>
        <span class="tag tag--right">{{ label() }}</span>
      } @else if (label()) {
        <span class="tag tag--left" [class.tag--held]="held()">{{ held() ? overlayLabel() : label() }}</span>
        @if (mode() === 'blink' && !held()) {
          <span class="tag tag--right tag--quiet">hold · {{ overlayLabel() }}</span>
        }
      }
    </div>
  `,
  styles: `
    :host { display: block; min-width: 0; }
    .vp { position: relative; aspect-ratio: 1; overflow: hidden; border-radius: var(--radius-sm); background: var(--panel-sunken); border: 1px solid var(--border); user-select: none; -webkit-user-select: none; -webkit-touch-callout: none; touch-action: pan-y; }
    .vp--pan { cursor: grab; touch-action: none; }
    .vp--pan:active { cursor: grabbing; }
    .vp--blink:not(.vp--pan) { cursor: pointer; }
    .img { position: absolute; display: block; height: auto; max-width: none; pointer-events: none; }
    .clip { position: absolute; inset: 0; }
    .divider { position: absolute; top: 0; bottom: 0; width: 14px; margin-left: -7px; cursor: ew-resize; display: grid; place-items: center; touch-action: none; }
    .divider::before { content: ''; position: absolute; top: 0; bottom: 0; left: 6px; width: 2px; background: var(--text); opacity: 0.8; }
    .divider__handle { position: relative; width: 12px; height: 26px; border-radius: 6px; background: var(--text); box-shadow: 0 1px 4px rgba(0, 0, 0, 0.5); }
    .tag { position: absolute; top: 5px; padding: 1px 6px; border-radius: var(--radius-sm); background: var(--overlay); color: var(--text); font-size: var(--fs-xs); font-weight: 600; pointer-events: none; }
    .tag--left { left: 5px; }
    .tag--right { right: 5px; }
    .tag--quiet { color: var(--text-muted); font-weight: 400; }
    .tag--held { background: var(--accent); color: #fff; }
  `,
})
export class ZoomView {
  readonly src = input.required<string>();
  readonly overlay = input<string | null>(null);
  readonly mode = input<'plain' | 'swipe' | 'blink'>('plain');
  readonly label = input('');
  readonly overlayLabel = input('');
  readonly zoom = input<ZoomLevel>(1);
  readonly crop = input<CropCenter>({ x: 0.5, y: 0.5 });
  readonly center = output<CropCenter>();

  /** Blink: the overlay is shown while this is true. */
  protected readonly held = signal(false);
  protected readonly divider = signal(0.5);
  protected readonly shown = computed(() => (this.mode() === 'blink' && this.held() ? (this.overlay() ?? this.src()) : this.src()));

  protected readonly pos = computed(() => {
    const z = this.zoom();
    const c = clampCrop(this.crop(), z);
    return { left: (0.5 - c.x * z) * 100, top: (0.5 - c.y * z) * 100 };
  });

  private drag: { id: number; x: number; y: number; crop: CropCenter; moved: boolean; divider: boolean; width: number } | null = null;

  protected down(event: PointerEvent): void {
    const el = event.currentTarget as HTMLElement;
    this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, crop: clampCrop(this.crop(), this.zoom()), moved: false, divider: false, width: el.clientWidth || 1 };
    if (this.zoom() > 1 || this.mode() === 'blink') {
      el.setPointerCapture(event.pointerId);
    }
    if (this.mode() === 'blink') {
      this.held.set(true);
    }
  }

  protected grabDivider(event: PointerEvent): void {
    event.stopPropagation();
    const vp = (event.currentTarget as HTMLElement).parentElement as HTMLElement;
    this.drag = { id: event.pointerId, x: event.clientX, y: event.clientY, crop: this.crop(), moved: true, divider: true, width: vp.clientWidth || 1 };
    vp.setPointerCapture(event.pointerId);
  }

  protected move(event: PointerEvent): void {
    const d = this.drag;
    if (!d || d.id !== event.pointerId) {
      return;
    }
    const el = event.currentTarget as HTMLElement;
    if (d.divider) {
      const rect = el.getBoundingClientRect();
      this.divider.set(Math.min(1, Math.max(0, (event.clientX - rect.left) / (rect.width || 1))));
      return;
    }
    const dx = event.clientX - d.x;
    const dy = event.clientY - d.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) {
      d.moved = true;
    }
    const z = this.zoom();
    if (z > 1 && d.moved) {
      // Panning is not a blink: show the base image while the crop moves.
      this.held.set(false);
      this.center.emit(clampCrop({ x: d.crop.x - dx / (d.width * z), y: d.crop.y - dy / (d.width * z) }, z));
    }
  }

  protected up(): void {
    this.drag = null;
    this.held.set(false);
  }

  protected key(event: KeyboardEvent, down: boolean): void {
    if (this.mode() === 'blink' && (event.key === ' ' || event.key === 'Enter')) {
      event.preventDefault();
      this.held.set(down);
    }
  }
}

export function clampCrop(c: CropCenter, zoom: number): CropCenter {
  const half = 0.5 / zoom;
  const clamp = (v: number) => Math.min(1 - half, Math.max(half, v));
  return { x: clamp(c.x), y: clamp(c.y) };
}
