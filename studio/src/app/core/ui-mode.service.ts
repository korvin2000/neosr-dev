import { Injectable, computed, signal } from '@angular/core';

/** The viewport class every screen reads to decide between side panels, drawers and sheets. */
export type Viewport = 'phone' | 'narrow' | 'wide';

@Injectable({ providedIn: 'root' })
export class UiModeService {
  readonly viewport = signal<Viewport>(measure());
  readonly isPhone = computed(() => this.viewport() === 'phone');
  readonly isNarrow = computed(() => this.viewport() !== 'wide');
  readonly isWide = computed(() => this.viewport() === 'wide');

  constructor() {
    if (typeof window !== 'undefined') {
      const update = () => this.viewport.set(measure());
      window.matchMedia('(max-width: 720px)').addEventListener('change', update);
      window.matchMedia('(max-width: 1400px)').addEventListener('change', update);
    }
  }
}

function measure(): Viewport {
  if (typeof window === 'undefined') {
    return 'wide';
  }
  if (window.matchMedia('(max-width: 720px)').matches) {
    return 'phone';
  }
  if (window.matchMedia('(max-width: 1400px)').matches) {
    return 'narrow';
  }
  return 'wide';
}
