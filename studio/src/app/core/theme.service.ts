import { Injectable, computed, effect, inject, signal } from '@angular/core';
import { SettingsService } from './settings.service';

export type Theme = 'light' | 'dark';

/**
 * The colour theme: the Settings choice, or the system's `prefers-color-scheme` when that is
 * "system". Dark is the tokens' default; light is `data-theme="light"` on `<html>`
 * (`styles/_tokens.scss`). Dark sets no attribute: Foblex reads `data-theme="dark"` as its own
 * switch and would repaint the canvas. `theme()` changes after the attribute does, so code that
 * reads computed styles (the charts) can depend on it.
 */
@Injectable({ providedIn: 'root' })
export class ThemeService {
  private readonly settings = inject(SettingsService);
  private readonly system = signal<Theme>('dark');
  private readonly wanted = computed<Theme>(() => {
    const choice = this.settings.settings().theme;
    return choice === 'system' ? this.system() : choice;
  });
  readonly theme = signal<Theme>('dark');

  constructor() {
    const light = window.matchMedia('(prefers-color-scheme: light)');
    const follow = () => this.system.set(light.matches ? 'light' : 'dark');
    follow();
    light.addEventListener('change', follow);
    effect(() => {
      const theme = this.wanted();
      if (theme === 'light') {
        document.documentElement.dataset['theme'] = theme;
      } else {
        delete document.documentElement.dataset['theme'];
      }
      this.theme.set(theme);
    });
  }
}
