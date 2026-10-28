import { Injectable, effect, signal } from '@angular/core';

export interface StudioSettings {
  readonly platform: 'windows' | 'linux';
  readonly neosrRoot: string;
  readonly experimentsRoot: string;
  readonly optionsDir: string;
  /** Simulated updates per second when launching a run. */
  readonly simSpeed: number;
  /** Show the NeoSR TOML key under every setting label. */
  readonly showKeys: boolean;
  /** Open every Build section by default instead of showing summaries. */
  readonly openSections: boolean;
  readonly inspectorDocked: boolean;
}

const KEY = 'neosr-studio.settings.v1';

const DEFAULTS: StudioSettings = {
  platform: typeof navigator !== 'undefined' && /win/i.test(navigator.platform) ? 'windows' : 'linux',
  neosrRoot: 'C:\\work2000\\neosr-dev',
  experimentsRoot: 'C:\\work2000\\neosr-dev\\experiments',
  optionsDir: 'C:\\work2000\\neosr-dev\\options',
  simSpeed: 400,
  showKeys: false,
  openSections: false,
  inspectorDocked: true,
};

@Injectable({ providedIn: 'root' })
export class SettingsService {
  readonly settings = signal<StudioSettings>(load());

  constructor() {
    effect(() => {
      try {
        localStorage.setItem(KEY, JSON.stringify(this.settings()));
      } catch {
        // Private mode or quota: settings simply do not persist.
      }
    });
  }

  patch(patch: Partial<StudioSettings>): void {
    this.settings.update((s) => ({ ...s, ...patch }));
  }

  reset(): void {
    this.settings.set(DEFAULTS);
  }
}

function load(): StudioSettings {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? { ...DEFAULTS, ...(JSON.parse(raw) as Partial<StudioSettings>) } : DEFAULTS;
  } catch {
    return DEFAULTS;
  }
}
