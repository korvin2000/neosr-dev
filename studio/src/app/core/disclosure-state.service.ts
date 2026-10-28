import { Injectable, effect, signal } from '@angular/core';

/** Remembers which "More options" and sections the user opened, per key, across reloads. */
const KEY = 'neosr-studio.disclosure.v1';

@Injectable({ providedIn: 'root' })
export class DisclosureState {
  private readonly state = signal<Readonly<Record<string, boolean>>>(load());

  constructor() {
    effect(() => {
      try {
        localStorage.setItem(KEY, JSON.stringify(this.state()));
      } catch {
        // best effort
      }
    });
  }

  isOpen(key: string, fallback = false): boolean {
    return this.state()[key] ?? fallback;
  }

  set(key: string, open: boolean): void {
    this.state.update((s) => ({ ...s, [key]: open }));
  }

  toggle(key: string, fallback = false): void {
    this.set(key, !this.isOpen(key, fallback));
  }
}

function load(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Record<string, boolean>) : {};
  } catch {
    return {};
  }
}
