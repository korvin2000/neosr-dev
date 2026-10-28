import { HttpErrorResponse } from '@angular/common/http';
import { signal } from '@angular/core';

/** A list fetched from the API with loading / error state. `error` is set when the backend is unreachable or refuses. */
export class RemoteList<T> {
  readonly data = signal<readonly T[]>([]);
  readonly loading = signal(false);
  readonly loaded = signal(false);
  readonly error = signal<string | null>(null);

  constructor(private readonly fetcher: () => Promise<T[]>) {}

  async load(): Promise<void> {
    this.loading.set(true);
    try {
      this.data.set(await this.fetcher());
      this.error.set(null);
    } catch (e) {
      this.error.set(apiErrorMessage(e));
    } finally {
      this.loading.set(false);
      this.loaded.set(true);
    }
  }

  /** Load once; later calls are no-ops unless the last attempt failed. */
  ensure(): void {
    if ((!this.loaded() || this.error()) && !this.loading()) {
      void this.load();
    }
  }
}

export function isOffline(e: unknown): boolean {
  // Status 0 = connection refused; the dev-server proxy answers 500/502/504 when the API is down.
  return e instanceof HttpErrorResponse && (e.status === 0 || ((e.status === 500 || e.status === 502 || e.status === 504) && (e.error === null || typeof e.error !== 'object')));
}

export function apiErrorMessage(e: unknown): string {
  if (isOffline(e)) {
    return 'Backend offline: start it with npm run api.';
  }
  if (e instanceof HttpErrorResponse) {
    const body = e.error as { error?: unknown } | null;
    return `HTTP ${e.status}: ${typeof body?.error === 'string' ? body.error : e.statusText}`;
  }
  return e instanceof Error ? e.message : String(e);
}

export function downloadText(fileName: string, text: string, type = 'text/plain'): void {
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }));
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Read the first file of an `<input type="file">` change event, then reset the input so the same file can be picked again. */
export async function readPickedFile(event: Event): Promise<{ name: string; text: string } | null> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {
    return null;
  }
  const text = await file.text();
  input.value = '';
  return { name: file.name, text };
}

export function safeFileName(name: string): string {
  return name.trim().replace(/[\\/:*?"<>|\s]+/g, '_').replace(/^_+|_+$/g, '') || 'recipe';
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

export function relativeTime(iso: string | undefined, now = Date.now()): string {
  if (!iso) {
    return '';
  }
  const sec = Math.round((now - new Date(iso).getTime()) / 1000);
  if (!Number.isFinite(sec)) {
    return '';
  }
  if (sec < 60) {
    return 'just now';
  }
  const min = Math.round(sec / 60);
  if (min < 60) {
    return `${min} min ago`;
  }
  const h = Math.round(min / 60);
  if (h < 48) {
    return `${h} h ago`;
  }
  return new Date(iso).toLocaleDateString();
}
