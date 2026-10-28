import { Injectable, signal } from '@angular/core';

export interface Toast {
  readonly id: number;
  readonly kind: 'info' | 'ok' | 'warn' | 'danger';
  readonly text: string;
  readonly action?: { readonly label: string; readonly run: () => void };
}

@Injectable({ providedIn: 'root' })
export class ToastService {
  readonly toasts = signal<readonly Toast[]>([]);
  private seq = 0;

  show(kind: Toast['kind'], text: string, action?: Toast['action'], ttlMs = 4500): void {
    const toast: Toast = { id: ++this.seq, kind, text, action };
    this.toasts.update((t) => [...t, toast]);
    setTimeout(() => this.dismiss(toast.id), ttlMs);
  }

  dismiss(id: number): void {
    this.toasts.update((t) => t.filter((x) => x.id !== id));
  }
}
