import { Injectable, inject, signal } from '@angular/core';
import type { EnvironmentInfo } from '@shared/run.model';
import { ApiService } from './api.service';

@Injectable({ providedIn: 'root' })
export class EnvironmentService {
  private readonly api = inject(ApiService);
  readonly info = signal<EnvironmentInfo | null>(null);
  readonly offline = signal(false);
  private loading: Promise<void> | null = null;

  load(): Promise<void> {
    this.loading ??= this.api
      .environment()
      .then((info) => {
        this.info.set(info);
        this.offline.set(false);
      })
      .catch(() => {
        this.offline.set(true);
        this.loading = null;
      });
    return this.loading;
  }
}
