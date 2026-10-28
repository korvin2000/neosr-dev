/**
 * Runs: the list, and `LiveRun` — one run's state rebuilt from its event stream, reconnecting
 * with the last sequence id so nothing is lost or duplicated.
 */
import { Injectable, computed, inject, signal } from '@angular/core';
import type { RunDetail, RunEvent, RunSnapshot, RunStatus, RunSummary, ScalarPoint, ValidationImage } from '@shared/run.model';
import { ApiService } from './api.service';

export interface ValidationPoint {
  readonly iter: number;
  readonly dataset: string;
  readonly metrics: Readonly<Record<string, number>>;
  readonly images: readonly ValidationImage[];
  /** Filled in when a checkpoint event arrives for the same iteration. */
  readonly checkpoint: boolean;
}

export interface CheckpointPoint {
  readonly iter: number;
  readonly files: readonly string[];
  readonly sizeMb: number;
  readonly state: boolean;
}

export const ACTIVE_STATUSES: readonly RunStatus[] = ['queued', 'starting', 'running', 'stopping', 'saving'];

@Injectable({ providedIn: 'root' })
export class RunStore {
  private readonly api = inject(ApiService);

  readonly runs = signal<RunSummary[]>([]);
  readonly loading = signal(false);
  readonly active = computed(() => this.runs().filter((r) => ACTIVE_STATUSES.includes(r.status)));

  async refresh(): Promise<void> {
    this.loading.set(true);
    try {
      this.runs.set(await this.api.listRuns());
    } finally {
      this.loading.set(false);
    }
  }

  live(id: string): LiveRun {
    return new LiveRun(id, this.api);
  }
}

const MAX_LOGS = 2000;
const MAX_EVENTS = 600;
/** Per-series point cap; above it every other point is dropped (the chart smooths anyway). */
const MAX_POINTS = 6000;

export class LiveRun {
  readonly run = signal<RunDetail | null>(null);
  readonly status = computed<RunStatus>(() => this.run()?.status ?? 'queued');
  readonly connected = signal(false);
  readonly error = signal<string | null>(null);
  readonly series = signal<Readonly<Record<string, readonly ScalarPoint[]>>>({});
  readonly validations = signal<readonly ValidationPoint[]>([]);
  readonly checkpoints = signal<readonly CheckpointPoint[]>([]);
  readonly logs = signal<readonly { iter: number; level: 'info' | 'warn' | 'error'; line: string; at: string }[]>([]);
  /** Non-log, non-scalar events for the timeline. */
  readonly events = signal<readonly RunEvent[]>([]);
  readonly system = signal<{ gpuUtil: number; vramUsedGb: number; vramTotalGb: number; dataWaitPct: number } | null>(null);
  /** NeoSR's own update counter for the current phase (restarts on a fine-tune phase). */
  readonly nativeIter = signal<number | null>(null);
  readonly lastSeq = signal(0);

  private source: EventSource | null = null;
  private closed = false;
  private pendingPoints = new Map<string, ScalarPoint[]>();
  private flushScheduled = false;
  private resyncing = false;
  /** Events that arrived while a snapshot was being fetched; replayed if newer than it. */
  private heldDuringResync: RunEvent[] = [];
  private closeTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    readonly id: string,
    private readonly api: ApiService,
  ) {}

  async open(): Promise<void> {
    try {
      const snapshot = await this.api.getRun(this.id);
      this.applySnapshot(snapshot);
      this.error.set(null);
    } catch (e) {
      this.error.set(e instanceof Error ? e.message : 'Failed to load run');
      return;
    }
    this.connect();
  }

  close(): void {
    this.closed = true;
    if (this.closeTimer) {
      clearTimeout(this.closeTimer);
      this.closeTimer = null;
    }
    this.source?.close();
    this.source = null;
    this.connected.set(false);
  }

  async stop(force = false): Promise<void> {
    const summary = await this.api.stopRun(this.id, force);
    this.patchRun(summary);
  }

  /** NeoSR restarts from the last checkpoint, so telemetry after it is discarded: reload the snapshot. */
  async resume(): Promise<void> {
    const summary = await this.api.resumeRun(this.id);
    this.patchRun(summary);
    this.source?.close();
    this.source = null;
    await this.open();
  }

  async chaos(kind: 'disconnect' | 'nan'): Promise<void> {
    const summary = await this.api.chaos(this.id, kind);
    this.patchRun(summary);
  }

  private patchRun(summary: RunSummary): void {
    const current = this.run();
    if (current) {
      this.run.set({ ...current, ...summary });
    }
  }

  private applySnapshot(snapshot: RunSnapshot): void {
    this.run.set(snapshot.run);
    this.series.set(snapshot.series);
    this.validations.set([]);
    this.checkpoints.set([]);
    this.logs.set([]);
    this.events.set([]);
    for (const event of snapshot.events) {
      this.applyEvent(event, false);
    }
    this.lastSeq.set(snapshot.lastSeq);
  }

  private connect(): void {
    if (this.closed || this.source) {
      return;
    }
    const source = new EventSource(this.api.eventsUrl(this.id, this.lastSeq()));
    this.source = source;
    source.onopen = () => this.connected.set(true);
    source.addEventListener('run', (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as RunEvent;
      if (event.seq <= this.lastSeq()) {
        return;
      }
      // The server keeps a bounded replay window: a hole in the sequence means events were
      // evicted before we subscribed, so rebuild from a fresh snapshot instead of charting a gap.
      if (this.resyncing) {
        this.heldDuringResync.push(event);
        return;
      }
      if (event.seq > this.lastSeq() + 1) {
        this.heldDuringResync = [event];
        void this.resync();
        return;
      }
      this.lastSeq.set(event.seq);
      this.applyEvent(event, true);
    });
    source.onerror = () => {
      // EventSource reconnects by itself; the `after` cursor is fixed at open time, so a manual
      // reopen with the current cursor avoids replaying the whole history.
      this.connected.set(false);
      source.close();
      this.source = null;
      const status = this.status();
      if (!this.closed && ACTIVE_STATUSES.includes(status)) {
        setTimeout(() => this.connect(), 1500);
      }
    };
  }

  private async resync(): Promise<void> {
    this.resyncing = true;
    try {
      const snapshot = await this.api.getRun(this.id);
      this.applySnapshot(snapshot);
      // Anything that streamed in meanwhile and is newer than the snapshot still counts.
      for (const held of this.heldDuringResync.sort((a, b) => a.seq - b.seq)) {
        if (held.seq > this.lastSeq()) {
          this.lastSeq.set(held.seq);
          this.applyEvent(held, true);
        }
      }
    } catch {
      // Keep the stream; the next event will try again.
    } finally {
      this.heldDuringResync = [];
      this.resyncing = false;
    }
  }

  /**
   * Scalars arrive several per update; batching them into one signal write per task keeps the
   * charts from re-rendering per point, and the cap keeps a long run from growing without bound.
   */
  private appendPoint(series: string, point: ScalarPoint): void {
    const list = this.pendingPoints.get(series) ?? [];
    list.push(point);
    this.pendingPoints.set(series, list);
    if (!this.flushScheduled) {
      this.flushScheduled = true;
      setTimeout(() => this.flushPoints(), 0);
    }
  }

  private flushPoints(): void {
    this.flushScheduled = false;
    const pending = this.pendingPoints;
    this.pendingPoints = new Map();
    if (pending.size === 0) {
      return;
    }
    this.series.update((s) => {
      const next = { ...s };
      for (const [key, points] of pending) {
        let merged = [...(next[key] ?? []), ...points];
        if (merged.length > MAX_POINTS) {
          merged = merged.filter((_, i) => i % 2 === 0 || i === merged.length - 1);
        }
        next[key] = merged;
      }
      return next;
    });
  }

  /** A replayed terminal status may be stale (the run was resumed meanwhile): confirm before closing. */
  private scheduleCloseIfTerminal(): void {
    if (this.closeTimer) {
      clearTimeout(this.closeTimer);
    }
    this.closeTimer = setTimeout(() => {
      this.closeTimer = null;
      if (!ACTIVE_STATUSES.includes(this.status())) {
        this.source?.close();
        this.source = null;
        this.connected.set(false);
      }
    }, 1500);
  }

  private applyEvent(event: RunEvent, live: boolean): void {
    const run = this.run();
    switch (event.type) {
      case 'status':
        if (run) {
          this.run.set({ ...run, status: event.status, iter: Math.max(run.iter, event.iter), finishedAt: ['completed', 'failed', 'stopped'].includes(event.status) ? event.at : run.finishedAt });
        }
        this.pushEvent(event);
        if (['completed', 'failed', 'stopped'].includes(event.status)) {
          this.scheduleCloseIfTerminal();
        }
        break;
      case 'progress':
        if (run) {
          this.run.set({ ...run, iter: event.iter, itersPerSec: event.itersPerSec, etaSec: event.etaSec, phaseIndex: event.phaseIndex });
        }
        this.nativeIter.set(event.nativeIter);
        break;
      case 'scalar':
        if (live) {
          this.appendPoint(event.series, { iter: event.iter, value: event.value });
        }
        break;
      case 'validation': {
        const point: ValidationPoint = { iter: event.iter, dataset: event.dataset, metrics: event.metrics, images: event.images, checkpoint: this.checkpoints().some((c) => c.iter === event.iter) };
        this.validations.update((v) => [...v, point]);
        if (live) {
          for (const [metric, value] of Object.entries(event.metrics)) {
            this.appendPoint(`val/${event.dataset}/${metric}`, { iter: event.iter, value });
          }
        }
        if (run) {
          this.run.set({ ...this.run()!, lastValidationIter: event.iter });
        }
        this.pushEvent(event);
        break;
      }
      case 'checkpoint':
        this.checkpoints.update((c) => [...c, { iter: event.iter, files: event.files, sizeMb: event.sizeMb, state: event.state }]);
        this.validations.update((v) => v.map((p) => (p.iter === event.iter ? { ...p, checkpoint: true } : p)));
        if (run) {
          this.run.set({ ...this.run()!, lastCheckpointIter: event.iter });
        }
        this.pushEvent(event);
        break;
      case 'phase':
        if (run) {
          this.run.set({ ...this.run()!, phaseIndex: event.phaseIndex });
        }
        this.pushEvent(event);
        break;
      case 'system':
        this.system.set({ gpuUtil: event.gpuUtil, vramUsedGb: event.vramUsedGb, vramTotalGb: event.vramTotalGb, dataWaitPct: event.dataWaitPct });
        if (live) {
          this.appendPoint('sys/gpu_util', { iter: event.iter, value: event.gpuUtil });
          this.appendPoint('sys/vram', { iter: event.iter, value: event.vramUsedGb });
          this.appendPoint('sys/data_wait', { iter: event.iter, value: event.dataWaitPct });
        }
        break;
      case 'warning':
        if (run) {
          this.run.set({ ...this.run()!, warnings: (this.run()!.warnings ?? 0) + (live ? 1 : 0) });
        }
        this.pushEvent(event);
        break;
      case 'log':
        this.logs.update((l) => {
          const next = [...l, { iter: event.iter, level: event.level, line: event.line, at: event.at }];
          return next.length > MAX_LOGS ? next.slice(next.length - MAX_LOGS) : next;
        });
        break;
    }
  }

  /** Timeline events are capped, but phase and status events are never dropped: they are the run's skeleton. */
  private pushEvent(event: RunEvent): void {
    this.events.update((e) => {
      const next = [...e, event];
      if (next.length <= MAX_EVENTS) {
        return next;
      }
      const keep = new Set<number>();
      let budget = MAX_EVENTS;
      for (let i = next.length - 1; i >= 0; i--) {
        const ev = next[i];
        if (ev.type === 'phase' || ev.type === 'status' || budget > 0) {
          keep.add(i);
          if (ev.type !== 'phase' && ev.type !== 'status') {
            budget--;
          }
        }
      }
      return next.filter((_, i) => keep.has(i));
    });
  }
}
