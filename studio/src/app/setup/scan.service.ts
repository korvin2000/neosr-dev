import { Injectable, inject, signal } from '@angular/core';
import { ApiService, type ScanReport } from '../core/api.service';
import { isOffline } from '../library/util';

export interface ScanRequest {
  readonly gt: string;
  readonly lq?: string;
  readonly scale: number;
  readonly patch: number;
}

export interface ScanState {
  readonly status: 'checking' | 'done' | 'offline' | 'error';
  readonly report?: ScanReport;
}

export interface ScanIssue {
  readonly severity: 'blocker' | 'note';
  readonly text: string;
}

/**
 * Dataset checks keyed by their request, shared by the Data and Review steps so a check runs
 * once per folder set and its result survives navigation between the two.
 */
@Injectable({ providedIn: 'root' })
export class ScanService {
  private readonly api = inject(ApiService);
  private readonly states = signal<Readonly<Record<string, ScanState>>>({});

  state(req: ScanRequest): ScanState | undefined {
    return this.states()[key(req)];
  }

  /** Start a check unless the same request already ran (or is running). */
  ensure(req: ScanRequest): void {
    if (!req.gt.trim() || this.state(req)) {
      return;
    }
    void this.run(req);
  }

  async run(req: ScanRequest): Promise<void> {
    const k = key(req);
    this.set(k, { status: 'checking' });
    try {
      const report = await this.api.scanDataset({ gt: req.gt, lq: req.lq, scale: req.scale, patch: req.patch });
      this.set(k, { status: 'done', report });
    } catch (e) {
      this.set(k, { status: isOffline(e) ? 'offline' : 'error' });
    }
  }

  private set(k: string, state: ScanState): void {
    this.states.update((s) => ({ ...s, [k]: state }));
  }
}

function key(req: ScanRequest): string {
  return [req.gt.trim(), (req.lq ?? '').trim(), req.scale, req.patch].join('|');
}

/** What the result line's "n issues" popover lists: blockers in red, notes muted. */
export function scanIssues(r: ScanReport): ScanIssue[] {
  const issues: ScanIssue[] = [];
  const names = (list: readonly string[]) => list.slice(0, 3).join(', ') + (list.length > 3 ? ', …' : '');
  if (r.scaleMismatch.length) {
    const m = r.scaleMismatch[0];
    issues.push({ severity: 'blocker', text: `${r.scaleMismatch.length} pair${r.scaleMismatch.length === 1 ? '' : 's'} not exactly scale× apart (${m.file}: ${m.gt.join('×')} vs ${m.lq.join('×')})` });
  }
  if (r.belowPatch.length) {
    const b = r.belowPatch[0];
    issues.push({ severity: 'blocker', text: `${r.belowPatch.length} image${r.belowPatch.length === 1 ? '' : 's'} smaller than the crop (${b.file}: ${b.w}×${b.h})` });
  }
  if (r.unmatchedGt.length) {
    issues.push({ severity: 'note', text: `${r.unmatchedGt.length} high-res without a low-res twin: ${names(r.unmatchedGt)}` });
  }
  if (r.unmatchedLq.length) {
    issues.push({ severity: 'note', text: `${r.unmatchedLq.length} low-res without a high-res twin: ${names(r.unmatchedLq)}` });
  }
  if (r.overlapWithVal.length) {
    issues.push({ severity: 'note', text: `${r.overlapWithVal.length} also in the validation set: ${names(r.overlapWithVal)}` });
  }
  if (r.channels.rgba) {
    issues.push({ severity: 'note', text: `${r.channels.rgba} RGBA image${r.channels.rgba === 1 ? '' : 's'} (alpha is dropped)` });
  }
  if (r.channels.gray) {
    issues.push({ severity: 'note', text: `${r.channels.gray} grayscale image${r.channels.gray === 1 ? '' : 's'} (trained as RGB)` });
  }
  return issues;
}
