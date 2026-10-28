/** Response shapes of the library and recipe endpoints (not part of `shared/`). */

export interface RecipeSummary {
  readonly id: string;
  readonly name: string;
  readonly updatedAt: string;
  readonly templateId?: string;
  readonly phases: number;
  readonly arch: string;
  readonly scale: number;
  readonly nodeCount: number;
  readonly description: string;
}

export interface ModelFile {
  readonly file: string;
  readonly path: string;
  readonly arch: string;
  readonly scale: number;
  readonly iter: number;
  readonly sizeMb: number;
  readonly runName: string;
  readonly variant: 'ema' | 'eval' | 'state';
  readonly updatedAt: string;
}

export interface DatasetInfo {
  readonly id: string;
  readonly name: string;
  readonly gtPath: string;
  readonly lqPath?: string;
  readonly images: number;
  readonly scale?: number;
  readonly minWidth: number;
  readonly minHeight: number;
  readonly format: string;
  readonly note: string;
}

export interface ScanReport {
  readonly gtCount: number;
  readonly lqCount: number;
  readonly pairs: number;
  readonly unmatchedGt: readonly string[];
  readonly unmatchedLq: readonly string[];
  readonly scaleMismatch: readonly { readonly file: string; readonly gt: readonly [number, number]; readonly lq: readonly [number, number] }[];
  readonly belowPatch: readonly { readonly file: string; readonly w: number; readonly h: number }[];
  readonly channels: { readonly rgb: number; readonly rgba: number; readonly gray: number };
  readonly overlapWithVal: readonly string[];
  readonly samples: readonly { readonly name: string; readonly w: number; readonly h: number }[];
  readonly durationMs: number;
}
