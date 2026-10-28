import type { Recipe } from './recipe.model';

/**
 * Runs and the event stream. The shapes are what a real NeoSR adapter would produce from the
 * trainer's stdout / tb_logger; the prototype's simulator produces the same shapes and labels
 * itself as simulated.
 */

export type RunStatus =
  | 'queued'
  | 'starting'
  | 'running'
  | 'stopping'
  | 'saving'
  | 'stopped'
  | 'completed'
  | 'failed'
  | 'disconnected';

export interface RunPhaseInfo {
  readonly id: string;
  readonly name: string;
  /** Curriculum clock. */
  readonly start: number;
  readonly end: number;
  /** NeoSR run name and TOML of this phase. */
  readonly runName: string;
  readonly fileName: string;
  readonly hasGan: boolean;
}

export interface RunSummary {
  readonly id: string;
  readonly name: string;
  readonly recipeId: string;
  readonly recipeName: string;
  readonly status: RunStatus;
  readonly simulated: boolean;
  readonly createdAt: string;
  readonly startedAt?: string;
  readonly finishedAt?: string;
  /** Curriculum iteration (sum over phases). */
  readonly iter: number;
  readonly totalIter: number;
  readonly phaseIndex: number;
  readonly phases: readonly RunPhaseInfo[];
  readonly itersPerSec: number;
  readonly etaSec: number;
  readonly arch: string;
  readonly scale: number;
  readonly bestMetric?: { readonly name: string; readonly value: number; readonly iter: number };
  readonly lastCheckpointIter?: number;
  readonly lastValidationIter?: number;
  readonly warnings: number;
  readonly gpu: string;
}

export interface RunDetail extends RunSummary {
  /** Frozen at launch. */
  readonly recipe: Recipe;
  /** One compiled TOML per phase, in order. */
  readonly tomls: readonly { readonly name: string; readonly text: string }[];
  readonly commands: readonly string[];
}

export interface ScalarPoint {
  readonly iter: number;
  readonly value: number;
}

export interface ValidationImage {
  readonly id: string;
  readonly name: string;
  readonly lqUrl: string;
  readonly srUrl: string;
  readonly gtUrl?: string;
  readonly metrics: Readonly<Record<string, number>>;
}

export type RunEventBody =
  | { readonly type: 'status'; readonly status: RunStatus; readonly message?: string }
  | { readonly type: 'progress'; readonly itersPerSec: number; readonly etaSec: number; readonly phaseIndex: number; readonly nativeIter: number }
  /** `series` such as `loss/g/mssim`, `loss/d/total`, `lr/g`, `grad_norm/g`, `skipped/g`. Values are logged window means. */
  | { readonly type: 'scalar'; readonly series: string; readonly value: number }
  | { readonly type: 'validation'; readonly dataset: string; readonly metrics: Readonly<Record<string, number>>; readonly images: readonly ValidationImage[] }
  | { readonly type: 'checkpoint'; readonly files: readonly string[]; readonly sizeMb: number; readonly state: boolean }
  | { readonly type: 'phase'; readonly phaseIndex: number; readonly name: string; readonly runName: string }
  | { readonly type: 'system'; readonly gpuUtil: number; readonly vramUsedGb: number; readonly vramTotalGb: number; readonly dataWaitPct: number }
  | { readonly type: 'warning'; readonly code: string; readonly message: string }
  | { readonly type: 'log'; readonly level: 'info' | 'warn' | 'error'; readonly line: string };

/** Envelope: sequence id for replay/reconnect, run and curriculum clock. */
export type RunEvent = RunEventBody & {
  readonly seq: number;
  readonly runId: string;
  readonly iter: number;
  readonly at: string;
};

/** Everything a cockpit needs after reconnecting: the event log replayed into state. */
export interface RunSnapshot {
  readonly run: RunDetail;
  readonly events: readonly RunEvent[];
  readonly series: Readonly<Record<string, readonly ScalarPoint[]>>;
  readonly lastSeq: number;
}

export interface MetricInfo {
  readonly key: string;
  readonly label: string;
  readonly better: 'higher' | 'lower';
  readonly unit?: string;
}

export const METRIC_INFO: Record<string, MetricInfo> = {
  psnr: { key: 'psnr', label: 'PSNR', better: 'higher', unit: 'dB' },
  ssim: { key: 'ssim', label: 'SSIM', better: 'higher' },
  dists: { key: 'dists', label: 'DISTS', better: 'lower' },
  topiq: { key: 'topiq', label: 'TOPIQ', better: 'higher' },
};

export interface GpuInfo {
  readonly name: string;
  readonly vramGb: number;
  readonly computeCapability: string;
  readonly driver: string;
}

export interface EnvironmentInfo {
  readonly platform: 'windows' | 'linux';
  readonly neosrPath: string;
  readonly neosrVersion: string;
  readonly experimentsRoot: string;
  readonly gpus: readonly GpuInfo[];
  readonly freeDiskGb: number;
  readonly simulated: boolean;
}
