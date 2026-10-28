/** Everything the simulator needs, derived once from the compiled recipe. */
import { LOSSES } from '../../shared/catalog/losses';
import { OPTIMIZERS } from '../../shared/catalog/optimizers';
import type { CompiledPhase, CompileResult } from '../../shared/compile/compile';
import { activeLossRows, lossRows, nodeValues, phaseNodeValues, scaleOf, slug, valDatasets, type PlanContext } from '../../shared/compile/graph';
import type { RecipeNode } from '../../shared/recipe.model';
import type { RunPhaseInfo } from '../../shared/run.model';
import { hashString } from './prng';

export interface LossSlot {
  readonly slot: string;
  readonly weight: number;
  /** Typical magnitude of the unweighted loss. */
  readonly base: number;
}

export interface LrPlan {
  readonly base: number;
  readonly warmup: number;
  readonly scheduleFree: boolean;
}

export interface SchedulePlan {
  readonly type: string;
  readonly milestones: readonly number[];
  readonly gamma: number;
  readonly tMax: number;
  readonly etaMin: number;
}

export interface SimPhase {
  readonly info: RunPhaseInfo;
  readonly index: number;
  readonly init: string;
  /** Curriculum clock. */
  readonly start: number;
  readonly end: number;
  /** Native clock: restarts at 0 for `finetune`, cumulative for `continue`. */
  readonly nativeStart: number;
  readonly hasGan: boolean;
  readonly batch: number;
  readonly losses: readonly LossSlot[];
  readonly lrG: LrPlan;
  readonly lrD?: LrPlan;
  readonly vramGb: number;
}

export interface ValSet {
  readonly name: string;
  readonly paired: boolean;
}

export interface SimPlan {
  readonly runId: string;
  readonly seed: number;
  readonly arch: string;
  readonly scale: number;
  readonly totalIter: number;
  readonly phases: readonly SimPhase[];
  readonly printFreq: number;
  readonly checkpointFreq: number;
  readonly valFreq?: number;
  readonly valSets: readonly ValSet[];
  readonly metrics: readonly string[];
  readonly schedule?: SchedulePlan;
  readonly nominalItS: number;
  readonly sizeMb: number;
  readonly paramsM: number;
  readonly psnrCeiling: number;
  readonly datasetImages: number;
  readonly vramTotalGb: number;
}

interface ArchProfile {
  readonly itS: number;
  readonly sizeMb: number;
  readonly paramsM: number;
  /** GB per batch sample at patch 64. */
  readonly vramPerSample: number;
  readonly psnrCeiling: number;
}

const ARCH_PROFILES: Readonly<Record<string, ArchProfile>> = {
  compact: { itS: 12, sizeMb: 5, paramsM: 0.6, vramPerSample: 0.2, psnrCeiling: 28.4 },
  span: { itS: 9, sizeMb: 9, paramsM: 2.23, vramPerSample: 0.32, psnrCeiling: 29.6 },
  realplksr: { itS: 4.2, sizeMb: 29, paramsM: 7.39, vramPerSample: 0.85, psnrCeiling: 31.0 },
  hat_s: { itS: 1.1, sizeMb: 38, paramsM: 9.62, vramPerSample: 4.5, psnrCeiling: 32.0 },
  esrgan: { itS: 3, sizeMb: 64, paramsM: 16.7, vramPerSample: 1.1, psnrCeiling: 30.4 },
};
const DEFAULT_PROFILE: ArchProfile = { itS: 3, sizeMb: 20, paramsM: 4.0, vramPerSample: 0.6, psnrCeiling: 29.5 };

const LOSS_BASE: Readonly<Record<string, number>> = {
  pixel: 0.05,
  mssim: 0.08,
  consistency: 0.02,
  ldl: 0.01,
  fdl: 0.4,
  dists: 0.2,
  perceptual: 0.3,
  gan: 0.6,
};

export function archProfile(arch: string): ArchProfile {
  return ARCH_PROFILES[arch] ?? DEFAULT_PROFILE;
}

export function buildSimPlan(runId: string, compiled: CompileResult, vramTotalGb: number): SimPlan {
  const ctx = compiled.context;
  const plan = ctx.plan ? nodeValues(ctx.plan) : {};
  const arch = ctx.model ? String(nodeValues(ctx.model)['type'] ?? 'unknown') : 'unknown';
  const profile = archProfile(arch);
  const seed = hashString(runId);
  const val = ctx.val ? nodeValues(ctx.val) : undefined;
  const phases = compiled.phases.map((phase) => simPhase(phase, ctx, profile, vramTotalGb));
  return {
    runId,
    seed,
    arch,
    scale: scaleOf(ctx.model),
    totalIter: phases.at(-1)?.end ?? 0,
    phases,
    printFreq: positiveInt(plan['print_freq'], 100),
    checkpointFreq: positiveInt(plan['save_checkpoint_freq'], 1000),
    valFreq: val ? positiveInt(val['val_freq'], 1000) : undefined,
    valSets: valDatasets(ctx.val).map((row, i) => ({ name: row.name || (i === 0 ? 'val' : `val_${i}`), paired: row.type === 'paired' })),
    metrics: Array.isArray(val?.['metrics']) ? (val['metrics'] as unknown[]).map(String) : [],
    schedule: ctx.schedule ? schedulePlan(ctx.schedule) : undefined,
    nominalItS: profile.itS,
    sizeMb: profile.sizeMb,
    paramsM: profile.paramsM,
    psnrCeiling: profile.psnrCeiling + ((seed % 61) - 30) / 100,
    datasetImages: 1200 + (seed % 2400),
    vramTotalGb,
  };
}

function simPhase(phase: CompiledPhase, ctx: PlanContext, profile: ArchProfile, vramTotalGb: number): SimPhase {
  const hasGan = phase.hasGan && !!ctx.dmodel && !!ctx.optimD;
  const data = ctx.data ? phaseNodeValues(ctx.data, phase.phase.id) : {};
  const patch = positiveInt(data['patch_size'], 64);
  const batch = positiveInt(data['batch_size'], 8);
  const activations = profile.vramPerSample * batch * (patch / 64) ** 2;
  return {
    info: {
      id: phase.phase.id,
      name: phase.phase.name,
      start: phase.start,
      end: phase.end,
      runName: phase.runName,
      fileName: phase.fileName,
      hasGan,
    },
    index: phase.index,
    init: phase.phase.init,
    start: phase.start,
    end: phase.end,
    nativeStart: phase.nativeStart,
    hasGan,
    batch,
    losses: lossSlots(ctx, phase.phase.id, hasGan),
    lrG: lrPlan(ctx.optimG, phase.phase.id, 1e-3),
    lrD: hasGan ? lrPlan(ctx.optimD, phase.phase.id, 1e-4) : undefined,
    vramGb: Math.min(vramTotalGb * 0.97, 1.4 + activations + (hasGan ? 1.6 : 0)),
  };
}

/** Same slot naming as the compiler's `[train.<slot>_opt]` sections. */
function lossSlots(ctx: PlanContext, phaseId: string, hasGan: boolean): LossSlot[] {
  return activeLossRows(lossRows(ctx.losses), phaseId)
    .filter(({ row }) => row.type !== 'gan_loss' || hasGan)
    .map(({ row, weight }) => {
      const info = LOSSES.find((l) => l.type === row.type);
      const slot = row.params['slot'] ? String(row.params['slot']) : (info?.slot ?? slug(row.type));
      return { slot, weight, base: LOSS_BASE[slot] ?? 0.05 };
    });
}

function lrPlan(node: RecipeNode | undefined, phaseId: string, fallback: number): LrPlan {
  const values = node ? phaseNodeValues(node, phaseId) : {};
  const info = OPTIMIZERS.find((o) => o.type === values['type']);
  return {
    base: positiveNumber(values['lr'], fallback),
    warmup: positiveInt(values['warmup_steps'], 0),
    scheduleFree: info?.scheduleFree ?? false,
  };
}

function schedulePlan(node: RecipeNode): SchedulePlan {
  const v = nodeValues(node);
  return {
    type: String(v['type'] ?? 'multisteplr'),
    milestones: Array.isArray(v['milestones']) ? (v['milestones'] as unknown[]).map(Number).filter(Number.isFinite) : [],
    gamma: positiveNumber(v['gamma'], 0.5),
    tMax: positiveNumber(v['T_max'], 350_000),
    etaMin: Number(v['eta_min']) || 0,
  };
}

function positiveNumber(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function positiveInt(value: unknown, fallback: number): number {
  return Math.round(positiveNumber(value, fallback));
}
