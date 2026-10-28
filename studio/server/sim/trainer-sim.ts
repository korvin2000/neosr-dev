/**
 * Deterministic NeoSR trainer simulator.
 *
 * The core is a set of transition functions over a plain `SimState` (`begin`, `startRunning`,
 * `step`, the stop/resume/chaos transitions) that take the clock as an argument and return the
 * events they emitted, so a run can be fast-forwarded synchronously. `SimRunner` is the thin timer
 * wrapper used for live runs. Every random draw is seeded by (run seed, iteration, channel), so the
 * numbers do not depend on tick size or wall time.
 */
import { METRIC_INFO, type RunEvent, type RunEventBody, type RunStatus, type RunSummary, type ScalarPoint, type ValidationImage } from '../../shared/run.model';
import { IMAGE_IDS, type ImageId } from './images';
import { between, intBetween, noise, rngAt, type Rng } from './prng';
import type { LrPlan, SimPhase, SimPlan, ValSet } from './sim-plan';

export const TICK_MS = 250;
export const EVENT_CAP = 4000;
const TRIM_SLACK = 256;
/** Dropped first when the event log is over the cap, each keeping its newest `keep` events. */
const DROPPABLE: readonly { readonly type: RunEventBody['type']; readonly keep: number }[] = [
  { type: 'progress', keep: 50 },
  { type: 'system', keep: 300 },
  { type: 'log', keep: 500 },
  { type: 'scalar', keep: 1000 },
];
const IMAGE_DIFFICULTY: Readonly<Record<ImageId, number>> = { building: -0.4, foliage: -1, portrait: 0.9, text: 0.5 };

type BestMetric = NonNullable<RunSummary['bestMetric']>;

export interface SimState {
  readonly plan: SimPlan;
  status: RunStatus;
  /** Curriculum clock. */
  iter: number;
  phaseIndex: number;
  seq: number;
  ticks: number;
  events: RunEvent[];
  series: Record<string, ScalarPoint[]>;
  itersPerSec: number;
  etaSec: number;
  bestMetric?: BestMetric;
  lastCheckpointIter?: number;
  lastValidationIter?: number;
  warnings: number;
  lastSystemAt: number;
  ganAnnounced: boolean;
  startedAt?: string;
  finishedAt?: string;
}

/** The persisted part of a state; the plan is re-derived from the recipe. */
export interface SimProgress {
  readonly run: RunSummary;
  readonly events: readonly RunEvent[];
  readonly series: Readonly<Record<string, readonly ScalarPoint[]>>;
  readonly lastSeq: number;
}

export function createSimState(plan: SimPlan): SimState {
  return {
    plan,
    status: 'queued',
    iter: 0,
    phaseIndex: 0,
    seq: 0,
    ticks: 0,
    events: [],
    series: {},
    itersPerSec: 0,
    etaSec: 0,
    warnings: 0,
    lastSystemAt: -Infinity,
    ganAnnounced: false,
  };
}

export function restoreSimState(plan: SimPlan, saved: SimProgress): SimState {
  const { run } = saved;
  const phaseIndex = Math.min(run.phaseIndex, Math.max(0, plan.phases.length - 1));
  return {
    ...createSimState(plan),
    status: isLiveStatus(run.status) ? 'disconnected' : run.status,
    iter: run.iter,
    phaseIndex,
    seq: saved.lastSeq,
    events: [...saved.events],
    series: Object.fromEntries(Object.entries(saved.series).map(([k, v]) => [k, [...v]])),
    itersPerSec: run.itersPerSec,
    etaSec: run.etaSec,
    bestMetric: run.bestMetric,
    lastCheckpointIter: run.lastCheckpointIter,
    lastValidationIter: run.lastValidationIter,
    warnings: run.warnings,
    ganAnnounced: plan.phases.slice(0, phaseIndex + 1).some((p) => p.hasGan),
    startedAt: run.startedAt,
    finishedAt: run.finishedAt,
  };
}

export function isLiveStatus(status: RunStatus): boolean {
  return status === 'queued' || status === 'starting' || status === 'running' || status === 'stopping' || status === 'saving';
}

export function isTerminalStatus(status: RunStatus): boolean {
  return status === 'stopped' || status === 'completed' || status === 'failed';
}

// --- transitions -----------------------------------------------------------------------------

/** `queued` → `starting`. */
export function begin(s: SimState, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  s.startedAt = iso(now);
  setStatus(s, out, now, 'starting');
  log(s, out, now, `Loading dataset… ${s.plan.datasetImages.toLocaleString('en-US')} images`);
  log(s, out, now, `Building network ${s.plan.arch}…`);
  log(s, out, now, `Params: ${s.plan.paramsM.toFixed(2)}M`);
  return out;
}

/** `starting` → `running`, entering the first phase. */
export function startRunning(s: SimState, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  setStatus(s, out, now, 'running');
  enterPhase(s, out, now);
  return out;
}

/** Advances up to `n` updates, never past the current phase boundary. */
export function step(s: SimState, n: number, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  const phase = currentPhase(s);
  if (s.status !== 'running' || !phase || n < 1) {
    return out;
  }
  s.ticks++;
  updateSpeed(s);
  const target = Math.min(s.iter + Math.floor(n), phase.end);
  while (s.iter < target) {
    s.iter++;
    periodicEvents(s, out, now, phase);
  }
  tickEvents(s, out, now, phase);
  if (s.iter >= phase.end) {
    finishPhase(s, out, now, phase);
  }
  return out;
}

export function requestStop(s: SimState, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  setStatus(s, out, now, 'stopping', 'stop requested');
  log(s, out, now, 'Stop requested: finishing the current update');
  return out;
}

/** `stopping` → `saving`, writing a checkpoint at the current native iteration. */
export function saveCheckpoint(s: SimState, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  const phase = currentPhase(s);
  setStatus(s, out, now, 'saving');
  if (phase) {
    log(s, out, now, `Saving training state at iter ${nativeOf(phase, s.iter).toLocaleString('en-US')}`);
    emitCheckpoint(s, out, now, phase);
  }
  return out;
}

export function finishStop(s: SimState, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  s.finishedAt = iso(now);
  s.etaSec = 0;
  setStatus(s, out, now, 'stopped');
  return out;
}

export function forceStop(s: SimState, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  log(s, out, now, 'Trainer process killed: no checkpoint written', 'warn');
  return [...out, ...finishStop(s, now)];
}

/** NeoSR `--auto_resume`: continue from the last checkpoint. */
export function resume(s: SimState, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  const from = s.lastCheckpointIter ?? 0;
  const savedIn = s.plan.phases.find((p) => from > p.start && from <= p.end);
  s.iter = from;
  s.phaseIndex = phaseIndexAt(s.plan, from);
  s.finishedAt = undefined;
  truncateSeries(s, from);
  setStatus(s, out, now, 'running', 'resumed');
  if (savedIn) {
    log(s, out, now, `Resuming from experiments/${savedIn.info.runName}/training_states/${nativeOf(savedIn, from)}.state`);
  } else {
    log(s, out, now, 'No training state found: starting from scratch', 'warn');
  }
  enterPhase(s, out, now);
  return out;
}

export function disconnect(s: SimState, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  log(s, out, now, 'Lost connection to the trainer process', 'error');
  setStatus(s, out, now, 'disconnected', 'no output from the trainer');
  return out;
}

export function injectNan(s: SimState, now: number): RunEvent[] {
  const out: RunEvent[] = [];
  warn(s, out, now, 'NONFINITE', 'non-finite loss, update skipped');
  emit(s, out, now, { type: 'scalar', series: 'skipped/g', value: 1 });
  return out;
}

/** Runs synchronously from `queued` up to `untilIter` (or completion) on a synthetic clock; returns that clock. */
export function fastForward(s: SimState, untilIter: number, startMs: number, chunk = 250): number {
  let now = startMs;
  if (s.status === 'queued') {
    begin(s, now);
    now += 1000;
    startRunning(s, now);
  }
  while (s.status === 'running' && s.iter < untilIter) {
    const n = Math.min(chunk, untilIter - s.iter);
    now += (n / s.plan.nominalItS) * 1000;
    step(s, n, now);
  }
  return now;
}

// --- phase handling --------------------------------------------------------------------------

function currentPhase(s: SimState): SimPhase | undefined {
  return s.plan.phases[s.phaseIndex];
}

function phaseIndexAt(plan: SimPlan, iter: number): number {
  const index = plan.phases.findIndex((p) => iter < p.end);
  return index === -1 ? Math.max(0, plan.phases.length - 1) : index;
}

function nativeOf(phase: SimPhase, iter: number): number {
  return phase.nativeStart + (iter - phase.start);
}

function enterPhase(s: SimState, out: RunEvent[], now: number): void {
  const phase = currentPhase(s);
  if (!phase) {
    return;
  }
  emit(s, out, now, { type: 'phase', phaseIndex: phase.index, name: phase.info.name, runName: phase.info.runName });
  log(s, out, now, `Starting ${phase.info.fileName} (${phase.init}, iter ${nativeOf(phase, s.iter).toLocaleString('en-US')})`);
  if (phase.hasGan && !s.ganAnnounced) {
    s.ganAnnounced = true;
    warn(s, out, now, 'GAN_START', 'discriminator enabled: expect l_g_total to rise');
  }
}

function finishPhase(s: SimState, out: RunEvent[], now: number, phase: SimPhase): void {
  const last = phase.index === s.plan.phases.length - 1;
  if (last && s.plan.valFreq && s.lastValidationIter !== s.iter) {
    emitValidation(s, out, now, phase);
  }
  if (s.lastCheckpointIter !== s.iter) {
    emitCheckpoint(s, out, now, phase);
  }
  if (last) {
    log(s, out, now, 'Training finished');
    s.finishedAt = iso(now);
    s.etaSec = 0;
    setStatus(s, out, now, 'completed');
    return;
  }
  log(s, out, now, `End of ${phase.info.runName}: saved net_g_latest.pth`);
  s.phaseIndex++;
  enterPhase(s, out, now);
}

// --- periodic events -------------------------------------------------------------------------

function periodicEvents(s: SimState, out: RunEvent[], now: number, phase: SimPhase): void {
  const native = nativeOf(phase, s.iter);
  const { printFreq, valFreq, checkpointFreq } = s.plan;
  if (native % printFreq === 0) {
    emitTrainScalars(s, out, now, phase, native);
  }
  if (valFreq && native % valFreq === 0) {
    emitValidation(s, out, now, phase);
  }
  if (native % checkpointFreq === 0) {
    emitCheckpoint(s, out, now, phase);
  }
}

function emitTrainScalars(s: SimState, out: RunEvent[], now: number, phase: SimPhase, native: number): void {
  const rng = rngAt(s.plan.seed, 'train', s.iter);
  const decay = 0.35 + 0.65 * Math.exp(-native / 60_000);
  const scalar = (series: string, value: number) => emit(s, out, now, { type: 'scalar', series, value: round(value, 6) });
  const parts: string[] = [];
  let total = 0;
  for (const loss of phase.losses) {
    const value = loss.weight * loss.base * decay * (1 + noise(rng, 0.08));
    total += value;
    scalar(`loss/g/${loss.slot}`, value);
    parts.push(`l_g_${loss.slot}: ${value.toFixed(4)}`);
  }
  scalar('loss/g/total', total);
  parts.push(`l_g_total: ${total.toFixed(4)}`);
  if (phase.hasGan) {
    const gan = ganProgress(native);
    const lossD = 0.69 - 0.15 * gan + noise(rng, 0.04);
    scalar('loss/d/total', lossD);
    scalar('out/d/real', 0.55 + 0.25 * gan + noise(rng, 0.03));
    scalar('out/d/fake', 0.45 - 0.25 * gan + noise(rng, 0.03));
    parts.push(`l_d_total: ${lossD.toFixed(4)}`);
  }
  const lrG = learningRate(s.plan, phase.lrG, native);
  const lrD = phase.lrD ? learningRate(s.plan, phase.lrD, native) : undefined;
  scalar('lr/g', lrG);
  if (lrD !== undefined) {
    scalar('lr/d', lrD);
  }
  scalar('grad_norm/g', 0.3 + 0.9 * Math.min(1, 0.55 * decay + 0.45 * rng()));
  const skipped = rng() < 0.004 ? 1 : 0;
  scalar('skipped/g', skipped);
  if (skipped) {
    warn(s, out, now, 'NONFINITE', 'non-finite loss, update skipped');
  }
  log(s, out, now, logLine(s, phase, native, lrD === undefined ? [lrG] : [lrG, lrD], parts));
}

/** `[epoch: 12, iter: 12,300, lr:(1.000e-03,)] [performance: 3.91 it/s] [eta: 0:14:32] l_g_…` */
function logLine(s: SimState, phase: SimPhase, native: number, lrs: readonly number[], parts: readonly string[]): string {
  const epoch = Math.floor((native * phase.batch) / s.plan.datasetImages);
  const lr = lrs.map((v) => `${scientific(v)},`).join('');
  const itersPerSec = s.itersPerSec || s.plan.nominalItS;
  const eta = (s.plan.totalIter - s.iter) / itersPerSec;
  return `[epoch: ${epoch}, iter: ${native.toLocaleString('en-US')}, lr:(${lr})] [performance: ${itersPerSec.toFixed(2)} it/s] [eta: ${formatEta(eta)}] ${parts.join(' ')}`;
}

function learningRate(plan: SimPlan, lr: LrPlan, native: number): number {
  if (lr.scheduleFree) {
    return lr.warmup > 0 ? lr.base * Math.min(1, native / lr.warmup) : lr.base;
  }
  const schedule = plan.schedule;
  if (!schedule) {
    return lr.base;
  }
  if (schedule.type === 'cosineannealing') {
    const t = Math.min(native, schedule.tMax) / schedule.tMax;
    return schedule.etaMin + ((lr.base - schedule.etaMin) * (1 + Math.cos(Math.PI * t))) / 2;
  }
  const passed = schedule.milestones.filter((m) => native >= m).length;
  return lr.base * schedule.gamma ** passed;
}

function ganProgress(native: number): number {
  return 1 - Math.exp(-native / 20_000);
}

function emitValidation(s: SimState, out: RunEvent[], now: number, phase: SimPhase): void {
  if (!s.plan.valSets.length) {
    return;
  }
  s.plan.valSets.forEach((set, i) => {
    const rng = rngAt(s.plan.seed, 'val', s.iter, set.name);
    const metrics = set.paired ? metricValues(s, phase, rng) : {};
    const images = IMAGE_IDS.map((id) => validationImage(s, set, id, metrics, rng));
    emit(s, out, now, { type: 'validation', dataset: set.name, metrics, images });
    if (i === 0) {
      updateBest(s, metrics);
    }
  });
  s.lastValidationIter = s.iter;
  if (rngAt(s.plan.seed, 'val-cost', s.iter)() < 0.04) {
    warn(s, out, now, 'VAL_COST', 'validation took 9% of the last interval');
  }
}

function metricValues(s: SimState, phase: SimPhase, rng: Rng): Record<string, number> {
  const tau = Math.max(s.plan.totalIter / 4, 8000);
  const f = 1 - Math.exp(-s.iter / tau);
  const g = phase.hasGan ? ganProgress(nativeOf(phase, s.iter)) : 0;
  const values: Record<string, number> = {};
  for (const name of s.plan.metrics) {
    values[name] = round(metricCurve(name, f, g, s.plan.psnrCeiling, rng), 4);
  }
  return values;
}

function metricCurve(name: string, f: number, g: number, ceiling: number, rng: Rng): number {
  switch (name) {
    case 'psnr':
      return 24 + (ceiling - 24) * f - 0.8 * g + noise(rng, 0.15);
    case 'ssim':
      return 0.7 + 0.2 * f - 0.02 * g + noise(rng, 0.004);
    case 'dists':
      return 0.3 - 0.18 * f - 0.03 * g + noise(rng, 0.004);
    case 'topiq':
      return 0.45 + 0.25 * f + 0.04 * g + noise(rng, 0.006);
    default:
      return 0.5 + 0.3 * f + noise(rng, 0.01);
  }
}

const METRIC_SPREAD: Readonly<Record<string, number>> = { psnr: 2, ssim: 0.04, dists: 0.04, topiq: 0.05 };

function validationImage(s: SimState, set: ValSet, id: ImageId, metrics: Readonly<Record<string, number>>, rng: Rng): ValidationImage {
  const url = (kind: string) => `/api/runs/${s.plan.runId}/images/${id}?kind=${kind}&iter=${s.iter}`;
  const perImage: Record<string, number> = {};
  for (const [name, value] of Object.entries(metrics)) {
    const sign = METRIC_INFO[name]?.better === 'lower' ? -1 : 1;
    const spread = METRIC_SPREAD[name] ?? 0.05;
    perImage[name] = round(value + sign * spread * (IMAGE_DIFFICULTY[id] + noise(rng, 0.1)), 4);
  }
  return {
    id,
    name: `${id}.png`,
    lqUrl: url('lq'),
    srUrl: url('sr'),
    gtUrl: set.paired ? url('gt') : undefined,
    metrics: perImage,
  };
}

function updateBest(s: SimState, metrics: Readonly<Record<string, number>>): void {
  const name = s.plan.metrics.find((m) => m in metrics);
  if (!name) {
    return;
  }
  const value = metrics[name] as number;
  const lower = METRIC_INFO[name]?.better === 'lower';
  const best = s.bestMetric;
  if (!best || (lower ? value < best.value : value > best.value)) {
    s.bestMetric = { name, value, iter: s.iter };
  }
}

function emitCheckpoint(s: SimState, out: RunEvent[], now: number, phase: SimPhase): void {
  const native = nativeOf(phase, s.iter);
  const files = [`net_g_${native}.pth`, ...(phase.hasGan ? [`net_d_${native}.pth`] : []), `${native}.state`];
  emit(s, out, now, { type: 'checkpoint', files, sizeMb: s.plan.sizeMb, state: true });
  s.lastCheckpointIter = s.iter;
}

// --- per-tick events -------------------------------------------------------------------------

function updateSpeed(s: SimState): void {
  const rng = rngAt(s.plan.seed, 'speed', s.ticks);
  s.itersPerSec = round(s.plan.nominalItS * between(rng, 0.9, 1.1), 2);
  s.etaSec = Math.round((s.plan.totalIter - s.iter) / s.itersPerSec);
}

function tickEvents(s: SimState, out: RunEvent[], now: number, phase: SimPhase): void {
  s.etaSec = Math.round((s.plan.totalIter - s.iter) / s.itersPerSec);
  emit(s, out, now, { type: 'progress', itersPerSec: s.itersPerSec, etaSec: s.etaSec, phaseIndex: phase.index, nativeIter: nativeOf(phase, s.iter) });
  if (now - s.lastSystemAt < 1000) {
    return;
  }
  s.lastSystemAt = now;
  const rng = rngAt(s.plan.seed, 'system', s.ticks);
  emit(s, out, now, {
    type: 'system',
    gpuUtil: intBetween(rng, 85, 99),
    vramUsedGb: round(phase.vramGb * (1 + noise(rng, 0.02)), 2),
    vramTotalGb: s.plan.vramTotalGb,
    dataWaitPct: round(between(rng, 2, 12), 1),
  });
}

// --- emission --------------------------------------------------------------------------------

function emit(s: SimState, out: RunEvent[], now: number, body: RunEventBody): void {
  const event: RunEvent = { ...body, seq: ++s.seq, runId: s.plan.runId, iter: s.iter, at: iso(now) };
  s.events.push(event);
  out.push(event);
  recordSeries(s, event);
  if (s.events.length > EVENT_CAP + TRIM_SLACK) {
    s.events = trimEvents(s.events, EVENT_CAP);
  }
}

function setStatus(s: SimState, out: RunEvent[], now: number, status: RunStatus, message?: string): void {
  s.status = status;
  emit(s, out, now, { type: 'status', status, message });
}

function log(s: SimState, out: RunEvent[], now: number, line: string, level: 'info' | 'warn' | 'error' = 'info'): void {
  emit(s, out, now, { type: 'log', level, line });
}

function warn(s: SimState, out: RunEvent[], now: number, code: string, message: string): void {
  s.warnings++;
  emit(s, out, now, { type: 'warning', code, message });
}

function recordSeries(s: SimState, event: RunEvent): void {
  const push = (series: string, value: number) => (s.series[series] ??= []).push({ iter: event.iter, value });
  switch (event.type) {
    case 'scalar':
      push(event.series, event.value);
      break;
    case 'validation':
      for (const [metric, value] of Object.entries(event.metrics)) {
        push(`val/${event.dataset}/${metric}`, value);
      }
      break;
    case 'system':
      push('sys/gpu_util', event.gpuUtil);
      push('sys/vram', event.vramUsedGb);
      push('sys/data_wait', event.dataWaitPct);
      break;
    default:
      break;
  }
}

/** Drops the oldest events of the droppable kinds (in order), then the oldest of any kind. */
export function trimEvents(events: readonly RunEvent[], cap: number): RunEvent[] {
  let excess = events.length - cap;
  if (excess <= 0) {
    return [...events];
  }
  const dropLeft = new Map<string, number>();
  for (const { type, keep } of DROPPABLE) {
    const count = events.filter((e) => e.type === type).length;
    const drop = Math.min(excess, Math.max(0, count - keep));
    dropLeft.set(type, drop);
    excess -= drop;
  }
  const kept = events.filter((e) => {
    const left = dropLeft.get(e.type) ?? 0;
    if (left > 0) {
      dropLeft.set(e.type, left - 1);
      return false;
    }
    return true;
  });
  return kept.slice(Math.max(0, kept.length - cap));
}

/** Stride-downsampled copy, always keeping the last point. */
export function downsample(points: readonly ScalarPoint[], max = 2000): ScalarPoint[] {
  if (points.length <= max) {
    return [...points];
  }
  const stride = Math.ceil(points.length / max);
  const out = points.filter((_, i) => i % stride === 0);
  const last = points[points.length - 1] as ScalarPoint;
  if (out[out.length - 1] !== last) {
    out[out.length - 1] = last;
  }
  return out;
}

function truncateSeries(s: SimState, iter: number): void {
  for (const key of Object.keys(s.series)) {
    s.series[key] = (s.series[key] ?? []).filter((p) => p.iter <= iter);
  }
}

// --- formatting ------------------------------------------------------------------------------

function iso(ms: number): string {
  return new Date(ms).toISOString();
}

function round(value: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}

/** `1.000e-03`, as Python prints it. */
function scientific(value: number): string {
  return value.toExponential(3).replace(/e([+-])(\d)$/, 'e$10$2');
}

function formatEta(sec: number): string {
  const total = Math.max(0, Math.round(sec));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const ss = total % 60;
  return `${h}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

// --- timer wrapper ---------------------------------------------------------------------------

export type Publish = (events: readonly RunEvent[]) => void;

/** Drives a `SimState` with timers; `speed` is simulated updates per real second. */
export class SimRunner {
  private ticker?: ReturnType<typeof setInterval>;
  private readonly timeouts = new Set<ReturnType<typeof setTimeout>>();

  constructor(
    private readonly state: SimState,
    private readonly speed: number,
    private readonly publish: Publish,
  ) {}

  /** `queued` → (0.5 s) `starting` → (1 s) `running`. */
  launch(): void {
    this.after(500, () => {
      this.emit(begin(this.state, Date.now()));
      this.after(1000, () => {
        this.emit(startRunning(this.state, Date.now()));
        this.startTicking();
      });
    });
  }

  stop(force: boolean): void {
    this.cancelAll();
    if (force) {
      this.emit(forceStop(this.state, Date.now()));
      return;
    }
    this.emit(requestStop(this.state, Date.now()));
    this.after(600, () => {
      this.emit(saveCheckpoint(this.state, Date.now()));
      this.after(700, () => this.emit(finishStop(this.state, Date.now())));
    });
  }

  resume(): void {
    this.cancelAll();
    this.emit(resume(this.state, Date.now()));
    this.startTicking();
  }

  disconnect(): void {
    this.cancelAll();
    this.emit(disconnect(this.state, Date.now()));
  }

  injectNan(): void {
    this.emit(injectNan(this.state, Date.now()));
  }

  dispose(): void {
    this.cancelAll();
  }

  private tick(): void {
    this.emit(step(this.state, Math.max(1, Math.round(this.speed / 4)), Date.now()));
    if (this.state.status !== 'running') {
      this.stopTicking();
    }
  }

  private emit(events: readonly RunEvent[]): void {
    if (events.length) {
      this.publish(events);
    }
  }

  private startTicking(): void {
    this.stopTicking();
    this.ticker = setInterval(() => this.tick(), TICK_MS);
  }

  private stopTicking(): void {
    if (this.ticker) {
      clearInterval(this.ticker);
      this.ticker = undefined;
    }
  }

  private after(ms: number, fn: () => void): void {
    const handle = setTimeout(() => {
      this.timeouts.delete(handle);
      fn();
    }, ms);
    this.timeouts.add(handle);
  }

  private cancelAll(): void {
    this.stopTicking();
    for (const handle of this.timeouts) {
      clearTimeout(handle);
    }
    this.timeouts.clear();
  }
}
