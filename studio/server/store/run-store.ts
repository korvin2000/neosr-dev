/**
 * Runs: simulator state plus the frozen launch data, kept in memory. A run is written to
 * `data/runs/<id>.json` whenever it reaches a terminal status (a real run: on every status change)
 * and reloaded at startup. A real run (`TRAINER_MODE` 'real') keeps its phase TOMLs in
 * `data/runs/<id>/` and is driven by `TrainerRunner` instead of `SimRunner`.
 */
import fs from 'node:fs';
import path from 'node:path';
import { compileRecipe, type CompileResult } from '../../shared/compile/compile';
import { nodeValues } from '../../shared/compile/graph';
import { newId, type Recipe } from '../../shared/recipe.model';
import type { RunDetail, RunEvent, RunPhaseInfo, RunSnapshot, RunSummary, ScalarPoint } from '../../shared/run.model';
import { buildTemplate } from '../../shared/templates';
import { DATA_DIR, GPU, NEOSR_PYTHON, NEOSR_ROOT, TRAINER_MODE } from '../config';
import { assertSafeId, badRequest, HttpError, isRecord, notFound } from '../http';
import { buildSimPlan } from '../sim/sim-plan';
import {
  createSimState,
  downsample,
  EVENT_CAP,
  fastForward,
  finishStop,
  isLiveStatus,
  isTerminalStatus,
  requestStop,
  restoreSimState,
  saveCheckpoint,
  SimRunner,
  trimEvents,
  type SimState,
} from '../sim/trainer-sim';
import { dirsOverlap, experimentsDir, lockOwner, lockPath, namespacePhases } from '../trainer/run-dirs';
import { trainPyLauncher, TrainerRunner } from '../trainer/trainer-runner';

export const DEFAULT_SPEED = 400;
const MAX_SPEED = 100_000;
/** A live real run is also saved on checkpoint and validation events, at most this often. */
const PERSIST_MS = 10_000;

/** Launch data frozen at creation. */
interface RunMeta {
  readonly id: string;
  readonly name: string;
  readonly recipeId: string;
  readonly recipeName: string;
  readonly createdAt: string;
  readonly arch: string;
  readonly scale: number;
  readonly totalIter: number;
  readonly phases: readonly RunPhaseInfo[];
  readonly recipe: Recipe;
  readonly tomls: RunDetail['tomls'];
  readonly commands: readonly string[];
}

/** What the store drives: `SimRunner` (chaos too) or `TrainerRunner`. */
interface Runner {
  launch(): void;
  stop(force: boolean): void;
  resume(): void;
  dispose(): void;
}

export interface RunListener {
  readonly events: (events: readonly RunEvent[]) => void;
  readonly end: () => void;
}

interface RunRecord {
  readonly meta: RunMeta;
  readonly state: SimState;
  readonly speed: number;
  /** Runs the actual trainer, not the simulator. */
  readonly real: boolean;
  runner?: Runner;
  persistedAt?: number;
  readonly listeners: Set<RunListener>;
}

interface PersistedRun {
  readonly format: 'neosr-studio-run';
  readonly speed: number;
  readonly real?: boolean;
  readonly snapshot: RunSnapshot;
}

interface SeedSpec {
  readonly id: string;
  readonly templateId: string;
  readonly hoursAgo: number;
  /** Stop gracefully at this iteration instead of completing. */
  readonly stopAt?: number;
}

const SEEDS: readonly SeedSpec[] = [
  { id: 'run_demo_span', templateId: '4x-span-fidelity', hoursAgo: 72 },
  { id: 'run_demo_curriculum', templateId: '4x-curriculum', hoursAgo: 48 },
  { id: 'run_demo_compact', templateId: '2x-compact-beginner', hoursAgo: 5, stopAt: 12_000 },
];

export class RunStore {
  private readonly runs = new Map<string, RunRecord>();
  /** Set when the backend shuts down: no trainer may start after the shutdown collected its runners. */
  private closing = false;
  /** Runs whose orphaned trainer was asked to stop: polled until its lock is released. */
  private readonly orphanWatch = new Map<string, ReturnType<typeof setInterval>>();

  constructor(private readonly dir: string) {}

  load(): void {
    fs.mkdirSync(this.dir, { recursive: true });
    for (const file of fs.readdirSync(this.dir).filter((f) => f.endsWith('.json'))) {
      try {
        this.restore(JSON.parse(fs.readFileSync(path.join(this.dir, file), 'utf8')));
      } catch (error) {
        console.warn(`[runs] skipped ${file}: ${(error as Error).message}`);
      }
    }
  }

  /** Two completed runs and one stopped run, fast-forwarded without timers. */
  seedIfEmpty(): void {
    if (this.runs.size) {
      return;
    }
    for (const seed of SEEDS) {
      const startMs = Date.now() - seed.hoursAgo * 3_600_000;
      const record = this.createRecord(seed.id, buildTemplate(seed.templateId), 'windows', DEFAULT_SPEED, new Date(startMs - 500).toISOString(), false);
      let now = fastForward(record.state, seed.stopAt ?? Infinity, startMs);
      if (seed.stopAt !== undefined) {
        requestStop(record.state, now);
        saveCheckpoint(record.state, (now += 600));
        finishStop(record.state, (now += 700));
      }
      this.runs.set(seed.id, record);
      this.persist(record);
    }
  }

  list(): RunSummary[] {
    return [...this.runs.values()].map(summarize).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }

  detail(id: string): RunDetail {
    return detailOf(this.get(id));
  }

  snapshot(id: string): RunSnapshot {
    return snapshotOf(this.get(id));
  }

  /** Total and current iteration, for the image endpoint. */
  clock(id: string): { iter: number; totalIter: number } {
    const record = this.get(id);
    return { iter: record.state.iter, totalIter: record.meta.totalIter };
  }

  completedRuns(): RunDetail[] {
    return [...this.runs.values()].filter((r) => r.state.status === 'completed').map(detailOf);
  }

  create(recipe: Recipe, platform: 'windows' | 'linux', speed = DEFAULT_SPEED): RunDetail {
    const record = this.createRecord(newId('run'), recipe, platform, speed, new Date().toISOString(), TRAINER_MODE === 'real');
    if (record.real) {
      this.assertCanTrain(record);
    }
    this.runs.set(record.meta.id, record);
    if (record.real) {
      this.persist(record);
    }
    this.runnerOf(record).launch();
    return detailOf(record);
  }

  stop(id: string, force: boolean): RunSummary {
    const record = this.get(id);
    const { status } = record.state;
    if (isTerminalStatus(status)) {
      throw badRequest(`Run is already ${status}`);
    }
    const orphan = this.orphanOf(record);
    if (orphan) {
      this.stopOrphan(record, orphan, force);
      return summarize(record);
    }
    const graceful = status === 'running' && !force;
    const busy = (status === 'stopping' || status === 'saving') && !force;
    if (!busy) {
      this.runnerOf(record).stop(!graceful);
    }
    return summarize(record);
  }

  resume(id: string): RunSummary {
    const record = this.get(id);
    const { status } = record.state;
    if (status !== 'stopped' && status !== 'disconnected') {
      throw badRequest(`Only stopped or disconnected runs can resume (status: ${status})`);
    }
    if (record.real) {
      this.assertCanTrain(record);
    }
    this.runnerOf(record).resume();
    return summarize(record);
  }

  chaos(id: string, kind: 'disconnect' | 'nan'): RunSummary {
    const record = this.get(id);
    if (record.state.status !== 'running') {
      throw badRequest(`Chaos needs a running run (status: ${record.state.status})`);
    }
    const runner = this.runnerOf(record);
    if (!(runner instanceof SimRunner)) {
      throw badRequest('Chaos is only available for simulated runs');
    }
    if (kind === 'disconnect') {
      runner.disconnect();
    } else {
      runner.injectNan();
    }
    return summarize(record);
  }

  remove(id: string): void {
    const record = this.get(id);
    const orphan = this.orphanOf(record);
    if (orphan) {
      throw new HttpError(409, `Trainer pid ${orphan.pid} still writes ${orphan.dir}: stop the run and wait until it has exited`);
    }
    clearInterval(this.orphanWatch.get(id));
    this.orphanWatch.delete(id);
    record.runner?.dispose();
    for (const listener of record.listeners) {
      listener.end();
    }
    this.runs.delete(id);
    fs.rmSync(path.join(this.dir, `${assertSafeId(id)}.json`), { force: true });
    fs.rmSync(path.join(this.dir, assertSafeId(id)), { recursive: true, force: true });
  }

  /** Backend shutdown: graceful stops of the live real runs, waited for up to `timeoutMs`, then every real run saved. */
  async shutdown(timeoutMs: number): Promise<void> {
    this.closing = true;
    const real = [...this.runs.values()].filter((r) => r.real);
    const exits = real.flatMap((r) => (r.runner instanceof TrainerRunner ? [r.runner.shutdown()] : []));
    let timer: ReturnType<typeof setTimeout> | undefined;
    await Promise.race([Promise.all(exits), new Promise((resolve) => (timer = setTimeout(resolve, timeoutMs)))]);
    clearTimeout(timer);
    real.filter((r) => this.runs.get(r.meta.id) === r).forEach((r) => this.persist(r)); // not one deleted meanwhile
  }

  eventsAfter(id: string, seq: number): RunEvent[] {
    return this.get(id).state.events.filter((e) => e.seq > seq);
  }

  subscribe(id: string, listener: RunListener): () => void {
    const record = this.get(id);
    record.listeners.add(listener);
    return () => record.listeners.delete(listener);
  }

  private get(id: string): RunRecord {
    const record = this.runs.get(id);
    if (!record) {
      throw notFound(`Run ${id}`);
    }
    return record;
  }

  private createRecord(id: string, recipe: Recipe, platform: 'windows' | 'linux', speed: number, createdAt: string, real: boolean): RunRecord {
    const checked = compileOrThrow(recipe, platform);
    const blockers = checked.problems.filter((p) => p.severity === 'blocker');
    if (blockers.length || !checked.phases.length) {
      throw new HttpError(400, 'Recipe has blocking problems', checked.problems);
    }
    const gpus = checked.context.runtime ? Number(nodeValues(checked.context.runtime)['num_gpu']) : 1;
    if (real && gpus > 1) {
      throw badRequest(`num_gpu = ${gpus} needs a distributed launch (torchrun), which the real trainer adapter does not run yet: set num_gpu = 1`);
    }
    const compiled = real ? namespaced(checked, id) : checked;
    const plan = buildSimPlan(id, compiled, GPU.vramGb);
    const tomls = compiled.phases.map((p) => ({ name: p.fileName, text: p.text }));
    const meta: RunMeta = {
      id,
      name: compiled.context.plan ? String(nodeValues(compiled.context.plan)['name'] || recipe.name) : recipe.name,
      recipeId: recipe.id,
      recipeName: recipe.name,
      createdAt,
      arch: plan.arch,
      scale: plan.scale,
      totalIter: plan.totalIter,
      phases: plan.phases.map((p) => p.info),
      recipe,
      tomls,
      commands: compiled.commands,
    };
    const clamped = Math.min(MAX_SPEED, Math.max(1, Math.round(speed)));
    return { meta, state: createSimState(plan), speed: clamped, real, listeners: new Set() };
  }

  private restore(value: unknown): void {
    if (!isRecord(value) || value['format'] !== 'neosr-studio-run' || !isRecord(value['snapshot'])) {
      throw new Error('not a neosr-studio-run file');
    }
    const persisted = value as unknown as PersistedRun;
    const { run } = persisted.snapshot;
    const compiled = compileOrThrow(run.recipe, 'windows');
    const plan = buildSimPlan(run.id, persisted.real ? namespaced(compiled, run.id) : compiled, GPU.vramGb);
    const meta: RunMeta = { ...pickMeta(run), recipe: run.recipe, tomls: run.tomls, commands: run.commands };
    const state = restoreSimState(plan, persisted.snapshot);
    this.runs.set(run.id, { meta, state, speed: persisted.speed || DEFAULT_SPEED, real: persisted.real === true, listeners: new Set() });
  }

  private runnerOf(record: RunRecord): Runner {
    record.runner ??= record.real ? this.trainerRunner(record) : new SimRunner(record.state, record.speed, (events) => this.publish(record, events));
    return record.runner;
  }

  /** Writes the frozen phase TOMLs (again, for a run restored from disk) next to the run file. */
  private trainerRunner(record: RunRecord): TrainerRunner {
    const dir = path.join(this.dir, assertSafeId(record.meta.id));
    fs.mkdirSync(dir, { recursive: true });
    for (const toml of record.meta.tomls) {
      fs.writeFileSync(path.join(dir, path.basename(toml.name)), toml.text, 'utf8');
    }
    return new TrainerRunner(record.state, dir, NEOSR_ROOT, trainPyLauncher(NEOSR_PYTHON, NEOSR_ROOT), (events) => this.publish(record, events));
  }

  /**
   * Refuses (503) while shutting down, (409) while a trainer process still holds one of this run's
   * output folders (an orphan of a crashed or timed-out backend), and, as a guard, while a live
   * real run's folders overlap them (each run has its own `<name>_<run id>`).
   */
  private assertCanTrain(record: RunRecord): void {
    if (this.closing) {
      throw new HttpError(503, 'The backend is shutting down');
    }
    const mine = outputDirs(record);
    for (const dir of mine) {
      const owner = lockOwner(lockPath(dir));
      if (owner) {
        throw new HttpError(409, `A trainer process (pid ${owner.pid}) still writes ${dir}: wait for it to exit, or stop the run`);
      }
    }
    for (const other of this.runs.values()) {
      if (other === record || !other.real || !isLiveStatus(other.state.status)) {
        continue;
      }
      const clash = outputDirs(other).find((d) => mine.some((m) => dirsOverlap(d, m)));
      if (clash) {
        throw new HttpError(409, `Run ${other.meta.id} (${other.state.status}) writes ${clash}: stop it first, or rename this recipe's run`);
      }
    }
  }

  /** A live trainer of this run that no runner here supervises: it survived a backend crash or a timed-out shutdown. */
  private orphanOf(record: RunRecord): { readonly pid: number; readonly verified: boolean; readonly dir: string } | undefined {
    if (!record.real || (record.runner instanceof TrainerRunner && record.runner.running)) {
      return undefined;
    }
    for (const dir of outputDirs(record)) {
      const owner = lockOwner(lockPath(dir));
      if (owner) {
        return { ...owner, dir };
      }
    }
    return undefined;
  }

  /** Signals a verified orphan's process group (it was spawned detached) and reports `stopped` once its lock is released. */
  private stopOrphan(record: RunRecord, orphan: { readonly pid: number; readonly verified: boolean; readonly dir: string }, force: boolean): void {
    if (!orphan.verified) {
      throw new HttpError(409, `Trainer pid ${orphan.pid} from an earlier backend still writes ${orphan.dir}; its identity cannot be verified on this platform, so Studio leaves it alone: end it yourself`);
    }
    const id = record.meta.id;
    if (record.state.status === 'stopping' && !force) {
      return; // a second SIGINT would make the trainer abort its save
    }
    try {
      process.kill(-orphan.pid, force ? 'SIGKILL' : 'SIGINT');
    } catch {
      // exited meanwhile: the watch below reports it
    }
    if (record.state.status !== 'stopping') {
      this.publish(record, requestStop(record.state, Date.now()));
    }
    if (this.orphanWatch.has(id)) {
      return;
    }
    const timer = setInterval(() => {
      if (lockOwner(lockPath(orphan.dir))) {
        return;
      }
      clearInterval(timer);
      this.orphanWatch.delete(id);
      this.publish(record, finishStop(record.state, Date.now()));
    }, 1000);
    timer.unref();
    this.orphanWatch.set(id, timer);
  }

  private publish(record: RunRecord, events: readonly RunEvent[]): void {
    if (this.runs.get(record.meta.id) !== record) {
      return; // deleted: a late event must not write the run back
    }
    for (const listener of record.listeners) {
      listener.events(events);
    }
    const status = events.some((e) => e.type === 'status');
    const progress = events.some((e) => e.type === 'checkpoint' || e.type === 'validation') && Date.now() - (record.persistedAt ?? 0) >= PERSIST_MS;
    if (record.real ? status || progress : status && isTerminalStatus(record.state.status)) {
      this.persist(record);
    }
  }

  private persist(record: RunRecord): void {
    const file: PersistedRun = { format: 'neosr-studio-run', speed: record.speed, real: record.real || undefined, snapshot: snapshotOf(record) };
    try {
      fs.writeFileSync(path.join(this.dir, `${assertSafeId(record.meta.id)}.json`), JSON.stringify(file));
      record.persistedAt = Date.now();
    } catch (error) {
      console.error(`[runs] could not persist ${record.meta.id}: ${(error as Error).message}`);
    }
  }
}

/** A real run's phases with its own output folders (`<name>_<run id>`): names, TOMLs and the plan's `runName`. */
function namespaced(compiled: CompileResult, id: string): CompileResult {
  return { ...compiled, phases: namespacePhases(compiled.phases, id.replace(/^run_/, '')) };
}

function outputDirs(record: RunRecord): string[] {
  return record.meta.tomls.map((t) => experimentsDir(t.text, NEOSR_ROOT));
}

function compileOrThrow(recipe: Recipe, platform: 'windows' | 'linux'): CompileResult {
  try {
    return compileRecipe(recipe, { platform });
  } catch (error) {
    throw badRequest(`Cannot compile recipe: ${(error as Error).message}`);
  }
}

function pickMeta(run: RunSummary): Omit<RunMeta, 'recipe' | 'tomls' | 'commands'> {
  const { id, name, recipeId, recipeName, createdAt, arch, scale, totalIter, phases } = run;
  return { id, name, recipeId, recipeName, createdAt, arch, scale, totalIter, phases };
}

function summarize(record: RunRecord): RunSummary {
  const { meta, state } = record;
  return {
    id: meta.id,
    name: meta.name,
    recipeId: meta.recipeId,
    recipeName: meta.recipeName,
    status: state.status,
    simulated: !record.real,
    createdAt: meta.createdAt,
    startedAt: state.startedAt,
    finishedAt: state.finishedAt,
    iter: state.iter,
    totalIter: meta.totalIter,
    phaseIndex: state.phaseIndex,
    phases: meta.phases,
    itersPerSec: state.itersPerSec,
    etaSec: state.etaSec,
    arch: meta.arch,
    scale: meta.scale,
    bestMetric: state.bestMetric,
    lastCheckpointIter: state.lastCheckpointIter,
    lastValidationIter: state.lastValidationIter,
    warnings: state.warnings,
    gpu: GPU.name,
  };
}

function detailOf(record: RunRecord): RunDetail {
  const { recipe, tomls, commands } = record.meta;
  return { ...summarize(record), recipe, tomls, commands };
}

function snapshotOf(record: RunRecord): RunSnapshot {
  const series: Record<string, ScalarPoint[]> = {};
  for (const [key, points] of Object.entries(record.state.series)) {
    series[key] = downsample(points);
  }
  return { run: detailOf(record), events: trimEvents(record.state.events, EVENT_CAP), series, lastSeq: record.state.seq };
}

export const runStore = new RunStore(path.join(DATA_DIR, 'runs'));
