/**
 * Runs: simulator state plus the frozen launch data, kept in memory. A run is written to
 * `data/runs/<id>.json` whenever it reaches a terminal status and reloaded at startup.
 */
import fs from 'node:fs';
import path from 'node:path';
import { compileRecipe, type CompileResult } from '../../shared/compile/compile';
import { nodeValues } from '../../shared/compile/graph';
import { newId, type Recipe } from '../../shared/recipe.model';
import type { RunDetail, RunEvent, RunPhaseInfo, RunSnapshot, RunSummary, ScalarPoint } from '../../shared/run.model';
import { buildTemplate } from '../../shared/templates';
import { DATA_DIR, GPU } from '../config';
import { assertSafeId, badRequest, HttpError, isRecord, notFound } from '../http';
import { buildSimPlan } from '../sim/sim-plan';
import {
  createSimState,
  downsample,
  EVENT_CAP,
  fastForward,
  finishStop,
  isTerminalStatus,
  requestStop,
  restoreSimState,
  saveCheckpoint,
  SimRunner,
  trimEvents,
  type SimState,
} from '../sim/trainer-sim';

export const DEFAULT_SPEED = 400;
const MAX_SPEED = 100_000;

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

export interface RunListener {
  readonly events: (events: readonly RunEvent[]) => void;
  readonly end: () => void;
}

interface RunRecord {
  readonly meta: RunMeta;
  readonly state: SimState;
  readonly speed: number;
  runner?: SimRunner;
  readonly listeners: Set<RunListener>;
}

interface PersistedRun {
  readonly format: 'neosr-studio-run';
  readonly speed: number;
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
      const record = this.createRecord(seed.id, buildTemplate(seed.templateId), 'windows', DEFAULT_SPEED, new Date(startMs - 500).toISOString());
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
    const record = this.createRecord(newId('run'), recipe, platform, speed, new Date().toISOString());
    this.runs.set(record.meta.id, record);
    this.runnerOf(record).launch();
    return detailOf(record);
  }

  stop(id: string, force: boolean): RunSummary {
    const record = this.get(id);
    const { status } = record.state;
    if (isTerminalStatus(status)) {
      throw badRequest(`Run is already ${status}`);
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
    this.runnerOf(record).resume();
    return summarize(record);
  }

  chaos(id: string, kind: 'disconnect' | 'nan'): RunSummary {
    const record = this.get(id);
    if (record.state.status !== 'running') {
      throw badRequest(`Chaos needs a running run (status: ${record.state.status})`);
    }
    const runner = this.runnerOf(record);
    if (kind === 'disconnect') {
      runner.disconnect();
    } else {
      runner.injectNan();
    }
    return summarize(record);
  }

  remove(id: string): void {
    const record = this.get(id);
    record.runner?.dispose();
    for (const listener of record.listeners) {
      listener.end();
    }
    this.runs.delete(id);
    fs.rmSync(path.join(this.dir, `${assertSafeId(id)}.json`), { force: true });
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

  private createRecord(id: string, recipe: Recipe, platform: 'windows' | 'linux', speed: number, createdAt: string): RunRecord {
    const compiled = compileOrThrow(recipe, platform);
    const blockers = compiled.problems.filter((p) => p.severity === 'blocker');
    if (blockers.length || !compiled.phases.length) {
      throw new HttpError(400, 'Recipe has blocking problems', compiled.problems);
    }
    const plan = buildSimPlan(id, compiled, GPU.vramGb);
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
      tomls: compiled.phases.map((p) => ({ name: p.fileName, text: p.text })),
      commands: compiled.commands,
    };
    const clamped = Math.min(MAX_SPEED, Math.max(1, Math.round(speed)));
    return { meta, state: createSimState(plan), speed: clamped, listeners: new Set() };
  }

  private restore(value: unknown): void {
    if (!isRecord(value) || value['format'] !== 'neosr-studio-run' || !isRecord(value['snapshot'])) {
      throw new Error('not a neosr-studio-run file');
    }
    const persisted = value as unknown as PersistedRun;
    const { run } = persisted.snapshot;
    const compiled = compileOrThrow(run.recipe, 'windows');
    const plan = buildSimPlan(run.id, compiled, GPU.vramGb);
    const meta: RunMeta = { ...pickMeta(run), recipe: run.recipe, tomls: run.tomls, commands: run.commands };
    const state = restoreSimState(plan, persisted.snapshot);
    this.runs.set(run.id, { meta, state, speed: persisted.speed || DEFAULT_SPEED, listeners: new Set() });
  }

  private runnerOf(record: RunRecord): SimRunner {
    record.runner ??= new SimRunner(record.state, record.speed, (events) => this.publish(record, events));
    return record.runner;
  }

  private publish(record: RunRecord, events: readonly RunEvent[]): void {
    for (const listener of record.listeners) {
      listener.events(events);
    }
    if (isTerminalStatus(record.state.status) && events.some((e) => e.type === 'status')) {
      this.persist(record);
    }
  }

  private persist(record: RunRecord): void {
    const file: PersistedRun = { format: 'neosr-studio-run', speed: record.speed, snapshot: snapshotOf(record) };
    try {
      fs.writeFileSync(path.join(this.dir, `${assertSafeId(record.meta.id)}.json`), JSON.stringify(file));
    } catch (error) {
      console.error(`[runs] could not persist ${record.meta.id}: ${(error as Error).message}`);
    }
  }
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
    simulated: true,
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
