/**
 * Runs the real trainer for one Studio run: one `train.py -opt <phase>.toml` process per phase, in
 * order, its output parsed into the same `SimState` and events the simulator produces. Stop sends
 * SIGINT to the process group (the trainer saves at the next update boundary and exits 0); force
 * stop and delete kill the whole tree. Resume points `path.resume_state` at the phase's own newest
 * state (see `withResumeState`).
 */
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import type { Readable } from 'node:stream';
import { METRIC_INFO, type RunEvent } from '../../shared/run.model';
import {
  emit,
  enterPhase,
  finishStop,
  forceStop,
  log,
  requestStop,
  setStatus,
  truncateSeries,
  updateBest,
  warn,
  type Publish,
  type SimState,
} from '../sim/trainer-sim';
import type { SimPhase } from '../sim/sim-plan';
import { TrainerLogParser, type ParsedEvent } from './log-parser';
import { experimentsDir, lockPath, newestState, withResumeState, writeLock } from './run-dirs';

export interface LaunchSpec {
  readonly command: string;
  readonly args: readonly string[];
  readonly cwd: string;
}

/** Builds the process for one phase's TOML; `resume` adds `--auto_resume`. */
export type Launcher = (tomlPath: string, resume: boolean) => LaunchSpec;

export function trainPyLauncher(python: string, neosrRoot: string): Launcher {
  return (tomlPath, resume) => ({ command: python, args: ['train.py', '-opt', tomlPath, ...(resume ? ['--auto_resume'] : [])], cwd: neosrRoot });
}

const WINDOWS = process.platform === 'win32';

export class TrainerRunner {
  private child?: ChildProcess;
  /** Resolves when the current process has exited and its outcome is applied. */
  private exited: Promise<void> = Promise.resolve();
  private parser = new TrainerLogParser();
  private stopping?: 'graceful' | 'force';
  private disposed = false;
  /** Set when the lock could not be written: the process is killed and the phase fails with this. */
  private lockError?: string;
  private bestDataset?: string;
  /** The current process reported `Starting at epoch …`. */
  private entered = false;

  constructor(
    private readonly s: SimState,
    private readonly tomlDir: string,
    private readonly neosrRoot: string,
    private readonly launcher: Launcher,
    private readonly publish: Publish,
  ) {}

  launch(): void {
    this.batch((out, now) => {
      this.s.startedAt = new Date(now).toISOString();
      setStatus(this.s, out, now, 'starting');
    });
    this.spawnPhase(false);
  }

  stop(force: boolean): void {
    const child = this.child;
    if (!child) {
      this.batch((out, now) => out.push(...finishStop(this.s, now)));
      return;
    }
    if (force || WINDOWS) {
      this.stopping = 'force';
      if (!force) {
        this.batch((out, now) => log(this.s, out, now, 'Graceful stop needs SIGINT, which Windows cannot send: killing the trainer', 'warn'));
      }
      signalTree(child, 'SIGKILL');
      return;
    }
    this.stopping = 'graceful';
    this.batch((out, now) => out.push(...requestStop(this.s, now)));
    signalTree(child, 'SIGINT');
  }

  /**
   * Continues the current phase from the newest `<update>.state` in its own experiments folder
   * (a resume TOML with that `resume_state`), else from the phase TOML as written. The series are
   * cut back to that update before the call returns, so the snapshot a client reloads is final.
   */
  resume(): void {
    const phase = this.phase();
    if (!phase) {
      return;
    }
    this.stopping = undefined;
    const toml = this.tomlPath(phase);
    const text = fs.readFileSync(toml, 'utf8');
    const state = newestState(experimentsDir(text, this.neosrRoot));
    let launchToml = toml;
    if (state) {
      launchToml = toml.replace(/\.toml$/, '.resume.toml');
      fs.writeFileSync(launchToml, withResumeState(text, state.file), 'utf8');
    }
    this.batch((out, now) => {
      const s = this.s;
      s.finishedAt = undefined;
      this.rewind(state ? this.curriculum(phase, state.update) : phase.start, state !== undefined);
      setStatus(s, out, now, 'starting', 'resuming');
      log(s, out, now, state ? `Resuming from ${state.file}` : `No training state of this run's ${phase.info.name} yet: starting the phase from its TOML`);
    });
    this.spawnPhase(true, launchToml);
  }

  /** Server shutdown: a graceful stop unless one is under way (a second SIGINT would abort its save). */
  shutdown(): Promise<void> {
    if (this.child && !this.stopping) {
      this.stop(false);
    }
    return this.exited;
  }

  /** A process of this runner is alive. */
  get running(): boolean {
    return this.child !== undefined;
  }

  /** Delete: kill the tree; nothing is published afterwards. */
  dispose(): void {
    this.disposed = true;
    if (this.child) {
      this.stopping = 'force';
      signalTree(this.child, 'SIGKILL');
    }
  }

  /**
   * Back to update `iter` of the curriculum: series cut there, and the summaries derived from them
   * again (best metric and last validation from the kept series; the last checkpoint is the resumed
   * state, or none past `iter`).
   */
  private rewind(iter: number, fromState: boolean): void {
    const s = this.s;
    s.iter = iter;
    truncateSeries(s, iter);
    if (s.lastCheckpointIter !== undefined && s.lastCheckpointIter > iter) {
      s.lastCheckpointIter = fromState ? iter : undefined;
    }
    const dataset = this.bestDataset ?? s.events.find((e) => e.type === 'validation')?.dataset;
    const name = dataset === undefined ? undefined : s.plan.metrics.find((m) => s.series[`val/${dataset}/${m}`]?.length);
    const points = name === undefined ? [] : (s.series[`val/${dataset}/${name}`] ?? []);
    const lower = name !== undefined && METRIC_INFO[name]?.better === 'lower';
    s.bestMetric = undefined;
    for (const p of points) {
      if (!s.bestMetric || (lower ? p.value < s.bestMetric.value : p.value > s.bestMetric.value)) {
        s.bestMetric = { name: name as string, value: p.value, iter: p.iter };
      }
    }
    s.lastValidationIter = points.at(-1)?.iter;
  }

  private phase(): SimPhase | undefined {
    return this.s.plan.phases[this.s.phaseIndex];
  }

  private tomlPath(phase: SimPhase): string {
    return path.join(this.tomlDir, path.basename(phase.info.fileName));
  }

  private curriculum(phase: SimPhase, native: number): number {
    return Math.min(phase.end, Math.max(phase.start, phase.start + native - phase.nativeStart));
  }

  private spawnPhase(resume: boolean, toml?: string): void {
    const phase = this.phase();
    if (!phase || this.disposed) {
      return;
    }
    const tomlFile = toml ?? this.tomlPath(phase);
    const spec = this.launcher(tomlFile, resume);
    this.parser = new TrainerLogParser(phase.index, phase.nativeStart + this.s.iter - phase.start);
    this.entered = false;
    this.batch((out, now) => log(this.s, out, now, `$ ${[spec.command, ...spec.args].join(' ')}`));
    // marks the output folder as owned while the process lives (an orphan keeps it after a backend crash)
    const lock = lockPath(experimentsDir(fs.readFileSync(tomlFile, 'utf8'), this.neosrRoot));
    const child = spawn(spec.command, [...spec.args], {
      cwd: spec.cwd,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: !WINDOWS, // own process group: one signal reaches the data-loader workers too
      windowsHide: true,
    });
    this.child = child;
    if (child.pid !== undefined) {
      try {
        writeLock(lock, child.pid);
      } catch (error) {
        // without a lock an orphan of this trainer could not be detected: do not let it run
        this.lockError = `Could not write ${lock}: ${(error as Error).message}`;
        signalTree(child, 'SIGKILL');
      }
    }
    const onLine = (line: string) => this.batch((out, now) => this.parser.push(line).forEach((e) => this.apply(e, out, now)));
    readLines(child.stdout, onLine);
    readLines(child.stderr, onLine);
    this.exited = new Promise((resolve) => {
      let settled = false;
      const settle = (code: number | null, signal: NodeJS.Signals | null, error?: Error) => {
        if (!settled) {
          settled = true;
          this.child = undefined;
          try {
            fs.rmSync(lock, { force: true });
          } catch (rmError) {
            this.batch((out, now) => log(this.s, out, now, `Could not remove ${lock}: ${(rmError as Error).message}`, 'warn'));
          }
          this.finish(code, signal, error);
          resolve();
        }
      };
      child.on('error', (error) => child.pid === undefined && settle(null, null, error)); // could not start
      child.on('close', (code, signal) => settle(code, signal));
    });
  }

  private finish(code: number | null, signal: NodeJS.Signals | null, failedToStart?: Error): void {
    this.batch((out, now) => {
      const s = this.s;
      this.parser.end(code === 0).forEach((e) => this.apply(e, out, now));
      const { outcome, error } = this.parser;
      const lockError = this.lockError;
      this.lockError = undefined;
      if (lockError) {
        log(s, out, now, lockError, 'error');
        s.finishedAt = new Date(now).toISOString();
        setStatus(s, out, now, 'failed', lockError);
      } else if (this.stopping === 'force') {
        out.push(...forceStop(s, now));
      } else if (outcome === 'completed' && (code === 0 || (this.stopping === 'graceful' && (code === 130 || signal === 'SIGINT')))) {
        // a stop that reaches the trainer after `End of training` (all saved) kills only its teardown
        if (s.phaseIndex === s.plan.phases.length - 1) {
          log(s, out, now, 'Training finished');
          s.finishedAt = new Date(now).toISOString();
          s.etaSec = 0;
          setStatus(s, out, now, 'completed');
          return;
        }
        s.phaseIndex++; // a resume after a stop here starts the next phase
        if (this.stopping === 'graceful') {
          out.push(...finishStop(s, now));
          return;
        }
        setStatus(s, out, now, 'starting', `phase ${s.phaseIndex + 1} of ${s.plan.phases.length}`);
        queueMicrotask(() => this.spawnPhase(false));
      } else if (outcome === 'stopped' || code === 130 || (this.stopping === 'graceful' && (code === 0 || signal === 'SIGINT'))) {
        // SIGINT during train.py's startup (before main's try) dies of the signal: a cancel, not a failure
        out.push(...finishStop(s, now));
      } else {
        const message =
          failedToStart?.message ??
          error ??
          (signal ? `trainer killed by ${signal}` : code === 0 ? 'trainer exited without finishing' : `trainer exited with code ${code}`);
        log(s, out, now, message, 'error');
        s.finishedAt = new Date(now).toISOString();
        s.etaSec = 0;
        setStatus(s, out, now, 'failed', message);
      }
    });
  }

  /** Native → curriculum clock, then the same state updates the simulator makes. */
  private apply(e: ParsedEvent, out: RunEvent[], now: number): void {
    const s = this.s;
    const phase = this.phase();
    if (phase) {
      s.iter = this.curriculum(phase, e.native);
    }
    const body = e.body;
    switch (body.type) {
      case 'status':
        if (body.status === 'running') {
          if (!this.entered) {
            this.entered = true;
            if (s.status === 'starting') {
              setStatus(s, out, now, 'running');
            }
            enterPhase(s, out, now);
          }
        } else if (s.status !== 'stopped') {
          setStatus(s, out, now, body.status, body.message);
        }
        return;
      case 'progress': {
        s.itersPerSec = body.itersPerSec;
        const later = phase ? s.plan.totalIter - phase.end : 0;
        s.etaSec = Math.round(body.etaSec + (body.itersPerSec > 0 ? later / body.itersPerSec : 0));
        emit(s, out, now, { ...body, etaSec: s.etaSec });
        return;
      }
      case 'validation':
        emit(s, out, now, body);
        this.bestDataset ??= body.dataset;
        if (body.dataset === this.bestDataset) {
          updateBest(s, body.metrics);
        }
        s.lastValidationIter = s.iter;
        return;
      case 'checkpoint':
        emit(s, out, now, body);
        if (body.state) {
          s.lastCheckpointIter = s.iter;
        }
        return;
      case 'warning':
        warn(s, out, now, body.code, body.message);
        return;
      default:
        emit(s, out, now, body);
    }
  }

  private batch(fn: (out: RunEvent[], now: number) => void): void {
    if (this.disposed) {
      return;
    }
    const out: RunEvent[] = [];
    fn(out, Date.now());
    if (out.length) {
      this.publish(out);
    }
  }
}

/** Splits on `\n`, `\r\n` and a lone `\r` (tqdm redraws), holding a trailing `\r` for a following `\n`. */
function readLines(stream: Readable | null, onLine: (line: string) => void): void {
  if (!stream) {
    return;
  }
  let rest = '';
  stream.setEncoding('utf8');
  stream.on('data', (chunk: string) => {
    let text = rest + chunk;
    const hold = text.endsWith('\r') ? '\r' : '';
    text = hold ? text.slice(0, -1) : text;
    const lines = text.split(/\r\n|\r|\n/);
    rest = (lines.pop() ?? '') + hold;
    lines.forEach(onLine);
  });
  stream.on('end', () => {
    if (rest.replace(/\r$/, '')) {
      onLine(rest.replace(/\r$/, ''));
    }
  });
}

function signalTree(child: ChildProcess, signal: 'SIGINT' | 'SIGKILL'): void {
  if (child.pid === undefined || child.exitCode !== null) {
    return;
  }
  if (WINDOWS) {
    spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
    return;
  }
  try {
    process.kill(-child.pid, signal);
  } catch {
    child.kill(signal); // the group is gone; the leader may still be exiting
  }
}
