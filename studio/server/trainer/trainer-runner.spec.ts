import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { compileRecipe } from '../../shared/compile/compile';
import type { RunEvent, RunStatus } from '../../shared/run.model';
import { buildTemplate } from '../../shared/templates';
import { buildSimPlan } from '../sim/sim-plan';
import { createSimState, type SimState } from '../sim/trainer-sim';
import { lockOwner, lockPath } from './run-dirs';
import { TrainerRunner, type Launcher } from './trainer-runner';

/** A node script standing in for `train.py` (no GPU, no Python): see fixtures/fake-train.mjs. */
const FAKE = path.join(import.meta.dirname, 'fixtures', 'fake-train.mjs');
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'studio-runner-spec-'));
afterAll(() => fs.rmSync(TMP, { recursive: true, force: true }));

/** A run of `templateId` whose phase TOMLs live in `<root>/toml` and whose trainer cwd is `<root>`. */
function start(templateId: string, mode: string, command = process.execPath) {
  const root = fs.mkdtempSync(path.join(TMP, 'run-'));
  const tomlDir = path.join(root, 'toml');
  fs.mkdirSync(tomlDir);
  const compiled = compileRecipe(buildTemplate(templateId), { platform: process.platform === 'win32' ? 'windows' : 'linux' });
  compiled.phases.forEach((p) => fs.writeFileSync(path.join(tomlDir, p.fileName), p.text));
  const state: SimState = createSimState(buildSimPlan('run_test', compiled, 24));
  const events: RunEvent[] = [];
  const launcher: Launcher = (toml, resume) => ({ command, args: [FAKE, mode, toml, ...(resume ? ['--auto_resume'] : [])], cwd: root });
  const runner = new TrainerRunner(state, tomlDir, root, launcher, (batch) => events.push(...batch));
  runner.launch();
  return { state, events, runner, root, tomlDir };
}

async function until(condition: () => boolean, ms = 10_000): Promise<void> {
  const t0 = Date.now();
  while (!condition()) {
    if (Date.now() - t0 > ms) {
      throw new Error('timed out');
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

const statuses = (events: readonly RunEvent[]): RunStatus[] => events.flatMap((e) => (e.type === 'status' ? [e.status] : []));
const terminal = (state: SimState) => () => ['completed', 'failed', 'stopped'].includes(state.status);
const commands = (events: readonly RunEvent[]): string[] => events.flatMap((e) => (e.type === 'log' && e.line.startsWith('$ ') ? [e.line] : []));
const posix = process.platform !== 'win32';

describe('TrainerRunner with a fake trainer process', () => {
  it('replays a recorded run into the simulator events and completes', async () => {
    const { state, events, tomlDir } = start('4x-span-fidelity', 'replay');
    await until(terminal(state));
    expect(statuses(events)).toEqual(['starting', 'running', 'completed']);
    expect(commands(events)[0]).toContain(` replay ${path.join(tomlDir, '4x_span_fidelity.toml')}`);
    expect(events.filter((e) => e.type === 'phase')).toHaveLength(1);
    expect(events.filter((e) => e.type === 'validation')).toHaveLength(4);
    expect(events.filter((e) => e.type === 'progress').at(-1)).toMatchObject({ nativeIter: 20_000, iter: 20_000 });
    expect(events.map((e) => e.seq)).toEqual(events.map((_, i) => i + 1));
    expect(state).toMatchObject({ iter: 20_000, lastCheckpointIter: 20_000, lastValidationIter: 20_000, bestMetric: { name: 'psnr', value: 30.3988, iter: 20_000 } });
    expect(state.series['loss/g/total']).toHaveLength(17);
  });

  it('runs the phases of a curriculum one process after another, reporting each start', async () => {
    const { state, events } = start('4x-curriculum', 'replay');
    await until(terminal(state));
    expect(state.plan.phases).toHaveLength(2);
    expect(commands(events)).toHaveLength(2);
    expect(events.flatMap((e) => (e.type === 'phase' ? [e.phaseIndex] : []))).toEqual([0, 1]);
    expect(statuses(events)).toEqual(['starting', 'running', 'starting', 'running', 'completed']);
  });

  it('fails with the exception line of the traceback', async () => {
    const { state, events } = start('4x-span-fidelity', 'fail');
    await until(terminal(state));
    expect(events.filter((e) => e.type === 'status').at(-1)).toMatchObject({
      status: 'failed',
      message: 'FloatingPointError: non-finite weights or buffers; the last committed checkpoint stands',
    });
  });

  it('fails when the trainer cannot start', async () => {
    const { state, events } = start('4x-span-fidelity', 'replay', path.join(TMP, 'no-such-python'));
    await until(terminal(state));
    expect(events.filter((e) => e.type === 'status').at(-1)).toMatchObject({ status: 'failed', message: expect.stringContaining('ENOENT') });
  });

  it.skipIf(!posix)('stops gracefully with SIGINT, then resumes the phase from its own newest state', async () => {
    const { state, events, runner, root, tomlDir } = start('4x-curriculum', 'loop');
    await until(() => (state.series['loss/g/total']?.length ?? 0) >= 5);
    const lock = lockPath(path.join(root, 'experiments', state.plan.phases[0]!.info.runName));
    expect(lockOwner(lock)?.pid).toBeGreaterThan(0); // held while the trainer runs
    runner.stop(false);
    await until(terminal(state));
    expect(fs.existsSync(lock)).toBe(false); // released on exit
    expect(statuses(events)).toEqual(['starting', 'running', 'stopping', 'saving', 'stopped']);
    const stoppedAt = state.iter;
    expect(events.filter((e) => e.type === 'checkpoint').at(-1)).toMatchObject({ iter: stoppedAt, state: true });

    // the trainer's states of this phase: resume must load the newest, not the TOML's own resume_state
    const runName = state.plan.phases[0]!.info.runName;
    const states = path.join(root, 'experiments', runName, 'training_states');
    fs.mkdirSync(states, { recursive: true });
    ['100.state', '300.state', '200.state'].forEach((f) => fs.writeFileSync(path.join(states, f), ''));
    const toml = path.join(tomlDir, `${runName}.toml`);
    fs.appendFileSync(toml, '\n[path]\nresume_state = "experiments/elsewhere/training_states/50.state"\n');

    // validations before and past the state (as restored from disk: events and series)
    state.events.push({ type: 'validation', dataset: 'val', metrics: { psnr: 25 }, images: [], seq: 0, runId: 'run_test', iter: 200, at: '' });
    state.series['val/val/psnr'] = [{ iter: 200, value: 25 }, { iter: 2000, value: 30 }];
    state.bestMetric = { name: 'psnr', value: 30, iter: 2000 };
    state.lastValidationIter = 2000;
    runner.resume();
    expect(state.iter).toBe(300); // cut back before the call returns
    expect(state).toMatchObject({ bestMetric: { name: 'psnr', value: 25, iter: 200 }, lastValidationIter: 200 });
    expect(Math.max(...(state.series['loss/g/total'] ?? []).map((p) => p.iter))).toBeLessThanOrEqual(300);
    const resumeToml = fs.readFileSync(path.join(tomlDir, `${runName}.resume.toml`), 'utf8');
    expect(resumeToml).toContain(`resume_state = ${JSON.stringify(path.join(states, '300.state'))}`);
    expect(resumeToml).not.toContain('elsewhere');
    expect(commands(events).at(-1)).toMatch(/\.resume\.toml --auto_resume$/);
    await until(() => state.status === 'running' && state.iter > 300);
    expect(events.filter((e) => e.type === 'log' && e.line.startsWith('fake trainer')).at(-1)).toMatchObject({ iter: 300 }); // before it reports its update
    runner.stop(true);
    await until(terminal(state));
    expect(state.status).toBe('stopped');
  });

  it.skipIf(!posix)('completes when the trainer ends normally although a graceful stop was pending', async () => {
    const { state, runner } = start('4x-span-fidelity', 'endonint');
    await until(() => state.status === 'running');
    runner.stop(false);
    await until(terminal(state));
    expect(state.status).toBe('completed');
  });

  it.skipIf(!posix)('completes when the stop arrives after `End of training` and kills the teardown', async () => {
    const { state, events, runner } = start('4x-span-fidelity', 'teardown');
    await until(() => events.some((e) => e.type === 'log' && e.line.startsWith('End of training')));
    runner.stop(false);
    await until(terminal(state));
    expect(state.status).toBe('completed');
  });

  it.skipIf(!posix)('a stop while the trainer starts up (dying of SIGINT) is a cancel, not a failure', async () => {
    const { state, runner } = start('4x-span-fidelity', 'startup');
    await new Promise((resolve) => setTimeout(resolve, 200));
    runner.stop(false);
    await until(terminal(state));
    expect(state.status).toBe('stopped');
  });

  it.skipIf(!posix)('publishes nothing after a delete', async () => {
    const { state, events, runner } = start('4x-span-fidelity', 'loop');
    await until(() => state.status === 'running');
    runner.dispose();
    const count = events.length;
    await new Promise((resolve) => setTimeout(resolve, 300));
    expect(events.length).toBe(count);
  });

  it.skipIf(!posix)('shutdown stops gracefully and resolves once the trainer has exited', async () => {
    const { state, runner } = start('4x-span-fidelity', 'loop');
    await until(() => state.status === 'running');
    await runner.shutdown();
    expect(state.status).toBe('stopped');
    await runner.shutdown(); // nothing left to stop
  });

  it.skipIf(!posix)('force stop kills the whole process tree', async () => {
    const { state, events, runner } = start('4x-span-fidelity', 'stubborn');
    await until(() => events.some((e) => e.type === 'log' && e.line.startsWith('grandchild ')));
    const line = events.find((e) => e.type === 'log' && e.line.startsWith('grandchild '));
    const pid = Number(line?.type === 'log' ? line.line.split(' ')[1] : NaN);
    runner.stop(true);
    await until(terminal(state));
    expect(events.filter((e) => e.type === 'log').at(-1)).toMatchObject({ line: 'Trainer process killed: no checkpoint written' });
    await until(() => {
      try {
        process.kill(pid, 0);
        return false;
      } catch {
        return true;
      }
    });
  });
});
