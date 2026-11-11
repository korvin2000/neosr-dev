import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { parse } from 'smol-toml';
import { describe, expect, it } from 'vitest';
import { compileRecipe } from '../../shared/compile/compile';
import { buildTemplate } from '../../shared/templates';
import { dirsOverlap, experimentsDir, lockOwner, lockPath, namespacePhases, newestState, withResumeState, writeLock } from './run-dirs';

const ROOT = path.resolve('/neosr');
const [p1, p2] = compileRecipe(buildTemplate('4x-curriculum'), { platform: 'linux' }).phases;

describe('run-dirs', () => {
  it('resolves experiments/<name>, honouring path.experiments_root', () => {
    expect(experimentsDir(p1!.text, ROOT)).toBe(path.join(ROOT, 'experiments', '4x_curriculum_realplksr_p1_pixel_pretrain'));
    expect(experimentsDir('name = "x"\n[path]\nexperiments_root = "/data/exp"\n', ROOT)).toBe(path.resolve('/data/exp', 'x'));
    expect(experimentsDir('name = "x"\n[path]\nexperiments_root = "~/exp"\n', ROOT)).toBe(path.join(os.homedir(), 'exp', 'x'));
  });

  it('gives each run its own folders, following the links between its phases only', () => {
    const imported = 'name = "a"\n[path]\nresume_state = "experiments/a/training_states/10.state"\n';
    const continued = 'name = "b"\n[datasets.val]\nname = "val"\n[path]\nresume_state = "experiments/a/training_states/100.state"\npretrain_network_g = "experiments/ab/models/x.pth"\n';
    const [a, b] = namespacePhases([{ runName: 'a', text: imported }, { runName: 'b', text: continued }], 'f00d');
    expect(a).toEqual({ runName: 'a_f00d', text: 'name = "a_f00d"\n[path]\nresume_state = "experiments/a/training_states/10.state"\n' }); // imported: kept
    expect(b!.text).toBe('name = "b_f00d"\n[datasets.val]\nname = "val"\n[path]\nresume_state = "experiments/a_f00d/training_states/100.state"\npretrain_network_g = "experiments/ab/models/x.pth"\n');
    const [q1, q2] = namespacePhases([p1!, p2!], 'f00d');
    expect([q1!.runName, q2!.runName]).toEqual(['4x_curriculum_realplksr_p1_pixel_pretrain_f00d', '4x_curriculum_realplksr_p2_gan_fine_tune_f00d']);
    expect(parse(q2!.text)['name']).toBe(q2!.runName);
    expect((parse(q2!.text)['path'] as Record<string, unknown>)['pretrain_network_g']).toBe('experiments/4x_curriculum_realplksr_p1_pixel_pretrain_f00d/models/net_g_latest.pth');
  });

  it('rewrites only the segment under the experiments root, and literal-string names too', () => {
    const p = "name = 'r\\un'\n[path]\nexperiments_root = 'D:\\r\\un\\out'\n";
    const q = "name = 'x'\n[path]\nexperiments_root = 'D:\\r\\un\\out'\npretrain_network_g = \"D:/r/un/out/r\\\\un/models/net_g_latest.pth\"\n";
    const [a, b] = namespacePhases([{ runName: 'r\\un', text: p }, { runName: 'x', text: q }], 'f00d');
    expect(parse(a!.text)['name']).toBe('r\\un_f00d');
    expect((parse(b!.text)['path'] as Record<string, unknown>)['pretrain_network_g']).toBe('D:/r/un/out/r\\un_f00d/models/net_g_latest.pth');
    expect((parse(b!.text)['path'] as Record<string, unknown>)['experiments_root']).toBe('D:\\r\\un\\out');
  });

  it('reports a lock as held only while its process lives, and deletes a stale one', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'studio-lock-'));
    try {
      const lock = lockPath(path.join(dir, 'experiments', 'x'));
      expect(lockOwner(lock)).toBeUndefined();
      writeLock(lock, process.pid);
      expect(lockOwner(lock)).toEqual({ pid: process.pid, verified: process.platform === 'linux' });
      fs.writeFileSync(lock, JSON.stringify({ pid: process.pid }));
      expect(lockOwner(lock)).toEqual({ pid: process.pid, verified: false }); // no start time recorded
      fs.writeFileSync(lock, JSON.stringify({ pid: 2 ** 22 + 12345 }));
      expect(lockOwner(lock)).toBeUndefined();
      expect(fs.existsSync(lock)).toBe(false);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('finds the newest numbered state', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'studio-run-dirs-'));
    try {
      expect(newestState(dir)).toBeUndefined();
      fs.mkdirSync(path.join(dir, 'training_states'));
      ['900.state', '1000.state', '20.state', 'x.state', '5000.state.tmp'].forEach((f) => fs.writeFileSync(path.join(dir, 'training_states', f), ''));
      expect(newestState(dir)).toEqual({ file: path.join(dir, 'training_states', '1000.state'), update: 1000 });
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('sets path.resume_state, replacing one the compiler wrote and keeping the rest', () => {
    const state = 'C:\\neosr\\experiments\\p2\\training_states\\1500.state';
    const continued = `${p2!.text.replace('[path]\n', '[path]\nresume_state = "experiments/p1/training_states/100000.state"\n')}`;
    const doc = parse(withResumeState(continued, state));
    const original = parse(p2!.text);
    expect((doc['path'] as Record<string, unknown>)['resume_state']).toBe(state);
    expect({ ...(doc['path'] as object), resume_state: undefined }).toEqual({ ...(original['path'] as object), resume_state: undefined });
    expect(doc['train']).toEqual(original['train']);
    expect((parse(withResumeState(p1!.text, state))['path'] as Record<string, unknown>)['resume_state']).toBe(state);
  });

  it('detects equal or nested output folders', () => {
    expect(dirsOverlap('/a/experiments/x', '/a/experiments/x/')).toBe(true);
    expect(dirsOverlap('/a/experiments', '/a/experiments/x')).toBe(true);
    expect(dirsOverlap('/a/experiments/x', '/a/experiments/x2')).toBe(false);
    expect(dirsOverlap('/a/experiments/Run', '/a/experiments/run', true)).toBe(true);
    expect(dirsOverlap('/a/experiments/Run', '/a/experiments/run', false)).toBe(false);
  });
});
