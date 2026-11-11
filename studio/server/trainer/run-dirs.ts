/**
 * Where a phase TOML makes `train.py` write, the per-run namespace and resume TOML the adapter
 * derives from it, and the lock that marks a folder as owned by a live trainer process.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { parse } from 'smol-toml';

/** `neosr/config.py`: `path.experiments_root` (default `experiments`, relative to the trainer's cwd; `~` expanded) / `name`. */
export function experimentsDir(tomlText: string, neosrRoot: string): string {
  const doc = parse(tomlText);
  const p = doc['path'];
  const value = p && typeof p === 'object' && !Array.isArray(p) ? (p as Record<string, unknown>)['experiments_root'] : undefined;
  const root = typeof value === 'string' && value ? value : 'experiments';
  const expanded = /^~(?=$|[\\/])/.test(root) ? path.join(os.homedir(), root.slice(1)) : root;
  return path.resolve(neosrRoot, expanded, String(doc['name']));
}

const PHASE_LINKS = ['resume_state', 'pretrain_network_g', 'pretrain_network_d'] as const;

function table(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

/**
 * One run, one set of output folders: every phase's `name` (and `runName`) gets `_<suffix>`, and
 * the `[path]` entries the compiler points into an earlier phase of the same run
 * (`<experiments_root>/<name>/…`, `compile.ts` `pathEntries`) follow it; only that segment changes.
 * Other paths (an imported pretrain or resume_state) stay as written. Values are read with the TOML
 * parser and the changed lines rewritten as basic strings.
 */
export function namespacePhases<T extends { readonly runName: string; readonly text: string }>(phases: readonly T[], suffix: string): T[] {
  const earlier: string[] = [];
  return phases.map((phase) => {
    const doc = parse(phase.text);
    const original = String(doc['name']);
    const name = `${original}_${suffix}`;
    const p = table(doc['path']);
    const rootValue = p['experiments_root'];
    const root = typeof rootValue === 'string' && rootValue.trim() ? rootValue.trim().replace(/[\\/]+$/, '').split('\\').join('/') : 'experiments';
    const links = new Map<string, string>();
    for (const key of PHASE_LINKS) {
      const value = p[key];
      const prev = typeof value === 'string' ? earlier.find((e) => value.startsWith(`${root}/${e}/`)) : undefined;
      if (typeof value === 'string' && prev !== undefined) {
        links.set(key, `${root}/${prev}_${suffix}/${value.slice(root.length + prev.length + 2)}`);
      }
    }
    earlier.push(original);
    let section = '';
    const text = phase.text
      .split('\n')
      .map((line) => {
        const header = /^\s*\[+([^\]]*)\]/.exec(line);
        if (header) {
          section = (header[1] as string).trim();
          return line;
        }
        const key = /^\s*([A-Za-z0-9_-]+)\s*=/.exec(line)?.[1];
        if (section === '' && key === 'name') {
          return `name = ${JSON.stringify(name)}`;
        }
        const link = section === 'path' && key !== undefined ? links.get(key) : undefined;
        return link === undefined ? line : `${key} = ${JSON.stringify(link)}`;
      })
      .join('\n');
    return { ...phase, runName: name, text };
  });
}

/** The newest `<update>.state` in `<experimentsDir>/training_states`, as `train.py --auto_resume` would pick it. */
export function newestState(experiments: string): { readonly file: string; readonly update: number } | undefined {
  const dir = path.join(experiments, 'training_states');
  let names: string[];
  try {
    names = fs.readdirSync(dir);
  } catch {
    return undefined;
  }
  const updates = names.flatMap((n) => /^(\d+)\.state$/.exec(n)?.[1] ?? []).map(Number);
  if (!updates.length) {
    return undefined;
  }
  const update = Math.max(...updates);
  return { file: path.join(dir, `${update}.state`), update };
}

/**
 * The phase TOML with `path.resume_state` set to `statePath`: an explicit resume_state wins over
 * `--auto_resume` in `train.py`'s `select_resume`, and the compiler writes one for `continue`
 * phases (the previous phase's state), so a resume must replace it. Text edit of the compiler's
 * own output (one `[path]` table, `key = value` lines), which keeps every other value as written.
 */
export function withResumeState(tomlText: string, statePath: string): string {
  const entry = `resume_state = ${JSON.stringify(statePath)}`;
  const lines = tomlText.split('\n');
  const header = lines.findIndex((l) => l.trim() === '[path]');
  if (header === -1) {
    return `${tomlText.trimEnd()}\n\n[path]\n${entry}\n`;
  }
  const next = lines.findIndex((l, i) => i > header && /^\s*\[/.test(l));
  const end = next === -1 ? lines.length : next;
  const body = lines.slice(header + 1, end).filter((l) => !/^\s*resume_state\s*=/.test(l));
  return [...lines.slice(0, header + 1), entry, ...body, ...lines.slice(end)].join('\n');
}

/** Equal or nested directories (case-insensitive where the file system usually is: Windows). */
export function dirsOverlap(a: string, b: string, ignoreCase = process.platform === 'win32'): boolean {
  const norm = (d: string) => (ignoreCase ? path.resolve(d).toLowerCase() : path.resolve(d));
  const [x, y] = [norm(a), norm(b)];
  return x === y || x.startsWith(y + path.sep) || y.startsWith(x + path.sep);
}

/**
 * `<experiments>.studio.lock`, next to the folder rather than in it: a fresh `train.py` start
 * archives an existing `experiments/<name>` (renames it), which would carry a lock inside away.
 */
export function lockPath(experiments: string): string {
  return `${experiments}.studio.lock`;
}

export function writeLock(lock: string, pid: number): void {
  fs.mkdirSync(path.dirname(lock), { recursive: true });
  fs.writeFileSync(lock, JSON.stringify({ pid, start: processStart(pid), at: new Date().toISOString() }), 'utf8');
}

/**
 * The process a lock names while it still runs, `verified` when its start time matches the one
 * recorded (Linux; elsewhere a reused pid cannot be told apart). A stale lock is deleted.
 */
export function lockOwner(lock: string): { readonly pid: number; readonly verified: boolean } | undefined {
  let data: { pid?: unknown; start?: unknown };
  try {
    data = JSON.parse(fs.readFileSync(lock, 'utf8')) as typeof data;
  } catch {
    return undefined;
  }
  const pid = Number(data.pid);
  const alive = Number.isInteger(pid) && pid > 0 && isAlive(pid);
  const start = alive ? processStart(pid) : undefined;
  if (!alive || (typeof data.start === 'string' && start !== undefined && start !== data.start)) {
    fs.rmSync(lock, { force: true }); // gone, or the pid was reused
    return undefined;
  }
  return { pid, verified: typeof data.start === 'string' && start === data.start };
}

function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM';
  }
}

/** Linux: field 22 of `/proc/<pid>/stat` (clock ticks after boot); undefined elsewhere. */
function processStart(pid: number): string | undefined {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19];
  } catch {
    return undefined;
  }
}
