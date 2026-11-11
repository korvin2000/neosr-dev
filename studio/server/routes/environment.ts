import fs from 'node:fs';
import path from 'node:path';
import { Router } from 'express';
import { parse } from 'smol-toml';
import type { EnvironmentInfo } from '../../shared/run.model';
import { EXPERIMENTS_ROOT, FREE_DISK_GB, GPU, NEOSR_PATH, NEOSR_PYTHON, NEOSR_ROOT, PYPROJECT_PATH, TRAINER_MODE } from '../config';
import { isRecord } from '../http';
import { probeEnvironment, type ProbeResult } from '../trainer/environment-probe';

export const environmentRouter = Router();

/** Real mode: one probe of NEOSR_PYTHON, started with the backend and cached. */
let probe: Promise<ProbeResult> | undefined;

export function startEnvironmentProbe(): Promise<ProbeResult> | undefined {
  if (TRAINER_MODE === 'real') {
    probe ??= probeEnvironment(NEOSR_PYTHON, NEOSR_ROOT).then((result) => {
      if (result.warning) {
        console.warn(`[api] ${result.warning}`);
      }
      return result;
    });
  }
  return probe;
}

environmentRouter.get('/environment', async (_req, res) => {
  const real = startEnvironmentProbe();
  res.json(real ? realEnvironment(await real) : simulatedEnvironment());
});

export function hostPlatform(): 'windows' | 'linux' {
  return process.platform === 'win32' ? 'windows' : 'linux';
}

function simulatedEnvironment(): EnvironmentInfo {
  return {
    platform: hostPlatform(),
    neosrPath: NEOSR_PATH,
    neosrVersion: neosrVersion(PYPROJECT_PATH),
    experimentsRoot: EXPERIMENTS_ROOT,
    gpus: [GPU],
    freeDiskGb: FREE_DISK_GB,
    simulated: true,
  };
}

/** The checkout `train.py` runs in; `experimentsRoot` is the default one (a recipe's `path.experiments_root` may differ). */
function realEnvironment(result: ProbeResult): EnvironmentInfo {
  return {
    platform: hostPlatform(),
    neosrPath: NEOSR_ROOT,
    neosrVersion: neosrVersion(path.join(NEOSR_ROOT, 'pyproject.toml')),
    experimentsRoot: path.join(NEOSR_ROOT, 'experiments'),
    gpus: result.gpus,
    freeDiskGb: freeDiskGb(NEOSR_ROOT),
    simulated: false,
    pythonPath: NEOSR_PYTHON,
    torchVersion: result.torchVersion,
    cudaVersion: result.cudaVersion,
    warnings: result.warning ? [result.warning] : [],
  };
}

function freeDiskGb(dir: string): number | null {
  try {
    const stats = fs.statfsSync(dir);
    return Math.round((stats.bavail * stats.bsize) / 1e9);
  } catch {
    return null;
  }
}

function neosrVersion(pyproject: string): string {
  try {
    const project = parse(fs.readFileSync(pyproject, 'utf8'))['project'];
    const version = isRecord(project) ? project['version'] : undefined;
    return typeof version === 'string' ? version : 'dev';
  } catch {
    return 'dev';
  }
}
