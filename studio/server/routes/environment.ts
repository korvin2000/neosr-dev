import fs from 'node:fs';
import { Router } from 'express';
import { parse } from 'smol-toml';
import type { EnvironmentInfo } from '../../shared/run.model';
import { EXPERIMENTS_ROOT, FREE_DISK_GB, GPU, NEOSR_PATH, PYPROJECT_PATH } from '../config';
import { isRecord } from '../http';

export const environmentRouter = Router();

environmentRouter.get('/environment', (_req, res) => {
  res.json(environmentInfo());
});

export function hostPlatform(): 'windows' | 'linux' {
  return process.platform === 'win32' ? 'windows' : 'linux';
}

function environmentInfo(): EnvironmentInfo {
  return {
    platform: hostPlatform(),
    neosrPath: NEOSR_PATH,
    neosrVersion: neosrVersion(),
    experimentsRoot: EXPERIMENTS_ROOT,
    gpus: [GPU],
    freeDiskGb: FREE_DISK_GB,
    simulated: true,
  };
}

function neosrVersion(): string {
  try {
    const project = parse(fs.readFileSync(PYPROJECT_PATH, 'utf8'))['project'];
    const version = isRecord(project) ? project['version'] : undefined;
    return typeof version === 'string' ? version : 'dev';
  } catch {
    return 'dev';
  }
}
