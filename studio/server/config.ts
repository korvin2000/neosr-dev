import path from 'node:path';
import type { GpuInfo } from '../shared/run.model';

export const PORT = 4310;
export const NEOSR_PATH = 'C:\\work2000\\neosr-dev';
export const EXPERIMENTS_ROOT = `${NEOSR_PATH}\\experiments`;
export const FREE_DISK_GB = 812;

export const GPU: GpuInfo = {
  name: 'NVIDIA GeForce RTX 4090',
  vramGb: 24,
  computeCapability: '8.9',
  driver: '581.57',
};

/** `server/` (tsx runs the sources as CommonJS, so `__dirname` is defined). */
export const SERVER_DIR = __dirname;
export const DATA_DIR = path.join(SERVER_DIR, 'data');
export const PYPROJECT_PATH = path.join(SERVER_DIR, '..', '..', 'pyproject.toml');

/** `STUDIO_TRAINER=real` or `--real` (`npm run api:real`) launches the actual `train.py` for new runs; the default simulates them. */
export const TRAINER_MODE: 'sim' | 'real' = process.env['STUDIO_TRAINER'] === 'real' || process.argv.includes('--real') ? 'real' : 'sim';
/** The trainer checkout `train.py` runs in (`NEOSR_ROOT`, default: the repository holding `studio/`). */
export const NEOSR_ROOT = path.resolve(process.env['NEOSR_ROOT'] ?? path.join(SERVER_DIR, '..', '..'));
/** Its Python (`NEOSR_PYTHON`, default: the checkout's `.venv`). */
export const NEOSR_PYTHON =
  process.env['NEOSR_PYTHON'] ?? path.join(NEOSR_ROOT, '.venv', ...(process.platform === 'win32' ? ['Scripts', 'python.exe'] : ['bin', 'python']));
