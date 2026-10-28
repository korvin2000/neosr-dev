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
