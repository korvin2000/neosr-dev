/**
 * The real trainer's environment, asked once from the configured Python (torch is imported there,
 * never in Node). GPU details come from `nvidia-smi`, which creates no CUDA context on a GPU that
 * may be busy training; torch's own device properties are the fallback. Fails soft: any error
 * gives null versions, no GPUs and a warning.
 */
import { execFile } from 'node:child_process';
import type { GpuInfo } from '../../shared/run.model';

export interface ProbeResult {
  readonly torchVersion: string | null;
  readonly cudaVersion: string | null;
  readonly gpus: readonly GpuInfo[];
  readonly warning?: string;
}

/** Prints one JSON line. `torch.cuda.device_count()` honours CUDA_VISIBLE_DEVICES without a context. */
export const PROBE_SCRIPT = `
import json, os, subprocess
import torch
out = {"torchVersion": torch.__version__, "cudaVersion": torch.version.cuda, "gpus": []}
count = torch.cuda.device_count()
if count:
    rows = []
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        rows = [[c.strip() for c in line.split(",")] for line in smi.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip().isdigit()]
    if visible:
        rows = [r for v in visible for r in rows if r[0] == v]
    if len(rows) >= count:
        for r in rows[:count]:
            out["gpus"].append({"name": r[1], "vramGb": round(float(r[2]) / 1024, 1), "computeCapability": r[4] if r[4][:1].isdigit() else None, "driver": r[3]})
    else:
        for i in range(count):
            p = torch.cuda.get_device_properties(i)
            out["gpus"].append({"name": p.name, "vramGb": round(p.total_memory / 2**30, 1), "computeCapability": f"{p.major}.{p.minor}", "driver": None})
print(json.dumps(out))
`;

const TIMEOUT_MS = 120_000;

export function probeEnvironment(python: string, cwd: string): Promise<ProbeResult> {
  return new Promise((resolve) => {
    execFile(python, ['-c', PROBE_SCRIPT], { cwd, timeout: TIMEOUT_MS, windowsHide: true }, (error, stdout, stderr) => {
      if (error) {
        const detail = String(stderr).trim().split('\n').at(-1) || error.message;
        resolve(failed(`Probe of ${python} failed: ${detail}`));
        return;
      }
      resolve(parseProbe(String(stdout)));
    });
  });
}

/** The probe's last stdout line, checked field by field. */
export function parseProbe(stdout: string): ProbeResult {
  let data: unknown;
  try {
    data = JSON.parse(stdout.trim().split('\n').at(-1) ?? '');
  } catch {
    return failed('The environment probe printed no JSON');
  }
  const d = (data ?? {}) as Record<string, unknown>;
  const text = (v: unknown) => (typeof v === 'string' ? v : null);
  const gpus = Array.isArray(d['gpus']) ? d['gpus'] : [];
  const valid = gpus.every((g) => typeof g?.name === 'string' && typeof g?.vramGb === 'number');
  if (!valid) {
    return failed('The environment probe printed malformed GPU entries');
  }
  return {
    torchVersion: text(d['torchVersion']),
    cudaVersion: text(d['cudaVersion']),
    gpus: gpus.map((g: Record<string, unknown>) => ({
      name: g['name'] as string,
      vramGb: g['vramGb'] as number,
      computeCapability: text(g['computeCapability']),
      driver: text(g['driver']),
    })),
  };
}

function failed(warning: string): ProbeResult {
  return { torchVersion: null, cudaVersion: null, gpus: [], warning };
}
