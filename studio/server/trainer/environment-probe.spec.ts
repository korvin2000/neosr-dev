import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { parseProbe, probeEnvironment } from './environment-probe';

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'studio-probe-'));
afterAll(() => fs.rmSync(TMP, { recursive: true, force: true }));

/** A "python" that ignores `-c <script>` and prints `stdout` (exit `code`). */
function fakePython(name: string, stdout: string, code = 0): string {
  const file = path.join(TMP, name);
  fs.writeFileSync(file, `#!${process.execPath}\nprocess.stdout.write(${JSON.stringify(stdout)});\nprocess.stderr.write('Traceback\\nModuleNotFoundError: no torch');\nprocess.exitCode = ${code};\n`, { mode: 0o755 });
  return file;
}

const GOOD = { torchVersion: '2.9.1+cu128', cudaVersion: '12.8', gpus: [{ name: 'NVIDIA GeForce RTX 4090', vramGb: 24, computeCapability: '8.9', driver: '615.78.08' }] };

describe('environment probe', () => {
  it.skipIf(process.platform === 'win32')('reads the JSON the probe prints', async () => {
    expect(await probeEnvironment(fakePython('ok', `warming up\n${JSON.stringify(GOOD)}\n`), TMP)).toEqual(GOOD);
  });

  it.skipIf(process.platform === 'win32')('fails soft with the last stderr line when the probe exits non-zero', async () => {
    expect(await probeEnvironment(fakePython('broken', '', 1), TMP)).toEqual({
      torchVersion: null,
      cudaVersion: null,
      gpus: [],
      warning: expect.stringContaining('ModuleNotFoundError: no torch'),
    });
  });

  it('fails soft when the Python does not exist', async () => {
    const result = await probeEnvironment(path.join(TMP, 'no-such-python'), TMP);
    expect(result).toMatchObject({ torchVersion: null, gpus: [], warning: expect.stringContaining('ENOENT') });
  });

  it('rejects output that is not the expected JSON', () => {
    expect(parseProbe('not json').warning).toMatch(/no JSON/);
    expect(parseProbe(JSON.stringify({ gpus: [{ name: 1 }] })).warning).toMatch(/malformed/);
    expect(parseProbe(JSON.stringify({ torchVersion: '2.9.1', cudaVersion: null, gpus: [] }))).toEqual({ torchVersion: '2.9.1', cudaVersion: null, gpus: [] });
  });
});
