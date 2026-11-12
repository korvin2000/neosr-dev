import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import type { RunEventBody } from '../../shared/run.model';
import { parseEta, seriesOf, TrainerLogParser, type ParsedEvent } from './log-parser';

/** 63 lines of a real `train.py` run (realplksr pretrain, 20k updates): progress, validation at 2k, the save at 5k, the end. */
const FIXTURE = fs.readFileSync(path.join(import.meta.dirname, 'fixtures', 'p0_pretrain.log'), 'utf8').split('\n');
const TS = '11-11-2026 09:00 AM |';

function parse(lines: readonly string[], cleanExit = true): { parser: TrainerLogParser; events: ParsedEvent[] } {
  const parser = new TrainerLogParser();
  const events = lines.flatMap((line) => parser.push(line));
  return { parser, events: [...events, ...parser.end(cleanExit)] };
}

function ofType<T extends RunEventBody['type']>(events: readonly ParsedEvent[], type: T): (ParsedEvent & { body: Extract<RunEventBody, { type: T }> })[] {
  return events.filter((e): e is ParsedEvent & { body: Extract<RunEventBody, { type: T }> } => e.body.type === type);
}

describe('TrainerLogParser on recorded output', () => {
  const { parser, events } = parse(FIXTURE);

  it('reports the start and every progress line', () => {
    expect(ofType(events, 'status').map((e) => [e.native, e.body.status])).toEqual([[0, 'running']]);
    const progress = ofType(events, 'progress');
    expect(progress.map((e) => e.body.nativeIter)).toEqual([100, 200, 300, 400, 500, 1800, 1900, 2000, 2100, 2200, 4900, 5000, 5100, 5200, 19800, 19900, 20000]);
    expect(progress[0]?.body).toEqual({ type: 'progress', itersPerSec: 4.176, etaSec: 4765, phaseIndex: 0, nativeIter: 100 });
  });

  it('maps the logged window means to the simulator series', () => {
    const at100 = ofType(events, 'scalar').filter((e) => e.native === 100);
    expect(Object.fromEntries(at100.map((e) => [e.body.series, e.body.value]))).toEqual({
      'sys/data_wait': 0,
      'lr/g': 2e-4,
      'loss/g/pix': 0.17889,
      'loss/g/mssim': 0.34647,
      'loss/g/consistency': 0.10501,
      'loss/g/ldl': 0.0091399,
      'loss/g/fdl': 0.63983,
      'loss/g/total': 1.2793,
      'grad_norm/g': 106.88,
      'clipped/g': 1,
      'skipped/g': 0,
    });
    expect(ofType(events, 'warning')).toEqual([]);
  });

  it('turns each validation block into one event at the update it ran', () => {
    const val = ofType(events, 'validation');
    expect(val.map((e) => [e.native, e.body.dataset])).toEqual([
      [2000, 'dev'],
      [2000, 'val8'],
      [20000, 'dev'],
      [20000, 'val8'],
    ]);
    expect(val[0]?.body.metrics).toEqual({ psnr: 29.7477, topiq: 0.6024, lpips: 0.2329, dists: 0.1873 });
    expect(val[3]?.body.metrics).toEqual({ psnr: 28.99, topiq: 0.6354, lpips: 0.2192, dists: 0.1489 });
    expect(val[0]?.body.images).toEqual([]);
  });

  it('reports a save once the next line shows it finished, with the export size, and the final save', () => {
    const checkpoints = ofType(events, 'checkpoint').map((e) => [e.native, e.body.files, e.body.sizeMb, e.body.state]);
    expect(checkpoints).toEqual([
      [5000, ['net_g_5000.pth', '5000.state'], 28.3, true],
      [20000, ['net_g_20000.pth', 'net_g_latest.pth', '20000.state'], 28.3, true],
    ]);
    expect(parser.outcome).toBe('completed');
    expect(parser.error).toBeUndefined();
  });

  it('logs every non-blank line without its timestamp and colour codes', () => {
    const logs = ofType(events, 'log');
    expect(logs.length).toBe(FIXTURE.filter((l) => l.trim()).length);
    expect(logs[0]?.body).toEqual({ type: 'log', level: 'info', line: 'PyTorch 2.9.1+cu128, device cuda:0 (NVIDIA GeForce RTX 4090)' });
    expect(logs.some((e) => e.body.line.includes('\x1b'))).toBe(false);
  });
});

describe('TrainerLogParser on synthetic lines', () => {
  it('reads a GAN progress line: D series, a skipped update, several lrs, a multi-day eta', () => {
    const { events } = parse([
      `${TS} INFO: [image] network_d: metagan2, parameters: 1,234`,
      `${TS} INFO: [ epoch:   3 ] \x1b[1;32m[ iter: 12,300 ]\x1b[0m [ performance: 3.910 it/s, data 7% ] [ lr: 1.00e-04 5.00e-05 ] [ lr_d: 2.00e-04 ] [ eta: 1 day, 2:03:04 ] [ l_g_total: 1.0000e+00 ] [ l_d_real: 3.0000e-01 ] [ out_d_fake: -2.0000e-01 ] [ grad_norm_d: 5.0000e-01 ] [ skipped_g: 2.0000e+00 ] [ nonfinite_l_g_gan: 1.0000e+00 ] `,
      `${TS} INFO: saving update 12300`,
      `${TS} INFO: one export set is 40.0 MiB; logger.keep_exports = 0 (0 keeps every set)`,
    ]);
    expect(ofType(events, 'progress')[0]?.body).toMatchObject({ itersPerSec: 3.91, etaSec: 93_784, nativeIter: 12_300 });
    const series = Object.fromEntries(ofType(events, 'scalar').map((e) => [e.body.series, e.body.value]));
    expect(series).toMatchObject({ 'sys/data_wait': 7, 'lr/g': 1e-4, 'lr/d': 2e-4, 'loss/d/real': 0.3, 'out/d/fake': -0.2, 'grad_norm/d': 0.5, 'skipped/g': 2, 'nonfinite/loss/g/gan': 1 });
    expect(ofType(events, 'warning').map((e) => e.body.code)).toEqual(['NONFINITE']);
    expect(ofType(events, 'checkpoint')[0]?.body).toEqual({ type: 'checkpoint', files: ['net_g_12300.pth', 'net_d_12300.pth', '12300.state'], sizeMb: 40, state: true });
  });

  it('turns the VRAM field of a CUDA progress line into a system event (no GPU utilisation)', () => {
    const { events } = parse([
      `${TS} INFO: [ epoch:   0 ] [ iter:     100 ] [ performance: 9.100 it/s, data 4% ] [ mem: 3.27 / 23.99 GiB ] [ lr: 2.00e-04 ] [ eta: 0:10:00 ] [ l_g_total: 1.0000e+00 ] `,
    ]);
    expect(ofType(events, 'system').map((e) => e.body)).toEqual([
      { type: 'system', gpuUtil: null, vramUsedGb: 3.27, vramTotalGb: 23.99, dataWaitPct: 4 },
    ]);
    const series = ofType(events, 'scalar').map((e) => e.body.series);
    expect(series).not.toContain('sys/data_wait'); // carried by the system event
    expect(series).toContain('lr/g');
  });

  it('takes the exception line of a traceback as the error and drops the save it interrupted', () => {
    const { parser, events } = parse(
      [
        `${TS} INFO: saving update 4000`,
        `${TS} ERROR: training failed`,
        'Traceback (most recent call last):',
        '  File "<neosr>/train.py", line 341, in main',
        '    train(cfg, rt)',
        'FloatingPointError: non-finite weights or buffers; the last committed checkpoint stands',
      ],
      false,
    );
    expect(parser.error).toBe('FloatingPointError: non-finite weights or buffers; the last committed checkpoint stands');
    expect(ofType(events, 'checkpoint')).toEqual([]);
    expect(ofType(events, 'log').slice(1).every((e) => e.body.level === 'error')).toBe(true);
  });

  it('keeps a save the process did not get to confirm only on a clean exit', () => {
    expect(ofType(parse([`${TS} INFO: saving update 10`], false).events, 'checkpoint')).toEqual([]);
    expect(ofType(parse([`${TS} INFO: saving update 10`], true).events, 'checkpoint')).toHaveLength(1);
  });

  it('reads ConfigError, argparse errors and warnings', () => {
    expect(parse(['config error: [datasets.val] has no images']).parser.error).toBe('[datasets.val] has no images');
    expect(parse(['usage: neosr [-h] -opt OPT', 'neosr: error: the following arguments are required: -opt']).parser.error).toBe(
      'the following arguments are required: -opt',
    );
    const warning = ofType(parse([`${TS} WARNING: input_size is set but torchprofile is not installed: MACs not reported`]).events, 'warning');
    expect(warning.map((e) => e.body)).toEqual([{ type: 'warning', code: 'TRAINER_WARNING', message: 'input_size is set but torchprofile is not installed: MACs not reported' }]);
  });

  it('follows a graceful stop and a best-model save, and ignores tqdm redraws', () => {
    const { parser, events } = parse([
      `${TS} INFO: Validation dev`,
      '',
      '\t # psnr: 30.1000\x1b[1;32m........ Best: 30.1000 @ 2000 iter\x1b[0m',
      '',
      `${TS} INFO: best psnr on dev: 30.1000 @ 2000, saved net_g_best_dev_psnr.pth`,
      'Test img_001:  45%|>>>>      | 135/300 [00:10<00:12, 13.20image/s]',
      `${TS} INFO: stop requested: saving update 2100`,
      `${TS} INFO: Stopped at update 2100. Time consumed: 0:03:12`,
    ]);
    expect(ofType(events, 'validation').map((e) => e.body.metrics)).toEqual([{ psnr: 30.1 }]);
    expect(ofType(events, 'status').map((e) => e.body)).toEqual([{ type: 'status', status: 'saving', message: 'stop requested' }]);
    expect(ofType(events, 'checkpoint').map((e) => [e.native, e.body.files, e.body.state])).toEqual([
      [0, ['net_g_best_dev_psnr.pth'], false],
      [2100, ['net_g_2100.pth', 'net_g_latest.pth', '2100.state'], true],
    ]);
    expect(ofType(events, 'log').some((e) => e.body.line.includes('image/s'))).toBe(false);
    expect(parser.outcome).toBe('stopped');
  });

  it('stamps a record line with the update it reports, not the previous one', () => {
    const { events } = parse([
      `${TS} INFO: [ epoch:   0 ] [ iter:     200 ] [ performance: 9.100 it/s ] [ lr: 2.00e-04 ] [ eta: 0:01:00 ] [ l_g_total: 1.0000e+00 ] `,
      `${TS} INFO: [ epoch:   0 ] [ iter:     250 ] [ performance: 9.100 it/s ] [ lr: 2.00e-04 ] [ eta: 0:00:55 ] [ l_g_total: 1.0000e+00 ] `,
      `${TS} INFO: stop requested: saving update 305`,
      `${TS} INFO: Stopped at update 305. Time consumed: 0:01:17`,
    ]);
    expect(ofType(events, 'log').map((e) => e.native)).toEqual([200, 250, 305, 305]);
    expect(events.findIndex((e) => e.body.type === 'log')).toBe(0); // the line still comes before what it reports
  });

  it('treats "nothing to do" as completed and reports the restored update', () => {
    const { parser, events } = parse([`${TS} INFO: nothing to do: update 20000 >= total_iter 20000`]);
    expect(parser.outcome).toBe('completed');
    expect(ofType(events, 'progress').map((e) => [e.native, e.body.nativeIter])).toEqual([[20000, 20000]]);
  });
});

describe('helpers', () => {
  it('parses timedelta strings', () => {
    expect(parseEta('0:00:08')).toBe(8);
    expect(parseEta('2 days, 0:00:01')).toBe(172_801);
    expect(parseEta('garbage')).toBe(0);
  });

  it('names series like the simulator', () => {
    expect(['l_g_total', 'l_d_r1', 'out_d_margin', 'grad_norm_g', 'clipped_g', 'gamma_r1'].map(seriesOf)).toEqual([
      'loss/g/total',
      'loss/d/r1',
      'out/d/margin',
      'grad_norm/g',
      'clipped/g',
      'train/gamma_r1',
    ]);
  });
});
