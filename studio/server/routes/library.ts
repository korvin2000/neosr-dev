import { Router } from 'express';
import type { DatasetInfo, ModelFile, ScanReport } from '../api-types';
import { EXPERIMENTS_ROOT } from '../config';
import { badRequest, bodyOf, optionalNumber, optionalString } from '../http';
import { archProfile } from '../sim/sim-plan';
import { hashString, intBetween, mulberry32, pick, type Rng } from '../sim/prng';
import { runStore } from '../store/run-store';

export const libraryRouter = Router();

libraryRouter.get('/library/models', (_req, res) => {
  res.json([...MOCK_MODELS, ...runModels()].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)));
});

libraryRouter.get('/library/datasets', (_req, res) => {
  res.json(DATASETS);
});

libraryRouter.post('/datasets/scan', (req, res) => {
  const body = bodyOf(req.body);
  const gt = optionalString(body['gt']);
  if (!gt) {
    throw badRequest('Body needs { gt: string, lq?: string, scale: number, patch: number }');
  }
  const lq = optionalString(body['lq']) || undefined;
  const scale = Math.max(1, Math.round(optionalNumber(body['scale']) ?? 4));
  const patch = Math.max(8, Math.round(optionalNumber(body['patch']) ?? 64));
  res.json(scanDataset(gt, lq, scale, patch));
});

// --- models ----------------------------------------------------------------------------------

const DAY_MS = 86_400_000;

function mockModel(runName: string, arch: string, scale: number, iter: number, variant: ModelFile['variant'], daysAgo: number): ModelFile {
  const file = variant === 'state' ? `${iter}.state` : `net_g_${variant === 'ema' ? 'ema_' : ''}${iter}.pth`;
  const folder = variant === 'state' ? 'training_states' : 'models';
  const size = archProfile(arch).sizeMb;
  return {
    file,
    path: `${EXPERIMENTS_ROOT}\\${runName}\\${folder}\\${file}`,
    arch,
    scale,
    iter,
    sizeMb: variant === 'state' ? size * 3 : size,
    runName,
    variant,
    updatedAt: new Date(Date.now() - daysAgo * DAY_MS).toISOString(),
  };
}

const MOCK_MODELS: readonly ModelFile[] = [
  mockModel('4x_esrgan_photo', 'esrgan', 4, 400_000, 'ema', 21),
  mockModel('4x_esrgan_photo', 'esrgan', 4, 400_000, 'eval', 21),
  mockModel('4x_esrgan_photo', 'esrgan', 4, 400_000, 'state', 21),
  mockModel('2x_span_anime', 'span', 2, 120_000, 'ema', 12),
  mockModel('2x_span_anime', 'span', 2, 80_000, 'ema', 13),
  mockModel('2x_span_anime', 'span', 2, 120_000, 'state', 12),
  mockModel('1x_compact_dejpeg', 'compact', 1, 40_000, 'eval', 6),
  mockModel('1x_compact_dejpeg', 'compact', 1, 40_000, 'state', 6),
];

/** The final generator of each phase of every completed simulated run. */
function runModels(): ModelFile[] {
  return runStore.completedRuns().flatMap((run) => {
    // NeoSR names files by its own update counter, which restarts on a fine-tune phase and
    // keeps counting on a continue phase.
    let nativeEnd = 0;
    return run.phases.map((phase, i) => {
      const init = run.recipe.phases[i]?.init;
      const nativeStart = i > 0 && init === 'continue' ? nativeEnd : 0;
      nativeEnd = nativeStart + (phase.end - phase.start);
      const iter = nativeEnd;
      const file = `net_g_${iter}.pth`;
      return {
        file,
        path: `${EXPERIMENTS_ROOT}\\${phase.runName}\\models\\${file}`,
        arch: run.arch,
        scale: run.scale,
        iter,
        sizeMb: archProfile(run.arch).sizeMb,
        runName: phase.runName,
        variant: 'ema' as const,
        updatedAt: run.finishedAt ?? run.createdAt,
      };
    });
  });
}

// --- datasets --------------------------------------------------------------------------------

const DATASETS: readonly DatasetInfo[] = [
  {
    id: 'ds_paired_x4',
    name: 'Photo pairs ×4',
    gtPath: 'C:\\datasets\\train\\gt\\',
    lqPath: 'C:\\datasets\\train\\lq\\',
    images: 3_450,
    scale: 4,
    minWidth: 512,
    minHeight: 384,
    format: 'png',
    note: 'Clean paired set: every GT has an LQ at exactly 1/4 size.',
  },
  {
    id: 'ds_otf_gt',
    name: 'Real-world GT (OTF)',
    gtPath: 'C:\\datasets\\otf\\gt\\',
    images: 9_812,
    minWidth: 768,
    minHeight: 512,
    format: 'png',
    note: 'GT only: low-quality inputs are synthesised on the fly by the degradation pipeline.',
  },
  {
    id: 'ds_val',
    name: 'Validation (10 images)',
    gtPath: 'C:\\datasets\\val\\gt\\',
    lqPath: 'C:\\datasets\\val\\lq\\',
    images: 10,
    scale: 4,
    minWidth: 480,
    minHeight: 320,
    format: 'png',
    note: 'Small varied set for val_freq checks; keep it disjoint from training data.',
  },
];

/** Deterministic mock scan: paths containing `bad` or `broken` produce problems. */
function scanDataset(gt: string, lq: string | undefined, scale: number, patch: number): ScanReport {
  const rng = mulberry32(hashString(`${gt}|${lq ?? ''}|${scale}|${patch}`));
  const broken = /bad|broken/i.test(`${gt} ${lq ?? ''}`);
  const gtCount = intBetween(rng, 800, 4000);
  const missing = broken ? intBetween(rng, 3, 9) : 0;
  const lqCount = lq ? gtCount - missing + (broken ? 2 : 0) : 0;
  const gray = intBetween(rng, 1, 3);
  const rgba = broken ? intBetween(rng, 2, 6) : 0;
  return {
    gtCount,
    lqCount,
    pairs: lq ? gtCount - missing : 0,
    unmatchedGt: lq ? fileNames(rng, missing, gtCount) : [],
    unmatchedLq: lq && broken ? ['thumbs.png', 'copy of 0042.png'] : [],
    scaleMismatch: lq && broken ? mismatches(rng, scale, gtCount) : [],
    belowPatch: broken ? belowPatch(rng, patch * scale, gtCount) : [],
    channels: { rgb: gtCount - gray - rgba, rgba, gray },
    overlapWithVal: broken || rng() < 0.5 ? fileNames(rng, broken ? 3 : 1, gtCount) : [],
    samples: Array.from({ length: 6 }, () => ({ name: fileName(rng, gtCount), w: pick(rng, [1280, 1920, 2048, 960]), h: pick(rng, [720, 1080, 1536, 640]) })),
    durationMs: Math.round(gtCount * 0.35 + intBetween(rng, 50, 400)),
  };
}

function fileName(rng: Rng, count: number): string {
  return `${String(intBetween(rng, 1, count)).padStart(4, '0')}.png`;
}

function fileNames(rng: Rng, n: number, count: number): string[] {
  return [...new Set(Array.from({ length: n }, () => fileName(rng, count)))].sort();
}

function mismatches(rng: Rng, scale: number, count: number): ScanReport['scaleMismatch'] {
  return fileNames(rng, 3, count).map((file) => {
    const w = pick(rng, [1280, 1920, 1023]);
    const h = pick(rng, [720, 1080, 767]);
    return { file, gt: [w, h] as const, lq: [Math.floor(w / scale) + 1, Math.floor(h / scale)] as const };
  });
}

function belowPatch(rng: Rng, gtPatch: number, count: number): ScanReport['belowPatch'] {
  return fileNames(rng, 2, count).map((file) => ({ file, w: intBetween(rng, gtPatch / 2, gtPatch - 1), h: intBetween(rng, gtPatch / 3, gtPatch - 1) }));
}
