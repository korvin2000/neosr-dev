/**
 * Built-in recipe templates. Each is a draft (see recipe-builder) so a template and a wizard
 * result are the same kind of thing; loading one materializes the nodes.
 */
import { DEGRADATION_TEMPLATE } from './catalog/degradations';
import type { ObjectiveId } from './catalog/node-catalog';
import { applyDraft, lossRowsFor, newPhase, type RecipeDraft } from './recipe-builder';
import { emptyRecipe, newId, snapshotPreset, type LossRow, type Phase, type Recipe } from './recipe.model';

export interface TemplateInfo {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  readonly scale: number;
  readonly arch: string;
  readonly objective: ObjectiveId;
  readonly phases: number;
  readonly totalIter: number;
  readonly tags: readonly string[];
  /** Where the numbers come from, shown on the card. */
  readonly provenance: 'neosr template' | 'wiki' | 'studio';
}

interface TemplateDef extends TemplateInfo {
  readonly draft: () => RecipeDraft;
}

const WIN = (p: string) => p;

function baseDraft(opts: {
  name: string;
  scale: number;
  arch: string;
  objective: ObjectiveId;
  iters: number;
  patch?: number;
  batch?: number;
  lr?: number;
  otf?: boolean;
  archParams?: Record<string, unknown>;
}): RecipeDraft {
  const gan = opts.objective === 'perceptual';
  return {
    plan: { name: opts.name, save_checkpoint_freq: 1000, ema: 0.999, use_tb_logger: true },
    runtime: { precision: 'bf16', fast_matmul: true },
    dataset: {
      type: opts.otf ? 'otf' : 'paired',
      dataroot_gt: WIN('C:\\datasets\\train\\gt\\'),
      dataroot_lq: opts.otf ? '' : WIN('C:\\datasets\\train\\lq\\'),
      patch_size: opts.patch ?? 64,
      batch_size: opts.batch ?? 8,
    },
    degradation: opts.otf ? { ...DEGRADATION_TEMPLATE } : null,
    validation: {
      datasets: [{ id: newId('v'), name: 'val', type: 'paired', dataroot_gt: WIN('C:\\datasets\\val\\gt\\'), dataroot_lq: WIN('C:\\datasets\\val\\lq\\') }],
      val_freq: 1000,
      metrics: ['psnr', 'ssim', 'dists'],
    },
    generator: { scale: String(opts.scale), type: opts.arch, params: opts.archParams ?? {} },
    discriminator: gan ? { type: 'metagan' } : null,
    init: { mode: 'scratch' },
    losses: lossRowsFor(opts.objective),
    optimG: { type: 'adan_sf', lr: opts.lr ?? 0.001, betas: [0.98, 0.92, 0.99], weight_decay: 0.01, warmup_steps: 1600 },
    optimD: gan ? { type: 'adan_sf', lr: 0.0001, betas: [0.98, 0.92, 0.99], weight_decay: 0.01, warmup_steps: 600 } : null,
    schedule: null,
    phases: [newPhase('Train', opts.iters, 'scratch')],
  };
}

/** Two phases: fidelity from scratch, then the full GAN recipe fine-tuned from phase 1 weights. */
function curriculumDraft(): RecipeDraft {
  const p1: Phase = newPhase('Pixel pretrain', 100_000, 'scratch', 'L1 + MS-SSIM only');
  const p2: Phase = newPhase('GAN fine-tune', 200_000, 'finetune', 'adds FDL, LDL and GAN');
  const fidelity = lossRowsFor('fidelity');
  const perceptual = lossRowsFor('perceptual');
  const byType = new Map<string, LossRow>();
  for (const row of [...fidelity, ...perceptual]) {
    if (!byType.has(row.type)) {
      byType.set(row.type, row);
    }
  }
  const rows: LossRow[] = [...byType.values()].map((row) => {
    const inP1 = fidelity.find((r) => r.type === row.type);
    const inP2 = perceptual.find((r) => r.type === row.type);
    return {
      ...row,
      weight: (inP2 ?? inP1)?.weight ?? row.weight,
      schedule: {
        [p1.id]: { active: !!inP1, weight: inP1?.weight ?? row.weight },
        [p2.id]: { active: !!inP2, weight: inP2?.weight ?? row.weight },
      },
    };
  });
  const base = baseDraft({ name: '4x_curriculum_realplksr', scale: 4, arch: 'realplksr', objective: 'perceptual', iters: 300_000, lr: 0.0005 });
  return { ...base, losses: rows, phases: [p1, p2] };
}

const TEMPLATES: readonly TemplateDef[] = [
  {
    id: '4x-span-fidelity',
    label: '×4 SPAN · fidelity',
    description: 'Fast conv net, pixel and structure losses only. A safe first run and a good pretrain for later GAN fine-tuning.',
    scale: 4,
    arch: 'span',
    objective: 'fidelity',
    phases: 1,
    totalIter: 150_000,
    tags: ['beginner', 'fast', 'pretrain'],
    provenance: 'studio',
    draft: () => baseDraft({ name: '4x_span_fidelity', scale: 4, arch: 'span', objective: 'fidelity', iters: 150_000 }),
  },
  {
    id: '2x-compact-beginner',
    label: '×2 Compact · beginner',
    description: 'The Upscale-Hub beginner setup: Compact converges in well under 40k updates and needs little VRAM.',
    scale: 2,
    arch: 'compact',
    objective: 'fidelity',
    phases: 1,
    totalIter: 40_000,
    tags: ['beginner', 'fast', 'low vram'],
    provenance: 'wiki',
    draft: () => baseDraft({ name: '2x_compact', scale: 2, arch: 'compact', objective: 'fidelity', iters: 40_000 }),
  },
  {
    id: '4x-realplksr-balanced',
    label: '×4 RealPLKSR · balanced',
    description: 'The NeoSR template losses without the GAN term: sharper than fidelity, no hallucinated texture.',
    scale: 4,
    arch: 'realplksr',
    objective: 'balanced',
    phases: 1,
    totalIter: 200_000,
    tags: ['balanced', 'no gan'],
    provenance: 'neosr template',
    draft: () => baseDraft({ name: '4x_realplksr_balanced', scale: 4, arch: 'realplksr', objective: 'balanced', iters: 200_000, lr: 0.0005 }),
  },
  {
    id: '4x-neosr-gan',
    label: '×4 NeoSR template · GAN',
    description: 'Exactly the shipped train_*.toml recipe: MS-SSIM + consistency + LDL + FDL + GAN (metagan), wavelet view from 80k.',
    scale: 4,
    arch: 'realplksr',
    objective: 'perceptual',
    phases: 1,
    totalIter: 1_000_000,
    tags: ['gan', 'template'],
    provenance: 'neosr template',
    draft: () => baseDraft({ name: '4x_realplksr_gan', scale: 4, arch: 'realplksr', objective: 'perceptual', iters: 1_000_000, lr: 0.0005 }),
  },
  {
    id: '4x-curriculum',
    label: '×4 Pixel pretrain → GAN fine-tune',
    description: 'Two phases: 100k fidelity from scratch, then 200k with the full GAN recipe fine-tuned from the saved weights.',
    scale: 4,
    arch: 'realplksr',
    objective: 'perceptual',
    phases: 2,
    totalIter: 300_000,
    tags: ['curriculum', 'gan', 'two runs'],
    provenance: 'studio',
    draft: curriculumDraft,
  },
  {
    id: '4x-otf-gan',
    label: '×4 On-the-fly degradations · GAN',
    description: 'GT-only dataset with the shipped two-stage Real-ESRGAN degradation table. Suits real-world sources more than anime.',
    scale: 4,
    arch: 'realplksr',
    objective: 'perceptual',
    phases: 1,
    totalIter: 500_000,
    tags: ['otf', 'gan', 'real-world'],
    provenance: 'neosr template',
    draft: () => baseDraft({ name: '4x_realplksr_otf', scale: 4, arch: 'realplksr', objective: 'perceptual', iters: 500_000, lr: 0.0005, otf: true }),
  },
  {
    id: '1x-restore-span',
    label: '×1 Restoration · SPAN',
    description: 'Scale 1 for denoising / de-JPEG with balanced losses. Pairs must be the same size.',
    scale: 1,
    arch: 'span',
    objective: 'balanced',
    phases: 1,
    totalIter: 150_000,
    tags: ['restore', '×1'],
    provenance: 'studio',
    draft: () => baseDraft({ name: '1x_span_restore', scale: 1, arch: 'span', objective: 'balanced', iters: 150_000, patch: 96 }),
  },
  {
    id: '2x-hat-s',
    label: '×2 HAT-S · quality',
    description: 'Window-attention transformer: patch 32, batch 4, slow but strong. Expect high VRAM.',
    scale: 2,
    arch: 'hat_s',
    objective: 'balanced',
    phases: 1,
    totalIter: 300_000,
    tags: ['transformer', 'quality', 'high vram'],
    provenance: 'neosr template',
    draft: () => baseDraft({ name: '2x_hat_s', scale: 2, arch: 'hat_s', objective: 'balanced', iters: 300_000, patch: 32, batch: 4 }),
  },
];

export const TEMPLATE_INFOS: readonly TemplateInfo[] = TEMPLATES.map(({ draft, ...info }) => {
  void draft;
  return info;
});

export function buildTemplate(id: string): Recipe {
  const def = TEMPLATES.find((t) => t.id === id);
  if (!def) {
    throw new Error(`Unknown template: ${id}`);
  }
  const recipe = applyDraft(emptyRecipe(def.label), def.draft());
  return snapshotPreset({ ...recipe, name: def.label, meta: { ...recipe.meta, templateId: id, tags: [...def.tags], description: def.description } }, def.label);
}
