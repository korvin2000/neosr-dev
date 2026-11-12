/**
 * End-to-end contract with the trainer: compile every template, a template with the opt-in trainer
 * options switched on, and the shipped realplksr recipes after an import round-trip; write the TOMLs
 * to a temporary folder and run the trainer's own `load_config` + `validate_train` on them
 * (`scripts/studio_surface.py --validate`). Run with `npm run contract`, which first checks that
 * neosr-surface.json is current.
 */
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { NODE_TYPES } from '../shared/catalog/node-catalog';
import { compileRecipe } from '../shared/compile/compile';
import { lossRows } from '../shared/compile/graph';
import { importToml } from '../shared/compile/import-toml';
import { applyDraft } from '../shared/recipe-builder';
import { emptyRecipe, type Recipe } from '../shared/recipe.model';
import { TEMPLATE_INFOS, buildTemplate } from '../shared/templates';

const SHIPPED = ['train_realplksr.toml', 'train_realplksr_otf.toml', 'train_realplksr_r3gan.toml'];

function withValues(recipe: Recipe, type: string, values: Record<string, unknown>): Recipe {
  return { ...recipe, nodes: recipe.nodes.map((n) => (n.type === type ? { ...n, values: { ...n.values, ...values } } : n)) };
}

/** The OTF GAN template with every opt-in key Studio writes set away from its default. */
function optionsRecipe(): Recipe {
  let recipe = applyDraft(buildTemplate('4x-otf-gan'), { schedule: { type: 'multisteplr', milestones: [100_000], nets: ['g', 'd'] } });
  recipe = withValues(recipe, NODE_TYPES.plan, { clamp: 'true', grad_clip: 'norm', grad_clip_norm: 0.5, max_skip_streak: 5, keep_exports: 2 });
  recipe = withValues(recipe, NODE_TYPES.runtime, { deterministic: true, detect_anomaly: true, fp16_accumulation: true, manual_seed: 7 });
  recipe = withValues(recipe, NODE_TYPES.trainDataset, { aug_multi_prob: 0.5, aug_antialias: true, tight_canvas: true });
  recipe = withValues(recipe, NODE_TYPES.validation, { metrics: ['psnr', 'topiq', 'lpips'], save_best: 'lpips', topiq_dataset: 'pipal', lpips_net: 'vgg', psnr_crop_border: 0, psnr_test_y_channel: true, psnr_input_order: 'CHW' });
  recipe = withValues(recipe, NODE_TYPES.generator, { params: { layer_norm: true, icnr: true } });
  recipe = withValues(recipe, NODE_TYPES.discriminator, { type: 'r3gan' });
  const rows = lossRows(recipe.nodes.find((n) => n.type === NODE_TYPES.losses)).map((row) =>
    row.type === 'gan_loss'
      ? { ...row, params: { gan_type: 'rpgan', r1_gamma: 1, r2_gamma: 1, reg_every: 1, gamma_decay: 0.1, gamma_iters: 200_000, start_iter: 1000, ramp_iters: 500 } }
      : row,
  );
  return withValues(recipe, NODE_TYPES.losses, { losses: [...rows, { id: 'psnr', type: 'psnr_loss', weight: 0.1, enabled: true, params: { end_iter: 50_000 }, schedule: {} }] });
}

const dir = mkdtempSync(join(tmpdir(), 'neosr-studio-contract-'));
let status = 1;
try {
  const files: string[] = [];
  let blocked = 0;
  const write = (prefix: string, recipe: Recipe) => {
    const result = compileRecipe(recipe, { platform: 'linux' });
    for (const b of result.problems.filter((p) => p.severity === 'blocker')) {
      console.log(`BLOCKER ${prefix}: ${b.message}`);
      blocked++;
    }
    for (const phase of result.phases) {
      const file = join(dir, `${prefix}__${phase.fileName}`);
      writeFileSync(file, phase.text, 'utf8');
      files.push(file);
    }
  };
  for (const info of TEMPLATE_INFOS) {
    write(info.id, buildTemplate(info.id));
  }
  write('options', optionsRecipe());
  for (const name of SHIPPED) {
    const imported = importToml(readFileSync(join('..', 'options', name), 'utf8'), emptyRecipe(name));
    if (imported.preserved.length) {
      console.log(`UNMAPPED ${name}: ${imported.preserved.join(', ')}`);
      blocked++;
    }
    write(`import-${name.replace(/\.toml$/, '')}`, imported.recipe);
  }
  const run = spawnSync('python', ['../scripts/studio_surface.py', '--validate', ...files], { stdio: 'inherit' });
  if (run.error) {
    console.error(`could not run python: ${run.error.message}`);
  }
  status = blocked ? 1 : (run.status ?? 1);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
process.exit(status);
