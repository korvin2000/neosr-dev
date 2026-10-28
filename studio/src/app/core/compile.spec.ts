import { describe, expect, it } from 'vitest';
import { compileRecipe } from '@shared/compile/compile';
import { importToml } from '@shared/compile/import-toml';
import { lossRows } from '@shared/compile/graph';
import { NODE_TYPES } from '@shared/catalog/node-catalog';
import { emptyRecipe, type LossRow, type Recipe } from '@shared/recipe.model';
import { TEMPLATE_INFOS, buildTemplate } from '@shared/templates';
import * as commands from './commands';

function blockers(recipe: Recipe) {
  return compileRecipe(recipe, { platform: 'windows' }).problems.filter((p) => p.severity === 'blocker');
}

describe('compileRecipe', () => {
  it('compiles every built-in template without blockers', () => {
    for (const info of TEMPLATE_INFOS) {
      const recipe = buildTemplate(info.id);
      expect(blockers(recipe), info.id).toEqual([]);
      const result = compileRecipe(recipe, { platform: 'windows' });
      expect(result.phases.length).toBe(info.phases);
      for (const phase of result.phases) {
        expect(phase.text).toContain('[datasets.train]');
        expect(phase.text).toContain('[network_g]');
        expect(phase.text).toContain('save_checkpoint_freq');
      }
    }
  });

  it('chains a fine-tune phase from the previous phase weights and restarts the native counter', () => {
    const result = compileRecipe(buildTemplate('4x-curriculum'), { platform: 'windows' });
    expect(result.phases).toHaveLength(2);
    const [p1, p2] = result.phases;
    expect(p1.hasGan).toBe(false);
    expect(p2.hasGan).toBe(true);
    expect(p2.nativeStart).toBe(0);
    expect(p2.text).toContain(`pretrain_network_g = "experiments/${p1.runName}/models/net_g_latest.pth"`);
    expect(p2.text).toContain('[network_d]');
    expect(p2.text).toContain('[train.optim_d]');
    expect(p1.text).not.toContain('[network_d]');
    expect(p2.text).toContain('total_iter = 200000');
  });

  it('refuses to continue a training state when the phase adds a discriminator', () => {
    const recipe = buildTemplate('4x-curriculum');
    const second = recipe.phases[1];
    const changed = commands.updatePhase(second.id, { init: 'continue' }).apply(recipe);
    const found = blockers(changed).find((p) => p.message.includes('adds the discriminator'));
    expect(found).toBeDefined();
    expect(found?.phaseId).toBe(second.id);
  });

  it('uses cumulative native counters and a resume_state for a compatible continue phase', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const withPhase = commands.addPhase({ id: 'p2', name: 'More', iters: 50_000, init: 'continue', note: '' }).apply(recipe);
    expect(blockers(withPhase)).toEqual([]);
    const result = compileRecipe(withPhase, { platform: 'windows' });
    expect(result.phases[1].nativeStart).toBe(150_000);
    expect(result.phases[1].text).toContain('total_iter = 200000');
    expect(result.phases[1].text).toContain(`resume_state = "experiments/${result.phases[0].runName}/training_states/150000.state"`);
  });

  it('enforces the GAN triple coupling', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const losses = recipe.nodes.find((n) => n.type === NODE_TYPES.losses)!;
    const rows: LossRow[] = [...lossRows(losses), { id: 'gan', type: 'gan_loss', weight: 0.3, enabled: true, params: {}, schedule: {} }];
    const withGan = commands.setValues(losses.id, { losses: rows }).apply(recipe);
    const messages = blockers(withGan).map((p) => p.message);
    expect(messages.some((m) => m.includes('Discriminator'))).toBe(true);
    expect(messages.some((m) => m.includes('Optimizer D'))).toBe(true);
  });

  it('blocks torch.compile on Windows only', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const runtime = recipe.nodes.find((n) => n.type === NODE_TYPES.runtime)!;
    const compiled = commands.setValues(runtime.id, { compile: true }).apply(recipe);
    expect(blockers(compiled).some((p) => p.message.includes('Windows'))).toBe(true);
    expect(compileRecipe(compiled, { platform: 'linux' }).problems.filter((p) => p.severity === 'blocker')).toEqual([]);
  });

  it('per-phase overrides change only that phase', () => {
    const recipe = buildTemplate('4x-curriculum');
    const data = recipe.nodes.find((n) => n.type === NODE_TYPES.trainDataset)!;
    const p2 = recipe.phases[1].id;
    const changed = commands.setPhaseValues(data.id, p2, { patch_size: 96 }).apply(recipe);
    const result = compileRecipe(changed, { platform: 'windows' });
    expect(result.phases[0].text).toContain('patch_size = 64');
    expect(result.phases[1].text).toContain('patch_size = 96');
  });
});

describe('review fixes', () => {
  it('blocks two active losses that share a slot and newLossRow avoids the clash', async () => {
    const { newLossRow } = await import('@shared/recipe-builder');
    const recipe = buildTemplate('4x-span-fidelity');
    const losses = recipe.nodes.find((n) => n.type === NODE_TYPES.losses)!;
    const rows = lossRows(losses);
    const clash = commands.setValues(losses.id, { losses: [...rows, { id: 'mse', type: 'MSELoss', weight: 1, enabled: true, params: {}, schedule: {} }] }).apply(recipe);
    expect(blockers(clash).some((p) => p.message.includes('[train.pixel_opt]'))).toBe(true);
    const unique = newLossRow('MSELoss', rows);
    expect(unique.params['slot']).toBe('pixel_2');
    const ok = commands.setValues(losses.id, { losses: [...rows, unique] }).apply(recipe);
    expect(blockers(ok)).toEqual([]);
    expect(compileRecipe(ok).phases[0].text).toContain('[train.pixel_2_opt]');
  });

  it('keeps a zero-weight gan row because NeoSR keeps training the discriminator', () => {
    const recipe = buildTemplate('4x-neosr-gan');
    const losses = recipe.nodes.find((n) => n.type === NODE_TYPES.losses)!;
    const rows = lossRows(losses).map((r) => (r.type === 'gan_loss' ? { ...r, weight: 0 } : r));
    const zero = commands.setValues(losses.id, { losses: rows }).apply(recipe);
    const text = compileRecipe(zero).phases[0].text;
    expect(text).toContain('[network_d]');
    expect(text).toContain('[train.optim_d]');
    expect(text).toContain('loss_weight = 0.0');
    expect(blockers(zero)).toEqual([]);
  });

  it('starts the native clock at the resumed update', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const init = recipe.nodes.find((n) => n.type === NODE_TYPES.initialization)!;
    const resumed = commands.setValues(init.id, { mode: 'resume', resume_state: 'experiments/old/training_states/50000.state' }).apply(recipe);
    const result = compileRecipe(resumed);
    expect(result.phases[0].nativeStart).toBe(50000);
    expect(result.phases[0].text).toContain('total_iter = 200000');
  });

  it('treats a disabled required source as missing', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const generator = recipe.nodes.find((n) => n.type === NODE_TYPES.generator)!;
    const disabled = commands.setNodeFlags(generator.id, { disabled: true }).apply(recipe);
    expect(blockers(disabled).some((p) => p.message.includes('disabled node'))).toBe(true);
  });

  it('enforces native numeric bounds', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const data = recipe.nodes.find((n) => n.type === NODE_TYPES.trainDataset)!;
    const bad = commands.setValues(data.id, { patch_size: 0, batch_size: -1 }).apply(recipe);
    const messages = blockers(bad).map((p) => p.message);
    expect(messages.some((m) => m.includes('LQ patch size'))).toBe(true);
    expect(messages.some((m) => m.includes('Batch size'))).toBe(true);
  });

  it('resets architecture options when the type changes', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const generator = recipe.nodes.find((n) => n.type === NODE_TYPES.generator)!;
    const tuned = commands.setValues(generator.id, { params: { feature_channels: 64 } }).apply(recipe);
    const switched = commands.setValues(generator.id, { type: 'compact' }).apply(tuned);
    expect(switched.nodes.find((n) => n.id === generator.id)!.values['params']).toEqual({});
  });

  it('chains phases under an imported experiments_root', () => {
    const base = buildTemplate('4x-curriculum');
    const recipe: Recipe = { ...base, nativeExtras: { path: { experiments_root: 'D:\\runs\\' } } };
    const result = compileRecipe(recipe);
    expect(result.phases[1].text).toContain('pretrain_network_g = "D:/runs/');
    expect(result.phases[0].text).toContain("experiments_root = 'D:\\runs\\'");
  });

  it('does not require a continue phase to end on a checkpoint boundary', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const odd = commands.setPhases([{ ...recipe.phases[0], iters: 1501 }, { id: 'p2', name: 'More', iters: 1000, init: 'continue', note: '' }]).apply(recipe);
    expect(blockers(odd)).toEqual([]);
    expect(compileRecipe(odd).phases[1].text).toContain('training_states/1501.state');
  });
});

describe('importToml', () => {
  const toml = `
name = "train_test"
model_type = "image"
scale = 2
use_amp = true
bfloat16 = true
mystery_root = 7

[datasets.train]
type = "paired"
dataroot_gt = 'C:\\\\gt\\\\'
dataroot_lq = 'C:\\\\lq\\\\'
patch_size = 48
batch_size = 4
custom_loader_flag = true

[datasets.val]
name = "val"
type = "paired"
dataroot_gt = 'C:\\\\val\\\\gt\\\\'
dataroot_lq = 'C:\\\\val\\\\lq\\\\'

[val]
val_freq = 500
[val.metrics.psnr]
type = "calculate_psnr"

[network_g]
type = "compact"
num_feat = 32

[train]
ema = 0.999
wavelet_guided = true
wavelet_init = 5000

[train.optim_g]
type = "adamw"
lr = 2e-4

[train.pixel_opt]
type = "L1Loss"
loss_weight = 1.0

[logger]
total_iter = 20000
save_checkpoint_freq = 500
`;

  it('maps known keys to nodes and preserves unknown ones', () => {
    const result = importToml(toml, emptyRecipe('t'));
    expect(result.recipe.nodes.map((n) => n.type)).toContain(NODE_TYPES.generator);
    expect(result.preserved).toEqual(expect.arrayContaining(['mystery_root', 'datasets.train.custom_loader_flag']));
    const generator = result.recipe.nodes.find((n) => n.type === NODE_TYPES.generator)!;
    expect(generator.values['type']).toBe('compact');
    expect(generator.values['params']).toEqual({ num_feat: 32 });
    expect(result.recipe.phases[0].iters).toBe(20000);
  });

  it('uses NeoSR defaults for keys the file omits', () => {
    const result = importToml(toml, emptyRecipe('t'));
    const text = compileRecipe(result.recipe).phases[0].text;
    expect(text).not.toContain('augmentation');
    expect(text).not.toContain('warmup_steps');
    expect(text).toContain('ema = 0.999');
    // NeoSR's reader default is EMA off; an omitted key must not become the template's 0.999.
    const noEma = importToml(toml.replace('ema = 0.999\n', ''), emptyRecipe('t'));
    expect(compileRecipe(noEma.recipe).phases[0].text).toContain('ema = 0.0');
  });

  it('preserves optimizer extras, schedule_free = false and metric parameters', () => {
    const extended = toml
      .replace('lr = 2e-4', 'lr = 2e-4\nr = 1.2\nschedule_free = false')
      .replace('[val.metrics.psnr]\ntype = "calculate_psnr"', '[val.metrics.psnr]\ntype = "calculate_psnr"\ncrop_border = 0\n[val.metrics.psnr_y]\ntype = "calculate_psnr"\ntest_y_channel = true')
      .replace('[datasets.val]\nname = "val"', '[datasets.val_2]\nname = "val2"\ntype = "paired"\ndataroot_gt = "/v2/gt"\ndataroot_lq = "/v2/lq"\ncolor = "y"\n[datasets.val]\nname = "val"');
    const result = importToml(extended, emptyRecipe('t'));
    const text = compileRecipe(result.recipe).phases[0].text;
    expect(text).toContain('r = 1.2');
    expect(text).toContain('schedule_free = false');
    expect(text).toContain('crop_border = 0');
    expect(text).toContain('[val.metrics.psnr_y]');
    expect(text).toContain('test_y_channel = true');
    expect(text).toContain('[datasets.val_2]');
    expect(text.indexOf('[datasets.val_2]')).toBeLessThan(text.indexOf('color = "y"'));
    expect(compileRecipe(result.recipe).problems.filter((p) => p.severity === 'blocker')).toEqual([]);
  });

  it('round-trips: compiled output keeps the imported values and the extras', () => {
    const result = importToml(toml, emptyRecipe('t'));
    const compiled = compileRecipe(result.recipe, { platform: 'windows' });
    const text = compiled.phases[0].text;
    expect(text).toContain('scale = 2');
    expect(text).toContain('patch_size = 48');
    expect(text).toContain('num_feat = 32');
    expect(text).toContain('mystery_root = 7');
    expect(text).toContain('custom_loader_flag = true');
    expect(text).toContain('lr = 2e-4');
    expect(text).toContain('[train.pixel_opt]');
    // wavelet keys without a GAN row are legacy noise; the importer drops them rather than emitting an invalid config.
    expect(text).not.toContain('wavelet_guided');
    expect(compiled.problems.filter((p) => p.severity === 'blocker')).toEqual([]);
  });
});

describe('GAN bundle commands', () => {
  it('enables GAN as one command: row + discriminator + optimizer D, all wired', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const losses = recipe.nodes.find((n) => n.type === NODE_TYPES.losses)!;
    const on = commands.setGanBundle(losses.id, true).apply(recipe);
    expect(on.nodes.some((n) => n.type === NODE_TYPES.discriminator)).toBe(true);
    expect(on.nodes.some((n) => n.type === NODE_TYPES.optimizerD)).toBe(true);
    expect(blockers(on)).toEqual([]);
    expect(compileRecipe(on).phases[0].hasGan).toBe(true);
    const off = commands.setGanBundle(losses.id, false).apply(on);
    expect(off.nodes.some((n) => n.type === NODE_TYPES.discriminator)).toBe(false);
    expect(off.nodes.some((n) => n.type === NODE_TYPES.optimizerD)).toBe(false);
    expect(blockers(off)).toEqual([]);
  });

  it('re-enabling GAN activates the row in every phase', () => {
    const recipe = buildTemplate('4x-curriculum');
    const losses = recipe.nodes.find((n) => n.type === NODE_TYPES.losses)!;
    const firstOnly = commands.removePhase(recipe.phases[1].id).apply(recipe);
    const on = commands.setGanBundle(losses.id, true).apply(firstOnly);
    expect(compileRecipe(on).phases[0].hasGan).toBe(true);
    expect(blockers(on)).toEqual([]);
  });

  it('a loss preset with gan reconciles the bundle', async () => {
    const { lossRowsFor } = await import('@shared/recipe-builder');
    const recipe = buildTemplate('4x-span-fidelity');
    const losses = recipe.nodes.find((n) => n.type === NODE_TYPES.losses)!;
    const perceptual = commands.setLossRows(losses.id, lossRowsFor('perceptual')).apply(recipe);
    expect(blockers(perceptual)).toEqual([]);
    expect(perceptual.nodes.some((n) => n.type === NODE_TYPES.discriminator)).toBe(true);
  });

  it('a disconnected duplicate loss node cannot touch the wired bundle', () => {
    const recipe = buildTemplate('4x-neosr-gan');
    const losses = recipe.nodes.find((n) => n.type === NODE_TYPES.losses)!;
    const dup = commands.duplicateNode(losses.id).apply(recipe);
    const copy = dup.nodes.find((n) => n.type === NODE_TYPES.losses && n.id !== losses.id)!;
    const off = commands.setGanBundle(copy.id, false).apply(dup);
    expect(off.nodes.some((n) => n.type === NODE_TYPES.discriminator)).toBe(true);
    expect(blockers(off)).toEqual([]);
  });

  it('adds degradations wired into the dataset', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const data = recipe.nodes.find((n) => n.type === NODE_TYPES.trainDataset)!;
    const otf = commands.setValues(data.id, { type: 'otf' }).apply(recipe);
    const withDeg = commands.addNodeWired(NODE_TYPES.degradation, { x: 0, y: 0 }).apply(otf);
    expect(withDeg.edges.some((e) => e.targetNode === data.id && e.targetPort === 'degradation')).toBe(true);
    expect(blockers(withDeg)).toEqual([]);
  });
});
