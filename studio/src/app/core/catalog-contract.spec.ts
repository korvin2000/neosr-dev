/**
 * Contract between the Studio catalog and the trainer. `shared/catalog/neosr-surface.json` is
 * generated from the trainer's own tables by `scripts/studio_surface.py` (`npm run surface`); when
 * the trainer changes, these cases name what the catalog lacks or has too much of.
 */
import { describe, expect, it } from 'vitest';
import { parse } from 'smol-toml';
import surface from '@shared/catalog/neosr-surface.json';
import { ARCHS } from '@shared/catalog/archs';
import type { ParamInfo } from '@shared/catalog/data-types';
import { DISCRIMINATORS } from '@shared/catalog/discriminators';
import { LOSS_SCHEDULE_PARAMS, LOSSES } from '@shared/catalog/losses';
import { METRIC_TYPES } from '@shared/catalog/metrics';
import { NODE_CATALOG, NODE_TYPES } from '@shared/catalog/node-catalog';
import { OPTIMIZERS } from '@shared/catalog/optimizers';
import { SCHEDULERS } from '@shared/catalog/schedulers';
import { compileRecipe } from '@shared/compile/compile';
import { lossRows } from '@shared/compile/graph';
import { importToml } from '@shared/compile/import-toml';
import { emptyRecipe, type Recipe } from '@shared/recipe.model';
import { TEMPLATE_INFOS, buildTemplate } from '@shared/templates';
import * as commands from './commands';

/** Loss types the catalog leaves out on purpose, with the reason. Empty: every trainer loss is offered. */
const OMITTED_LOSSES: readonly string[] = [];

function expectSameSet(what: string, catalog: readonly string[], trainer: readonly string[]): void {
  const missing = trainer.filter((t) => !catalog.includes(t));
  const extra = catalog.filter((c) => !trainer.includes(c));
  const duplicates = catalog.filter((c, i) => catalog.indexOf(c) !== i);
  expect({ missing, extra, duplicates }, `${what}: "missing" = in the trainer, not in the catalog; "extra" = the reverse`).toEqual({ missing: [], extra: [], duplicates: [] });
}

const schema: Readonly<Record<string, Readonly<Record<string, unknown>>>> = surface.schema;

/** A constructor read by `ast`: literal defaults (null when `nonliteral`, `required` or None), `fixed` = bound by a preset. */
interface Signature {
  readonly params: Readonly<Record<string, unknown>>;
  readonly fixed?: readonly string[];
  readonly nonliteral?: readonly string[];
  readonly required?: readonly string[];
  readonly var_kwargs?: boolean;
}
/** A constructor outside the repository (torch.optim): not readable without torch, so not checked. */
type Signatures = Readonly<Record<string, Signature | { readonly external: string }>>;
const archSignatures: Readonly<Record<string, Signature>> = surface.arch_params;

/**
 * Catalog defaults that differ from a None constructor default on purpose: the catalog shows the
 * value the constructor resolves None to.
 */
const RESOLVED_DEFAULTS: readonly string[] = ['fdl_loss.vgg_weights', 'fdl_loss.dino_layers', 'fdl_loss.dino_weights', 'vgg_perceptual_loss.layer_weights'];

/** What a catalog entry's parameters (and window size) get wrong about the constructor. */
function signatureProblems(table: Signatures, type: string, params: readonly ParamInfo[], windowSize?: number): string[] {
  const sig = table[type];
  if (!sig) {
    return [`${type}: no constructor in the surface`];
  }
  if ('external' in sig) {
    return [];
  }
  const literal = (key: string) => key in sig.params && !sig.nonliteral?.includes(key) && !sig.required?.includes(key);
  const bad: string[] = [];
  for (const p of params) {
    if (!(p.key in sig.params)) {
      if (!sig.var_kwargs) {
        bad.push(`${type}.${p.key}: not a constructor parameter`);
      }
    } else {
      const fixed = sig.fixed?.includes(p.key) ?? false;
      if (fixed !== (p.fixed === true)) {
        bad.push(`${type}.${p.key}: ${fixed ? 'bound by the preset (a value raises TypeError) but not marked fixed' : 'marked fixed, but the constructor takes it'}`);
      }
      const resolved = sig.params[p.key] === null && RESOLVED_DEFAULTS.includes(`${type}.${p.key}`);
      if (literal(p.key) && !resolved && JSON.stringify(p.default) !== JSON.stringify(sig.params[p.key])) {
        bad.push(`${type}.${p.key}: default ${JSON.stringify(p.default)}, constructor ${JSON.stringify(sig.params[p.key])}`);
      }
    }
  }
  if (windowSize !== undefined) {
    const source = windowSource(type, sig);
    if (source === undefined) {
      bad.push(`${type}: windowSize ${windowSize} has no source in the constructor: add one to WINDOW_SOURCES`);
    } else if (source !== 'computed' && source !== windowSize) {
      bad.push(`${type}: windowSize ${windowSize}, constructor ${source}`);
    }
  }
  return bad;
}

const maxOf = (key: string) => (p: Readonly<Record<string, unknown>>) => (Array.isArray(p[key]) ? Math.max(...(p[key] as number[])) : undefined);
const firstOf = (key: string) => (p: Readonly<Record<string, unknown>>) => (Array.isArray(p[key]) ? (p[key] as number[])[0] : undefined);
/**
 * Where `windowSize` comes from when the constructor has no numeric `window_size` default:
 * another parameter, or 'computed' (a None default the constructor resolves itself).
 */
const WINDOW_SOURCES: Readonly<Record<string, 'computed' | ((p: Readonly<Record<string, unknown>>) => number | undefined)>> = {
  ...Object.fromEntries(['dat_s', 'dat_m', 'dat_2', 'rgt', 'rgt_s'].map((t) => [t, maxOf('split_size')])),
  ...Object.fromEntries(['hit_srf', 'hit_srf_medium', 'hit_srf_large'].map((t) => [t, firstOf('base_win_size')])),
  ...Object.fromEntries(['dct', 'grformer', 'grformer_medium', 'grformer_large'].map((t) => [t, 'computed' as const])),
};

function windowSource(type: string, sig: Signature): number | 'computed' | undefined {
  const own = sig.params['window_size'];
  if (typeof own === 'number' && !sig.nonliteral?.includes('window_size')) {
    return own;
  }
  const source = WINDOW_SOURCES[type];
  return source === 'computed' ? source : source?.(sig.params);
}
const opaque = surface.opaque.map((p) => new RegExp(`^${p.replace(/[.]/g, '\\.').replace(/\*/g, '.*')}$`));

/** Dotted TOML paths under schema-validated sections that the trainer does not know. */
function unknownKeys(table: Readonly<Record<string, unknown>>, section = ''): string[] {
  const bad: string[] = [];
  for (const [key, value] of Object.entries(table)) {
    const dotted = section ? `${section}.${key}` : key;
    if (opaque.some((re) => re.test(dotted))) {
      continue;
    }
    // [datasets.<phase>] names are free-form; their keys follow datasets.*
    const known = section === 'datasets' || key in (schema[section] ?? (section.startsWith('datasets.') ? schema['datasets.*'] : {}));
    if (!known) {
      bad.push(dotted);
    } else if (value && typeof value === 'object' && !Array.isArray(value)) {
      bad.push(...unknownKeys(value as Record<string, unknown>, dotted));
    }
  }
  return bad;
}

/** The catalog's TOML keys as a tree: a key that prefixes another becomes a table. */
function catalogKeyTree(): Record<string, unknown> {
  const root: Record<string, unknown> = {};
  for (const field of NODE_CATALOG.flatMap((n) => n.fields)) {
    if (!field.tomlKey) {
      continue;
    }
    const parts = field.tomlKey.split('.');
    let node = root;
    for (const part of parts.slice(0, -1)) {
      if (typeof node[part] !== 'object') {
        node[part] = {};
      }
      node = node[part] as Record<string, unknown>;
    }
    const last = parts[parts.length - 1];
    node[last] ??= true;
  }
  return root;
}

function blockers(recipe: Recipe): string[] {
  return compileRecipe(recipe).problems.filter((p) => p.severity === 'blocker').map((p) => p.message);
}

function node(recipe: Recipe, type: string) {
  return recipe.nodes.find((n) => n.type === type)!;
}

describe('catalog contract with the trainer (neosr-surface.json)', () => {
  it('offers exactly the trainer generators and discriminators', () => {
    expectSameSet('generators', ARCHS.map((a) => a.type), surface.generators);
    expectSameSet('discriminators', DISCRIMINATORS.map((d) => d.type), surface.discriminators);
  });

  it('offers exactly the trainer losses, optimizers, schedulers and metrics', () => {
    expectSameSet('losses', [...LOSSES.map((l) => l.type), ...OMITTED_LOSSES], surface.losses);
    expectSameSet('optimizers', OPTIMIZERS.map((o) => o.type), surface.optimizers);
    expectSameSet('schedulers', SCHEDULERS.map((s) => s.type), surface.schedulers);
    expectSameSet('metrics', METRIC_TYPES.map((m) => m.type), surface.metrics);
  });

  it('matches gan_type choices, loss schedule keys and metric directions and choices', () => {
    const ganType = LOSSES.find((l) => l.type === 'gan_loss')?.params.find((p) => p.key === 'gan_type');
    expectSameSet('gan_loss gan_type', ganType?.options ?? [], surface.gan_types);
    expectSameSet('loss schedule keys', LOSS_SCHEDULE_PARAMS.map((p) => p.key), surface.loss_schedule_keys);
    expectSameSet('metrics with better = lower', METRIC_TYPES.filter((m) => m.better === 'lower').map((m) => m.type), surface.metric_lower);
    const choices: Readonly<Record<string, Readonly<Record<string, readonly string[]>>>> = surface.metric_choices;
    for (const m of METRIC_TYPES) {
      const catalog = Object.fromEntries(m.params.filter((p) => p.options).map((p) => [p.key, [...(p.options ?? [])]]));
      expect(catalog, `${m.type} keyword choices`).toEqual(choices[m.type] ?? {});
    }
  });

  it('lists constructor parameters with their defaults, and window sizes, as the trainer has them', () => {
    const bad = [...ARCHS.flatMap((a) => signatureProblems(archSignatures, a.type, a.params, a.windowSize)), ...DISCRIMINATORS.flatMap((d) => signatureProblems(archSignatures, d.type, d.params))];
    expect(bad, 'catalog params vs the constructors (surface.arch_params)').toEqual([]);
    for (const info of TEMPLATE_INFOS) {
      for (const phase of compileRecipe(buildTemplate(info.id)).phases) {
        const doc = parse(phase.text);
        for (const section of ['network_g', 'network_d']) {
          const { type, ...kwargs } = (doc[section] ?? {}) as Record<string, unknown>;
          const sig = archSignatures[String(type)];
          const rejected = Object.keys(kwargs).filter((k) => sig?.fixed?.includes(k) || (sig && !(k in sig.params) && !sig.var_kwargs));
          expect(rejected, `${phase.fileName} [${section}]: keys the constructor rejects`).toEqual([]);
        }
      }
    }
  });

  it('lists loss, optimizer, scheduler and metric parameters with their defaults as the trainer has them', () => {
    const bad = [
      ...LOSSES.flatMap((l) => signatureProblems(surface.loss_params, l.type, l.params)),
      ...OPTIMIZERS.flatMap((o) => signatureProblems(surface.optimizer_params, o.type, o.params)),
      ...SCHEDULERS.flatMap((s) => signatureProblems(surface.scheduler_params, s.type, s.params)),
      ...METRIC_TYPES.flatMap((m) => signatureProblems(surface.metric_params, m.type, m.params)),
    ];
    expect(bad, 'catalog params vs the constructors (surface.*_params)').toEqual([]);
    const lossParams: Signatures = surface.loss_params;
    const stale = RESOLVED_DEFAULTS.filter((key) => {
      const [type, param] = key.split('.');
      const sig = lossParams[type];
      return !(sig && 'params' in sig && sig.params[param] === null);
    });
    expect(stale, 'RESOLVED_DEFAULTS entries whose constructor default is not None').toEqual([]);
    const metricParams: Signatures = surface.metric_params;
    for (const m of METRIC_TYPES) {
      const sig = metricParams[m.type];
      const keys = sig && 'params' in sig ? Object.keys(sig.params) : [];
      expectSameSet(`${m.type} parameters (the Validation node models each)`, m.params.map((p) => p.key), keys);
    }
  });

  it('uses only trainer scales', () => {
    const bad = ARCHS.filter((a) => a.scales.some((s) => !surface.scales.includes(s))).map((a) => a.type);
    expect(bad, 'archs with a scale the trainer rejects').toEqual([]);
  });

  it('writes only keys the trainer schema knows', () => {
    expect(unknownKeys(catalogKeyTree()), 'catalog tomlKeys missing from surface.schema').toEqual([]);
    for (const info of TEMPLATE_INFOS) {
      for (const phase of compileRecipe(buildTemplate(info.id)).phases) {
        expect(unknownKeys(parse(phase.text)), `${phase.fileName}: keys missing from surface.schema`).toEqual([]);
      }
    }
  });
});

describe('validator mirrors the trainer rules', () => {
  it('refuses a value for a key the preset binds, also from an imported or saved recipe', () => {
    const recipe = buildTemplate('2x-hat-s');
    const generator = node(recipe, NODE_TYPES.generator);
    expect(blockers(recipe)).toEqual([]);
    const edited = commands.setValues(generator.id, { params: { embed_dim: 144 } }).apply(recipe);
    expect(blockers(edited).some((m) => m.includes('"embed_dim"') && m.includes('preset sets it'))).toBe(true);
    const saved = JSON.parse(JSON.stringify(edited)) as Recipe;
    expect(blockers(saved).some((m) => m.includes('"embed_dim"'))).toBe(true);
    const toml = compileRecipe(edited).phases[0]!.text;
    const imported = importToml(toml, emptyRecipe('imported')).recipe;
    expect(blockers(imported).some((m) => m.includes('"embed_dim"'))).toBe(true);
    const open = commands.setValues(generator.id, { params: { mlp_ratio: 3 } }).apply(recipe);
    expect(blockers(open)).toEqual([]);
  });

  it('refuses nafnet at a scale other than 1', () => {
    const recipe = buildTemplate('1x-restore-span');
    const generator = node(recipe, NODE_TYPES.generator);
    const ok = commands.setValues(generator.id, { type: 'nafnet' }).apply(recipe);
    expect(blockers(ok)).toEqual([]);
    const x2 = commands.setValues(generator.id, { scale: '2' }).apply(ok);
    expect(blockers(x2).some((m) => m.includes('NAFNet does not support ×2'))).toBe(true);
  });

  it('needs the exact GT crop for an R3GAN scalar head', () => {
    const recipe = buildTemplate('4x-neosr-gan');
    const d = node(recipe, NODE_TYPES.discriminator);
    const scalar = commands.setValues(d.id, { type: 'r3gan_scalar128' }).apply(recipe);
    expect(blockers(scalar).some((m) => m.includes('exactly 128 px'))).toBe(true);
    const data = node(recipe, NODE_TYPES.trainDataset);
    expect(blockers(commands.setValues(data.id, { patch_size: 32 }).apply(scalar))).toEqual([]);
    expect(blockers(commands.setValues(d.id, { type: 'r3gan' }).apply(recipe))).toEqual([]);
  });

  it('refuses R1/R2 with dunet and a save_best that is not a metric', () => {
    const recipe = buildTemplate('4x-neosr-gan');
    const losses = node(recipe, NODE_TYPES.losses);
    const rows = lossRows(losses).map((r) => (r.type === 'gan_loss' ? { ...r, params: { ...r.params, r1_gamma: 1 } } : r));
    const penalised = commands.setValues(losses.id, { losses: rows }).apply(recipe);
    expect(blockers(penalised)).toEqual([]);
    const dunet = commands.setValues(node(recipe, NODE_TYPES.discriminator).id, { type: 'dunet' }).apply(penalised);
    expect(blockers(dunet).some((m) => m.includes('dunet'))).toBe(true);
    const val = node(recipe, NODE_TYPES.validation);
    expect(blockers(commands.setValues(val.id, { save_best: 'lpips' }).apply(recipe)).some((m) => m.includes('save_best'))).toBe(true);
    const best = commands.setValues(val.id, { metrics: ['psnr', 'lpips'], save_best: 'lpips', lpips_net: 'vgg' }).apply(recipe);
    expect(blockers(best)).toEqual([]);
    const text = compileRecipe(best).phases[0].text;
    expect(text).toContain('save_best = "lpips"');
    expect(text).toContain('[val.metrics.lpips]\ntype = "calculate_lpips"\nbetter = "lower"\nnet = "vgg"');
  });

  it('keeps scheduler nets to existing optimizers', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const withSchedule = commands.addNodeWired(NODE_TYPES.schedule, { x: 0, y: 0 }, { nets: ['g', 'd'] }).apply(recipe);
    expect(blockers(withSchedule).some((m) => m.includes('nets lists "d"'))).toBe(true);
    const gOnly = commands.setValues(node(withSchedule, NODE_TYPES.schedule).id, { nets: ['g'] }).apply(withSchedule);
    expect(blockers(gOnly)).toEqual([]);
    expect(compileRecipe(gOnly).phases[0].text).toContain('nets = [ "g" ]');
  });

  it('imports the new keys into fields and writes them back', () => {
    const toml = `
name = "t"
model_type = "otf"
scale = 4
deterministic = true
detect_anomaly = true
[datasets.train]
type = "otf"
dataroot_gt = "/gt"
patch_size = 64
batch_size = 8
aug_multi_prob = 0.5
aug_antialias = true
tight_canvas = true
[datasets.val]
name = "val"
type = "paired"
dataroot_gt = "/v/gt"
dataroot_lq = "/v/lq"
[val]
val_freq = 1000
save_best = "topiq"
[val.metrics.topiq]
type = "calculate_topiq"
dataset = "pipal"
[network_g]
type = "realplksr"
layer_norm = true
[train]
clamp = true
grad_clip = 0.5
max_skip_streak = 5
[train.optim_g]
type = "adamw"
lr = 2e-4
[train.scheduler]
type = "cosineannealing"
T_max = 20000
nets = ["g"]
[train.pixel_opt]
type = "psnr_loss"
loss_weight = 1.0
start_iter = 100
ramp_iters = 50
[logger]
total_iter = 20000
save_checkpoint_freq = 500
keep_exports = 2
`;
    const result = importToml(toml, emptyRecipe('t'));
    expect(result.preserved).toEqual([]);
    expect(node(result.recipe, NODE_TYPES.plan).values).toMatchObject({ clamp: 'true', grad_clip: 'norm', grad_clip_norm: 0.5, keep_exports: 2 });
    expect(blockers(result.recipe)).toEqual([]);
    const text = compileRecipe(result.recipe).phases[0].text;
    for (const line of ['deterministic = true', 'detect_anomaly = true', 'aug_multi_prob = 0.5', 'aug_antialias = true', 'tight_canvas = true', 'save_best = "topiq"', 'dataset = "pipal"', 'layer_norm = true', 'clamp = true', 'grad_clip = 0.5', 'max_skip_streak = 5', 'nets = [ "g" ]', 'start_iter = 100', 'keep_exports = 2']) {
      expect(text, line).toContain(line);
    }
  });
});
