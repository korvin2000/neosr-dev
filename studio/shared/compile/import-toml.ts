/**
 * NeoSR TOML → Recipe. Known keys become node values; everything else is preserved in
 * `nativeExtras` at its original TOML path and re-emitted by the compiler.
 */
import { parse } from 'smol-toml';
import { LOSSES } from '../catalog/losses';
import { METRIC_TYPES } from '../catalog/metrics';
import { NODE_TYPES, nodeSpec } from '../catalog/node-catalog';
import type { LossRow, Recipe } from '../recipe.model';
import { newId } from '../recipe.model';
import { applyDraft, type RecipeDraft } from '../recipe-builder';
import type { ValDatasetRow } from './graph';

export interface ImportResult {
  readonly recipe: Recipe;
  readonly warnings: readonly string[];
  /** TOML paths that were preserved as extras. */
  readonly preserved: readonly string[];
}

type Table = Record<string, unknown>;

function isTable(v: unknown): v is Table {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Take a scalar out of a table (deleting it) so what remains is "unknown". */
function take<T = unknown>(table: Table | undefined, key: string): T | undefined {
  if (!table || !(key in table)) {
    return undefined;
  }
  const v = table[key] as T;
  delete table[key];
  return v;
}

export function importToml(text: string, base: Recipe, name?: string): ImportResult {
  const warnings: string[] = [];
  const doc = parse(text) as Table;
  const draft: { -readonly [K in keyof RecipeDraft]: RecipeDraft[K] } = {};

  // --- root ---------------------------------------------------------------------
  const runName = take<string>(doc, 'name') ?? name ?? 'imported';
  const modelType = take<string>(doc, 'model_type');
  const scale = take<number>(doc, 'scale');
  const useAmp = take<boolean>(doc, 'use_amp');
  const bf16 = take<boolean>(doc, 'bfloat16');
  draft.runtime = {
    precision: bf16 ? 'bf16' : useAmp ? 'amp' : 'fp32',
    fast_matmul: take<boolean>(doc, 'fast_matmul') ?? false,
    compile: take<boolean>(doc, 'compile') ?? false,
    compile_mode: take<string>(doc, 'compile_mode') ?? 'default',
    manual_seed: take<number>(doc, 'manual_seed'),
    num_gpu: typeof doc['num_gpu'] === 'number' ? take<number>(doc, 'num_gpu') : (take(doc, 'num_gpu'), undefined),
  };

  // --- datasets -------------------------------------------------------------------
  const datasets = isTable(doc['datasets']) ? (doc['datasets'] as Table) : undefined;
  const train = datasets && isTable(datasets['train']) ? (datasets['train'] as Table) : undefined;
  const degradations = isTable(doc['degradations']) ? (doc['degradations'] as Table) : undefined;
  if (train) {
    const spec = nodeSpec(NODE_TYPES.trainDataset);
    const values: Record<string, unknown> = {};
    for (const f of spec.fields) {
      const key = f.tomlKey.replace(/^datasets\.train\./, '');
      if (f.tomlKey.startsWith('datasets.train.') && key in train) {
        values[f.key] = take(train, key);
      }
    }
    if (modelType === 'otf' || values['type'] === 'otf') {
      values['type'] = 'otf';
    }
    // Absent keys mean NeoSR's own defaults, not the Studio template's: no augmentation.
    if (!('augmentation' in values)) {
      values['augmentation'] = [];
      values['aug_prob'] = [];
    }
    draft.dataset = values;
    if (values['type'] === 'otf') {
      const spec2 = nodeSpec(NODE_TYPES.degradation);
      const dv: Record<string, unknown> = {};
      for (const f of spec2.fields) {
        const key = f.key;
        if (degradations && key in degradations) {
          dv[key] = take(degradations, key);
        } else if (key in train) {
          dv[key] = take(train, key);
        }
      }
      draft.degradation = dv;
    }
    if (Object.keys(train).length === 0) {
      delete datasets!['train'];
    }
  } else {
    warnings.push('No [datasets.train] table: the Train Dataset node was left empty.');
  }
  if (degradations && Object.keys(degradations).length === 0) {
    delete doc['degradations'];
  }

  // Validation sets: val, val_1, val_2 …
  const valRows: ValDatasetRow[] = [];
  if (datasets) {
    for (const key of Object.keys(datasets).filter((k) => /^val(_\d+)?$/.test(k)).sort()) {
      const t = datasets[key] as Table;
      if (!isTable(t)) {
        continue;
      }
      valRows.push({
        id: newId('v'),
        name: take<string>(t, 'name') ?? key,
        type: take<string>(t, 'type') === 'single' ? 'single' : 'paired',
        dataroot_gt: take<string>(t, 'dataroot_gt') ?? '',
        dataroot_lq: take<string>(t, 'dataroot_lq') ?? '',
        section: `datasets.${key}`,
      });
      if (Object.keys(t).length === 0) {
        delete datasets[key];
      }
    }
    for (const key of Object.keys(datasets)) {
      if (/^test(_\d+)?$/.test(key)) {
        warnings.push(`[datasets.${key}] belongs to an inference config and was preserved as an extra.`);
      }
    }
    if (Object.keys(datasets).length === 0) {
      delete doc['datasets'];
    }
  }
  const val = isTable(doc['val']) ? (doc['val'] as Table) : undefined;
  if (valRows.length || val) {
    const spec = nodeSpec(NODE_TYPES.validation);
    const values: Record<string, unknown> = { datasets: valRows };
    for (const f of spec.fields) {
      const key = f.tomlKey.replace(/^val\./, '');
      if (f.tomlKey.startsWith('val.') && f.tomlKey !== 'val.metrics' && val && key in val) {
        values[f.key] = take(val, key);
      }
    }
    const metricsTable = val && isTable(val['metrics']) ? (val['metrics'] as Table) : undefined;
    if (metricsTable) {
      const names: string[] = [];
      for (const [mName, mv] of Object.entries(metricsTable)) {
        if (!isTable(mv)) {
          continue;
        }
        const type = String(mv['type']);
        const info = METRIC_TYPES.find((m) => m.type === type);
        const better = mv['better'];
        const betterMatches = better === undefined || better === info?.better;
        const extraKeys = Object.keys(mv).filter((k) => k !== 'type' && k !== 'better');
        if (info && info.name === mName && betterMatches && extraKeys.length === 0) {
          // Exactly what the Validation node would write: model it.
          names.push(info.name);
          delete metricsTable[mName];
        } else if (info && info.name === mName && betterMatches) {
          // Known metric with extra parameters: model the name, keep the parameters as extras.
          names.push(info.name);
          delete mv['type'];
          delete mv['better'];
          warnings.push(`Metric "${mName}" keeps its parameters (${extraKeys.join(', ')}) as preserved extras.`);
        } else {
          warnings.push(`Metric "${mName}" (${type}) is not one the Validation node models; its table is preserved verbatim.`);
        }
      }
      values['metrics'] = names;
      if (Object.keys(metricsTable).length === 0) {
        delete val!['metrics'];
      }
    } else {
      values['metrics'] = [];
    }
    draft.validation = values;
    if (val && Object.keys(val).length === 0) {
      delete doc['val'];
    }
  }

  // --- networks ---------------------------------------------------------------------
  const netG = isTable(doc['network_g']) ? (doc['network_g'] as Table) : undefined;
  if (netG) {
    const type = take<string>(netG, 'type');
    draft.generator = { type, scale: scale !== undefined ? String(scale) : undefined, params: { ...netG } };
    delete doc['network_g'];
  } else {
    warnings.push('No [network_g] table.');
  }
  const netD = isTable(doc['network_d']) ? (doc['network_d'] as Table) : undefined;
  if (netD) {
    const type = take<string>(netD, 'type');
    draft.discriminator = { type, params: { ...netD } };
    delete doc['network_d'];
  }

  // --- path -----------------------------------------------------------------------
  const path = isTable(doc['path']) ? (doc['path'] as Table) : undefined;
  if (path) {
    const resume = take<string>(path, 'resume_state');
    const pg = take<string>(path, 'pretrain_network_g');
    draft.init = {
      mode: resume ? 'resume' : pg ? 'pretrain' : 'scratch',
      resume_state: resume ?? '',
      pretrain_network_g: pg ?? '',
      pretrain_network_d: take<string>(path, 'pretrain_network_d') ?? '',
      strict_load_g: take<boolean>(path, 'strict_load_g') ?? true,
      param_key_g: take<string>(path, 'param_key_g') ?? '',
    };
    if (Object.keys(path).length === 0) {
      delete doc['path'];
    }
  } else {
    draft.init = { mode: 'scratch' };
  }

  // --- train ------------------------------------------------------------------------
  const trainT = isTable(doc['train']) ? (doc['train'] as Table) : undefined;
  // NeoSR's reader default for EMA is off; the template default (0.999) must not appear by itself.
  const planValues: Record<string, unknown> = { name: runName, ema: 0 };
  if (trainT) {
    const planSpec = nodeSpec(NODE_TYPES.plan);
    for (const f of planSpec.fields) {
      if (f.tomlKey.startsWith('train.')) {
        const key = f.tomlKey.slice('train.'.length);
        if (key in trainT) {
          planValues[f.key] = take(trainT, key);
        }
      }
    }
    // Legacy wavelet keys map onto the gan row.
    const waveletGuided = take<boolean>(trainT, 'wavelet_guided');
    const waveletInit = take<number>(trainT, 'wavelet_init');
    const matchLq = take<boolean>(trainT, 'match_lq_colors');

    for (const side of ['optim_g', 'optim_d'] as const) {
      const t = isTable(trainT[side]) ? (trainT[side] as Table) : undefined;
      if (!t) {
        continue;
      }
      const spec = nodeSpec(side === 'optim_g' ? NODE_TYPES.optimizerG : NODE_TYPES.optimizerD);
      const values: Record<string, unknown> = {};
      for (const f of spec.fields) {
        const key = f.tomlKey.replace(`train.${side}.`, '');
        if (f.key !== 'extra' && key in t) {
          values[f.key] = take(t, key);
        }
      }
      // `schedule_free = true` is the catalog default for the SF optimizers; only an explicit false is information.
      const sf = take<boolean>(t, 'schedule_free');
      if (sf === false) {
        t['schedule_free'] = false;
      }
      if (!('warmup_steps' in values)) {
        values['warmup_steps'] = 0;
      }
      if (Object.keys(t).length) {
        values['extra'] = { ...t };
      }
      delete trainT[side];
      if (side === 'optim_g') {
        draft.optimG = values;
      } else {
        draft.optimD = values;
      }
    }
    const sched = isTable(trainT['scheduler']) ? (trainT['scheduler'] as Table) : undefined;
    if (sched) {
      const spec = nodeSpec(NODE_TYPES.schedule);
      const values: Record<string, unknown> = {};
      for (const f of spec.fields) {
        const key = f.tomlKey.replace('train.scheduler.', '');
        if (key in sched) {
          values[f.key] = take(sched, key);
        }
      }
      draft.schedule = values;
      if (Object.keys(sched).length === 0) {
        delete trainT['scheduler'];
      }
    }
    const rows: LossRow[] = [];
    for (const key of Object.keys(trainT).filter((k) => k.endsWith('_opt'))) {
      const t = trainT[key] as Table;
      if (!isTable(t)) {
        continue;
      }
      const slot = key.slice(0, -'_opt'.length);
      const type = String(take(t, 'type') ?? '');
      const weight = Number(take(t, 'loss_weight') ?? take(t, 'perceptual_weight') ?? LOSSES.find((l) => l.type === type)?.defaultWeight ?? 1);
      const params: Record<string, unknown> = { ...t };
      const info = LOSSES.find((l) => l.type === type);
      if (!info) {
        warnings.push(`Loss "${type}" in [train.${key}] is unknown; kept as a row but NeoSR may reject it.`);
      }
      if (info && info.slot !== slot) {
        params['slot'] = slot;
      }
      if (type === 'gan_loss' && waveletGuided && waveletInit !== undefined && !('wavelet_init' in params)) {
        params['wavelet_init'] = waveletInit;
      }
      if ((slot === 'consistency' || slot === 'msswd') && matchLq && !('target' in params)) {
        params['target'] = 'lq';
      }
      rows.push({ id: newId('l'), type, weight, enabled: weight > 0, params, schedule: {} });
      delete trainT[key];
    }
    draft.losses = rows;
    if (Object.keys(trainT).length === 0) {
      delete doc['train'];
    }
  }

  // --- logger -----------------------------------------------------------------------
  const logger = isTable(doc['logger']) ? (doc['logger'] as Table) : undefined;
  let totalIter = 1_000_000;
  if (logger) {
    const planSpec = nodeSpec(NODE_TYPES.plan);
    totalIter = take<number>(logger, 'total_iter') ?? totalIter;
    for (const f of planSpec.fields) {
      if (f.tomlKey.startsWith('logger.') && !f.tomlKey.includes('wandb')) {
        const key = f.tomlKey.slice('logger.'.length);
        if (key in logger) {
          planValues[f.key] = take(logger, key);
        }
      }
    }
    const wandb = isTable(logger['wandb']) ? (logger['wandb'] as Table) : undefined;
    if (wandb && typeof wandb['project'] === 'string') {
      planValues['wandb_project'] = take(wandb, 'project');
      if (Object.keys(wandb).length === 0) {
        delete logger['wandb'];
      }
    }
    if (Object.keys(logger).length === 0) {
      delete doc['logger'];
    }
  } else {
    warnings.push('No [logger] table: save_checkpoint_freq defaults to 1000.');
  }
  draft.plan = planValues;
  draft.phases = [{ id: newId('p'), name: 'Train', iters: totalIter, init: 'scratch', note: 'imported' }];

  // --- what is left is preserved ----------------------------------------------------------
  const preserved: string[] = [];
  const collect = (obj: Table, path: string[]) => {
    for (const [k, v] of Object.entries(obj)) {
      if (isTable(v)) {
        collect(v, [...path, k]);
      } else {
        preserved.push([...path, k].join('.'));
      }
    }
  };
  collect(doc, []);

  const recipe: Recipe = {
    ...applyDraft({ ...base, name: runName }, draft),
    nativeExtras: doc,
  };
  return { recipe, warnings, preserved };
}
