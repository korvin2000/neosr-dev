/**
 * Recipe → one NeoSR TOML per phase, plus the launch commands.
 *
 * Phases are sequential `neosr-train` runs. A `finetune` boundary starts a new run from the previous
 * phase's last saved weights (native counters restart); a `continue` boundary resumes the previous
 * phase's last `.state` (native counters are cumulative). Every TOML is complete on its own.
 */
import { ARCH_BY_TYPE } from '../catalog/archs';
import { LOSSES } from '../catalog/losses';
import { METRIC_TYPES } from '../catalog/metrics';
import { NODE_TYPES, nodeSpec } from '../catalog/node-catalog';
import { OPTIMIZERS } from '../catalog/optimizers';
import { fieldVisible, type NodeSpec } from '../catalog/types';
import { phaseRanges, type Phase, type Recipe, type RecipeNode } from '../recipe.model';
import {
  activeLossRows,
  lossRows,
  lossSlot,
  nodeValues,
  phaseHasGan,
  phaseNodeValues,
  planContext,
  scaleOf,
  slug,
  stateIter,
  valDatasets,
  valSection,
  type PlanContext,
} from './graph';
import type { Problem } from './problems';
import { annotate, tomlText, writeToml, type TomlDoc, type TomlLine, type TomlOrigin, type TomlSection } from './toml-writer';
import { validateRecipe } from './validate';

export interface CompileOptions {
  /** Affects the `compile` check and the shape of launch commands. */
  readonly platform?: 'windows' | 'linux';
}

export interface CompiledPhase {
  readonly phase: Phase;
  readonly index: number;
  /** NeoSR run name (`name` key). */
  readonly runName: string;
  readonly fileName: string;
  /** Curriculum clock. */
  readonly start: number;
  readonly end: number;
  /** Native clock: the update counter NeoSR will show. */
  readonly nativeStart: number;
  readonly nativeEnd: number;
  readonly hasGan: boolean;
  readonly lines: readonly TomlLine[];
  readonly text: string;
  readonly command: string;
}

export interface CompileResult {
  readonly phases: readonly CompiledPhase[];
  readonly commands: readonly string[];
  readonly problems: readonly Problem[];
  readonly context: PlanContext;
}

export function compileRecipe(recipe: Recipe, options: CompileOptions = {}): CompileResult {
  const context = planContext(recipe);
  const problems = validateRecipe(recipe, context, options);
  if (!context.plan) {
    return { phases: [], commands: [], problems, context };
  }
  const ranges = phaseRanges(recipe.phases);
  const phases: CompiledPhase[] = [];
  // Resuming a .state restores NeoSR's update counter, so the first phase's native clock starts there.
  const initialNative = initialNativeStart(context);
  ranges.forEach(({ phase, start, end }, index) => {
    const previous = phases[index - 1];
    const continues = index > 0 && phase.init === 'continue';
    const nativeStart = index === 0 ? initialNative : continues && previous ? previous.nativeEnd : 0;
    const nativeEnd = nativeStart + phase.iters;
    const runName = recipe.phases.length === 1 ? planName(context.plan!) : `${planName(context.plan!)}_p${index + 1}_${slug(phase.name)}`;
    const doc = buildDoc(recipe, context, phase, index, runName, nativeEnd, previous, options);
    const lines = writeToml(doc);
    const text = tomlText(lines);
    const fileName = `${runName}.toml`;
    const command = launchCommand(context, fileName, options);
    phases.push({
      phase,
      index,
      runName,
      fileName,
      start,
      end,
      nativeStart,
      nativeEnd,
      hasGan: phaseHasGan(lossRows(context.losses), phase.id),
      lines,
      text,
      command,
    });
  });
  return { phases, commands: phases.map((p) => p.command), problems, context };
}

function initialNativeStart(context: PlanContext): number {
  if (!context.init) {
    return 0;
  }
  const v = nodeValues(context.init);
  if (v['mode'] !== 'resume') {
    return 0;
  }
  return stateIter(String(v['resume_state'] ?? '')) ?? 0;
}

/** NeoSR's experiments root, honouring an imported `path.experiments_root`; always forward slashes. */
function experimentsRoot(recipe: Recipe): string {
  const path = recipe.nativeExtras['path'];
  const root = path && typeof path === 'object' ? (path as Record<string, unknown>)['experiments_root'] : undefined;
  if (typeof root !== 'string' || !root.trim()) {
    return 'experiments';
  }
  return root.trim().replace(/[\\/]+$/, '').split('\\').join('/');
}

function planName(plan: RecipeNode): string {
  const name = String(nodeValues(plan)['name'] ?? '').trim();
  return name || 'neosr_run';
}

function launchCommand(context: PlanContext, fileName: string, options: CompileOptions): string {
  const gpus = context.runtime ? Number(nodeValues(context.runtime)['num_gpu']) : NaN;
  const path = options.platform === 'windows' ? `options\\${fileName}` : `options/${fileName}`;
  if (Number.isFinite(gpus) && gpus > 1) {
    return `torchrun --nproc_per_node=${gpus} train.py --launcher pytorch -opt ${path}`;
  }
  return `neosr-train ${path}`;
}

function origin(node: RecipeNode | undefined, fieldKey?: string): TomlOrigin | undefined {
  return node ? { nodeId: node.id, fieldKey } : undefined;
}

/** Keys NeoSR declares as floats; printed with a decimal point so `1` reads as `1.0`. */
const FLOAT_KEYS = new Set(['loss_weight', 'lr', 'weight_decay', 'ema', 'ema_power', 'gamma', 'eta_min', 'eps', 'aug_prob']);

function isFloatKey(key: string, sectionPath: string): boolean {
  return FLOAT_KEYS.has(key) || sectionPath === 'degradations' && (key.includes('prob') || key.includes('range') || key.includes('sigma'));
}

/**
 * Emit every visible field of a node that has a TOML key under `sectionPath`, honouring phase
 * overrides. Native-level fields are only written when they differ from NeoSR's default, which
 * keeps the output as short as the shipped templates.
 */
function sectionEntries(node: RecipeNode, spec: NodeSpec, sectionPath: string, phaseId: string): Record<string, unknown> {
  const values = phaseNodeValues(node, phaseId, spec);
  const entries: Record<string, unknown> = {};
  for (const field of spec.fields) {
    if (!field.tomlKey || field.widget.kind === 'readout' || field.widget.kind === 'custom' || field.widget.kind === 'multiselect' && field.key === 'metrics') {
      continue;
    }
    if (!fieldVisible(field, values)) {
      continue;
    }
    // A key/value table addressed at the section itself (optimizer extra kwargs) spreads into it.
    if (field.widget.kind === 'keyvalue' && field.tomlKey === sectionPath) {
      const table = values[field.key];
      if (table && typeof table === 'object') {
        for (const [k, v] of Object.entries(table as Record<string, unknown>)) {
          entries[k] = annotate(v, origin(node, field.key), isFloatKey(k, sectionPath));
        }
      }
      continue;
    }
    if (field.level === 'native' && JSON.stringify(values[field.key]) === JSON.stringify(field.default)) {
      continue;
    }
    const prefix = `${sectionPath}.`;
    if (!field.tomlKey.startsWith(prefix)) {
      continue;
    }
    const key = field.tomlKey.slice(prefix.length);
    if (key.includes('.')) {
      continue;
    }
    const value = values[field.key];
    if (field.widget.kind === 'keyvalue' && value && typeof value === 'object') {
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        entries[k] = annotate(v, origin(node, field.key));
      }
      continue;
    }
    entries[key] = annotate(value, origin(node, field.key), isFloatKey(key, sectionPath));
  }
  return entries;
}

function buildDoc(
  recipe: Recipe,
  ctx: PlanContext,
  phase: Phase,
  index: number,
  runName: string,
  nativeEnd: number,
  previous: CompiledPhase | undefined,
  options: CompileOptions,
): TomlDoc {
  const plan = ctx.plan as RecipeNode;
  const planSpec = nodeSpec(NODE_TYPES.plan);
  const rows = lossRows(ctx.losses);
  const hasGan = phaseHasGan(rows, phase.id) && !!ctx.dmodel && !!ctx.optimD;
  const dataValues = ctx.data ? phaseNodeValues(ctx.data, phase.id) : {};
  const isOtf = dataValues['type'] === 'otf';
  const scale = scaleOf(ctx.model);

  // --- root --------------------------------------------------------------
  const root: Record<string, unknown> = {
    name: annotate(runName, origin(plan, 'name')),
    model_type: annotate(isOtf ? 'otf' : 'image', origin(ctx.data, 'type')),
    scale: annotate(scale, origin(ctx.model, 'scale')),
  };
  if (ctx.runtime) {
    const rt = nodeValues(ctx.runtime);
    const precision = rt['precision'];
    root['use_amp'] = annotate(precision === 'amp' || precision === 'bf16', origin(ctx.runtime, 'precision'));
    root['bfloat16'] = annotate(precision === 'bf16' ? true : undefined, origin(ctx.runtime, 'precision'));
    root['fast_matmul'] = annotate(rt['fast_matmul'] === true ? true : undefined, origin(ctx.runtime, 'fast_matmul'));
    root['compile'] = annotate(rt['compile'] === true ? true : undefined, origin(ctx.runtime, 'compile'));
    root['compile_mode'] = annotate(rt['compile'] === true && rt['compile_mode'] !== 'default' ? rt['compile_mode'] : undefined, origin(ctx.runtime, 'compile_mode'));
    root['manual_seed'] = annotate(numberOrUndefined(rt['manual_seed']), origin(ctx.runtime, 'manual_seed'));
    root['num_gpu'] = annotate(numberOrUndefined(rt['num_gpu']), origin(ctx.runtime, 'num_gpu'));
  }

  const sections: TomlSection[] = [];

  // --- datasets.train ----------------------------------------------------
  if (ctx.data) {
    const spec = nodeSpec(NODE_TYPES.trainDataset);
    const entries = sectionEntries(ctx.data, spec, 'datasets.train', phase.id);
    if (isOtf) {
      delete entries['dataroot_lq'];
    }
    if (Array.isArray(dataValues['augmentation']) && (dataValues['augmentation'] as unknown[]).length === 0) {
      delete entries['augmentation'];
      delete entries['aug_prob'];
    }
    sections.push({ path: 'datasets.train', origin: origin(ctx.data), entries });
  }
  if (isOtf && ctx.degradation) {
    const spec = nodeSpec(NODE_TYPES.degradation);
    sections.push({ path: 'degradations', comment: 'OTF degradation pipeline', origin: origin(ctx.degradation), entries: sectionEntries(ctx.degradation, spec, 'degradations', phase.id) });
  }

  // --- datasets.val* and [val] ------------------------------------------
  if (ctx.val) {
    const spec = nodeSpec(NODE_TYPES.validation);
    valDatasets(ctx.val).forEach((row, i) => {
      const path = valSection(row, i);
      sections.push({
        path,
        origin: origin(ctx.val, 'datasets'),
        entries: {
          name: annotate(row.name || (i === 0 ? 'val' : `val_${i}`), origin(ctx.val, 'datasets')),
          type: annotate(row.type, origin(ctx.val, 'datasets')),
          dataroot_gt: annotate(row.type === 'paired' ? row.dataroot_gt : undefined, origin(ctx.val, 'datasets')),
          dataroot_lq: annotate(row.dataroot_lq, origin(ctx.val, 'datasets')),
        },
      });
    });
    const valEntries = sectionEntries(ctx.val, spec, 'val', phase.id);
    sections.push({ path: 'val', origin: origin(ctx.val), entries: valEntries });
    const metrics = nodeValues(ctx.val)['metrics'];
    if (Array.isArray(metrics)) {
      for (const name of metrics as string[]) {
        const info = METRIC_TYPES.find((m) => m.name === name);
        if (!info) {
          continue;
        }
        sections.push({
          path: `val.metrics.${name}`,
          origin: origin(ctx.val, 'metrics'),
          entries: {
            type: annotate(info.type, origin(ctx.val, 'metrics')),
            better: annotate(info.better === 'lower' ? 'lower' : undefined, origin(ctx.val, 'metrics')),
          },
        });
      }
    }
  }

  // --- path --------------------------------------------------------------
  sections.push({ path: 'path', origin: origin(ctx.init), entries: pathEntries(ctx, phase, index, previous, hasGan, experimentsRoot(recipe)) });

  // --- networks ----------------------------------------------------------
  if (ctx.model) {
    const mv = nodeValues(ctx.model);
    const entries: Record<string, unknown> = { type: annotate(mv['type'], origin(ctx.model, 'type')) };
    for (const [k, v] of Object.entries((mv['params'] as Record<string, unknown>) ?? {})) {
      entries[k] = annotate(v, origin(ctx.model, 'params'));
    }
    sections.push({ path: 'network_g', origin: origin(ctx.model), entries });
  }
  if (hasGan && ctx.dmodel) {
    const dv = nodeValues(ctx.dmodel);
    const entries: Record<string, unknown> = { type: annotate(dv['type'], origin(ctx.dmodel, 'type')) };
    for (const [k, v] of Object.entries((dv['params'] as Record<string, unknown>) ?? {})) {
      entries[k] = annotate(v, origin(ctx.dmodel, 'params'));
    }
    sections.push({ path: 'network_d', origin: origin(ctx.dmodel), entries });
  }

  // --- train -------------------------------------------------------------
  sections.push({ path: 'train', origin: origin(plan), entries: sectionEntries(plan, planSpec, 'train', phase.id) });
  if (ctx.optimG) {
    sections.push({ path: 'train.optim_g', origin: origin(ctx.optimG), entries: optimizerEntries(ctx.optimG, NODE_TYPES.optimizerG, 'train.optim_g', phase) });
  }
  if (hasGan && ctx.optimD) {
    sections.push({ path: 'train.optim_d', origin: origin(ctx.optimD), entries: optimizerEntries(ctx.optimD, NODE_TYPES.optimizerD, 'train.optim_d', phase) });
  }
  if (ctx.schedule) {
    sections.push({ path: 'train.scheduler', origin: origin(ctx.schedule), entries: sectionEntries(ctx.schedule, nodeSpec(NODE_TYPES.schedule), 'train.scheduler', phase.id) });
  }

  // --- losses ------------------------------------------------------------
  let first = true;
  for (const { row, weight } of activeLossRows(rows, phase.id)) {
    if (row.type === 'gan_loss' && !hasGan) {
      continue;
    }
    const info = LOSSES.find((l) => l.type === row.type);
    const slotName = lossSlot(row, info?.slot ?? slug(row.type));
    const entries: Record<string, unknown> = {
      type: annotate(row.type, origin(ctx.losses, 'losses')),
      loss_weight: annotate(weight, origin(ctx.losses, 'losses'), true),
    };
    for (const [k, v] of Object.entries(row.params)) {
      if (k === 'slot') {
        continue;
      }
      entries[k] = annotate(v, origin(ctx.losses, 'losses'));
    }
    sections.push({ path: `train.${slotName}_opt`, comment: first ? 'losses' : undefined, origin: origin(ctx.losses, 'losses'), entries });
    first = false;
  }

  // --- logger ------------------------------------------------------------
  const loggerEntries = sectionEntries(plan, planSpec, 'logger', phase.id);
  loggerEntries['total_iter'] = annotate(nativeEnd, origin(plan, 'total_iter'));
  const wandb = nodeValues(plan)['wandb_project'];
  sections.push({ path: 'logger', origin: origin(plan), entries: orderKeys(loggerEntries, ['total_iter', 'save_checkpoint_freq', 'keep_states', 'print_freq', 'use_tb_logger', 'save_tb_img']) });
  if (typeof wandb === 'string' && wandb.trim()) {
    sections.push({ path: 'logger.wandb', origin: origin(plan, 'wandb_project'), entries: { project: annotate(wandb.trim(), origin(plan, 'wandb_project')) } });
  }

  // --- native extras -----------------------------------------------------
  mergeExtras(root, sections, recipe.nativeExtras);
  void options;
  return { root, sections };
}

function optimizerEntries(node: RecipeNode, type: string, sectionPath: string, phase: Phase): Record<string, unknown> {
  const spec = nodeSpec(type);
  const entries = sectionEntries(node, spec, sectionPath, phase.id);
  const values = phaseNodeValues(node, phase.id, spec);
  const info = OPTIMIZERS.find((o) => o.type === values['type']);
  if (info && !info.scheduleFree) {
    delete entries['warmup_steps'];
  }
  // Adam-family optimizers take two betas; trim a three-value list left over from Adan.
  const betas = values['betas'];
  if (info && !info.type.startsWith('adan') && Array.isArray(betas) && betas.length > 2) {
    entries['betas'] = annotate(betas.slice(0, 2), { nodeId: node.id, fieldKey: 'betas' });
  }
  return orderKeys(entries, ['type', 'lr', 'betas', 'weight_decay', 'warmup_steps', 'eps']);
}

function pathEntries(ctx: PlanContext, phase: Phase, index: number, previous: CompiledPhase | undefined, hasGan: boolean, root: string): Record<string, unknown> {
  const entries: Record<string, unknown> = {};
  if (index === 0 || !previous) {
    if (!ctx.init) {
      return entries;
    }
    const v = nodeValues(ctx.init);
    const o = (key: string) => ({ nodeId: ctx.init!.id, fieldKey: key });
    if (v['mode'] === 'pretrain') {
      entries['pretrain_network_g'] = annotate(v['pretrain_network_g'], o('pretrain_network_g'));
      entries['pretrain_network_d'] = annotate(hasGan ? v['pretrain_network_d'] : undefined, o('pretrain_network_d'));
      entries['strict_load_g'] = annotate(v['strict_load_g'] === false ? false : undefined, o('strict_load_g'));
      entries['param_key_g'] = annotate(v['param_key_g'], o('param_key_g'));
    } else if (v['mode'] === 'resume') {
      entries['resume_state'] = annotate(v['resume_state'], o('resume_state'));
    }
    return entries;
  }
  const o = ctx.plan ? { nodeId: ctx.plan.id, fieldKey: 'phases' } : undefined;
  if (phase.init === 'continue') {
    entries['resume_state'] = annotate(`${root}/${previous.runName}/training_states/${previous.nativeEnd}.state`, o);
  } else {
    entries['pretrain_network_g'] = annotate(`${root}/${previous.runName}/models/net_g_latest.pth`, o);
    if (hasGan && previous.hasGan) {
      entries['pretrain_network_d'] = annotate(`${root}/${previous.runName}/models/net_d_latest.pth`, o);
    }
  }
  return entries;
}

function numberOrUndefined(value: unknown): number | undefined {
  if (value === '' || value === null || value === undefined) {
    return undefined;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

function orderKeys(entries: Record<string, unknown>, order: readonly string[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of order) {
    if (key in entries) {
      out[key] = entries[key];
    }
  }
  for (const [key, value] of Object.entries(entries)) {
    if (!(key in out)) {
      out[key] = value;
    }
  }
  return out;
}

/**
 * Native extras are stored as a nested object mirroring the TOML tree. Leaves land in the matching
 * section (created if needed) unless the UI already owns that key: the validator reports those.
 */
function mergeExtras(root: Record<string, unknown>, sections: TomlSection[], extras: Readonly<Record<string, unknown>>): void {
  const walk = (obj: Record<string, unknown>, path: string[]) => {
    for (const [key, value] of Object.entries(obj)) {
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        walk(value as Record<string, unknown>, [...path, key]);
        continue;
      }
      if (path.length === 0) {
        if (!(key in root)) {
          root[key] = value;
        }
        continue;
      }
      const sectionPath = path.join('.');
      const existing = sections.find((s) => s.path === sectionPath);
      if (existing) {
        if (!(key in existing.entries)) {
          (existing.entries as Record<string, unknown>)[key] = value;
        }
      } else {
        sections.push({ path: sectionPath, comment: 'preserved from imported config', entries: { [key]: value } });
      }
    }
  };
  walk(extras as Record<string, unknown>, []);
}

/** Keys the UI owns, flattened; used by the validator to flag extras collisions. */
export function ownedTomlKeys(): Set<string> {
  const keys = new Set<string>();
  for (const type of Object.values(NODE_TYPES)) {
    for (const f of nodeSpec(type).fields) {
      if (f.tomlKey) {
        keys.add(f.tomlKey);
      }
    }
  }
  return keys;
}

export { ARCH_BY_TYPE };
