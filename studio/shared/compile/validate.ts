/**
 * Static validation of a Recipe. Rules marked `native` are NeoSR's own (it would refuse the TOML);
 * `arch` rules come from architecture metadata; `heuristic` rules are advice the user may ignore.
 * Source: design/neosr_config_surface.md §10.
 */
import { ARCH_BY_TYPE } from '../catalog/archs';
import { LOSSES } from '../catalog/losses';
import { NODE_TYPES, nodeSpec } from '../catalog/node-catalog';
import { OPTIMIZERS } from '../catalog/optimizers';
import { fieldVisible, portVisible } from '../catalog/types';
import { phaseRanges, type Recipe, type RecipeNode } from '../recipe.model';
import {
  activeLossRows,
  findNodes,
  incomingEdge,
  lossRows,
  lossSlot,
  nodeValues,
  phaseHasGan,
  phaseNodeValues,
  stateIter,
  unreachableNodes,
  upstreamNode,
  valDatasets,
  valSection,
  type PlanContext,
} from './graph';
import type { Problem, Severity } from './problems';

export interface ValidateOptions {
  readonly platform?: 'windows' | 'linux';
}

export function validateRecipe(recipe: Recipe, ctx: PlanContext, options: ValidateOptions = {}): Problem[] {
  const problems: Problem[] = [];
  let seq = 0;
  const add = (severity: Severity, source: Problem['source'], message: string, extra: Partial<Problem> = {}) => {
    problems.push({ id: `p${seq++}`, severity, source, message, ...extra });
  };

  // --- structure ---------------------------------------------------------
  const plans = findNodes(recipe, NODE_TYPES.plan).filter((n) => !n.disabled);
  if (plans.length === 0) {
    add('blocker', 'studio', 'Add a Training Plan node: nothing can be compiled without one.');
    return problems;
  }
  if (plans.length > 1) {
    add('blocker', 'studio', 'Only one Training Plan per recipe. Disable or delete the others.', { nodeId: plans[1].id });
  }
  const plan = ctx.plan as RecipeNode;
  const planSpec = nodeSpec(NODE_TYPES.plan);
  for (const port of planSpec.inputs) {
    if (!port.required) {
      continue;
    }
    const edge = incomingEdge(recipe, plan.id, port.key);
    if (!edge) {
      add('blocker', 'studio', `Training Plan is missing its ${port.label} input.`, { nodeId: plan.id, hint: `Connect a node with a ${port.type} output.` });
    } else if (!upstreamNode(recipe, plan.id, port.key)) {
      add('blocker', 'studio', `Training Plan's ${port.label} input comes from a disabled node.`, { nodeId: edge.sourceNode, hint: 'Enable the node or connect another one.' });
    }
  }

  // Singleton families and duplicated section owners.
  const owners = new Map<string, RecipeNode[]>();
  for (const node of recipe.nodes.filter((n) => !n.disabled)) {
    for (const section of nodeSpec(node.type).owns) {
      owners.set(section, [...(owners.get(section) ?? []), node]);
    }
  }
  for (const [section, nodes] of owners) {
    const wired = nodes.filter((n) => !unreachableNodes(recipe).includes(n));
    if (wired.length > 1) {
      add('blocker', 'studio', `Two nodes write [${section}]: ${wired.map((n) => n.title || nodeSpec(n.type).label).join(' and ')}.`, { nodeId: wired[1].id });
    }
  }
  for (const node of unreachableNodes(recipe)) {
    if (node.type !== NODE_TYPES.plan) {
      add('info', 'studio', `${node.title || nodeSpec(node.type).label} is not connected to the Training Plan and will not be compiled.`, { nodeId: node.id });
    }
  }

  // Required fields and visible required ports on every wired node.
  for (const node of recipe.nodes.filter((n) => !n.disabled)) {
    const spec = nodeSpec(node.type);
    const values = nodeValues(node, spec);
    for (const field of spec.fields) {
      if (!field.required || !fieldVisible(field, values)) {
        continue;
      }
      const v = values[field.key];
      const empty = v === undefined || v === null || v === '' || (Array.isArray(v) && v.length === 0);
      if (empty) {
        add('blocker', 'native', `${spec.label}: "${field.label}" is required.`, { nodeId: node.id, fieldKey: field.key });
      }
    }
    for (const port of spec.inputs) {
      if (port.required && portVisible(port, values) && !upstreamNode(recipe, node.id, port.key)) {
        add('blocker', 'studio', `${spec.label}: input "${port.label}" must be connected to an enabled node.`, { nodeId: node.id });
      }
    }
    checkNumbers(node, spec.label, add);
  }

  // --- phases --------------------------------------------------------------
  if (recipe.phases.length === 0) {
    add('blocker', 'studio', 'The Training Plan needs at least one phase.', { nodeId: plan.id });
  }
  const rows = lossRows(ctx.losses);
  const ranges = phaseRanges(recipe.phases);
  const saveFreq = Number(nodeValues(plan)['save_checkpoint_freq']);
  ranges.forEach(({ phase }, i) => {
    if (!(phase.iters > 0)) {
      add('blocker', 'studio', `Phase "${phase.name}" has no duration.`, { nodeId: plan.id, phaseId: phase.id });
    }
    if (i === 0 && phase.init !== 'scratch') {
      add('info', 'studio', `The first phase starts from the Initialization node; its transition is ignored.`, { phaseId: phase.id });
    }
    if (i > 0) {
      const prev = ranges[i - 1].phase;
      const ganNow = phaseHasGan(rows, phase.id);
      const ganPrev = phaseHasGan(rows, prev.id);
      if (phase.init === 'continue') {
        if (ganNow !== ganPrev) {
          add('blocker', 'native', `Phase "${phase.name}" ${ganNow ? 'adds' : 'removes'} the discriminator but is set to continue the training state. NeoSR resume requires the same networks and optimizers.`, {
            phaseId: phase.id,
            nodeId: plan.id,
            hint: 'Set the transition to "fine-tune from weights".',
          });
        }
        for (const node of [ctx.optimG, ctx.optimD, ctx.data].filter(Boolean) as RecipeNode[]) {
          const overrides = node.phaseValues[phase.id] ?? {};
          if ('lr' in overrides) {
            add('warning', 'native', `Phase "${phase.name}" overrides the learning rate, but a resumed state restores the optimizer LR; the override will not be effective.`, { phaseId: phase.id, nodeId: node.id, fieldKey: 'lr', hint: 'Use "fine-tune from weights" to apply a new LR.' });
          }
        }
      }
      if (phase.init === 'finetune' && !ganPrev && ganNow && !ctx.dmodel) {
        add('blocker', 'studio', `Phase "${phase.name}" enables gan_loss but no Discriminator is connected.`, { phaseId: phase.id, nodeId: plan.id });
      }
    }
    const active = activeLossRows(rows, phase.id);
    if (!active.some(({ row, weight }) => row.type !== 'gan_loss' && weight > 0)) {
      add('blocker', 'native', `Phase "${phase.name}" has no generator loss with a non-zero weight.`, { phaseId: phase.id, nodeId: ctx.losses?.id ?? plan.id, fieldKey: 'losses' });
    }
    const slots = new Map<string, number>();
    for (const { row } of active) {
      const slot = lossSlot(row, LOSSES.find((l) => l.type === row.type)?.slot ?? row.type);
      slots.set(slot, (slots.get(slot) ?? 0) + 1);
    }
    for (const [slot, count] of slots) {
      if (count > 1) {
        add('blocker', 'native', `Two active loss terms write [train.${slot}_opt]${recipe.phases.length > 1 ? ` in phase "${phase.name}"` : ''}. Give one of them another slot name.`, { phaseId: phase.id, nodeId: ctx.losses?.id, fieldKey: 'losses', hint: 'Open the row parameters and set a slot such as pixel_2.' });
      }
    }
  });
  if (ctx.init) {
    const iv = nodeValues(ctx.init);
    if (iv['mode'] === 'resume' && String(iv['resume_state'] ?? '').trim() && stateIter(String(iv['resume_state'])) === null) {
      add('warning', 'studio', 'The resume state file name does not end in <update>.state, so the restored update counter is unknown; total_iter is computed from 0.', { nodeId: ctx.init.id, fieldKey: 'resume_state', hint: 'Use the file NeoSR wrote, e.g. training_states/50000.state.' });
    }
  }

  // --- GAN triple coupling ---------------------------------------------------
  const anyGan = recipe.phases.some((p) => phaseHasGan(rows, p.id));
  if (anyGan && !ctx.dmodel) {
    add('blocker', 'native', 'gan_loss needs a Discriminator connected to the Training Plan.', { nodeId: ctx.losses?.id ?? plan.id, fieldKey: 'losses', hint: 'Add a Discriminator node (metagan) and an Optimizer D.' });
  }
  if (anyGan && !ctx.optimD) {
    add('blocker', 'native', 'gan_loss needs an Optimizer D connected to the Training Plan.', { nodeId: plan.id, hint: 'Add an Optimizer D node.' });
  }
  if (!anyGan && (ctx.dmodel || ctx.optimD)) {
    add('warning', 'native', 'A Discriminator / Optimizer D is connected but no phase enables gan_loss; they will be left out of the TOML.', { nodeId: (ctx.dmodel ?? ctx.optimD)?.id });
  }
  for (const row of rows) {
    if (row.type === 'gan_loss' && row.enabled && row.weight === 0) {
      add('warning', 'native', 'gan_loss with weight 0 still trains the discriminator; disable the row instead.', { nodeId: ctx.losses?.id, fieldKey: 'losses' });
    }
    if (!LOSSES.some((l) => l.type === row.type)) {
      add('blocker', 'native', `Unknown loss type "${row.type}".`, { nodeId: ctx.losses?.id, fieldKey: 'losses' });
    }
  }

  // --- model / scale / patch -------------------------------------------------
  const scale = Number(ctx.model ? nodeValues(ctx.model)['scale'] : 4);
  if (ctx.model) {
    const arch = ARCH_BY_TYPE.get(String(nodeValues(ctx.model)['type']));
    if (!arch) {
      add('blocker', 'native', `Unknown architecture "${nodeValues(ctx.model)['type']}".`, { nodeId: ctx.model.id, fieldKey: 'type' });
    } else {
      if (!arch.scales.includes(scale)) {
        add('blocker', 'arch', `${arch.label} does not support ×${scale} (supported: ${arch.scales.map((s) => `×${s}`).join(', ')}).`, { nodeId: ctx.model.id, fieldKey: 'scale' });
      }
      const params = (nodeValues(ctx.model)['params'] as Record<string, unknown>) ?? {};
      for (const key of ['scale', 'upscale', 'upscaling_factor', 'upsampling', 'sr_rate', 'upscale_factor']) {
        if (key in params) {
          add('blocker', 'native', `Remove "${key}" from the architecture options: NeoSR injects the run scale and rejects a conflicting value.`, { nodeId: ctx.model.id, fieldKey: 'params' });
        }
      }
      if (ctx.data && arch.windowSize) {
        for (const { phase } of ranges) {
          const patch = Number(phaseNodeValues(ctx.data, phase.id)['patch_size']);
          if (patch % arch.windowSize !== 0) {
            add('warning', 'arch', `patch_size ${patch} is not a multiple of ${arch.label}'s window size ${arch.windowSize}${recipe.phases.length > 1 ? ` (phase "${phase.name}")` : ''}.`, { nodeId: ctx.data.id, fieldKey: 'patch_size', phaseId: phase.id, hint: `Use ${Math.ceil(patch / arch.windowSize) * arch.windowSize}.` });
          }
        }
      }
      if (['hat_l', 'drct_l', 'drct_xl', 'hit_srf_large', 'atd'].includes(arch.type)) {
        add('info', 'heuristic', `${arch.label} is VRAM-heavy; expect to lower batch or patch size.`, { nodeId: ctx.model.id, fieldKey: 'type' });
      }
    }
  }
  if (ctx.dmodel) {
    const t = String(nodeValues(ctx.dmodel)['type']);
    if (ARCH_BY_TYPE.has(t)) {
      add('blocker', 'native', `"${t}" is a generator; network_d needs a discriminator type.`, { nodeId: ctx.dmodel.id, fieldKey: 'type' });
    }
  }

  // --- dataset -----------------------------------------------------------------
  if (ctx.data) {
    const spec = nodeSpec(NODE_TYPES.trainDataset);
    const dv = nodeValues(ctx.data, spec);
    const isOtf = dv['type'] === 'otf';
    if (isOtf && String(dv['dataroot_lq'] ?? '').trim()) {
      add('warning', 'native', 'On-the-fly degradations generate the low-res images; the low-res folder is ignored (NeoSR rejects dataroot_lq for otf).', { nodeId: ctx.data.id, fieldKey: 'dataroot_lq' });
    }
    if (!isOtf && ctx.degradation) {
      add('warning', 'studio', 'Degradations are connected but the dataset is paired; they will be ignored. Switch the dataset type to OTF to use them.', { nodeId: ctx.data.id, fieldKey: 'type' });
    }
    const aug = dv['augmentation'];
    const prob = dv['aug_prob'];
    if (Array.isArray(aug) && Array.isArray(prob) && aug.length > 0 && aug.length !== prob.length) {
      add('blocker', 'native', `augmentation has ${aug.length} entries but aug_prob has ${prob.length}.`, { nodeId: ctx.data.id, fieldKey: 'aug_prob' });
    }
    for (const { phase } of ranges) {
      const pv = phaseNodeValues(ctx.data, phase.id, spec);
      const batch = Number(pv['batch_size']);
      if (batch === 1 && Array.isArray(aug) && (aug as string[]).some((a) => a !== 'none' && a !== 'cutblur')) {
        add('blocker', 'native', 'mixup / cutmix / resizemix need batch_size > 1.', { nodeId: ctx.data.id, fieldKey: 'batch_size', phaseId: phase.id });
      }
      if (isOtf && Number(pv['queue_size']) < batch) {
        add('blocker', 'native', 'OTF queue_size must be ≥ batch_size.', { nodeId: ctx.data.id, fieldKey: 'queue_size', phaseId: phase.id });
      }
      const patch = Number(pv['patch_size']);
      if (patch > 0 && patch % 8 !== 0) {
        add('info', 'heuristic', `patch_size ${patch}: the wiki recommends a multiple of 8 or 16.`, { nodeId: ctx.data.id, fieldKey: 'patch_size', phaseId: phase.id });
      }
    }
    if (Number(dv['num_worker_per_gpu']) === 0) {
      add('info', 'native', 'With 0 data workers persistent_workers is disabled and loading runs in the main process.', { nodeId: ctx.data.id, fieldKey: 'num_worker_per_gpu' });
    }
  }

  // --- validation --------------------------------------------------------------
  if (ctx.val) {
    const vv = nodeValues(ctx.val);
    const sets = valDatasets(ctx.val);
    if (sets.length === 0) {
      add('blocker', 'studio', 'Add at least one validation set, or disconnect the Validation node.', { nodeId: ctx.val.id, fieldKey: 'datasets' });
    }
    const names = new Set<string>();
    const sectionsSeen = new Set<string>();
    sets.forEach((row, i) => {
      const name = row.name || (i === 0 ? 'val' : `val_${i}`);
      if (names.has(name)) {
        add('blocker', 'native', `Two validation sets are named "${name}".`, { nodeId: ctx.val!.id, fieldKey: 'datasets' });
      }
      names.add(name);
      const section = valSection(row, i);
      if (sectionsSeen.has(section)) {
        add('blocker', 'native', `Two validation sets write [${section}].`, { nodeId: ctx.val!.id, fieldKey: 'datasets' });
      }
      sectionsSeen.add(section);
      if (!row.dataroot_lq.trim()) {
        add('blocker', 'native', `Validation set "${name}" needs a low-res folder.`, { nodeId: ctx.val!.id, fieldKey: 'datasets' });
      }
      if (row.type === 'paired' && !row.dataroot_gt.trim()) {
        add('blocker', 'native', `Validation set "${name}" needs a high-res folder (or make it a visual-only "single" set).`, { nodeId: ctx.val!.id, fieldKey: 'datasets' });
      }
    });
    const tile = Number(vv['tile']);
    const overlap = Number(vv['tile_overlap']);
    if (tile > 0 && !(tile > 2 * overlap && overlap >= 0)) {
      add('blocker', 'native', `val.tile (${tile}) must be greater than 2 × tile_overlap (${overlap}).`, { nodeId: ctx.val.id, fieldKey: 'tile' });
    }
    const metrics = vv['metrics'];
    if (sets.every((s) => s.type === 'single') && Array.isArray(metrics) && metrics.length) {
      add('info', 'native', 'Metrics are skipped for visual-only ("single") validation sets.', { nodeId: ctx.val.id, fieldKey: 'metrics' });
    }
    const valFreq = Number(vv['val_freq']);
    if (Number.isFinite(saveFreq) && valFreq > 0 && valFreq * 4 < saveFreq) {
      add('info', 'heuristic', `Validating every ${valFreq} updates while saving every ${saveFreq} costs time; most validation points will have no saved weights.`, { nodeId: ctx.val.id, fieldKey: 'val_freq' });
    }
  } else {
    add('warning', 'heuristic', 'No validation set: you will only see losses, never images or metrics, until training ends.', { nodeId: plan.id });
  }

  // --- optimizers / schedule ---------------------------------------------------
  const sfTypes = OPTIMIZERS.filter((o) => o.scheduleFree).map((o) => o.type);
  for (const node of [ctx.optimG, ctx.optimD].filter(Boolean) as RecipeNode[]) {
    const v = nodeValues(node);
    if (!OPTIMIZERS.some((o) => o.type === v['type'])) {
      add('blocker', 'native', `Unknown optimizer "${v['type']}".`, { nodeId: node.id, fieldKey: 'type' });
    }
    if (!(Number(v['lr']) > 0)) {
      add('blocker', 'native', 'Learning rate must be > 0.', { nodeId: node.id, fieldKey: 'lr' });
    }
  }
  if (ctx.schedule && ctx.optimG && sfTypes.includes(String(nodeValues(ctx.optimG)['type']))) {
    add('warning', 'heuristic', 'An LR Schedule is connected, but the generator optimizer is schedule-free; the scheduler will fight the optimizer.', { nodeId: ctx.schedule.id, hint: 'Remove the LR Schedule or switch to adamw / adan.' });
  }

  // --- runtime -----------------------------------------------------------------
  if (ctx.runtime) {
    const rv = nodeValues(ctx.runtime);
    if (rv['compile'] === true && options.platform === 'windows') {
      add('blocker', 'native', 'torch.compile is not available on Windows (no Triton); NeoSR refuses to start.', { nodeId: ctx.runtime.id, fieldKey: 'compile' });
    }
    if (Number(rv['num_gpu']) > 1) {
      add('info', 'native', `${rv['num_gpu']} GPUs: the launch command uses torchrun with --launcher pytorch.`, { nodeId: ctx.runtime.id, fieldKey: 'num_gpu' });
    }
    if (rv['precision'] === 'bf16') {
      add('info', 'heuristic', 'bf16 needs an Ampere-class GPU or newer and can be unstable on some ops; fall back to AMP fp16 if losses spike.', { nodeId: ctx.runtime.id, fieldKey: 'precision' });
    }
  }

  // --- plan ----------------------------------------------------------------------
  const pv = nodeValues(plan);
  const name = String(pv['name'] ?? '');
  if (name && !/^[A-Za-z0-9._-]+$/.test(name)) {
    add('warning', 'heuristic', 'Run names are folder names: avoid spaces and special characters.', { nodeId: plan.id, fieldKey: 'name' });
  }
  if (/debug/i.test(name)) {
    add('info', 'native', '"debug" in the run name forces debug frequencies (validate/save every 8 updates).', { nodeId: plan.id, fieldKey: 'name' });
  }
  if (pv['use_tb_logger'] !== true) {
    add('warning', 'heuristic', 'TensorBoard logger is off: Studio charts and TensorBoard will have no scalars.', { nodeId: plan.id, fieldKey: 'use_tb_logger' });
  }
  const ema = Number(pv['ema']);
  if (ema === 0 && ctx.optimG && sfTypes.includes(String(nodeValues(ctx.optimG)['type']))) {
    add('info', 'heuristic', 'EMA is off with a schedule-free optimizer; the wiki recommends ema = 0.999.', { nodeId: plan.id, fieldKey: 'ema' });
  }
  if (Number.isFinite(saveFreq) && saveFreq > 0) {
    const total = ranges.length ? ranges[ranges.length - 1].end : 0;
    const states = Math.floor(total / saveFreq);
    if (states > 400) {
      add('info', 'heuristic', `${states} checkpoints will be written (weights kept, states pruned to keep_states). Consider a larger save interval.`, { nodeId: plan.id, fieldKey: 'save_checkpoint_freq' });
    }
  }

  // --- native extras collisions ----------------------------------------------------
  const owned = new Set<string>();
  for (const type of Object.values(NODE_TYPES)) {
    for (const f of nodeSpec(type).fields) {
      if (f.tomlKey) {
        owned.add(f.tomlKey);
      }
    }
  }
  const walk = (obj: Record<string, unknown>, path: string[]) => {
    for (const [key, value] of Object.entries(obj)) {
      const full = [...path, key].join('.');
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        walk(value as Record<string, unknown>, [...path, key]);
      } else if (owned.has(full)) {
        add('warning', 'studio', `Imported extra "${full}" is also set by a node; the node's value wins.`, { nodeId: plan.id });
      }
    }
  };
  walk(recipe.nativeExtras as Record<string, unknown>, []);

  return problems;
}

/** Native numeric rules NeoSR enforces at load time (config.py validate_train, train.py). */
const NUMERIC_RULES: Record<string, { readonly min?: number; readonly exclusiveMin?: number; readonly integer?: boolean; readonly when?: (v: Record<string, unknown>) => boolean }> = {
  patch_size: { min: 1, integer: true },
  batch_size: { min: 1, integer: true },
  accumulate: { min: 1, integer: true },
  queue_size: { min: 1, integer: true },
  num_worker_per_gpu: { min: 0, integer: true },
  val_freq: { min: 1, integer: true },
  tile_overlap: { min: 0, integer: true },
  save_checkpoint_freq: { min: 1, integer: true },
  print_freq: { min: 1, integer: true },
  keep_states: { min: 0, integer: true },
  max_skip_streak: { min: 1, integer: true },
  ema: { min: 0 },
  ema_power: { exclusiveMin: 0, when: (v) => Number(v['ema']) > 0 },
  warmup_steps: { min: 0, integer: true },
  weight_decay: { min: 0 },
  gamma: { exclusiveMin: 0 },
  T_max: { min: 1, integer: true },
  eta_min: { min: 0 },
  manual_seed: { min: 0, integer: true },
  num_gpu: { min: 1, integer: true },
};

function checkNumbers(node: RecipeNode, label: string, add: (severity: Severity, source: Problem['source'], message: string, extra?: Partial<Problem>) => void): void {
  const spec = nodeSpec(node.type);
  const values = nodeValues(node, spec);
  for (const field of spec.fields) {
    const rule = NUMERIC_RULES[field.key];
    if (!rule || field.widget.kind !== 'number' || !fieldVisible(field, values)) {
      continue;
    }
    const raw = values[field.key];
    if (raw === undefined || raw === null || raw === '') {
      continue;
    }
    if (rule.when && !rule.when(values)) {
      continue;
    }
    const n = Number(raw);
    const bad =
      !Number.isFinite(n) ||
      (rule.integer && !Number.isInteger(n)) ||
      (rule.min !== undefined && n < rule.min) ||
      (rule.exclusiveMin !== undefined && n <= rule.exclusiveMin);
    if (bad) {
      const bound = rule.exclusiveMin !== undefined ? `> ${rule.exclusiveMin}` : `≥ ${rule.min}`;
      add('blocker', 'native', `${label}: ${field.label} must be ${rule.integer ? 'an integer ' : ''}${bound} (got ${String(raw)}).`, { nodeId: node.id, fieldKey: field.key });
    }
  }
}
