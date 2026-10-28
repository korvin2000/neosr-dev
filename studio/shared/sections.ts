/**
 * The seven editing sections of a recipe (Data · Model · Losses · Training plan · Optimizer ·
 * Validation & saving · Runtime). One inspector renders them everywhere: the Build list, the
 * graph drawer, the Review tokens. Each section maps to node fields split into Basics (≤ 5) and
 * More options; whatever a part does not list is not shown there.
 */
import { ARCH_BY_TYPE } from './catalog/archs';
import { LOSSES } from './catalog/losses';
import { NODE_TYPES, nodeSpec } from './catalog/node-catalog';
import { defaultValues, fieldVisible, type FieldSpec } from './catalog/types';
import { activeLossRows, lossRows, nodeValues, planContext, valDatasets, type PlanContext } from './compile/graph';
import type { Problem } from './compile/problems';
import { totalIters, type Recipe, type RecipeNode } from './recipe.model';

export type SectionId = 'data' | 'model' | 'losses' | 'plan' | 'optimizer' | 'validation' | 'runtime';

export type CustomEditor = 'losses' | 'val-datasets' | 'arch-params' | 'phases';

export interface SectionPart {
  readonly nodeType: string;
  /** Heading when a section shows several parts; defaults to the node label. */
  readonly title?: string;
  /** Field keys shown when the section opens. */
  readonly basics: readonly string[];
  /** Field keys behind "More options", in order. `'*'` = every other visible field. */
  readonly more: readonly string[];
  /** Custom editors to render (in Basics) for this node. */
  readonly custom?: readonly CustomEditor[];
  /** Only render this part when the predicate holds (e.g. discriminator only with a GAN row). */
  readonly when?: (ctx: PlanContext, recipe: Recipe) => boolean;
  /** Text for the "add" button when the node is missing. */
  readonly addLabel?: string;
}

export interface SectionDef {
  readonly id: SectionId;
  readonly title: string;
  readonly icon: string;
  readonly parts: readonly SectionPart[];
  readonly summary: (recipe: Recipe, ctx: PlanContext) => string;
}

const hasGanRow = (ctx: PlanContext, recipe: Recipe) => recipe.phases.some((p) => activeLossRows(lossRows(ctx.losses), p.id).some(({ row }) => row.type === 'gan_loss'));

function fmtK(n: number): string {
  return n >= 1000 ? `${Math.round(n / 1000)}k` : String(n);
}

function fmtLr(v: unknown): string {
  const n = Number(v);
  return Number.isFinite(n) ? (n < 0.001 ? n.toExponential(0).replace('e-', 'e-') : String(n)) : '?';
}

export const SECTIONS: readonly SectionDef[] = [
  {
    id: 'data',
    title: 'Data',
    icon: 'database',
    parts: [
      {
        nodeType: NODE_TYPES.trainDataset,
        basics: ['type', 'dataroot_gt', 'dataroot_lq', 'patch_size', 'batch_size'],
        more: ['gt_crop', 'accumulate', 'effective_batch', 'augmentation', 'aug_prob', 'use_hflip', 'use_rot', 'num_worker_per_gpu', 'color', 'meta_info', 'queue_size', '*'],
        addLabel: 'Add training data',
      },
      {
        nodeType: NODE_TYPES.degradation,
        basics: [],
        more: ['*'],
        when: (ctx) => !!ctx.data && nodeValues(ctx.data)['type'] === 'otf',
        addLabel: 'Add on-the-fly degradations',
      },
    ],
    summary: (recipe, ctx) => {
      if (!ctx.data) {
        return 'No training data yet';
      }
      const v = nodeValues(ctx.data);
      const scale = ctx.model ? Number(nodeValues(ctx.model)['scale']) : 4;
      const patch = Number(v['patch_size']);
      return `${v['type'] === 'otf' ? 'On-the-fly degradations' : 'Paired images'} · ${patch} px → ${patch * scale} · batch ${v['batch_size']}${Number(v['accumulate']) > 1 ? ` × ${v['accumulate']}` : ''}`;
    },
  },
  {
    id: 'model',
    title: 'Model',
    icon: 'cpu',
    parts: [
      { nodeType: NODE_TYPES.generator, basics: ['type', 'scale'], more: [], custom: ['arch-params'], addLabel: 'Add a model' },
      { nodeType: NODE_TYPES.initialization, title: 'Start from', basics: ['mode', 'pretrain_network_g', 'resume_state'], more: ['pretrain_network_d', 'strict_load_g', 'param_key_g'], addLabel: 'Choose where to start from' },
      { nodeType: NODE_TYPES.discriminator, basics: ['type'], more: [], custom: ['arch-params'], when: hasGanRow, addLabel: 'Add a discriminator' },
    ],
    summary: (recipe, ctx) => {
      if (!ctx.model) {
        return 'No model yet';
      }
      const v = nodeValues(ctx.model);
      const arch = ARCH_BY_TYPE.get(String(v['type']));
      const init = ctx.init ? nodeValues(ctx.init)['mode'] : 'scratch';
      const from = init === 'pretrain' ? 'from weights' : init === 'resume' ? 'resuming a state' : 'from scratch';
      return `${arch?.label ?? v['type']} · ×${v['scale']} · ${from}${ctx.dmodel && hasGanRow(ctx, recipe) ? ` · D ${nodeValues(ctx.dmodel)['type']}` : ''}`;
    },
  },
  {
    id: 'losses',
    title: 'Losses',
    icon: 'target',
    parts: [{ nodeType: NODE_TYPES.losses, basics: [], more: [], custom: ['losses'], addLabel: 'Add losses' }],
    summary: (recipe, ctx) => {
      const rows = lossRows(ctx.losses);
      const phaseId = recipe.phases[recipe.phases.length - 1]?.id ?? '';
      const active = activeLossRows(rows, phaseId);
      if (!active.length) {
        return 'No loss terms';
      }
      const names = active.slice(0, 4).map(({ row, weight }) => `${LOSSES.find((l) => l.type === row.type)?.label ?? row.type} ${weight}`);
      return `${active.length} term${active.length === 1 ? '' : 's'} · ${names.join(', ')}${active.length > 4 ? ', …' : ''}`;
    },
  },
  {
    id: 'plan',
    title: 'Training plan',
    icon: 'timeline',
    parts: [{ nodeType: NODE_TYPES.plan, basics: ['total_iter'], more: [], custom: ['phases'] }],
    summary: (recipe) => {
      const total = totalIters(recipe.phases);
      const n = recipe.phases.length;
      return `${fmtK(total)} updates · ${n === 1 ? 'one run' : `${n} phases, ${n} sequential runs`}`;
    },
  },
  {
    id: 'optimizer',
    title: 'Optimizer',
    icon: 'sliders',
    parts: [
      { nodeType: NODE_TYPES.optimizerG, basics: ['type', 'lr'], more: ['betas', 'weight_decay', 'warmup_steps', 'eps', 'extra'], addLabel: 'Add an optimizer' },
      { nodeType: NODE_TYPES.optimizerD, basics: ['type', 'lr'], more: ['betas', 'weight_decay', 'warmup_steps', 'eps', 'extra'], when: hasGanRow, addLabel: 'Add a discriminator optimizer' },
      { nodeType: NODE_TYPES.schedule, title: 'Learning-rate schedule', basics: ['type', 'milestones', 'gamma', 'T_max', 'eta_min'], more: [], when: (ctx) => !!ctx.schedule },
    ],
    summary: (recipe, ctx) => {
      if (!ctx.optimG) {
        return 'No optimizer yet';
      }
      const g = nodeValues(ctx.optimG);
      const d = ctx.optimD && hasGanRow(ctx, recipe) ? ` · D lr ${fmtLr(nodeValues(ctx.optimD)['lr'])}` : '';
      return `${g['type']} · lr ${fmtLr(g['lr'])}${d}${ctx.schedule ? ` · ${nodeValues(ctx.schedule)['type']}` : ''}`;
    },
  },
  {
    id: 'validation',
    title: 'Validation & saving',
    icon: 'check-image',
    parts: [
      { nodeType: NODE_TYPES.validation, basics: ['val_freq', 'metrics'], more: ['save_img', 'tile', 'tile_overlap', 'save_lq'], custom: ['val-datasets'], addLabel: 'Add validation' },
      { nodeType: NODE_TYPES.plan, title: 'Saving', basics: ['save_checkpoint_freq'], more: ['keep_states', 'print_freq', 'use_tb_logger', 'save_tb_img', 'wandb_project'] },
    ],
    summary: (recipe, ctx) => {
      const plan = ctx.plan ? nodeValues(ctx.plan) : {};
      const ckpt = `checkpoints every ${fmtK(Number(plan['save_checkpoint_freq'] ?? 1000))}`;
      if (!ctx.val) {
        return `No validation · ${ckpt}`;
      }
      const v = nodeValues(ctx.val);
      const sets = valDatasets(ctx.val);
      const metrics = Array.isArray(v['metrics']) ? (v['metrics'] as string[]).map((m) => m.toUpperCase()).join(' ') : '';
      return `${sets.length} set${sets.length === 1 ? '' : 's'} · every ${fmtK(Number(v['val_freq']))}${metrics ? ` · ${metrics}` : ''} · ${ckpt}`;
    },
  },
  {
    id: 'runtime',
    title: 'Runtime',
    icon: 'gpu',
    parts: [
      { nodeType: NODE_TYPES.runtime, basics: ['precision', 'fast_matmul'], more: ['compile', 'compile_mode', 'manual_seed', 'num_gpu'], addLabel: 'Add runtime settings' },
      { nodeType: NODE_TYPES.plan, title: 'Stability', basics: ['ema'], more: ['ema_power', 'grad_clip', 'clamp', 'max_skip_streak'] },
    ],
    summary: (recipe, ctx) => {
      const rt = ctx.runtime ? nodeValues(ctx.runtime) : undefined;
      const plan = ctx.plan ? nodeValues(ctx.plan) : {};
      const precision = rt ? ({ fp32: 'fp32', amp: 'AMP fp16', bf16: 'AMP bf16' } as Record<string, string>)[String(rt['precision'])] ?? String(rt['precision']) : 'default precision';
      const ema = Number(plan['ema'] ?? 0);
      return `${precision}${rt?.['compile'] === true ? ' · compiled' : ''} · ${ema > 0 ? `EMA ${ema}` : 'no EMA'}`;
    },
  },
];

export const SECTION_BY_ID: ReadonlyMap<SectionId, SectionDef> = new Map(SECTIONS.map((s) => [s.id, s]));

export function sectionOf(id: string): SectionDef {
  const s = SECTION_BY_ID.get(id as SectionId);
  if (!s) {
    throw new Error(`Unknown section ${id}`);
  }
  return s;
}

/** Fields of a part, resolved against the catalog, split into basics and more. */
export function partFields(part: SectionPart, node: RecipeNode | undefined): { basics: FieldSpec[]; more: FieldSpec[] } {
  const spec = nodeSpec(part.nodeType);
  const values = node ? nodeValues(node, spec) : defaultValues(spec);
  const visible = spec.fields.filter((f) => fieldVisible(f, values) && f.widget.kind !== 'custom');
  const byKey = new Map(visible.map((f) => [f.key, f]));
  const basics = part.basics.map((k) => byKey.get(k)).filter((f): f is FieldSpec => !!f);
  const listed = new Set([...part.basics, ...part.more]);
  const more: FieldSpec[] = [];
  for (const key of part.more) {
    if (key === '*') {
      for (const f of visible) {
        if (!listed.has(f.key) && !more.includes(f)) {
          more.push(f);
        }
      }
    } else {
      const f = byKey.get(key);
      if (f) {
        more.push(f);
      }
    }
  }
  return { basics, more };
}

/** The node a part edits: the one wired into the plan, else the first of its type. */
export function partNode(part: SectionPart, recipe: Recipe, ctx: PlanContext = planContext(recipe)): RecipeNode | undefined {
  const wired = [ctx.plan, ctx.data, ctx.degradation, ctx.val, ctx.model, ctx.dmodel, ctx.init, ctx.losses, ctx.optimG, ctx.optimD, ctx.schedule, ctx.runtime].find((n) => n?.type === part.nodeType);
  return wired ?? recipe.nodes.find((n) => n.type === part.nodeType && !n.disabled);
}

/** The section that owns a node type (first match). */
export function sectionForNodeType(nodeType: string): SectionDef | undefined {
  return SECTIONS.find((s) => s.parts.some((p) => p.nodeType === nodeType));
}

// --- changes against the preset snapshot ----------------------------------------------------------

/** Baseline a field is compared against: the preset snapshot when the recipe has one, else the catalog default. */
export function baselineValue(recipe: Recipe, node: RecipeNode, key: string): unknown {
  const snap = recipe.meta.presetSnapshot?.[node.type];
  if (snap && key in snap) {
    return snap[key];
  }
  return defaultValues(nodeSpec(node.type))[key];
}

export function isChanged(recipe: Recipe, node: RecipeNode, key: string): boolean {
  const current = nodeValues(node)[key];
  return JSON.stringify(current ?? null) !== JSON.stringify(baselineValue(recipe, node, key) ?? null);
}

export interface Change {
  readonly section: SectionId;
  readonly nodeId: string;
  readonly nodeType: string;
  readonly key: string;
  readonly label: string;
  readonly from: unknown;
  readonly to: unknown;
  /** `phases`: the phase list itself; `override`: per-phase values of a node. Resetting these is not a plain setValue. */
  readonly kind?: 'phases' | 'override';
  /** For `override` changes: the phase the override belongs to. */
  readonly phaseId?: string;
}

/** Every field that differs from the preset, in section order. Custom editors and the phase list count as one change each. */
export function recipeChanges(recipe: Recipe, ctx: PlanContext = planContext(recipe)): Change[] {
  const changes: Change[] = [];
  const presetPhases = recipe.meta.presetPhases;
  if (presetPhases && ctx.plan) {
    const before = presetPhases.map((p) => `${p.name}:${p.iters}:${p.init}`).join('|');
    const after = recipe.phases.map((p) => `${p.name}:${p.iters}:${p.init}`).join('|');
    if (before !== after) {
      changes.push({
        section: 'plan',
        nodeId: ctx.plan.id,
        nodeType: ctx.plan.type,
        key: 'phases',
        kind: 'phases',
        label: 'Training length',
        from: `${totalIters(presetPhases).toLocaleString()} updates · ${presetPhases.length} phase${presetPhases.length === 1 ? '' : 's'}`,
        to: `${totalIters(recipe.phases).toLocaleString()} updates · ${recipe.phases.length} phase${recipe.phases.length === 1 ? '' : 's'}`,
      });
    }
  }
  for (const section of SECTIONS) {
    for (const part of section.parts) {
      if (part.when && !part.when(ctx, recipe)) {
        continue;
      }
      const node = partNode(part, recipe, ctx);
      if (!node) {
        continue;
      }
      const spec = nodeSpec(node.type);
      const { basics, more } = partFields(part, node);
      for (const field of [...basics, ...more]) {
        if (field.widget.kind === 'readout') {
          continue;
        }
        if (isChanged(recipe, node, field.key)) {
          changes.push({ section: section.id, nodeId: node.id, nodeType: node.type, key: field.key, label: field.label, from: baselineValue(recipe, node, field.key), to: nodeValues(node)[field.key] });
        }
      }
      for (const editor of part.custom ?? []) {
        const key = editor === 'losses' ? 'losses' : editor === 'val-datasets' ? 'datasets' : editor === 'arch-params' ? 'params' : null;
        if (key && isChanged(recipe, node, key)) {
          const label = spec.fields.find((f) => f.key === key)?.label ?? key;
          changes.push({ section: section.id, nodeId: node.id, nodeType: node.type, key, label, from: baselineValue(recipe, node, key), to: nodeValues(node)[key] });
        }
      }
      // Per-phase overrides are edits too, even when the shared value is untouched.
      const baselineOverrides = recipe.meta.presetOverrides?.[node.type] ?? {};
      for (const [phaseId, values] of Object.entries(node.phaseValues)) {
        const phase = recipe.phases.find((p) => p.id === phaseId);
        for (const [key, value] of Object.entries(values)) {
          const field = spec.fields.find((f) => f.key === key);
          const inPart = [...basics, ...more].some((f) => f.key === key);
          if (!field || !inPart || !phase) {
            continue;
          }
          const before = baselineOverrides[phaseId]?.[key];
          if (JSON.stringify(before ?? null) !== JSON.stringify(value ?? null)) {
            changes.push({ section: section.id, nodeId: node.id, nodeType: node.type, key, kind: 'override', phaseId, label: `${field.label} · ${phase.name}`, from: before ?? nodeValues(node)[key], to: value });
          }
        }
      }
    }
  }
  return changes;
}

/** The section (and whether the field is behind More options) that shows `fieldKey` of a node type. */
export function sectionForField(nodeType: string, fieldKey: string | undefined): SectionDef | undefined {
  if (fieldKey) {
    for (const section of SECTIONS) {
      for (const part of section.parts) {
        if (part.nodeType === nodeType && (part.basics.includes(fieldKey) || part.more.includes(fieldKey) || part.more.includes('*'))) {
          const custom = part.custom ?? [];
          const customKey = fieldKey === 'losses' ? 'losses' : fieldKey === 'datasets' ? 'val-datasets' : fieldKey === 'params' ? 'arch-params' : null;
          if (!customKey || custom.includes(customKey)) {
            return section;
          }
        }
      }
    }
    for (const section of SECTIONS) {
      for (const part of section.parts) {
        const customKey = fieldKey === 'losses' ? 'losses' : fieldKey === 'datasets' ? 'val-datasets' : fieldKey === 'params' ? 'arch-params' : fieldKey === 'phases' ? 'phases' : null;
        if (part.nodeType === nodeType && customKey && part.custom?.includes(customKey)) {
          return section;
        }
      }
    }
  }
  return sectionForNodeType(nodeType);
}

export function sectionProblems(problems: readonly Problem[], recipe: Recipe, section: SectionDef, ctx: PlanContext = planContext(recipe)): Problem[] {
  const ids = new Set(section.parts.map((p) => partNode(p, recipe, ctx)?.id).filter((id): id is string => !!id));
  return problems.filter((p) => p.nodeId && ids.has(p.nodeId));
}

// --- find a setting ---------------------------------------------------------------------------------

export interface SettingHit {
  readonly section: SectionDef;
  readonly part: SectionPart;
  readonly field: FieldSpec;
  readonly inMore: boolean;
}

/** Matches labels, help and TOML keys so a TOML user can type `val_freq` and land on the field. */
export function findSetting(query: string, recipe: Recipe): SettingHit[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return [];
  }
  const ctx = planContext(recipe);
  const scored: { hit: SettingHit; score: number }[] = [];
  for (const section of SECTIONS) {
    for (const part of section.parts) {
      const node = partNode(part, recipe, ctx);
      const { basics, more } = partFields(part, node);
      for (const [list, inMore] of [[basics, false], [more, true]] as const) {
        for (const field of list) {
          // Exact key or TOML key first, then the label, then a mention in the help text.
          const key = field.key.toLowerCase();
          const toml = field.tomlKey.toLowerCase();
          const label = field.label.toLowerCase();
          const score = key === q || toml === q || toml.endsWith(`.${q}`) ? 3 : key.includes(q) || toml.includes(q) || label.includes(q) ? 2 : field.help.toLowerCase().includes(q) ? 1 : 0;
          if (score > 0) {
            scored.push({ hit: { section, part, field, inMore }, score });
          }
        }
      }
    }
  }
  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, 12)
    .map((s) => s.hit);
}
