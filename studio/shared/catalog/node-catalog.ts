/**
 * The node catalog: one NodeSpec per NeoSR TOML section family. Fields carry the TOML key they
 * write, so the compiler, the inspector, the wizard and the TOML source map all read one table.
 */
import { ARCHS } from './archs';
import { AUGMENTATIONS } from './augmentations';
import { DEGRADATION_FIELDS } from './degradations';
import { DISCRIMINATORS } from './discriminators';
import { LOSSES } from './losses';
import { METRIC_TYPES } from './metrics';
import { OPTIMIZERS } from './optimizers';
import { SCHEDULERS } from './schedulers';
import type { DegradationFieldInfo, OptimizerInfo } from './data-types';
import type { FieldSpec, NodeSpec, SelectOption, Widget } from './types';

export const NODE_TYPES = {
  trainDataset: 'train-dataset',
  degradation: 'otf-degradation',
  validation: 'validation',
  generator: 'generator',
  discriminator: 'discriminator',
  initialization: 'initialization',
  losses: 'loss-recipe',
  optimizerG: 'optimizer-g',
  optimizerD: 'optimizer-d',
  schedule: 'lr-schedule',
  runtime: 'runtime',
  plan: 'training-plan',
} as const;

export type NodeTypeId = (typeof NODE_TYPES)[keyof typeof NODE_TYPES];

const archOptions: SelectOption[] = ARCHS.map((a) => ({
  value: a.type,
  label: a.label,
  hint: `${a.sizeClass} · ${a.speed}${a.windowSize ? ` · window ${a.windowSize}` : ''}`,
}));

const discriminatorOptions: SelectOption[] = DISCRIMINATORS.map((d) => ({
  value: d.type,
  label: d.label,
  hint: d.description,
}));

function optimizerOptions(side: 'g' | 'd'): SelectOption[] {
  return OPTIMIZERS.map((o) => ({
    value: o.type,
    label: o.label,
    hint: o.description,
    fills: optimizerFills(o, side),
  }));
}

/** Picking an optimizer type fills lr/betas/weight_decay/warmup with the template or ctor defaults. */
function optimizerFills(o: OptimizerInfo, side: 'g' | 'd'): Record<string, unknown> {
  const fromTemplate = o.template?.[side] ?? {};
  const fromCtor: Record<string, unknown> = {};
  for (const p of o.params) {
    if (['lr', 'betas', 'weight_decay', 'warmup_steps', 'eps'].includes(p.key)) {
      fromCtor[p.key] = p.default;
    }
  }
  const fills = { ...fromCtor, ...fromTemplate };
  if (!o.scheduleFree) {
    delete fills['warmup_steps'];
  }
  return fills;
}

function optimizerFields(side: 'g' | 'd'): FieldSpec[] {
  const isG = side === 'g';
  const prefix = isG ? 'train.optim_g' : 'train.optim_d';
  const sfTypes = OPTIMIZERS.filter((o) => o.scheduleFree).map((o) => o.type);
  return [
    {
      key: 'type',
      label: 'Optimizer',
      help: 'Schedule-free optimizers (adan_sf, adamw_sf) need no LR schedule and are what the NeoSR templates use.',
      tomlKey: `${prefix}.type`,
      widget: { kind: 'select', options: optimizerOptions(side) },
      default: 'adan_sf',
      required: true,
      level: 'essentials',
    },
    {
      key: 'lr',
      label: 'Learning rate',
      help: isG
        ? 'Generator learning rate. Templates: 1e-3 for most archs, 5e-4 for PLKSR-like, 8e-4 for ESRGAN.'
        : 'Discriminator learning rate. Templates use 1e-4, ten times lower than the generator.',
      tomlKey: `${prefix}.lr`,
      widget: { kind: 'number', min: 0, step: 0.00001 },
      default: isG ? 0.001 : 0.0001,
      required: true,
      level: 'essentials',
      phaseAware: true,
      suggestions: isG
        ? [
            { label: '1e-3', value: 0.001, why: 'template default for most archs' },
            { label: '5e-4', value: 0.0005, why: 'PLKSR / ESC / FlexNet templates' },
            { label: '1e-4', value: 0.0001, why: 'conservative fine-tune' },
          ]
        : [
            { label: '1e-4', value: 0.0001, why: 'template default' },
            { label: '5e-5', value: 0.00005, why: 'calmer discriminator' },
          ],
    },
    {
      key: 'betas',
      label: 'Betas',
      help: 'Momentum coefficients. Adan uses three values, Adam-family two.',
      tomlKey: `${prefix}.betas`,
      widget: { kind: 'numberlist', step: 0.001 },
      default: [0.98, 0.92, 0.99],
      level: 'advanced',
    },
    {
      key: 'weight_decay',
      label: 'Weight decay',
      help: 'L2 regularisation. Templates use 0.01.',
      tomlKey: `${prefix}.weight_decay`,
      widget: { kind: 'number', min: 0, step: 0.001 },
      default: 0.01,
      level: 'advanced',
    },
    {
      key: 'warmup_steps',
      label: 'Warm-up steps',
      help: 'Linear LR ramp at the start (schedule-free optimizers only). Templates: 1600 for G, 600 for D.',
      tomlKey: `${prefix}.warmup_steps`,
      widget: { kind: 'number', min: 0, step: 100, unit: 'updates' },
      default: isG ? 1600 : 600,
      level: 'advanced',
      showWhen: { key: 'type', values: sfTypes },
      support: 'native',
    },
    {
      key: 'eps',
      label: 'Epsilon',
      help: 'Numerical stability term.',
      tomlKey: `${prefix}.eps`,
      widget: { kind: 'number', min: 0, step: 1e-9 },
      default: 1e-8,
      level: 'native',
    },
    {
      key: 'extra',
      label: 'Extra kwargs',
      help: 'Any other constructor argument, passed through verbatim.',
      tomlKey: prefix,
      widget: { kind: 'keyvalue' },
      default: {},
      level: 'native',
    },
  ];
}

function degradationWidget(f: DegradationFieldInfo): Widget {
  switch (f.type) {
    case 'bool':
      return { kind: 'toggle' };
    case 'float':
      return { kind: 'number', min: 0, max: f.key.includes('prob') ? 1 : undefined, step: 0.05 };
    case 'int':
      return { kind: 'number', min: 0, step: 1 };
    case 'list':
      if (f.key.startsWith('kernel_list')) {
        return {
          kind: 'multiselect',
          options: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso', 'skewed', 'skewed_aniso'].map(
            (k) => ({ value: k, label: k }),
          ),
        };
      }
      return { kind: 'numberlist', step: 0.05 };
    default:
      return { kind: 'text' };
  }
}

const DEGRADATION_ESSENTIALS = new Set([
  'second_blur_prob',
  'final_sinc_prob',
  'jpeg_compress',
  'jpeg_range',
  'blur_sigma',
  'noise_range',
  'resize_range',
]);

const degradationFields: FieldSpec[] = DEGRADATION_FIELDS.map((f) => ({
  key: f.key,
  label: f.label,
  help: f.help,
  tomlKey: `degradations.${f.key}`,
  widget: degradationWidget(f),
  default: f.default,
  level: DEGRADATION_ESSENTIALS.has(f.key) ? 'essentials' : 'advanced',
  group: f.stage === 'final' ? 'Final' : `Stage ${f.stage} · ${f.group}`,
}));

export const NODE_CATALOG: readonly NodeSpec[] = [
  {
    id: NODE_TYPES.trainDataset,
    label: 'Train Dataset',
    family: 'data',
    icon: 'database',
    description: 'Training images: paired LQ/GT folders, or GT only with on-the-fly degradations.',
    owns: ['datasets.train'],
    inputs: [
      { key: 'degradation', label: 'Degradations', type: 'Degradation', required: true, showWhen: { key: 'type', values: ['otf'] } },
    ],
    outputs: [{ key: 'data', label: 'Train data', type: 'TrainData' }],
    fields: [
      {
        key: 'type',
        label: 'Dataset type',
        help: 'Paired: you provide LQ and GT folders with matching file names. OTF: GT only, LQ is degraded on the fly (sets model_type = otf).',
        tomlKey: 'datasets.train.type',
        widget: {
          kind: 'select',
          options: [
            { value: 'paired', label: 'Paired LQ / GT', hint: 'Pre-made pairs, exact control over degradations' },
            { value: 'otf', label: 'On-the-fly (OTF)', hint: 'GT only; Real-ESRGAN style two-stage degradations' },
          ],
        },
        default: 'paired',
        required: true,
        level: 'essentials',
      },
      {
        key: 'dataroot_gt',
        label: 'GT folder',
        help: 'High-resolution ground truth. Scanned recursively; LMDB folders work too.',
        tomlKey: 'datasets.train.dataroot_gt',
        widget: { kind: 'path', mode: 'directory' },
        default: '',
        required: true,
        level: 'essentials',
      },
      {
        key: 'dataroot_lq',
        label: 'LQ folder',
        help: 'Low-quality inputs; every GT file needs the same relative path here and must be exactly scale× smaller.',
        tomlKey: 'datasets.train.dataroot_lq',
        widget: { kind: 'path', mode: 'directory' },
        default: '',
        required: true,
        level: 'essentials',
        showWhen: { key: 'type', values: ['paired'] },
      },
      {
        key: 'patch_size',
        label: 'LQ patch size',
        help: 'Crop size of the LQ input in pixels; the GT crop is patch_size × scale. Window-attention transformers need a multiple of their window size.',
        tomlKey: 'datasets.train.patch_size',
        widget: { kind: 'number', min: 8, step: 8, unit: 'px' },
        default: 64,
        required: true,
        level: 'essentials',
        phaseAware: true,
        suggestions: [
          { label: '32', value: 32, why: 'window transformers (HAT, DAT, ATD…)' },
          { label: '48', value: 48 },
          { label: '64', value: 64, why: 'template default for conv nets' },
          { label: '96', value: 96, why: 'more context, more VRAM' },
        ],
      },
      {
        key: 'gt_crop',
        label: 'GT crop',
        help: 'Derived: patch_size × scale.',
        tomlKey: '',
        widget: { kind: 'readout', compute: 'gt_crop', unit: 'px' },
        default: undefined,
        level: 'essentials',
      },
      {
        key: 'batch_size',
        label: 'Batch size',
        help: 'Images per micro-batch per process. Templates use 8.',
        tomlKey: 'datasets.train.batch_size',
        widget: { kind: 'number', min: 1, step: 1 },
        default: 8,
        required: true,
        level: 'essentials',
        phaseAware: true,
        suggestions: [
          { label: '4', value: 4, why: 'low VRAM' },
          { label: '8', value: 8, why: 'template default' },
          { label: '16', value: 16, why: 'large GPU' },
        ],
      },
      {
        key: 'accumulate',
        label: 'Gradient accumulation',
        help: 'Micro-batches per optimizer update. Effective batch = batch_size × accumulate. Does not reduce VRAM per micro-batch.',
        tomlKey: 'datasets.train.accumulate',
        widget: { kind: 'number', min: 1, step: 1 },
        default: 1,
        level: 'advanced',
        phaseAware: true,
      },
      {
        key: 'effective_batch',
        label: 'Effective batch',
        help: 'Derived: batch_size × accumulate.',
        tomlKey: '',
        widget: { kind: 'readout', compute: 'effective_batch' },
        default: undefined,
        level: 'essentials',
      },
      {
        key: 'augmentation',
        label: 'Augmentations',
        help: 'One entry is drawn per batch by weight. CutBlur suits real-world SR and may hurt bicubic-only training.',
        tomlKey: 'datasets.train.augmentation',
        widget: { kind: 'multiselect', options: AUGMENTATIONS.map((a) => ({ value: a.value, label: a.label, hint: a.description })) },
        default: AUGMENTATIONS.map((a) => a.value),
        level: 'advanced',
        group: 'Augmentation',
      },
      {
        key: 'aug_prob',
        label: 'Augmentation weights',
        help: 'Relative weights, same length and order as the augmentation list. 0 disables an entry.',
        tomlKey: 'datasets.train.aug_prob',
        widget: { kind: 'numberlist', step: 0.05 },
        default: AUGMENTATIONS.map((a) => a.templateProb),
        level: 'advanced',
        group: 'Augmentation',
      },
      {
        key: 'use_hflip',
        label: 'Horizontal flip',
        help: 'Random horizontal flips.',
        tomlKey: 'datasets.train.use_hflip',
        widget: { kind: 'toggle' },
        default: true,
        level: 'advanced',
        group: 'Augmentation',
      },
      {
        key: 'use_rot',
        label: 'Rotation',
        help: 'Random vertical flip and transpose.',
        tomlKey: 'datasets.train.use_rot',
        widget: { kind: 'toggle' },
        default: true,
        level: 'advanced',
        group: 'Augmentation',
      },
      {
        key: 'num_worker_per_gpu',
        label: 'Data workers',
        help: 'Loader processes per GPU. Default 4; "auto" uses all CPU threads.',
        tomlKey: 'datasets.train.num_worker_per_gpu',
        widget: { kind: 'number', min: 0, step: 1, optional: true },
        default: 4,
        level: 'advanced',
        group: 'Loader',
      },
      {
        key: 'color',
        label: 'Color mode',
        help: '"y" trains on the luma channel only (arch in_ch must be 1).',
        tomlKey: 'datasets.train.color',
        widget: { kind: 'select', options: [{ value: 'rgb', label: 'RGB' }, { value: 'y', label: 'Y (grayscale)' }] },
        default: 'rgb',
        level: 'advanced',
        group: 'Loader',
      },
      {
        key: 'meta_info',
        label: 'Meta info file',
        help: 'Optional text file listing image names; catches filename mismatches and skips the folder scan.',
        tomlKey: 'datasets.train.meta_info',
        widget: { kind: 'path', mode: 'file', extensions: ['txt'] },
        default: '',
        level: 'advanced',
        group: 'Loader',
      },
      {
        key: 'queue_size',
        label: 'OTF queue size',
        help: 'Pool of degraded pairs; must be ≥ batch_size.',
        tomlKey: 'datasets.train.queue_size',
        widget: { kind: 'number', min: 1, step: 1 },
        default: 180,
        level: 'advanced',
        group: 'Loader',
        showWhen: { key: 'type', values: ['otf'] },
      },
      {
        key: 'dataset_enlarge_ratio',
        label: 'Dataset enlarge ratio',
        help: 'Repeats the dataset per epoch (affects epoch length only).',
        tomlKey: 'datasets.train.dataset_enlarge_ratio',
        widget: { kind: 'number', min: 1, step: 1 },
        default: 1,
        level: 'native',
        group: 'Loader',
      },
      {
        key: 'pin_memory',
        label: 'Pin memory',
        help: 'Pinned host memory for faster transfers.',
        tomlKey: 'datasets.train.pin_memory',
        widget: { kind: 'toggle' },
        default: true,
        level: 'native',
        group: 'Loader',
      },
    ],
  },
  {
    id: NODE_TYPES.degradation,
    label: 'OTF Degradations',
    family: 'data',
    icon: 'waves',
    description: 'Two-stage Real-ESRGAN style degradation pipeline applied on the fly (blur → resize → noise → JPEG, twice, then sinc).',
    owns: ['degradations'],
    inputs: [],
    outputs: [{ key: 'degradation', label: 'Degradations', type: 'Degradation' }],
    fields: degradationFields,
  },
  {
    id: NODE_TYPES.validation,
    label: 'Validation',
    family: 'data',
    icon: 'check-image',
    description: 'A small paired set (8–10 varied images) evaluated every val_freq updates with metrics and saved images.',
    owns: ['datasets.val', 'val'],
    inputs: [],
    outputs: [{ key: 'val', label: 'Validation', type: 'ValSpec' }],
    fields: [
      {
        key: 'datasets',
        label: 'Validation sets',
        help: 'Named sets become [datasets.val], [datasets.val_1], … Paired sets get metrics; "single" sets are visual-only probes of real inputs.',
        tomlKey: 'datasets.val',
        widget: { kind: 'custom', editor: 'val-datasets' },
        default: [],
        required: true,
        level: 'essentials',
      },
      {
        key: 'val_freq',
        label: 'Validate every',
        help: 'Updates between validations. Each validation costs time; 1000 is the template default.',
        tomlKey: 'val.val_freq',
        widget: { kind: 'number', min: 1, step: 100, unit: 'updates' },
        default: 1000,
        required: true,
        level: 'essentials',
        suggestions: [
          { label: '1000', value: 1000, why: 'template default' },
          { label: '2500', value: 2500 },
          { label: '5000', value: 5000, why: 'long runs' },
        ],
      },
      {
        key: 'metrics',
        label: 'Metrics',
        help: 'PSNR/SSIM higher is better, DISTS lower is better, TOPIQ higher. Metrics are loose guides; inspect the images.',
        tomlKey: 'val.metrics',
        widget: { kind: 'multiselect', options: METRIC_TYPES.map((m) => ({ value: m.name, label: m.label, hint: m.description })) },
        default: ['psnr', 'ssim', 'dists'],
        level: 'essentials',
      },
      {
        key: 'save_img',
        label: 'Save images',
        help: 'Writes experiments/<name>/visualization/<img>/<img>_<update>.png.',
        tomlKey: 'val.save_img',
        widget: { kind: 'toggle' },
        default: true,
        level: 'advanced',
      },
      {
        key: 'tile',
        label: 'Tile size',
        help: 'Tiled inference for large validation images (avoids OOM). Leave empty for whole-image.',
        tomlKey: 'val.tile',
        widget: { kind: 'number', min: 0, step: 8, unit: 'px', optional: true },
        default: undefined,
        level: 'advanced',
        suggestions: [{ label: '200', value: 200, why: 'wiki suggestion' }],
      },
      {
        key: 'tile_overlap',
        label: 'Tile overlap',
        help: 'Context kept around each tile; tile must be > 2 × overlap.',
        tomlKey: 'val.tile_overlap',
        widget: { kind: 'number', min: 0, step: 4, unit: 'px' },
        default: 16,
        level: 'advanced',
      },
      {
        key: 'save_lq',
        label: 'Save LQ once',
        help: 'Also writes the LQ input next to the outputs.',
        tomlKey: 'val.save_lq',
        widget: { kind: 'toggle' },
        default: true,
        level: 'native',
      },
    ],
  },
  {
    id: NODE_TYPES.generator,
    label: 'Generator',
    family: 'model',
    icon: 'cpu',
    description: 'The super-resolution network (network_g) and the run scale.',
    owns: ['network_g', 'scale'],
    inputs: [],
    outputs: [{ key: 'model', label: 'Model', type: 'Model' }],
    fields: [
      {
        key: 'scale',
        label: 'Scale',
        help: 'Upscale factor. ×1 = restoration only. NeoSR injects it into the architecture; never set the arch scale kwarg yourself.',
        tomlKey: 'scale',
        widget: {
          kind: 'select',
          options: [
            { value: '1', label: '×1 (restore)' },
            { value: '2', label: '×2' },
            { value: '3', label: '×3' },
            { value: '4', label: '×4' },
            { value: '8', label: '×8' },
          ],
        },
        default: '4',
        required: true,
        level: 'essentials',
      },
      {
        key: 'type',
        label: 'Architecture',
        help: 'Compact/SPAN are fast and stable; PLKSR/MoSR balanced; HAT/DAT/DRCT heavy transformers (high VRAM, patch 32).',
        tomlKey: 'network_g.type',
        widget: { kind: 'select', options: archOptions },
        default: 'span',
        required: true,
        level: 'essentials',
        resets: ['params'],
      },
      {
        key: 'params',
        label: 'Architecture options',
        help: 'Constructor arguments that differ from the defaults. Unknown keys are rejected by NeoSR.',
        tomlKey: 'network_g',
        widget: { kind: 'custom', editor: 'arch-params' },
        default: {},
        level: 'advanced',
      },
    ],
  },
  {
    id: NODE_TYPES.discriminator,
    label: 'Discriminator',
    family: 'model',
    icon: 'shield',
    description: 'Adversarial critic (network_d). Required by, and only by, a gan_loss row.',
    owns: ['network_d'],
    inputs: [],
    outputs: [{ key: 'dmodel', label: 'Discriminator', type: 'DModel' }],
    fields: [
      {
        key: 'type',
        label: 'Architecture',
        help: 'metagan is used by every shipped template.',
        tomlKey: 'network_d.type',
        widget: { kind: 'select', options: discriminatorOptions },
        default: 'metagan',
        required: true,
        level: 'essentials',
        resets: ['params'],
      },
      {
        key: 'params',
        label: 'Options',
        help: 'Constructor arguments that differ from the defaults.',
        tomlKey: 'network_d',
        widget: { kind: 'custom', editor: 'arch-params' },
        default: {},
        level: 'advanced',
      },
    ],
  },
  {
    id: NODE_TYPES.initialization,
    label: 'Initialization',
    family: 'model',
    icon: 'download',
    description: 'Where the first phase starts: random weights, a pretrained .pth, or a resumable .state.',
    owns: ['path'],
    inputs: [],
    outputs: [{ key: 'init', label: 'Init', type: 'Init' }],
    fields: [
      {
        key: 'mode',
        label: 'Start from',
        help: 'Pretrained weights speed training up a lot; a .state resumes an interrupted run exactly (optimizer included).',
        tomlKey: '',
        widget: {
          kind: 'select',
          options: [
            { value: 'scratch', label: 'Scratch (random init)' },
            { value: 'pretrain', label: 'Pretrained weights (.pth)' },
            { value: 'resume', label: 'Resume training state (.state)' },
          ],
        },
        default: 'scratch',
        required: true,
        level: 'essentials',
      },
      {
        key: 'pretrain_network_g',
        label: 'Generator weights',
        help: 'net_g_*.pth of the same arch (ideally same scale). Use strict_load off for a different scale.',
        tomlKey: 'path.pretrain_network_g',
        widget: { kind: 'path', mode: 'file', extensions: ['pth', 'safetensors'] },
        default: '',
        required: true,
        level: 'essentials',
        showWhen: { key: 'mode', values: ['pretrain'] },
      },
      {
        key: 'pretrain_network_d',
        label: 'Discriminator weights',
        help: 'Optional net_d_*.pth when continuing a GAN run.',
        tomlKey: 'path.pretrain_network_d',
        widget: { kind: 'path', mode: 'file', extensions: ['pth'] },
        default: '',
        level: 'advanced',
        showWhen: { key: 'mode', values: ['pretrain'] },
      },
      {
        key: 'strict_load_g',
        label: 'Strict load',
        help: 'Off skips tensors with mismatched shapes (different scale or slightly different arch).',
        tomlKey: 'path.strict_load_g',
        widget: { kind: 'toggle' },
        default: true,
        level: 'advanced',
        showWhen: { key: 'mode', values: ['pretrain'] },
      },
      {
        key: 'resume_state',
        label: 'Training state',
        help: 'experiments/<name>/training_states/<update>.state. Pretrain paths are ignored while resuming.',
        tomlKey: 'path.resume_state',
        widget: { kind: 'path', mode: 'file', extensions: ['state'] },
        default: '',
        required: true,
        level: 'essentials',
        showWhen: { key: 'mode', values: ['resume'] },
      },
      {
        key: 'param_key_g',
        label: 'Param key',
        help: 'Dict key inside the .pth; auto-detects params_ema / params.',
        tomlKey: 'path.param_key_g',
        widget: { kind: 'text', placeholder: 'auto' },
        default: '',
        level: 'native',
        showWhen: { key: 'mode', values: ['pretrain'] },
      },
    ],
  },
  {
    id: NODE_TYPES.losses,
    label: 'Loss Recipe',
    family: 'objective',
    icon: 'target',
    description: 'The training objective: a list of loss terms with weights, optionally different per phase.',
    owns: ['train.losses'],
    inputs: [],
    outputs: [{ key: 'objective', label: 'Objective', type: 'Objective' }],
    fields: [
      {
        key: 'losses',
        label: 'Loss terms',
        help: 'Each row becomes [train.<slot>_opt]. A gan_loss row requires a Discriminator and an Optimizer D.',
        tomlKey: 'train',
        widget: { kind: 'custom', editor: 'losses' },
        default: [],
        required: true,
        level: 'essentials',
        phaseAware: true,
      },
    ],
  },
  {
    id: NODE_TYPES.optimizerG,
    label: 'Optimizer G',
    family: 'optimization',
    icon: 'sliders',
    description: 'Generator optimizer (train.optim_g).',
    owns: ['train.optim_g'],
    inputs: [],
    outputs: [{ key: 'optim', label: 'Optimizer G', type: 'OptimG' }],
    fields: optimizerFields('g'),
  },
  {
    id: NODE_TYPES.optimizerD,
    label: 'Optimizer D',
    family: 'optimization',
    icon: 'sliders',
    description: 'Discriminator optimizer (train.optim_d). Needed only with a gan_loss.',
    owns: ['train.optim_d'],
    inputs: [],
    outputs: [{ key: 'optim', label: 'Optimizer D', type: 'OptimD' }],
    fields: optimizerFields('d'),
  },
  {
    id: NODE_TYPES.schedule,
    label: 'LR Schedule',
    family: 'optimization',
    icon: 'trending-down',
    description: 'One learning-rate scheduler applied to both optimizers. Redundant with schedule-free optimizers.',
    owns: ['train.scheduler'],
    inputs: [],
    outputs: [{ key: 'schedule', label: 'Schedule', type: 'Schedule' }],
    fields: [
      {
        key: 'type',
        label: 'Scheduler',
        help: 'MultiStepLR multiplies the LR by gamma at each milestone; CosineAnnealing decays smoothly to eta_min.',
        tomlKey: 'train.scheduler.type',
        widget: { kind: 'select', options: SCHEDULERS.map((s) => ({ value: s.type, label: s.label, hint: s.description })) },
        default: 'multisteplr',
        required: true,
        level: 'essentials',
      },
      {
        key: 'milestones',
        label: 'Milestones',
        help: 'Updates at which the LR is multiplied by gamma.',
        tomlKey: 'train.scheduler.milestones',
        widget: { kind: 'numberlist', step: 1000 },
        default: [60000, 120000],
        level: 'essentials',
        showWhen: { key: 'type', values: ['multisteplr'] },
      },
      {
        key: 'gamma',
        label: 'Gamma',
        help: 'LR multiplier at each milestone.',
        tomlKey: 'train.scheduler.gamma',
        widget: { kind: 'number', min: 0, max: 1, step: 0.05 },
        default: 0.5,
        level: 'essentials',
        showWhen: { key: 'type', values: ['multisteplr'] },
      },
      {
        key: 'T_max',
        label: 'T_max',
        help: 'Length of the cosine cycle in updates (usually total iterations).',
        tomlKey: 'train.scheduler.T_max',
        widget: { kind: 'number', min: 1, step: 1000, unit: 'updates' },
        default: 350000,
        level: 'essentials',
        showWhen: { key: 'type', values: ['cosineannealing'] },
      },
      {
        key: 'eta_min',
        label: 'Minimum LR',
        help: 'LR at the end of the cosine cycle.',
        tomlKey: 'train.scheduler.eta_min',
        widget: { kind: 'number', min: 0, step: 0.00001 },
        default: 0.00004,
        level: 'essentials',
        showWhen: { key: 'type', values: ['cosineannealing'] },
      },
    ],
  },
  {
    id: NODE_TYPES.runtime,
    label: 'Runtime',
    family: 'run',
    icon: 'gpu',
    description: 'Precision, compilation, seed and devices (top-level TOML keys).',
    owns: ['runtime'],
    singleton: true,
    inputs: [],
    outputs: [{ key: 'runtime', label: 'Runtime', type: 'Runtime' }],
    fields: [
      {
        key: 'precision',
        label: 'Precision',
        help: 'AMP fp16 on Turing or newer; bf16 on Ampere or newer (faster, occasionally unstable). fp32 is the safe reference.',
        tomlKey: 'use_amp',
        widget: {
          kind: 'select',
          options: [
            { value: 'fp32', label: 'fp32', hint: 'slowest, most stable' },
            { value: 'amp', label: 'AMP fp16', hint: 'Turing+' },
            { value: 'bf16', label: 'AMP bf16', hint: 'Ampere+, template default' },
          ],
        },
        default: 'bf16',
        required: true,
        level: 'essentials',
      },
      {
        key: 'fast_matmul',
        label: 'Fast matmul (TF32)',
        help: 'Reduced-precision matmul on Ampere+. Template default on.',
        tomlKey: 'fast_matmul',
        widget: { kind: 'toggle' },
        default: true,
        level: 'essentials',
      },
      {
        key: 'compile',
        label: 'torch.compile',
        help: 'Compiles the generator. Linux only (needs Triton); errors on Windows.',
        tomlKey: 'compile',
        widget: { kind: 'toggle' },
        default: false,
        level: 'advanced',
        support: 'native',
      },
      {
        key: 'compile_mode',
        label: 'Compile mode',
        help: 'torch.compile mode.',
        tomlKey: 'compile_mode',
        widget: {
          kind: 'select',
          options: [
            { value: 'default', label: 'default' },
            { value: 'reduce-overhead', label: 'reduce-overhead' },
            { value: 'max-autotune', label: 'max-autotune' },
          ],
        },
        default: 'default',
        level: 'native',
        showWhen: { key: 'compile', values: [true] },
      },
      {
        key: 'manual_seed',
        label: 'Seed',
        help: 'Fixed seed enables deterministic algorithms (slower). Empty = random seed, logged.',
        tomlKey: 'manual_seed',
        widget: { kind: 'number', min: 0, step: 1, optional: true },
        default: undefined,
        level: 'advanced',
      },
      {
        key: 'num_gpu',
        label: 'GPUs',
        help: 'Leave empty for auto. More than one needs `--launcher pytorch` (torchrun).',
        tomlKey: 'num_gpu',
        widget: { kind: 'number', min: 1, step: 1, optional: true },
        default: undefined,
        level: 'advanced',
      },
    ],
  },
  {
    id: NODE_TYPES.plan,
    label: 'Training Plan',
    family: 'run',
    icon: 'play',
    description: 'Ties everything together: run name, phases, EMA, checkpoints and logging. Compiles to one TOML per phase.',
    owns: ['name', 'train.scalars', 'logger'],
    singleton: true,
    inputs: [
      { key: 'data', label: 'Train data', type: 'TrainData', required: true },
      { key: 'val', label: 'Validation', type: 'ValSpec' },
      { key: 'model', label: 'Model', type: 'Model', required: true },
      { key: 'discriminator', label: 'Discriminator', type: 'DModel' },
      { key: 'init', label: 'Init', type: 'Init' },
      { key: 'losses', label: 'Objective', type: 'Objective', required: true },
      { key: 'optimG', label: 'Optimizer G', type: 'OptimG', required: true },
      { key: 'optimD', label: 'Optimizer D', type: 'OptimD' },
      { key: 'schedule', label: 'Schedule', type: 'Schedule' },
      { key: 'runtime', label: 'Runtime', type: 'Runtime', required: true },
    ],
    outputs: [],
    fields: [
      {
        key: 'name',
        label: 'Run name',
        help: 'Folder under experiments/. Convention: scale prefix, e.g. 4x_mymodel. "debug" in the name forces debug frequencies.',
        tomlKey: 'name',
        widget: { kind: 'text', placeholder: '4x_mymodel' },
        default: '4x_mymodel',
        required: true,
        level: 'essentials',
      },
      {
        key: 'total_iter',
        label: 'Total updates',
        help: 'Derived: sum of all phases. NeoSR counts optimizer updates, not epochs.',
        tomlKey: 'logger.total_iter',
        widget: { kind: 'readout', compute: 'total_iter', unit: 'updates' },
        default: undefined,
        level: 'essentials',
      },
      {
        key: 'save_checkpoint_freq',
        label: 'Checkpoint every',
        help: 'Required by NeoSR. Each checkpoint writes net_g, net_d and a .state; keep_states limits the states kept.',
        tomlKey: 'logger.save_checkpoint_freq',
        widget: { kind: 'number', min: 1, step: 500, unit: 'updates' },
        default: 1000,
        required: true,
        level: 'essentials',
        suggestions: [
          { label: '1000', value: 1000, why: 'template default' },
          { label: '2500', value: 2500 },
          { label: '5000', value: 5000, why: 'long runs, less disk' },
        ],
      },
      {
        key: 'ema',
        label: 'EMA decay',
        help: 'Exponential moving average of the generator used for validation and export. 0.999 recommended, especially with schedule-free optimizers. 0 = off.',
        tomlKey: 'train.ema',
        widget: { kind: 'number', min: 0, max: 1, step: 0.001 },
        default: 0.999,
        level: 'essentials',
      },
      {
        key: 'ema_power',
        label: 'EMA power',
        help: 'Ramps the decay up over the first updates (ema_pytorch style). Templates use 0.75.',
        tomlKey: 'train.ema_power',
        widget: { kind: 'number', min: 0.01, max: 1, step: 0.05 },
        default: 0.75,
        level: 'advanced',
      },
      {
        key: 'use_tb_logger',
        label: 'TensorBoard logger',
        help: 'Writes experiments/tb_logger/<name>. Studio charts read the same scalars.',
        tomlKey: 'logger.use_tb_logger',
        widget: { kind: 'toggle' },
        default: true,
        level: 'essentials',
        group: 'Logging',
      },
      {
        key: 'save_tb_img',
        label: 'Validation images to TensorBoard',
        help: 'Slow and large; the visualization folder is written anyway.',
        tomlKey: 'logger.save_tb_img',
        widget: { kind: 'toggle' },
        default: false,
        level: 'advanced',
        group: 'Logging',
      },
      {
        key: 'print_freq',
        label: 'Log every',
        help: 'Updates between log lines.',
        tomlKey: 'logger.print_freq',
        widget: { kind: 'number', min: 1, step: 10, unit: 'updates' },
        default: 100,
        level: 'advanced',
        group: 'Logging',
      },
      {
        key: 'keep_states',
        label: 'Keep training states',
        help: 'Newest .state files kept; 0 keeps all.',
        tomlKey: 'logger.keep_states',
        widget: { kind: 'number', min: 0, step: 1 },
        default: 3,
        level: 'advanced',
        group: 'Logging',
      },
      {
        key: 'grad_clip',
        label: 'Gradient clipping',
        help: 'Clips each network’s gradient norm to 1.0.',
        tomlKey: 'train.grad_clip',
        widget: { kind: 'toggle' },
        default: true,
        level: 'advanced',
        group: 'Stability',
      },
      {
        key: 'clamp',
        label: 'Clamp output',
        help: 'Clamps the generator output to [1/255, 1] before the losses.',
        tomlKey: 'train.clamp',
        widget: { kind: 'toggle' },
        default: true,
        level: 'native',
        group: 'Stability',
      },
      {
        key: 'max_skip_streak',
        label: 'Max skipped updates',
        help: 'Consecutive non-finite updates before the run stops.',
        tomlKey: 'train.max_skip_streak',
        widget: { kind: 'number', min: 1, step: 1 },
        default: 10,
        level: 'native',
        group: 'Stability',
      },
      {
        key: 'wandb_project',
        label: 'W&B project',
        help: 'Optional Weights & Biases project (needs the TensorBoard logger).',
        tomlKey: 'logger.wandb.project',
        widget: { kind: 'text', placeholder: 'none' },
        default: '',
        level: 'native',
        group: 'Logging',
      },
    ],
  },
];

export const NODE_BY_ID: ReadonlyMap<string, NodeSpec> = new Map(NODE_CATALOG.map((n) => [n.id, n]));

export function nodeSpec(id: string): NodeSpec {
  const spec = NODE_BY_ID.get(id);
  if (!spec) {
    throw new Error(`Unknown node type: ${id}`);
  }
  return spec;
}

/** Objective presets used by the wizard and the loss editor. */
export const OBJECTIVES = [
  {
    id: 'fidelity',
    label: 'Fidelity',
    description: 'Pixel and structure losses only. Best PSNR, smoothest output. Good first phase and for ×1 restoration.',
  },
  {
    id: 'balanced',
    label: 'Balanced',
    description: 'Template losses without GAN: MS-SSIM + consistency + LDL + FDL. Sharper than fidelity, no hallucination.',
  },
  {
    id: 'perceptual',
    label: 'Perceptual',
    description: 'The full NeoSR template recipe with GAN (metagan). Most texture, needs a discriminator and careful review.',
  },
] as const;

export type ObjectiveId = (typeof OBJECTIVES)[number]['id'];

/** Loss rows (type → weight) that an objective preset enables. */
export function objectiveLosses(objective: ObjectiveId): { type: string; slot: string; weight: number }[] {
  return LOSSES.filter((l) => l.presets[objective] !== undefined).map((l) => ({
    type: l.type,
    slot: l.slot,
    weight: l.presets[objective] as number,
  }));
}
