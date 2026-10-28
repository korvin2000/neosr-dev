import type { OptimizerInfo } from './data-types';

/** `[train.optim_g]` / `[train.optim_d]` types (design/neosr_config_surface.md section 6.2). */
export const OPTIMIZERS: readonly OptimizerInfo[] = [
  {
    type: 'adam',
    label: 'Adam',
    description: 'torch.optim.Adam; any torch keyword is accepted.',
    scheduleFree: false,
    params: [
      // (?) torch.optim default; the reference lists no default values for torch optimizers
      { key: 'lr', type: 'float', default: 1e-3 },
      { key: 'betas', type: 'list', default: [0.9, 0.999] },
      { key: 'eps', type: 'float', default: 1e-8 },
      { key: 'weight_decay', type: 'float', default: 0.0 },
      { key: 'amsgrad', type: 'bool', default: false },
      { key: 'foreach', type: 'bool', default: null },
      { key: 'fused', type: 'bool', default: null },
    ],
  },
  {
    type: 'adamw',
    label: 'AdamW',
    description: 'torch.optim.AdamW with decoupled weight decay; used by the Aether templates.',
    scheduleFree: false,
    params: [
      // (?) torch.optim default; the reference lists no default values for torch optimizers
      { key: 'lr', type: 'float', default: 1e-3 },
      { key: 'betas', type: 'list', default: [0.9, 0.999] },
      { key: 'eps', type: 'float', default: 1e-8 },
      { key: 'weight_decay', type: 'float', default: 0.01 },
      { key: 'amsgrad', type: 'bool', default: false },
      { key: 'foreach', type: 'bool', default: null },
      { key: 'fused', type: 'bool', default: null },
    ],
    template: {
      g: { lr: 2e-4, betas: [0.9, 0.99], weight_decay: 0 },
      d: { lr: 1e-4 },
    },
  },
  {
    type: 'nadam',
    label: 'NAdam',
    description: 'torch.optim.NAdam: Adam with Nesterov momentum.',
    scheduleFree: false,
    params: [
      // (?) torch.optim default; the reference lists no default values for torch optimizers
      { key: 'lr', type: 'float', default: 2e-3 },
      { key: 'betas', type: 'list', default: [0.9, 0.999] },
      { key: 'eps', type: 'float', default: 1e-8 },
      { key: 'weight_decay', type: 'float', default: 0.0 },
      { key: 'momentum_decay', type: 'float', default: 4e-3 },
    ],
  },
  {
    type: 'adan',
    label: 'Adan',
    description: 'Adan optimizer (adaptive Nesterov momentum) with a fixed base learning rate; not schedule-free.',
    scheduleFree: false,
    params: [
      { key: 'lr', type: 'float', default: 5e-4 },
      { key: 'betas', type: 'list', default: [0.98, 0.92, 0.99] },
      { key: 'eps', type: 'float', default: 1e-8 },
      { key: 'weight_decay', type: 'float', default: 0.02 },
      { key: 'max_grad_norm', type: 'float', default: 0.0 },
      { key: 'no_prox', type: 'bool', default: true },
      { key: 'foreach', type: 'bool', default: true },
    ],
  },
  {
    type: 'adamw_sf',
    label: 'AdamW schedule-free',
    description: 'Schedule-free AdamW; needs no LR scheduler and supports a linear warmup.',
    scheduleFree: true,
    params: [
      { key: 'lr', type: 'float', default: 8e-4 },
      { key: 'betas', type: 'list', default: [0.9, 0.99] },
      { key: 'eps', type: 'float', default: 1e-8 },
      { key: 'weight_decay', type: 'float', default: 0.0 },
      { key: 'warmup_steps', type: 'int', default: 0 },
      { key: 'r', type: 'float', default: 0.0 },
      { key: 'weight_lr_power', type: 'float', default: 2.0 },
      // must stay true for adamw_sf
      { key: 'schedule_free', type: 'bool', default: true },
      { key: 'foreach', type: 'bool', default: true },
    ],
  },
  {
    type: 'adan_sf',
    label: 'Adan schedule-free',
    description: 'Schedule-free Adan, the optimizer in every shipped train template; schedule_free = false gives plain Adan.',
    scheduleFree: true,
    params: [
      { key: 'lr', type: 'float', default: 1.6e-3 },
      { key: 'betas', type: 'list', default: [0.98, 0.92, 0.99] },
      { key: 'eps', type: 'float', default: 1e-8 },
      { key: 'weight_decay', type: 'float', default: 0.02 },
      { key: 'max_grad_norm', type: 'float', default: 0.0 },
      { key: 'warmup_steps', type: 'int', default: 0 },
      { key: 'r', type: 'float', default: 0.0 },
      { key: 'weight_lr_power', type: 'float', default: 2.0 },
      { key: 'schedule_free', type: 'bool', default: true },
    ],
    template: {
      g: { lr: 1e-3, betas: [0.98, 0.92, 0.99], weight_decay: 0.01, warmup_steps: 1600 },
      d: { lr: 1e-4, betas: [0.98, 0.92, 0.99], weight_decay: 0.01, warmup_steps: 600 },
    },
  },
];
