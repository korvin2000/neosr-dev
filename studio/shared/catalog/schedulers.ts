import type { SchedulerInfo } from './data-types';

/** `[train.scheduler]` types (design/neosr_config_surface.md section 6.3). One table serves both G and D. */
export const SCHEDULERS: readonly SchedulerInfo[] = [
  {
    type: 'multisteplr',
    label: 'MultiStepLR',
    description: 'Multiplies the learning rate by gamma at each milestone (counted in updates).',
    params: [
      // (?) required; the reference lists no default (wiki example: [60000, 120000])
      { key: 'milestones', type: 'list', default: null },
      { key: 'gamma', type: 'float', default: 0.1 },
    ],
  },
  {
    type: 'cosineannealing',
    label: 'Cosine annealing',
    description: 'Decays the learning rate along a cosine curve over T_max updates.',
    params: [
      // (?) required; the reference lists no default (wiki example: 350000)
      { key: 'T_max', type: 'int', default: null },
      { key: 'eta_min', type: 'float', default: 0 },
    ],
  },
];
