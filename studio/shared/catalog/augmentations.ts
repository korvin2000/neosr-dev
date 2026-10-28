import type { AugmentationInfo } from './data-types';

/** `[datasets.train] augmentation` values with the shipped aug_prob (design/neosr_config_surface.md section 4.1). */
export const AUGMENTATIONS: readonly AugmentationInfo[] = [
  {
    value: 'none',
    label: 'None',
    description: 'No batch augmentation for the drawn sample.',
    templateProb: 0.5,
  },
  {
    value: 'mixup',
    label: 'MixUp',
    description: 'Blends two samples of a batch; needs batch_size above 1.',
    templateProb: 0.1,
  },
  {
    value: 'cutmix',
    label: 'CutMix',
    description: 'Pastes a patch of one sample onto another; needs batch_size above 1.',
    templateProb: 0.1,
  },
  {
    value: 'resizemix',
    label: 'ResizeMix',
    description: 'Pastes a resized copy of one sample into another; needs batch_size above 1.',
    templateProb: 0.1,
  },
  {
    value: 'cutblur',
    label: 'CutBlur',
    description: 'Swaps a patch between the LQ and GT pair; suited to real-world SR and may hurt bicubic-only training.',
    templateProb: 0.5,
  },
];
