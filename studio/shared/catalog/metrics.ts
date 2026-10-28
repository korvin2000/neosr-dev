import type { MetricTypeInfo } from './data-types';

/** `[val.metrics.<name>]` types (design/neosr_config_surface.md section 8.2). */
export const METRIC_TYPES: readonly MetricTypeInfo[] = [
  {
    type: 'calculate_psnr',
    name: 'psnr',
    label: 'PSNR',
    better: 'higher',
    params: [
      { key: 'crop_border', type: 'int', default: 4 },
      { key: 'input_order', type: 'str', default: 'HWC' },
      { key: 'test_y_channel', type: 'bool', default: false },
    ],
    description: 'Peak signal-to-noise ratio between the output and the ground truth.',
  },
  {
    type: 'calculate_ssim',
    name: 'ssim',
    label: 'SSIM',
    better: 'higher',
    params: [
      { key: 'crop_border', type: 'int', default: 4 },
      { key: 'input_order', type: 'str', default: 'HWC' },
      { key: 'test_y_channel', type: 'bool', default: false },
    ],
    description: 'Structural similarity; needs at least 11x11 pixels after the border crop.',
  },
  {
    type: 'calculate_dists',
    name: 'dists',
    label: 'DISTS',
    better: 'lower',
    params: [],
    description: 'Learned perceptual distance; downloads weights on first use, and better must be "lower".',
  },
  {
    type: 'calculate_topiq',
    name: 'topiq',
    label: 'TOPIQ',
    better: 'higher',
    params: [],
    description: 'TOPIQ learned image-quality score; higher is better.',
  },
];
