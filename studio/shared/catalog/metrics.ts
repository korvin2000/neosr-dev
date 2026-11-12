import type { MetricTypeInfo } from './data-types';

/** `[val.metrics.<name>]` types (design/neosr_config_surface.md section 8.2); `params` are all keywords the trainer takes. */
export const METRIC_TYPES: readonly MetricTypeInfo[] = [
  {
    type: 'calculate_psnr',
    name: 'psnr',
    label: 'PSNR',
    better: 'higher',
    params: [
      { key: 'crop_border', type: 'int', default: 4, help: 'Pixels cropped from every edge before scoring; usually the scale (0: none).' },
      { key: 'test_y_channel', type: 'bool', default: false, help: 'Score the luma (Y of YCbCr) only, the usual SR benchmark convention.' },
      { key: 'input_order', type: 'str', default: 'HWC', help: 'HWC or CHW: the layout of numpy inputs; validation passes tensors, so it only has to be valid.' },
    ],
    description: 'Peak signal-to-noise ratio between the output and the ground truth.',
  },
  {
    type: 'calculate_ssim',
    name: 'ssim',
    label: 'SSIM',
    better: 'higher',
    params: [
      { key: 'crop_border', type: 'int', default: 4, help: 'Pixels cropped from every edge before scoring; usually the scale (0: none).' },
      { key: 'test_y_channel', type: 'bool', default: false, help: 'Score the luma (Y of YCbCr) only, the usual SR benchmark convention.' },
      { key: 'input_order', type: 'str', default: 'HWC', help: 'HWC or CHW: the layout of numpy inputs; validation passes tensors, so it only has to be valid.' },
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
    params: [{ key: 'dataset', type: 'str', default: 'kadid', options: ['kadid', 'pipal'], help: 'Weights: kadid (KADID-10k) or pipal (trained on SR and GAN outputs).' }],
    description: 'TOPIQ learned image-quality score; higher is better.',
  },
  {
    type: 'calculate_lpips',
    name: 'lpips',
    label: 'LPIPS',
    better: 'lower',
    params: [{ key: 'net', type: 'str', default: 'alex', options: ['alex', 'vgg'], help: 'LPIPS v0.1 backbone: AlexNet or VGG16.' }],
    description: 'LPIPS v0.1 learned perceptual distance; downloads weights on first use, lower is better.',
  },
];
