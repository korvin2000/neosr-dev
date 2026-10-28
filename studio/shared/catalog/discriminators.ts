import type { DiscriminatorInfo } from './data-types';

/** `[network_d]` types (design/neosr_config_surface.md section 5.2). Discriminators take no scale. */
export const DISCRIMINATORS: readonly DiscriminatorInfo[] = [
  {
    type: 'unet',
    label: 'U-Net',
    description: 'U-Net discriminator with spectral normalization.',
    params: [
      { key: 'num_in_ch', type: 'int', default: 3 },
      { key: 'num_feat', type: 'int', default: 64 },
      { key: 'skip_connection', type: 'bool', default: true },
    ],
  },
  {
    type: 'patchgan',
    label: 'PatchGAN',
    description: 'Patch-based discriminator with optional spectral norm and a configurable depth.',
    params: [
      { key: 'num_in_ch', type: 'int', default: 3 },
      { key: 'num_feat', type: 'int', default: 64 },
      { key: 'num_layers', type: 'int', default: 3 },
      { key: 'max_nf_mult', type: 'int', default: 8 },
      { key: 'norm_type', type: 'str', default: 'none' },
      { key: 'use_sigmoid', type: 'bool', default: false },
      { key: 'use_sn', type: 'bool', default: true },
    ],
  },
  {
    type: 'resgan',
    label: 'ResGAN',
    description: 'Residual discriminator with an optional attention block.',
    params: [
      { key: 'num_in_ch', type: 'int', default: 3 },
      { key: 'num_feat', type: 'int', default: 64 },
      { key: 'skip_connection', type: 'bool', default: true },
      { key: 'upscale', type: 'bool', default: true },
      { key: 'attention', type: 'bool', default: false },
    ],
  },
  {
    type: 'metagan',
    label: 'MetaGAN',
    description: 'Multi-stage discriminator (dims 48 to 288); the default in every shipped train template.',
    params: [
      { key: 'in_ch', type: 'int', default: 3 },
      { key: 'n_class', type: 'int', default: 1 },
      { key: 'dims', type: 'list', default: [48, 96, 192, 288] },
      { key: 'blocks', type: 'list', default: [3, 3, 9, 3] },
      { key: 'downs', type: 'list', default: [4, 4, 2, 2] },
      { key: 'drop_path', type: 'float', default: 0.02 },
      { key: 'end_drop', type: 'float', default: 0.2 },
    ],
    note: 'template default',
  },
  {
    type: 'dunet',
    label: 'DUNet',
    description: 'U-Net style discriminator with a base width of 64.',
    params: [
      { key: 'in_ch', type: 'int', default: 3 },
      { key: 'dim', type: 'int', default: 64 },
    ],
  },
  {
    type: 'ea2fpn',
    label: 'EA2FPN',
    description: 'A2-FPN derived discriminator with a torchvision ResNet18 encoder.',
    params: [
      // (?) default class_num 6 looks segmentation-oriented; unverified for GAN use
      { key: 'class_num', type: 'int', default: 6 },
      { key: 'encoder_channels', type: 'list', default: [512, 256, 128, 64] },
      { key: 'pyramid_channels', type: 'int', default: 64 },
      { key: 'segmentation_channels', type: 'int', default: 64 },
      { key: 'dropout', type: 'float', default: 0.0 },
    ],
    note: 'downloads torchvision ResNet18 weights on first use',
  },
];
