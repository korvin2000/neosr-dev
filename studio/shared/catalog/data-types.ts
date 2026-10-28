/**
 * Shapes of the NeoSR reference data (architectures, losses, optimizers, ...). Values come from
 * `design/neosr_config_surface.md`, extracted from this repository's `neosr/` package.
 */

export type ParamType = 'int' | 'float' | 'bool' | 'str' | 'list';

export interface ParamInfo {
  readonly key: string;
  readonly type: ParamType;
  readonly default: unknown;
  readonly help?: string;
  /** Enumerable values, when known. */
  readonly options?: readonly string[];
}

export type SizeClass = 'ultra-light' | 'light' | 'medium' | 'medium-heavy' | 'heavy' | 'ultra-heavy';

export interface ArchInfo {
  /** Exact `[network_g] type` string. */
  readonly type: string;
  /** Family / module name, e.g. `span`. */
  readonly family: string;
  /** Display label, e.g. `SPAN`. */
  readonly label: string;
  readonly description: string;
  readonly sizeClass: SizeClass;
  /** Supported training scales. */
  readonly scales: readonly number[];
  /** Window size: `patch_size` must be a multiple of it. Undefined for conv nets. */
  readonly windowSize?: number;
  /** patch_size used by the shipped template (32 for window-attention transformers, else 64). */
  readonly templatePatch: number;
  /** Generator lr used by the shipped template. */
  readonly templateLr: number;
  /** Constructor kwargs (without the scale kwarg, which NeoSR injects). */
  readonly params: readonly ParamInfo[];
  readonly tags: readonly ('conv' | 'transformer' | 'attention' | 'mamba' | 'qat' | 'testing' | 'legacy')[];
  /** Rough training-speed class for the architecture cards. */
  readonly speed: 'fast' | 'medium' | 'slow';
  /** False when `convert.py` cannot export it (craft, ditn). */
  readonly onnx: boolean;
  /** Short suitability note shown on the card, e.g. "beginner-friendly". */
  readonly note?: string;
}

export interface DiscriminatorInfo {
  readonly type: string;
  readonly label: string;
  readonly description: string;
  readonly params: readonly ParamInfo[];
  readonly note?: string;
}

export type LossCategory = 'pixel' | 'structural' | 'perceptual' | 'frequency' | 'color' | 'adversarial' | 'distribution';

export interface LossInfo {
  /** Exact `type` string, e.g. `mssim_loss`. */
  readonly type: string;
  /** Usual slot name → `[train.<slot>_opt]`. */
  readonly slot: string;
  readonly label: string;
  readonly description: string;
  readonly category: LossCategory;
  /** NeoSR reader default. */
  readonly defaultWeight: number;
  /** Weight used by the shipped templates, when the templates use it. */
  readonly templateWeight?: number;
  readonly params: readonly ParamInfo[];
  readonly requiresDiscriminator?: boolean;
  /** `target = "lq"` is meaningful (colour losses). */
  readonly supportsTargetLq?: boolean;
  readonly cost: 'low' | 'medium' | 'high';
  /** Objective presets that include this loss by default, with the weight they use. */
  readonly presets: Readonly<Partial<Record<'fidelity' | 'balanced' | 'perceptual', number>>>;
}

export interface OptimizerInfo {
  readonly type: string;
  readonly label: string;
  readonly description: string;
  readonly scheduleFree: boolean;
  readonly params: readonly ParamInfo[];
  /** Values the shipped templates use for G and D. */
  readonly template?: { readonly g: Record<string, unknown>; readonly d: Record<string, unknown> };
}

export interface SchedulerInfo {
  readonly type: string;
  readonly label: string;
  readonly description: string;
  readonly params: readonly ParamInfo[];
}

export interface MetricTypeInfo {
  /** `type` string, e.g. `calculate_psnr`. */
  readonly type: string;
  /** Usual `[val.metrics.<name>]` name. */
  readonly name: string;
  readonly label: string;
  readonly better: 'higher' | 'lower';
  readonly params: readonly ParamInfo[];
  readonly description: string;
}

export interface AugmentationInfo {
  readonly value: string;
  readonly label: string;
  readonly description: string;
  readonly templateProb: number;
}

export interface DegradationFieldInfo {
  readonly key: string;
  readonly label: string;
  readonly help: string;
  readonly type: ParamType;
  /** Value in the shipped `*_otf.toml` templates. */
  readonly default: unknown;
  readonly stage: 1 | 2 | 'final';
  readonly group: 'blur' | 'resize' | 'noise' | 'jpeg' | 'sinc' | 'general';
}
