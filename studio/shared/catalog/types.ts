/**
 * Node catalog types. The catalog is data: adding a NeoSR option is a catalog entry, not a
 * component. The same `FieldSpec` drives the wizard, the node card and the inspector.
 */

export type PortType =
  | 'TrainData'
  | 'Degradation'
  | 'ValSpec'
  | 'Model'
  | 'DModel'
  | 'Init'
  | 'Objective'
  | 'OptimG'
  | 'OptimD'
  | 'Schedule'
  | 'Runtime';

export type NodeFamily = 'data' | 'model' | 'objective' | 'optimization' | 'run';

/** How a setting reaches NeoSR. Shown as a badge so nothing is misrepresented. */
export type SupportLevel = 'native' | 'orchestrated' | 'estimate';

export interface SelectOption {
  readonly value: string;
  readonly label: string;
  readonly hint?: string;
  /** Picking this option also sets sibling fields (e.g. an optimizer preset sets betas). */
  readonly fills?: Record<string, unknown>;
}

export type Widget =
  | { readonly kind: 'text'; readonly placeholder?: string; readonly monospace?: boolean }
  | { readonly kind: 'number'; readonly min?: number; readonly max?: number; readonly step?: number; readonly unit?: string; readonly optional?: boolean }
  | { readonly kind: 'slider'; readonly min: number; readonly max: number; readonly step: number; readonly unit?: string }
  | { readonly kind: 'toggle' }
  | { readonly kind: 'select'; readonly options: readonly SelectOption[]; readonly allowCustom?: boolean }
  | { readonly kind: 'multiselect'; readonly options: readonly SelectOption[] }
  | { readonly kind: 'path'; readonly mode: 'directory' | 'file'; readonly extensions?: readonly string[] }
  | { readonly kind: 'numberlist'; readonly length?: number; readonly step?: number }
  | { readonly kind: 'keyvalue' }
  /** Rendered by a dedicated editor (loss rows, metrics, augmentation list). */
  | { readonly kind: 'custom'; readonly editor: 'losses' | 'val-datasets' | 'arch-params' }
  /** Read-only derived value, computed from sibling fields. */
  | { readonly kind: 'readout'; readonly compute: string; readonly unit?: string };

export interface FieldSpec {
  readonly key: string;
  readonly label: string;
  /** Short explanation shown as tooltip and in the wizard rail. */
  readonly help: string;
  /** NeoSR TOML path, e.g. `datasets.train.patch_size`. Empty for UI-only fields. */
  readonly tomlKey: string;
  readonly widget: Widget;
  readonly default: unknown;
  readonly required?: boolean;
  /** `essentials` is shown on the card and in the wizard; `advanced` folds; `native` is TOML-only. */
  readonly level: 'essentials' | 'advanced' | 'native';
  /** Sub-section inside the inspector. */
  readonly group?: string;
  /** Visible only when a sibling field has one of these values. */
  readonly showWhen?: { readonly key: string; readonly values: readonly unknown[] };
  /** Can be overridden per phase (loss weights, lr, patch size). */
  readonly phaseAware?: boolean;
  readonly support?: SupportLevel;
  /** Typical values offered as one-click chips. */
  readonly suggestions?: readonly { readonly label: string; readonly value: unknown; readonly why?: string }[];
  /** Sibling fields reset to their default whenever this field's value changes (arch type → params). */
  readonly resets?: readonly string[];
}

export interface PortSpec {
  readonly key: string;
  readonly label: string;
  readonly type: PortType;
  readonly required?: boolean;
  /** Input only: show this port only when a sibling field has one of these values. */
  readonly showWhen?: { readonly key: string; readonly values: readonly unknown[] };
}

export interface NodeSpec {
  readonly id: string;
  readonly label: string;
  readonly family: NodeFamily;
  readonly icon: string;
  readonly description: string;
  /** TOML sections this node owns; two enabled nodes owning the same section is a conflict. */
  readonly owns: readonly string[];
  readonly inputs: readonly PortSpec[];
  readonly outputs: readonly PortSpec[];
  readonly fields: readonly FieldSpec[];
  /** At most one per recipe (Training Plan, Runtime). */
  readonly singleton?: boolean;
}

export const FAMILY_LABEL: Record<NodeFamily, string> = {
  data: 'Data',
  model: 'Model',
  objective: 'Objective',
  optimization: 'Optimization',
  run: 'Run',
};

export const FAMILY_ACCENT: Record<NodeFamily, string> = {
  data: 'emerald',
  model: 'violet',
  objective: 'rose',
  optimization: 'amber',
  run: 'sky',
};

export function defaultValues(spec: NodeSpec): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const field of spec.fields) {
    if (field.default !== undefined) {
      values[field.key] = field.default;
    }
  }
  return values;
}

export function fieldVisible(field: FieldSpec, values: Readonly<Record<string, unknown>>): boolean {
  if (!field.showWhen) {
    return true;
  }
  return field.showWhen.values.includes(values[field.showWhen.key]);
}

export function portVisible(port: PortSpec, values: Readonly<Record<string, unknown>>): boolean {
  if (!port.showWhen) {
    return true;
  }
  return port.showWhen.values.includes(values[port.showWhen.key]);
}
