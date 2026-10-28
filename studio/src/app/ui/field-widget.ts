import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import type { FieldSpec, SelectOption } from '@shared/catalog/types';
import { Icon } from './icon';

/**
 * Renders one catalog field from its `FieldSpec`. Shared by the wizard, the node card and the
 * inspector, so a NeoSR option looks and behaves the same everywhere. Custom editors (loss rows,
 * validation sets, arch params) are hosted by the caller, not here.
 *
 * Emits `valueChange` for the field itself and `patch` when a select option carries `fills`.
 */
@Component({
  selector: 'app-field-widget',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  templateUrl: './field-widget.html',
  styleUrl: './field-widget.scss',
  host: {
    '[class.compact]': 'compact()',
    '[class.overridden]': 'overridden()',
  },
})
export class FieldWidget {
  readonly field = input.required<FieldSpec>();
  readonly value = input<unknown>();
  readonly disabled = input(false);
  /** Dense layout for node cards. */
  readonly compact = input(false);
  /** The value shown is a per-phase override. */
  readonly overridden = input(false);
  /** Text for a readout widget, computed by the host. */
  readonly readout = input<string>('');
  /** Suggestion chips under the control (the v1 style); FieldRow turns them off and shows one contextual suggestion instead. */
  readonly showSuggestions = input(true);

  readonly valueChange = output<unknown>();
  readonly patch = output<Record<string, unknown>>();

  /** Id for the control so a host label can point at it. */
  readonly controlId = input<string>('');

  protected readonly widget = computed(() => this.field().widget);
  private readonly ownId = `f_${Math.random().toString(36).slice(2, 7)}`;
  protected readonly inputId = computed(() => this.controlId() || `${this.ownId}_${this.field().key}`);

  protected readonly textValue = computed(() => {
    const v = this.value();
    return v === undefined || v === null ? '' : String(v);
  });

  protected readonly numberListText = computed(() => {
    const v = this.value();
    return Array.isArray(v) ? v.join(', ') : '';
  });

  protected readonly selectedSet = computed(() => new Set(Array.isArray(this.value()) ? (this.value() as string[]) : []));

  protected readonly keyValueRows = computed(() => {
    const v = this.value();
    return v && typeof v === 'object' ? Object.entries(v as Record<string, unknown>) : [];
  });

  protected readonly selectHint = computed(() => {
    const w = this.widget();
    if (w.kind !== 'select') {
      return '';
    }
    return w.options.find((o) => o.value === String(this.value() ?? ''))?.hint ?? '';
  });

  protected onText(event: Event): void {
    this.valueChange.emit((event.target as HTMLInputElement).value);
  }

  protected onNumber(event: Event): void {
    const raw = (event.target as HTMLInputElement).value.trim();
    if (raw === '') {
      this.valueChange.emit(undefined);
      return;
    }
    const n = Number(raw);
    if (Number.isFinite(n)) {
      this.valueChange.emit(n);
    }
  }

  protected onToggle(event: Event): void {
    this.valueChange.emit((event.target as HTMLInputElement).checked);
  }

  protected onSelect(event: Event): void {
    const raw = (event.target as HTMLSelectElement).value;
    const w = this.widget();
    const option: SelectOption | undefined = w.kind === 'select' ? w.options.find((o) => o.value === raw) : undefined;
    this.valueChange.emit(raw);
    if (option?.fills) {
      this.patch.emit(option.fills);
    }
  }

  protected toggleMulti(value: string): void {
    const next = new Set(this.selectedSet());
    if (next.has(value)) {
      next.delete(value);
    } else {
      next.add(value);
    }
    const w = this.widget();
    const order = w.kind === 'multiselect' ? w.options.map((o) => o.value) : [...next];
    this.valueChange.emit(order.filter((v) => next.has(v)));
  }

  protected onNumberList(event: Event): void {
    const raw = (event.target as HTMLInputElement).value;
    const parts = raw
      .split(/[,\s]+/)
      .map((p) => p.trim())
      .filter(Boolean)
      .map(Number);
    if (parts.every((n) => Number.isFinite(n))) {
      this.valueChange.emit(parts);
    }
  }

  protected setKeyValue(index: number, key: string, value: string): void {
    const rows = [...this.keyValueRows()];
    rows[index] = [key, parseScalar(value)];
    this.valueChange.emit(Object.fromEntries(rows.filter(([k]) => k.trim())));
  }

  protected addKeyValue(): void {
    const rows = [...this.keyValueRows(), ['', '']];
    this.valueChange.emit(Object.fromEntries(rows));
  }

  protected removeKeyValue(index: number): void {
    const rows = this.keyValueRows().filter((_, i) => i !== index);
    this.valueChange.emit(Object.fromEntries(rows));
  }

  protected applySuggestion(value: unknown): void {
    this.valueChange.emit(value);
  }

  protected readonly suggestionActive = (value: unknown) => JSON.stringify(value) === JSON.stringify(this.value());
}

/** Key/value tables carry TOML scalars; numbers and booleans should not become strings. */
export function parseScalar(raw: string): unknown {
  const t = raw.trim();
  if (t === 'true') {
    return true;
  }
  if (t === 'false') {
    return false;
  }
  if (t !== '' && Number.isFinite(Number(t))) {
    return Number(t);
  }
  if (t.startsWith('[') && t.endsWith(']')) {
    try {
      return JSON.parse(t);
    } catch {
      return t;
    }
  }
  return t;
}
