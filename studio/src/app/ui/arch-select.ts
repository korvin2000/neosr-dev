import { NgTemplateOutlet } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, input, output, signal } from '@angular/core';
import { ARCHS, ARCH_BY_TYPE } from '@shared/catalog/archs';
import type { ArchInfo, SizeClass } from '@shared/catalog/data-types';
import type { FieldSpec } from '@shared/catalog/types';
import { Drawer } from './drawer';
import { HelpPopover } from './help-popover';
import { Icon } from './icon';
import { Tooltip } from './tooltip';

type SizeFacet = 'all' | 'light' | 'medium' | 'heavy';
type KindFacet = 'all' | 'conv' | 'transformer';

const SIZE_GROUP: Record<SizeClass, SizeFacet> = {
  'ultra-light': 'light',
  light: 'light',
  medium: 'medium',
  'medium-heavy': 'medium',
  heavy: 'heavy',
  'ultra-heavy': 'heavy',
};

const SIZE_LABEL: Record<SizeClass, string> = {
  'ultra-light': 'tiny',
  light: 'small',
  medium: 'medium',
  'medium-heavy': 'medium+',
  heavy: 'large',
  'ultra-heavy': 'very large',
};

/** Four safe starting points, shown first; everything else is one search away. */
const RECOMMENDED = ['compact', 'span', 'realplksr', 'hat_s'];

/**
 * The architecture choice: a button that states the current model in one line and opens a
 * searchable sheet with size and type facets. 93 types never appear at once.
 */
@Component({
  selector: 'app-arch-select',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Drawer, HelpPopover, Icon, Tooltip, NgTemplateOutlet],
  template: `
    <div class="label">
      <span>{{ field().label }}</span>
      <app-help [field]="field()" />
    </div>
    <div class="control">
      <button type="button" class="current" (click)="open.set(true)" [attr.aria-haspopup]="'dialog'">
        <span class="current__name">{{ current()?.label ?? value() }}</span>
        <span class="current__meta">{{ meta(current()) }}</span>
        <app-icon name="chevron-right" [size]="14" class="current__chev" />
      </button>
      @if (changed()) {
        <button type="button" class="reset" (click)="reset.emit()" [appTooltip]="'Preset: ' + baselineLabel()" aria-label="Reset to preset"><app-icon name="undo" [size]="13" /></button>
      }
    </div>

    <app-drawer [open]="open()" title="Choose a model" [width]="460" (closed)="open.set(false)">
      <input class="input search" type="search" placeholder="Search by name or family…" [value]="query()" (input)="query.set($any($event.target).value)" autofocus aria-label="Search architectures" />
      <div class="facets">
        <div class="facets__group" role="group" aria-label="Size">
          @for (s of sizes; track s.value) {
            <button type="button" class="chip" [class.chip--active]="size() === s.value" (click)="size.set(s.value)">{{ s.label }}</button>
          }
        </div>
        <div class="facets__group" role="group" aria-label="Type">
          @for (k of kinds; track k.value) {
            <button type="button" class="chip" [class.chip--active]="kind() === k.value" (click)="kind.set(k.value)">{{ k.label }}</button>
          }
        </div>
        @if (scale()) {
          <label class="chip chip--toggle" [class.chip--active]="onlyScale()">
            <input type="checkbox" [checked]="onlyScale()" (change)="onlyScale.set($any($event.target).checked)" />
            supports ×{{ scale() }}
          </label>
        }
      </div>

      @if (!query() && size() === 'all' && kind() === 'all') {
        <p class="group">Recommended</p>
        @for (a of recommended(); track a.type) {
          <ng-container *ngTemplateOutlet="row; context: { $implicit: a }" />
        }
        <p class="group">All models ({{ filtered().length }})</p>
      }
      @for (a of filtered(); track a.type) {
        <ng-container *ngTemplateOutlet="row; context: { $implicit: a }" />
      }
      @if (filtered().length === 0) {
        <p class="faint empty">Nothing matches.</p>
      }
    </app-drawer>

    <ng-template #row let-a>
      <button type="button" class="arch" [class.arch--active]="a.type === value()" (click)="pick(a)">
        <span class="arch__name">{{ a.label }}<span class="arch__family faint"> · {{ a.family }}</span></span>
        <span class="arch__meta">{{ meta(a) }}</span>
        @if (a.note) {
          <span class="arch__note faint">{{ a.note }}</span>
        }
        @if (a.type === value()) {
          <app-icon name="check" [size]="14" class="arch__check" />
        }
      </button>
    </ng-template>
  `,
  styles: `
    :host { display: grid; grid-template-columns: minmax(120px, 160px) minmax(0, 1fr); column-gap: 14px; align-items: start; padding: 8px 0 8px 10px; margin-left: -10px; border-left: 2px solid transparent; }
    :host(.changed) { border-left-color: var(--accent); }
    .label { display: flex; align-items: center; gap: 4px; min-height: var(--control-h); }
    .control { display: flex; align-items: center; gap: 8px; }
    .current { display: flex; align-items: center; gap: 10px; max-width: 420px; flex: 1; min-height: var(--control-h); padding: 4px 10px; border-radius: var(--radius); border: 1px solid var(--node-border); background: var(--panel-sunken); text-align: left; color: var(--text); }
    .current:hover { border-color: var(--node-border-hover); }
    .current__name { font-weight: 600; }
    .current__meta { color: var(--text-muted); font-size: var(--fs-sm); flex: 1; }
    .current__chev { color: var(--text-faint); }
    .reset { display: inline-flex; color: var(--text-faint); padding: 4px; border-radius: 50%; }
    .reset:hover { color: var(--text); background: var(--panel-raised); }
    .search { margin-bottom: 10px; }
    .facets { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-bottom: 12px; }
    .facets__group { display: flex; gap: 4px; }
    .chip--toggle input { position: absolute; opacity: 0; width: 0; height: 0; }
    .group { margin: 10px 0 4px; color: var(--text-faint); font-size: var(--fs-sm); }
    .arch { position: relative; display: grid; grid-template-columns: 1fr; gap: 2px; width: 100%; padding: 8px 32px 8px 10px; border-radius: var(--radius); text-align: left; color: var(--text); }
    .arch:hover { background: var(--panel-raised); }
    .arch--active { background: var(--accent-soft); }
    .arch__meta { color: var(--text-muted); font-size: var(--fs-sm); }
    .arch__note { font-size: var(--fs-sm); }
    .arch__check { position: absolute; right: 10px; top: 12px; color: var(--accent-text); }
    .empty { padding: 12px; }
    @media (max-width: 720px) { :host { grid-template-columns: 1fr; row-gap: 6px; } .current { max-width: none; } }
  `,
  host: { '[class.changed]': 'changed()' },
})
export class ArchSelect {
  readonly field = input.required<FieldSpec>();
  readonly value = input<string>('');
  readonly changed = input(false);
  readonly baseline = input<unknown>();
  /** The recipe's scale, for the "supports ×N" facet. */
  readonly scale = input<number | null>(null);
  readonly valueChange = output<string>();
  readonly reset = output<void>();

  protected readonly open = signal(false);
  protected readonly query = signal('');
  protected readonly size = signal<SizeFacet>('all');
  protected readonly kind = signal<KindFacet>('all');
  protected readonly onlyScale = signal(true);

  protected readonly sizes: readonly { value: SizeFacet; label: string }[] = [
    { value: 'all', label: 'Any size' },
    { value: 'light', label: 'Small' },
    { value: 'medium', label: 'Medium' },
    { value: 'heavy', label: 'Large' },
  ];
  protected readonly kinds: readonly { value: KindFacet; label: string }[] = [
    { value: 'all', label: 'Any type' },
    { value: 'conv', label: 'Conv' },
    { value: 'transformer', label: 'Transformer' },
  ];

  protected readonly current = computed(() => ARCH_BY_TYPE.get(this.value()));
  protected readonly baselineLabel = computed(() => ARCH_BY_TYPE.get(String(this.baseline() ?? ''))?.label ?? String(this.baseline() ?? ''));

  protected readonly filtered = computed<ArchInfo[]>(() => {
    const q = this.query().trim().toLowerCase();
    const size = this.size();
    const kind = this.kind();
    const scale = this.scale();
    return ARCHS.filter((a) => {
      if (q && !`${a.label} ${a.family} ${a.type} ${a.note ?? ''}`.toLowerCase().includes(q)) {
        return false;
      }
      if (size !== 'all' && SIZE_GROUP[a.sizeClass] !== size) {
        return false;
      }
      if (kind === 'conv' && a.tags.includes('transformer')) {
        return false;
      }
      if (kind === 'transformer' && !a.tags.includes('transformer')) {
        return false;
      }
      if (scale && this.onlyScale() && !a.scales.includes(scale)) {
        return false;
      }
      return true;
    });
  });

  protected readonly recommended = computed(() => RECOMMENDED.map((t) => ARCH_BY_TYPE.get(t)).filter((a): a is ArchInfo => !!a && (!this.scale() || !this.onlyScale() || a.scales.includes(this.scale()!))));

  protected meta(a: ArchInfo | undefined): string {
    if (!a) {
      return '';
    }
    const parts = [SIZE_LABEL[a.sizeClass], a.speed === 'fast' ? 'fast' : a.speed === 'slow' ? 'slow' : 'medium speed', a.tags.includes('transformer') ? 'transformer' : 'conv'];
    if (a.windowSize) {
      parts.push(`patch ×${a.windowSize}`);
    }
    return parts.join(' · ');
  }

  protected pick(a: ArchInfo): void {
    this.open.set(false);
    if (a.type !== this.value()) {
      this.valueChange.emit(a.type);
    }
  }
}
