import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import type { ParamInfo } from '@shared/catalog/data-types';
import { Icon } from '../icon';
import { parseScalar } from '../field-widget';

/**
 * Constructor kwargs of a network. Only values that differ from the default are stored (and
 * emitted to TOML), so the config stays as short as NeoSR's own templates.
 */
@Component({
  selector: 'app-arch-params-editor',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  template: `
    @if (params().length === 0) {
      <p class="faint">This architecture has no documented options besides the scale, which NeoSR injects.</p>
    }
    <div class="grid">
      @for (p of params(); track p.key) {
        <label class="param" [class.param--set]="isSet(p.key)" [attr.title]="p.help ?? ''">
          <span class="param__k mono">{{ p.key }}</span>
          @if (p.options?.length) {
            <select class="select" [value]="shown(p)" (change)="set(p, $any($event.target).value)">
              @for (o of p.options; track o) {
                <option [value]="o" [selected]="o === shown(p)">{{ o }}</option>
              }
            </select>
          } @else if (p.type === 'bool') {
            <select class="select" [value]="shown(p)" (change)="set(p, $any($event.target).value)">
              <option value="true" [selected]="shown(p) === 'true'">true</option>
              <option value="false" [selected]="shown(p) === 'false'">false</option>
            </select>
          } @else {
            <input class="input mono" type="text" [value]="shown(p)" [placeholder]="'' + (p.default ?? '')" (change)="set(p, $any($event.target).value)" />
          }
          @if (isSet(p.key)) {
            <button type="button" class="reset" (click)="reset(p.key)" title="Reset to default"><app-icon name="undo" [size]="11" /></button>
          }
        </label>
      }
    </div>
    @if (extraKeys().length) {
      <p class="faint extra">Also set (unknown to the catalog): <span class="mono">{{ extraKeys().join(', ') }}</span></p>
    }
  `,
  styles: `
    :host { display: block; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px 12px; }
    .param { display: grid; grid-template-columns: 1fr; gap: 2px; position: relative; }
    .param__k { font-size: var(--fs-xs); color: var(--text-faint); }
    .param--set .param__k { color: var(--accent-text); }
    .param .input, .param .select { height: 26px; font-size: var(--fs-sm); }
    .reset { position: absolute; right: 6px; bottom: 6px; color: var(--text-faint); display: inline-flex; }
    .reset:hover { color: var(--text); }
    .extra { margin-top: 8px; font-size: var(--fs-xs); }
  `,
})
export class ArchParamsEditor {
  readonly params = input.required<readonly ParamInfo[]>();
  readonly value = input.required<Readonly<Record<string, unknown>>>();
  readonly valueChange = output<Record<string, unknown>>();

  protected readonly extraKeys = computed(() => Object.keys(this.value()).filter((k) => !this.params().some((p) => p.key === k)));

  protected isSet(key: string): boolean {
    return this.value()[key] !== undefined;
  }

  protected shown(p: ParamInfo): string {
    const v = this.value()[p.key];
    const d = v === undefined ? p.default : v;
    return d === null || d === undefined ? '' : typeof d === 'object' ? JSON.stringify(d) : String(d);
  }

  protected set(p: ParamInfo, raw: string): void {
    const next = { ...this.value() };
    const value = p.type === 'bool' ? raw === 'true' : p.type === 'str' ? raw : parseScalar(raw);
    if (raw.trim() === '' || JSON.stringify(value) === JSON.stringify(p.default)) {
      delete next[p.key];
    } else {
      next[p.key] = value;
    }
    this.valueChange.emit(next);
  }

  protected reset(key: string): void {
    const next = { ...this.value() };
    delete next[key];
    this.valueChange.emit(next);
  }
}
