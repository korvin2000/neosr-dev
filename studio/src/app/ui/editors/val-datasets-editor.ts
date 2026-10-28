import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import type { ValDatasetRow } from '@shared/compile/graph';
import { newId } from '@shared/recipe.model';
import { Icon } from '../icon';
import { Tooltip } from '../tooltip';

/** Named validation sets → `[datasets.val]`, `[datasets.val_1]`, … Paired sets get metrics; single sets are visual-only. */
@Component({
  selector: 'app-val-datasets-editor',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, Tooltip],
  template: `
    @for (row of rows(); track row.id; let i = $index) {
      <div class="set" [class.set--single]="row.type === 'single'">
        <div class="set__head">
          <span class="set__section" [appTooltip]="'[datasets.' + (i === 0 ? 'val' : 'val_' + i) + ']'">Set {{ i + 1 }}</span>
          <input class="input input--name" type="text" [value]="row.name" placeholder="name" (change)="patch(row.id, { name: $any($event.target).value })" aria-label="Validation set name" />
          <select class="select select--type" [value]="row.type" (change)="patch(row.id, { type: $any($event.target).value })" aria-label="Type">
            <option value="paired" [selected]="row.type === 'paired'">paired · metrics</option>
            <option value="single" [selected]="row.type === 'single'">single · visual only</option>
          </select>
          <button type="button" class="btn btn--ghost btn--sm btn--icon btn--danger" (click)="remove(row.id)" title="Remove set" [disabled]="disabled()"><app-icon name="trash" [size]="13" /></button>
        </div>
        @if (row.type === 'paired') {
          <label class="set__row"><span class="k" title="dataroot_gt">High-res</span><input class="input mono" type="text" [value]="row.dataroot_gt" placeholder="C:\\datasets\\val\\gt\\" (change)="patch(row.id, { dataroot_gt: $any($event.target).value })" /></label>
        }
        <label class="set__row"><span class="k" title="dataroot_lq">Low-res</span><input class="input mono" type="text" [value]="row.dataroot_lq" [placeholder]="row.type === 'single' ? probePlaceholder : lqPlaceholder" (change)="patch(row.id, { dataroot_lq: $any($event.target).value })" /></label>
      </div>
    }
    <div class="add">
      <button type="button" class="btn btn--sm" [disabled]="disabled()" (click)="add('paired')"><app-icon name="plus" [size]="12" /> Paired set</button>
      <button type="button" class="btn btn--ghost btn--sm" [disabled]="disabled()" (click)="add('single')" title="Real inputs without ground truth: images only, no metrics"><app-icon name="plus" [size]="12" /> Visual-only probe</button>
    </div>
  `,
  styles: `
    :host { display: flex; flex-direction: column; gap: 8px; }
    .set { padding: 8px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--panel-sunken); display: flex; flex-direction: column; gap: 6px; }
    .set--single { border-style: dashed; }
    .set__head { display: flex; align-items: center; gap: 6px; }
    .set__section { color: var(--text-faint); font-size: var(--fs-sm); white-space: nowrap; min-width: 36px; }
    .input--name { width: 110px; height: 26px; }
    .select--type { width: 150px; height: 26px; font-size: var(--fs-sm); }
    .set__row { display: grid; grid-template-columns: 64px 1fr; align-items: center; gap: 6px; }
    .set__row .input { height: 26px; font-size: var(--fs-sm); }
    .k { color: var(--text-faint); font-size: var(--fs-xs); }
    .add { display: flex; gap: 6px; }
  `,
})
export class ValDatasetsEditor {
  readonly rows = input.required<readonly ValDatasetRow[]>();
  readonly disabled = input(false);
  readonly rowsChange = output<ValDatasetRow[]>();
  protected readonly gtPlaceholder = ['C:', 'datasets', 'val', 'gt', ''].join('\\');
  protected readonly lqPlaceholder = ['C:', 'datasets', 'val', 'lq', ''].join('\\');
  protected readonly probePlaceholder = ['C:', 'datasets', 'real_inputs', ''].join('\\');

  protected patch(id: string, patch: Partial<ValDatasetRow>): void {
    this.rowsChange.emit(this.rows().map((r) => (r.id === id ? { ...r, ...patch } : r)));
  }

  protected remove(id: string): void {
    this.rowsChange.emit(this.rows().filter((r) => r.id !== id));
  }

  protected add(type: 'paired' | 'single'): void {
    const n = this.rows().length;
    this.rowsChange.emit([...this.rows(), { id: newId('v'), name: n === 0 ? 'val' : type === 'single' ? `probe_${n}` : `val_${n}`, type, dataroot_gt: '', dataroot_lq: '' }]);
  }
}
