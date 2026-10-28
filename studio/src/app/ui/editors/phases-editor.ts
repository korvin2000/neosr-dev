import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { newPhase } from '@shared/recipe-builder';
import { phaseRanges, type Phase, type PhaseInit } from '@shared/recipe.model';
import { Icon } from '../icon';
import { Tooltip } from '../tooltip';

const INIT_OPTIONS: readonly { value: PhaseInit; label: string; help: string }[] = [
  { value: 'finetune', label: 'fine-tune from the previous weights', help: 'A new run starts from the previous phase’s net_g_latest.pth; optimizer and warm-up restart. Required when the phase adds or removes the discriminator.' },
  { value: 'continue', label: 'continue the previous state', help: 'Resumes the previous phase’s .state: counters and optimizer continue, so a new learning rate here would not apply.' },
];

/**
 * The curriculum: a list of phases with duration and transition. One phase is the normal case and
 * shows a single row; more phases mean more sequential neosr-train runs, said in plain words.
 */
@Component({
  selector: 'app-phases-editor',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, Tooltip],
  template: `
    @for (r of rows(); track r.phase.id; let i = $index) {
      <div class="phase">
        <span class="phase__n">{{ i + 1 }}</span>
        <input class="input phase__name" type="text" [value]="r.phase.name" (change)="update(r.phase.id, { name: $any($event.target).value })" aria-label="Phase name" />
        <div class="phase__iters">
          <input class="input mono" type="number" min="1" step="1000" [value]="r.phase.iters" (change)="setIters(r.phase.id, $any($event.target).value)" aria-label="Updates" />
          <span class="faint">updates</span>
        </div>
        @if (i > 0) {
          <select class="select phase__init" [value]="r.phase.init" (change)="update(r.phase.id, { init: $any($event.target).value })" aria-label="Transition">
            @for (o of initOptions; track o.value) {
              <option [value]="o.value" [selected]="o.value === r.phase.init">{{ o.label }}</option>
            }
          </select>
        } @else {
          <span class="phase__first faint">starts as chosen under Model</span>
        }
        @if (rows().length > 1) {
          <button type="button" class="btn btn--ghost btn--sm btn--icon" (click)="remove(r.phase.id)" appTooltip="Remove phase" aria-label="Remove phase"><app-icon name="close" [size]="13" /></button>
        }
      </div>
    }
    <div class="foot">
      <button type="button" class="btn btn--ghost btn--sm" (click)="add()"><app-icon name="plus" [size]="12" /> Add a phase</button>
      @if (rows().length > 1) {
        <span class="faint">{{ rows().length }} phases = {{ rows().length }} sequential neosr-train runs, {{ total() }} updates in total.</span>
      }
    </div>
  `,
  styles: `
    :host { display: block; }
    .phase { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 6px 0; }
    .phase__name { flex: 1 1 140px; min-width: 120px; }
    .phase__n { color: var(--text-faint); font-size: var(--fs-sm); text-align: right; }
    .phase__iters { display: flex; align-items: center; gap: 6px; }
    .phase__iters .input { width: 110px; }
    .phase__init { width: 250px; }
    .phase__first { font-size: var(--fs-sm); }
    .foot { display: flex; align-items: center; gap: 12px; padding-top: 6px; font-size: var(--fs-sm); }
    .phase__init { flex: 1 1 220px; }
  `,
})
export class PhasesEditor {
  readonly phases = input.required<readonly Phase[]>();
  readonly phasesChange = output<Phase[]>();
  protected readonly initOptions = INIT_OPTIONS;
  protected readonly rows = computed(() => phaseRanges(this.phases()));
  protected readonly total = computed(() => this.phases().reduce((s, p) => s + p.iters, 0).toLocaleString());

  protected update(id: string, patch: Partial<Phase>): void {
    this.phasesChange.emit(this.phases().map((p) => (p.id === id ? { ...p, ...patch } : p)));
  }

  protected setIters(id: string, raw: string): void {
    const n = Number(raw);
    if (Number.isFinite(n) && n > 0) {
      this.update(id, { iters: Math.round(n) });
    }
  }

  protected add(): void {
    const n = this.phases().length + 1;
    this.phasesChange.emit([...this.phases(), newPhase(`Phase ${n}`, 100_000, 'finetune')]);
  }

  protected remove(id: string): void {
    const rest = this.phases().filter((p) => p.id !== id);
    this.phasesChange.emit(rest.map((p, i) => (i === 0 ? { ...p, init: 'scratch' } : p)));
  }
}
