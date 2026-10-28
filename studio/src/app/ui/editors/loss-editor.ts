import { ChangeDetectionStrategy, Component, computed, input, output, signal } from '@angular/core';
import { LOSSES } from '@shared/catalog/losses';
import { OBJECTIVES, type ObjectiveId } from '@shared/catalog/node-catalog';
import type { LossInfo, ParamInfo } from '@shared/catalog/data-types';
import { lossRowsFor, newLossRow } from '@shared/recipe-builder';
import type { LossRow, Phase } from '@shared/recipe.model';
import { Icon } from '../icon';
import { Tooltip } from '../tooltip';
import { parseScalar } from '../field-widget';

interface RowView {
  readonly row: LossRow;
  readonly info: LossInfo | undefined;
  readonly cells: readonly CellView[];
  readonly expanded: boolean;
  readonly needsD: boolean;
  readonly params: readonly ParamInfo[];
}

interface CellView {
  readonly phase: Phase;
  readonly active: boolean;
  readonly weight: number;
  readonly overridden: boolean;
}

/**
 * The Loss Recipe editor: one row per loss term, one column per phase. A cell without an override
 * inherits the row's shared weight (shown faded); editing a cell creates a per-phase override.
 * Weights are piecewise-constant per phase because NeoSR has no native loss schedule — a change of
 * weight means a new run, which is exactly what a phase compiles to.
 */
@Component({
  selector: 'app-loss-editor',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, Tooltip],
  templateUrl: './loss-editor.html',
  styleUrl: './loss-editor.scss',
})
export class LossEditor {
  readonly rows = input.required<readonly LossRow[]>();
  readonly phases = input.required<readonly Phase[]>();
  readonly activePhaseId = input<string | null>(null);
  readonly hasDiscriminator = input(false);
  readonly disabled = input(false);
  /** Hide the preset bar and params (node card). */
  readonly compact = input(false);
  /** Show the `[train.<slot>_opt]` section next to each row (the "Show NeoSR keys" preference). */
  readonly showKeys = input(false);

  readonly rowsChange = output<LossRow[]>();
  readonly applyObjective = output<ObjectiveId>();

  protected readonly objectives = OBJECTIVES;
  protected readonly expanded = signal<ReadonlySet<string>>(new Set());
  protected readonly adding = signal(false);
  protected readonly multiPhase = computed(() => this.phases().length > 1);

  protected readonly catalogByCategory = computed(() => {
    const groups = new Map<string, LossInfo[]>();
    for (const loss of LOSSES) {
      groups.set(loss.category, [...(groups.get(loss.category) ?? []), loss]);
    }
    return [...groups.entries()].map(([category, losses]) => ({ category, losses }));
  });

  protected readonly views = computed<RowView[]>(() =>
    this.rows().map((row) => {
      const info = LOSSES.find((l) => l.type === row.type);
      return {
        row,
        info,
        expanded: this.expanded().has(row.id),
        needsD: row.type === 'gan_loss' && row.enabled && !this.hasDiscriminator(),
        params: info?.params.filter((p) => p.key !== 'loss_weight') ?? [],
        cells: this.phases().map((phase) => {
          const s = row.schedule[phase.id];
          return { phase, active: s ? s.active : row.enabled, weight: s ? s.weight : row.weight, overridden: !!s };
        }),
      };
    }),
  );

  protected readonly totalActive = computed(() => this.rows().filter((r) => r.enabled).length);

  private emit(rows: LossRow[]): void {
    this.rowsChange.emit(rows);
  }

  private update(id: string, fn: (row: LossRow) => LossRow): void {
    this.emit(this.rows().map((r) => (r.id === id ? fn(r) : r)));
  }

  protected toggleEnabled(id: string, enabled: boolean): void {
    this.update(id, (r) => ({ ...r, enabled }));
  }

  protected setWeight(id: string, raw: string): void {
    const weight = Number(raw);
    if (Number.isFinite(weight) && weight >= 0) {
      this.update(id, (r) => ({ ...r, weight }));
    }
  }

  protected setType(id: string, type: string): void {
    const info = LOSSES.find((l) => l.type === type);
    this.update(id, (r) => ({ ...r, type, params: {}, weight: info?.templateWeight ?? info?.defaultWeight ?? r.weight }));
  }

  protected setCell(id: string, phaseId: string, patch: Partial<{ active: boolean; weight: number }>): void {
    this.update(id, (r) => {
      const current = r.schedule[phaseId] ?? { active: r.enabled, weight: r.weight };
      return { ...r, schedule: { ...r.schedule, [phaseId]: { ...current, ...patch } } };
    });
  }

  protected setCellWeight(id: string, phaseId: string, raw: string): void {
    const weight = Number(raw);
    if (Number.isFinite(weight) && weight >= 0) {
      this.setCell(id, phaseId, { weight });
    }
  }

  protected clearCell(id: string, phaseId: string): void {
    this.update(id, (r) => {
      const schedule = { ...r.schedule };
      delete schedule[phaseId];
      return { ...r, schedule };
    });
  }

  protected setParam(id: string, key: string, raw: string, info: ParamInfo): void {
    this.update(id, (r) => {
      const params = { ...r.params };
      const value = info.type === 'bool' ? raw === 'true' : info.type === 'str' ? raw : parseScalar(raw);
      if (raw === '' || JSON.stringify(value) === JSON.stringify(info.default)) {
        delete params[key];
      } else {
        params[key] = value;
      }
      return { ...r, params };
    });
  }

  protected setTarget(id: string, target: string): void {
    this.update(id, (r) => {
      const params = { ...r.params };
      if (target === 'lq') {
        params['target'] = 'lq';
      } else {
        delete params['target'];
      }
      return { ...r, params };
    });
  }

  protected paramValue(row: LossRow, info: ParamInfo): string {
    const v = row.params[info.key];
    const d = v === undefined ? info.default : v;
    return d === null || d === undefined ? '' : typeof d === 'object' ? JSON.stringify(d) : String(d);
  }

  protected toggleExpanded(id: string): void {
    this.expanded.update((set) => {
      const next = new Set(set);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  protected remove(id: string): void {
    this.emit(this.rows().filter((r) => r.id !== id));
  }

  protected move(id: string, delta: number): void {
    const rows = [...this.rows()];
    const i = rows.findIndex((r) => r.id === id);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= rows.length) {
      return;
    }
    [rows[i], rows[j]] = [rows[j], rows[i]];
    this.emit(rows);
  }

  protected add(type: string): void {
    this.adding.set(false);
    this.emit([...this.rows(), newLossRow(type, this.rows())]);
  }

  protected preset(objective: ObjectiveId): void {
    this.emit(lossRowsFor(objective));
    this.applyObjective.emit(objective);
  }

  protected readonly categoryLabel: Record<string, string> = {
    pixel: 'Pixel',
    structural: 'Structure',
    perceptual: 'Perceptual',
    frequency: 'Frequency',
    color: 'Colour',
    adversarial: 'Adversarial',
    distribution: 'Distribution',
  };
}
