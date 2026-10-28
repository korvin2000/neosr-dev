import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import type { FieldSpec } from '@shared/catalog/types';
import type { Problem } from '@shared/compile/problems';
import { FieldWidget } from './field-widget';
import { HelpPopover } from './help-popover';
import { Icon } from './icon';
import { Tooltip } from './tooltip';

export interface PhaseState {
  readonly phaseId: string;
  readonly phaseName: string;
  readonly overridden: boolean;
}

/**
 * One setting: label · ⓘ · widget, plus what the user needs at the moment of editing and nothing
 * else — a changed-from-preset bar with reset, an inline problem, a contextual suggestion.
 */
@Component({
  selector: 'app-field-row',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FieldWidget, HelpPopover, Icon, Tooltip],
  templateUrl: './field-row.html',
  styleUrl: './field-row.scss',
  host: {
    '[class.changed]': 'changed()',
    '[class.has-problem]': 'worst() === "blocker"',
    '[class.has-warning]': 'worst() === "warning"',
    '[attr.data-field]': 'field().key',
  },
})
export class FieldRow {
  readonly field = input.required<FieldSpec>();
  readonly value = input<unknown>();
  readonly readout = input<string>('');
  readonly changed = input(false);
  /** Baseline value shown in the reset tooltip. */
  readonly baseline = input<unknown>();
  readonly problems = input<readonly Problem[]>([]);
  readonly suggestion = input<{ label: string; value: unknown; why?: string } | null>(null);
  readonly showKey = input(false);
  readonly phase = input<PhaseState | null>(null);
  readonly disabled = input(false);

  readonly valueChange = output<unknown>();
  readonly patch = output<Record<string, unknown>>();
  readonly reset = output<void>();
  readonly overrideToggle = output<boolean>();

  /** Shared with the widget so the label reaches the control (and the ⓘ never steals its name). */
  protected readonly controlId = `fr_${Math.random().toString(36).slice(2, 9)}`;
  /** Blockers and warnings belong under the field; notes stay in the readiness popover. */
  protected readonly inlineProblems = computed(() => this.problems().filter((p) => p.severity !== 'info'));

  protected readonly worst = computed<'blocker' | 'warning' | 'info' | null>(() => {
    const p = this.problems();
    return p.some((x) => x.severity === 'blocker') ? 'blocker' : p.some((x) => x.severity === 'warning') ? 'warning' : p.length ? 'info' : null;
  });

  protected readonly baselineText = computed(() => {
    const b = this.baseline();
    if (b === undefined || b === null || b === '') {
      return 'empty';
    }
    return Array.isArray(b) ? b.join(', ') : typeof b === 'object' ? 'default' : String(b);
  });

  protected readonly phaseInherits = computed(() => {
    const p = this.phase();
    return !!p && !p.overridden && !!this.field().phaseAware;
  });

  protected readonly phaseOverridden = computed(() => {
    const p = this.phase();
    return !!p && p.overridden;
  });
}
