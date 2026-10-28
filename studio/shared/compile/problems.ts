export type Severity = 'blocker' | 'warning' | 'info';

/**
 * One validation finding. `nodeId`/`fieldKey` let the UI focus the offending control; `phaseId`
 * ties it to a phase on the timeline. `source` says whether the rule is NeoSR's own (native), an
 * architecture-specific check, or a heuristic the user may ignore.
 */
export interface Problem {
  readonly id: string;
  readonly severity: Severity;
  readonly message: string;
  readonly hint?: string;
  readonly nodeId?: string;
  readonly fieldKey?: string;
  readonly phaseId?: string;
  readonly source: 'native' | 'arch' | 'heuristic' | 'studio';
}

export function worstSeverity(problems: readonly Problem[]): Severity | null {
  if (problems.some((p) => p.severity === 'blocker')) {
    return 'blocker';
  }
  if (problems.some((p) => p.severity === 'warning')) {
    return 'warning';
  }
  return problems.length ? 'info' : null;
}

export function countBy(problems: readonly Problem[]): Record<Severity, number> {
  const counts: Record<Severity, number> = { blocker: 0, warning: 0, info: 0 };
  for (const p of problems) {
    counts[p.severity]++;
  }
  return counts;
}
