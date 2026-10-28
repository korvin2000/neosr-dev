import { ARCH_BY_TYPE } from '@shared/catalog/archs';
import { objectiveLosses, type ObjectiveId } from '@shared/catalog/node-catalog';
import { activeLossRows } from '@shared/compile/graph';
import type { LossRow } from '@shared/recipe.model';

/** The three outcomes a user picks between; each maps to a loss objective and a start template. */
export interface Look {
  readonly id: ObjectiveId;
  readonly label: 'Clean' | 'Balanced' | 'Detailed';
  readonly line: string;
  readonly templateId: string;
}

export const LOOKS: readonly Look[] = [
  { id: 'fidelity', label: 'Clean', line: 'Pixel losses · best PSNR, smoothest result', templateId: '4x-span-fidelity' },
  { id: 'balanced', label: 'Balanced', line: 'Template losses without GAN · sharper, no invented texture', templateId: '4x-realplksr-balanced' },
  { id: 'perceptual', label: 'Detailed', line: 'Full GAN recipe · most texture, may invent detail', templateId: '4x-neosr-gan' },
];

/** The look whose loss set matches the rows active in a phase; `null` when the rows are custom. */
export function lookOf(rows: readonly LossRow[], phaseId: string): Look | null {
  const active = activeLossRows(rows, phaseId).map(({ row }) => row.type).sort();
  for (const look of LOOKS) {
    const expected = objectiveLosses(look.id).map((l) => l.type).sort();
    if (expected.length === active.length && expected.every((t, i) => t === active[i])) {
      return look;
    }
  }
  return null;
}

/** Nominal simulated throughput per architecture (updates per second), as the backend uses. */
const NOMINAL_ITS: Readonly<Record<string, number>> = { span: 9, compact: 12, realplksr: 4.2, hat_s: 1.1 };

export function nominalIts(arch: string): number {
  return NOMINAL_ITS[arch] ?? 3;
}

/** "≈ 2 h" style duration from updates and the nominal speed. */
export function estimateDuration(arch: string, updates: number): string {
  const hours = updates / nominalIts(arch) / 3600;
  if (hours < 1) {
    return `≈ ${Math.max(5, Math.round((hours * 60) / 5) * 5)} min`;
  }
  if (hours < 48) {
    return `≈ ${Math.round(hours)} h`;
  }
  return `≈ ${Math.round(hours / 24)} days`;
}

export function sizeLabel(arch: string): string {
  const size = ARCH_BY_TYPE.get(arch)?.sizeClass;
  switch (size) {
    case 'ultra-light':
    case 'light':
      return 'Small model';
    case 'medium':
    case 'medium-heavy':
      return 'Medium model';
    case 'heavy':
    case 'ultra-heavy':
      return 'Large model';
    default:
      return 'Model';
  }
}

export function fmtUpdates(n: number): string {
  if (n >= 1_000_000) {
    return `${+(n / 1_000_000).toFixed(1)}M`;
  }
  return n >= 1000 ? `${+(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k` : String(n);
}
