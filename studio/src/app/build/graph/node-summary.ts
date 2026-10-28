/** The two or three plain-text lines a summary card shows for a node. */
import { ARCH_BY_TYPE } from '@shared/catalog/archs';
import { LOSSES } from '@shared/catalog/losses';
import { NODE_TYPES, nodeSpec } from '@shared/catalog/node-catalog';
import { fieldVisible } from '@shared/catalog/types';
import { activeLossRows, lossRows, nodeValues, phaseNodeValues, valDatasets } from '@shared/compile/graph';
import { totalIters, type Recipe, type RecipeNode } from '@shared/recipe.model';

function fmtK(n: number): string {
  return n >= 1000 ? `${Math.round(n / 100) / 10}k` : String(n);
}

function fmtLr(v: unknown): string {
  const n = Number(v);
  return Number.isFinite(n) ? (n < 0.001 ? n.toExponential(0) : String(n)) : '?';
}

function baseName(path: unknown): string {
  const text = String(path ?? '').trim();
  return text.split(/[\\/]/).filter(Boolean).at(-1) ?? '';
}

const PRECISION: Record<string, string> = { fp32: 'fp32', amp: 'AMP fp16', bf16: 'AMP bf16' };

export function summaryLines(node: RecipeNode, recipe: Recipe, phaseId: string | null): string[] {
  const spec = nodeSpec(node.type);
  const v = phaseId ? phaseNodeValues(node, phaseId, spec) : nodeValues(node, spec);
  const phases = recipe.phases;
  switch (node.type) {
    case NODE_TYPES.trainDataset: {
      const acc = Number(v['accumulate'] ?? 1);
      return [v['type'] === 'otf' ? 'on-the-fly' : 'paired', `${v['patch_size']} px · batch ${v['batch_size']}${acc > 1 ? ` × ${acc}` : ''}`];
    }
    case NODE_TYPES.validation: {
      const sets = valDatasets(node);
      const metrics = Array.isArray(v['metrics']) ? (v['metrics'] as string[]).map((m) => m.toUpperCase()).join(' ') : '';
      return [`${sets.length} set${sets.length === 1 ? '' : 's'} · every ${fmtK(Number(v['val_freq']))}`, ...(metrics ? [metrics] : [])];
    }
    case NODE_TYPES.generator: {
      const arch = ARCH_BY_TYPE.get(String(v['type']));
      return [`${arch?.label ?? v['type']} · ×${v['scale']}`];
    }
    case NODE_TYPES.discriminator:
      return [String(v['type'])];
    case NODE_TYPES.initialization: {
      const mode = String(v['mode']);
      const file = baseName(mode === 'resume' ? v['resume_state'] : v['pretrain_network_g']);
      return mode === 'pretrain' ? ['from weights', ...(file ? [file] : [])] : mode === 'resume' ? ['resume state', ...(file ? [file] : [])] : ['from scratch'];
    }
    case NODE_TYPES.losses: {
      const rows = lossRows(node);
      const active = activeLossRows(rows, phaseId ?? phases.at(-1)?.id ?? '');
      const names = active.slice(0, 3).map(({ row, weight }) => `${LOSSES.find((l) => l.type === row.type)?.label ?? row.type} ${weight}`);
      return [`${active.length} term${active.length === 1 ? '' : 's'}`, ...(names.length ? [names.join(', ') + (active.length > 3 ? ', …' : '')] : [])];
    }
    case NODE_TYPES.optimizerG:
    case NODE_TYPES.optimizerD:
      return [`${v['type']} · ${fmtLr(v['lr'])}`];
    case NODE_TYPES.schedule:
      return [String(v['type'])];
    case NODE_TYPES.runtime:
      return [`${PRECISION[String(v['precision'])] ?? v['precision']}${v['compile'] === true ? ' · compiled' : ''}`];
    case NODE_TYPES.plan: {
      const n = phases.length;
      return [`${fmtK(totalIters(phases))} updates · ${n} phase${n === 1 ? '' : 's'}`, `checkpoints every ${fmtK(Number(v['save_checkpoint_freq'] ?? 1000))}`];
    }
    default: {
      const lines = spec.fields
        .filter((f) => f.level === 'essentials' && fieldVisible(f, v) && f.widget.kind !== 'custom' && f.widget.kind !== 'readout')
        .slice(0, 3)
        .map((f) => `${f.label} ${Array.isArray(v[f.key]) ? `[${(v[f.key] as unknown[]).length}]` : String(v[f.key] ?? '')}`);
      return lines.length ? lines : [spec.description];
    }
  }
}
