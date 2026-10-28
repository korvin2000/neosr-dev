import { Injectable, computed, effect, signal } from '@angular/core';
import { nodeSpec } from '@shared/catalog/node-catalog';
import type { RecipeNode } from '@shared/recipe.model';

/** A saved node configuration: the values of one node, reusable from the palette. */
export interface NodePreset {
  readonly id: string;
  readonly name: string;
  readonly nodeType: string;
  readonly nodeLabel: string;
  readonly description: string;
  readonly values: Readonly<Record<string, unknown>>;
  readonly createdAt: string;
}

const KEY = 'neosr-studio.node-presets.v1';

/** Built-in node presets: the values the NeoSR templates use, one per optimizer side. */
const BUILT_IN: readonly NodePreset[] = [
  {
    id: 'builtin-optim-g-template',
    name: 'Template G (adan_sf 1e-3)',
    nodeType: 'optimizer-g',
    nodeLabel: 'Optimizer G',
    description: 'adan_sf, lr 1e-3, betas 0.98/0.92/0.99, wd 0.01, warm-up 1600 — the shipped train_*.toml values.',
    values: { type: 'adan_sf', lr: 0.001, betas: [0.98, 0.92, 0.99], weight_decay: 0.01, warmup_steps: 1600 },
    createdAt: '2026-10-27T00:00:00.000Z',
  },
  {
    id: 'builtin-optim-g-plksr',
    name: 'PLKSR-family G (adan_sf 5e-4)',
    nodeType: 'optimizer-g',
    nodeLabel: 'Optimizer G',
    description: 'The lower generator LR the templates use for PLKSR, RealPLKSR, ESC, FlexNet, KRGN, PlainUSR, Sebica.',
    values: { type: 'adan_sf', lr: 0.0005, betas: [0.98, 0.92, 0.99], weight_decay: 0.01, warmup_steps: 1600 },
    createdAt: '2026-10-27T00:00:00.000Z',
  },
  {
    id: 'builtin-optim-d-template',
    name: 'Template D (adan_sf 1e-4)',
    nodeType: 'optimizer-d',
    nodeLabel: 'Optimizer D',
    description: 'Discriminator optimizer from the shipped templates: lr 1e-4, warm-up 600.',
    values: { type: 'adan_sf', lr: 0.0001, betas: [0.98, 0.92, 0.99], weight_decay: 0.01, warmup_steps: 600 },
    createdAt: '2026-10-27T00:00:00.000Z',
  },
  {
    id: 'builtin-runtime-safe',
    name: 'Safe fp32',
    nodeType: 'runtime',
    nodeLabel: 'Runtime',
    description: 'No AMP, no bf16, no compile: the configuration to fall back to when losses spike.',
    values: { precision: 'fp32', fast_matmul: false, compile: false },
    createdAt: '2026-10-27T00:00:00.000Z',
  },
];

@Injectable({ providedIn: 'root' })
export class PresetService {
  private readonly user = signal<readonly NodePreset[]>(load());
  readonly presets = computed<readonly NodePreset[]>(() => [...BUILT_IN, ...this.user()]);

  constructor() {
    effect(() => {
      try {
        localStorage.setItem(KEY, JSON.stringify(this.user()));
      } catch {
        // best effort
      }
    });
  }

  byType(nodeType: string): readonly NodePreset[] {
    return this.presets().filter((p) => p.nodeType === nodeType);
  }

  isBuiltIn(id: string): boolean {
    return BUILT_IN.some((p) => p.id === id);
  }

  save(name: string, node: RecipeNode, description = ''): NodePreset {
    const spec = nodeSpec(node.type);
    const preset: NodePreset = {
      id: `preset_${Math.random().toString(36).slice(2, 10)}`,
      name,
      nodeType: node.type,
      nodeLabel: spec.label,
      description,
      values: { ...node.values },
      createdAt: new Date().toISOString(),
    };
    this.user.update((list) => [...list, preset]);
    return preset;
  }

  remove(id: string): void {
    this.user.update((list) => list.filter((p) => p.id !== id));
  }
}

function load(): NodePreset[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as NodePreset[]) : [];
  } catch {
    return [];
  }
}
