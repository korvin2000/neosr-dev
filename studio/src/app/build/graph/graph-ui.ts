import { Injectable, inject, signal } from '@angular/core';
import { nodeSpec } from '@shared/catalog/node-catalog';
import type { Point } from '@shared/recipe.model';
import * as commands from '../../core/commands';
import { PresetService } from '../../core/preset.service';
import { RecipeStore } from '../../core/recipe-store';
import { ToastService } from '../../ui/toast.service';

/** A request to show one field: the drawer scrolls to it and focuses its control. */
export interface FieldFocus {
  readonly nodeId: string;
  readonly key: string | null;
  readonly seq: number;
}

/** Palette drag payloads: a node type, or `preset:<id>` for a saved configuration. */
export const PRESET_PREFIX = 'preset:';

/** Foblex connector ids: `<nodeId>:in:<port>` / `<nodeId>:out:<port>`. */
export const connectorId = {
  input: (nodeId: string, port: string): string => `${nodeId}:in:${port}`,
  output: (nodeId: string, port: string): string => `${nodeId}:out:${port}`,
  parse(id: string): { nodeId: string; direction: 'in' | 'out'; port: string } | null {
    const m = /^(.+?):(in|out):(.+)$/.exec(id);
    return m ? { nodeId: m[1], direction: m[2] as 'in' | 'out', port: m[3] } : null;
  },
};

/** Presentation state shared by the canvas, the drawer, the palette sheet and the phone outline. */
@Injectable()
export class GraphUi {
  private readonly store = inject(RecipeStore);
  private readonly presets = inject(PresetService);
  private readonly toasts = inject(ToastService);
  private seq = 0;

  readonly focus = signal<FieldFocus | null>(null);
  /** The "+" sheet. */
  readonly addOpen = signal(false);
  /** Bumped when the canvas should fit the graph (after a template load or auto layout). */
  readonly fitRequests = signal(0);
  /** Phone: "View graph" shows the read-only canvas instead of the outline. */
  readonly canvasOnPhone = signal(false);

  openInspector(nodeId: string, field: string | null = null): void {
    this.store.selectOne(nodeId);
    this.focus.set({ nodeId, key: field, seq: ++this.seq });
  }

  requestFit(): void {
    this.fitRequests.update((n) => n + 1);
  }

  /** A spot to the right of the right-most node, so an added node never lands on another. */
  freeSpot(): Point {
    const nodes = this.store.recipe().nodes;
    if (nodes.length === 0) {
      return { x: 80, y: 80 };
    }
    const right = nodes.reduce((a, b) => (b.position.x > a.position.x ? b : a));
    return { x: right.position.x + 220 + 60, y: Math.max(40, right.position.y) };
  }

  /** Adds (and wires) a node, selects it, and returns its id. */
  addNode(type: string, position = this.freeSpot(), values: Record<string, unknown> = {}): string | null {
    const spec = nodeSpec(type);
    const before = this.store.recipe().nodes;
    if (spec.singleton && before.some((n) => n.type === type)) {
      this.toasts.show('warn', `${spec.label} already exists: a recipe has at most one.`);
      return null;
    }
    const known = new Set(before.map((n) => n.id));
    this.store.dispatch(commands.addNodeWired(type, position, values));
    const added = this.store.recipe().nodes.find((n) => !known.has(n.id));
    if (added) {
      this.openInspector(added.id);
    }
    return added?.id ?? null;
  }

  saveAsPreset(nodeId: string): void {
    const node = this.store.node(nodeId);
    if (!node) {
      return;
    }
    const spec = nodeSpec(node.type);
    const name = window.prompt(`Preset name for this ${spec.label}:`, node.title || spec.label);
    if (!name?.trim()) {
      return;
    }
    const preset = this.presets.save(name.trim(), node);
    this.toasts.show('ok', `Saved preset "${preset.name}" (${spec.label}). Find it under + › Presets.`);
  }
}
