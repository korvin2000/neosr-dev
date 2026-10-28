import { ChangeDetectionStrategy, Component, ElementRef, computed, inject, input, output } from '@angular/core';
import { NODE_TYPES, nodeSpec } from '@shared/catalog/node-catalog';
import { portVisible, type PortSpec } from '@shared/catalog/types';
import { nodeValues } from '@shared/compile/graph';
import type { RecipeEdge } from '@shared/recipe.model';
import * as commands from '../../core/commands';
import { RecipeStore } from '../../core/recipe-store';
import { Drawer } from '../../ui/drawer';
import { Menu, type MenuItem } from '../../ui/menu';
import { SectionEditor } from '../../ui/section-editor';
import { GraphUi } from './graph-ui';

interface SourceOption {
  readonly label: string;
  readonly nodeId: string;
  readonly port: string;
}

interface Picker {
  readonly port: PortSpec;
  readonly edge: RecipeEdge | undefined;
  /** Index into `options`, or -1 for "none". */
  readonly current: number;
  readonly options: readonly SourceOption[];
}

/**
 * The graph's inspector: a drawer with the node's connections as pickers (the keyboard and touch
 * twin of drag-to-connect) and the one section editor in single-node mode. Docked as a column on
 * wide screens, an overlay below 1400 px, a sheet on phones.
 */
@Component({
  selector: 'app-node-drawer',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Drawer, Menu, SectionEditor],
  template: `
    <!-- Named slots cannot sit inside control flow, so the header binds through optional access. -->
    <app-drawer [open]="!!node()" title="" [width]="400" (closed)="closed.emit()">
      <span drawer-title class="head">
        <span class="head__kind">{{ spec().label }}</span>
        <input class="head__title" type="text" [value]="node()?.title ?? ''" [placeholder]="spec().label" (change)="rename($any($event.target).value)" aria-label="Node title" />
      </span>
      <app-menu drawer-actions [items]="menu()" (picked)="onMenu($event)" />

      @if (pickers().length) {
        <div class="pickers">
          @for (p of pickers(); track p.port.key) {
            <label class="picker">
              <span class="picker__label">{{ p.port.label }}</span>
              <select class="select" (change)="wire(p, $any($event.target).value)" [attr.aria-label]="'Source for ' + p.port.label">
                <option value="-1" [selected]="p.current < 0">none</option>
                @for (o of p.options; track o.nodeId + ':' + o.port; let i = $index) {
                  <option [value]="i" [selected]="i === p.current">{{ o.label }}</option>
                }
              </select>
            </label>
          }
        </div>
      }

      <app-section-editor [nodeId]="nodeId()" [focusField]="focusField()" />
    </app-drawer>
  `,
  styles: `
    :host { display: contents; }
    :host(.docked) ::ng-deep .backdrop { display: none; }
    :host(.docked) ::ng-deep .drawer { position: relative; top: auto; right: auto; bottom: auto; height: 100%; flex: none; animation: none; box-shadow: none; }
    .head { display: flex; flex-direction: column; min-width: 0; flex: 1; }
    .head__kind { color: var(--text-faint); font-size: var(--fs-xs); line-height: 1.2; }
    .head__title { width: 100%; height: 24px; padding: 0 4px; margin-left: -4px; border: 1px solid transparent; border-radius: var(--radius-sm); background: transparent; color: var(--text); font-size: var(--fs-md); font-weight: 600; }
    .head__title:hover { border-color: var(--border-strong); }
    .head__title:focus { border-color: var(--accent); background: var(--panel-sunken); box-shadow: none; }
    .pickers { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid var(--border-soft); }
    .picker { display: grid; grid-template-columns: 110px 1fr; align-items: center; gap: 8px; }
    .picker__label { color: var(--text-muted); font-size: var(--fs-sm); }
    .picker .select { min-height: 32px; }
  `,
  host: { '[class.docked]': 'docked()' },
})
export class NodeDrawer {
  readonly nodeId = input<string | null>(null);
  readonly focusField = input<string | null>(null);
  readonly docked = input(false);
  readonly closed = output<void>();

  private readonly store = inject(RecipeStore);
  private readonly ui = inject(GraphUi);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);

  protected readonly node = computed(() => {
    const id = this.nodeId();
    return id ? this.store.node(id) : undefined;
  });
  protected readonly spec = computed(() => nodeSpec(this.node()?.type ?? NODE_TYPES.plan));

  protected readonly pickers = computed<Picker[]>(() => {
    const node = this.node();
    if (!node) {
      return [];
    }
    const recipe = this.store.recipe();
    const values = nodeValues(node);
    const edges = recipe.edges.filter((e) => e.targetNode === node.id);
    return this.spec()
      .inputs.filter((port) => portVisible(port, values) || edges.some((e) => e.targetPort === port.key))
      .map((port) => {
        const edge = edges.find((e) => e.targetPort === port.key);
        const options = recipe.nodes
          .filter((n) => n.id !== node.id && !n.disabled)
          .flatMap((n) =>
            nodeSpec(n.type)
              .outputs.filter((o) => o.type === port.type)
              .map<SourceOption>((o) => ({ label: n.title || nodeSpec(n.type).label, nodeId: n.id, port: o.key })),
          );
        const current = edge ? options.findIndex((o) => o.nodeId === edge.sourceNode && o.port === edge.sourcePort) : -1;
        return { port, edge, current, options };
      });
  });

  protected readonly menu = computed<MenuItem[]>(() => {
    const node = this.node();
    return [
      { id: 'rename', label: 'Rename', icon: 'tag' },
      { id: 'duplicate', label: 'Duplicate', icon: 'copy', hint: 'Ctrl D', disabled: !!this.spec().singleton },
      { id: 'disable', label: node?.disabled ? 'Enable' : 'Disable', icon: 'power' },
      { id: 'disconnect', label: 'Disconnect', icon: 'close' },
      { id: 'preset', label: 'Save as preset…', icon: 'bookmark', divider: true },
      { id: 'delete', label: 'Delete', icon: 'trash', hint: 'Del', danger: true, divider: true },
    ];
  });

  protected wire(picker: Picker, raw: string): void {
    const node = this.node();
    const index = Number(raw);
    const option = picker.options[index];
    if (!node) {
      return;
    }
    if (!option) {
      if (picker.edge) {
        this.store.dispatch(commands.removeEdges([picker.edge.id]));
      }
      return;
    }
    if (picker.edge && picker.edge.sourceNode === option.nodeId && picker.edge.sourcePort === option.port) {
      return;
    }
    this.store.dispatch(commands.connect({ sourceNode: option.nodeId, sourcePort: option.port, targetNode: node.id, targetPort: picker.port.key }));
  }

  protected rename(raw: string): void {
    const node = this.node();
    const title = raw.trim();
    if (node && title !== node.title && title !== this.spec().label) {
      this.store.dispatch(commands.setNodeFlags(node.id, { title }));
    }
  }

  protected onMenu(id: string): void {
    const node = this.node();
    if (!node) {
      return;
    }
    switch (id) {
      case 'rename':
        this.host.nativeElement.querySelector<HTMLInputElement>('.head__title')?.focus();
        break;
      case 'duplicate':
        this.store.dispatch(commands.duplicateNode(node.id));
        break;
      case 'disable':
        this.store.dispatch(commands.setNodeFlags(node.id, { disabled: !node.disabled }));
        break;
      case 'disconnect':
        this.store.dispatch(commands.disconnectNode(node.id));
        break;
      case 'preset':
        this.ui.saveAsPreset(node.id);
        break;
      case 'delete':
        this.store.dispatch(commands.removeNodes([node.id]));
        break;
    }
  }
}
