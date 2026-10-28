import { ChangeDetectionStrategy, Component, ElementRef, computed, effect, inject, input, signal, untracked, viewChild } from '@angular/core';
import {
  FCanvasComponent,
  FCreateConnectionEvent,
  FCreateNodeEvent,
  FDeleteSelectedEvent,
  FFlowComponent,
  FFlowModule,
  FMoveNodesEvent,
  FReassignConnectionEvent,
  FSelectionChangeEvent,
} from '@foblex/flow';
import { NODE_TYPES, nodeSpec } from '@shared/catalog/node-catalog';
import type { RecipeNode } from '@shared/recipe.model';
import * as commands from '../../core/commands';
import { DisclosureState } from '../../core/disclosure-state.service';
import { PresetService } from '../../core/preset.service';
import { RecipeStore } from '../../core/recipe-store';
import { UiModeService } from '../../core/ui-mode.service';
import { Icon } from '../../ui/icon';
import { Menu, type MenuItem } from '../../ui/menu';
import { Tooltip } from '../../ui/tooltip';
import { GraphNode } from './graph-node';
import { GraphUi, PRESET_PREFIX, connectorId } from './graph-ui';

const GAN_PORTS = new Set(['discriminator', 'optimD']);
const GAN_TYPES = new Set<string>([NODE_TYPES.discriminator, NODE_TYPES.optimizerD]);

export const MINIMAP_KEY = 'graph:minimap';
export const PALETTE_KEY = 'graph:palette';

/**
 * The Foblex canvas in classic mode: the store owns the recipe, the library reports gestures and
 * every gesture becomes a command. Every gesture has a visible twin: the drawer's connection
 * pickers, the ⋯ menus, the + button.
 */
@Component({
  selector: 'app-graph-canvas',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FFlowModule, Icon, Menu, Tooltip, GraphNode],
  templateUrl: './graph-canvas.html',
  styleUrl: './graph-canvas.scss',
  host: { '(window:keydown)': 'onKey($event)' },
})
export class GraphCanvas {
  /** Pan and zoom only (the phone's "View graph"). */
  readonly readonly = input(false);

  protected readonly store = inject(RecipeStore);
  protected readonly ui = inject(GraphUi);
  private readonly mode = inject(UiModeService);
  private readonly disclosure = inject(DisclosureState);
  private readonly presets = inject(PresetService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly canvas = viewChild.required(FCanvasComponent);
  private readonly flow = viewChild(FFlowComponent);

  protected readonly MIN_ZOOM = 0.2;
  protected readonly MAX_ZOOM = 2.5;
  protected readonly zoom = signal(100);
  private fitted = false;

  protected readonly placed = computed(() => this.store.recipe().nodes.map((node) => ({ node, spec: nodeSpec(node.type) })));
  protected readonly isEmpty = computed(() => this.store.recipe().nodes.length === 0);
  protected readonly minimap = computed(() => this.disclosure.isOpen(MINIMAP_KEY));
  private readonly paletteDocked = computed(() => this.disclosure.isOpen(PALETTE_KEY));

  /** Whether the active phase trains a discriminator; shared values count as "everything relevant". */
  private readonly activeGan = computed(() => {
    const id = this.store.activePhaseId();
    if (!id) {
      return true;
    }
    return this.store.compiled().phases.find((p) => p.phase.id === id)?.hasGan ?? true;
  });

  protected readonly edges = computed(() => {
    const recipe = this.store.recipe();
    const nodes = new Map(recipe.nodes.map((n) => [n.id, n]));
    const gan = this.activeGan();
    const selected = this.store.selectedNode()?.id ?? null;
    return recipe.edges.map((edge) => {
      const source = nodes.get(edge.sourceNode);
      const target = nodes.get(edge.targetNode);
      const out = source ? nodeSpec(source.type).outputs.find((p) => p.key === edge.sourcePort) : undefined;
      const inp = target ? nodeSpec(target.type).inputs.find((p) => p.key === edge.targetPort) : undefined;
      const offSelection = !!selected && edge.sourceNode !== selected && edge.targetNode !== selected;
      return {
        edge,
        source: connectorId.output(edge.sourceNode, edge.sourcePort),
        target: connectorId.input(edge.targetNode, edge.targetPort),
        colourClass: out ? `port-${out.type}` : '',
        invalid: !out || !inp || out.type !== inp.type,
        muted: !!(source?.disabled || target?.disabled),
        dimmed: offSelection || (!gan && (GAN_PORTS.has(edge.targetPort) || GAN_TYPES.has(source?.type ?? '') || GAN_TYPES.has(target?.type ?? ''))),
      };
    });
  });

  protected readonly menu = computed<MenuItem[]>(() => [
    { id: 'layout', label: 'Auto layout', icon: 'layout', disabled: this.isEmpty() },
    { id: 'minimap', label: 'Minimap', icon: 'grid', hint: this.minimap() ? 'on' : 'off' },
    { id: 'zoom-in', label: 'Zoom in', icon: 'plus', divider: true },
    { id: 'zoom-out', label: 'Zoom out', icon: 'minus' },
    { id: 'zoom-reset', label: 'Zoom 100%', icon: 'fit', hint: `${this.zoom()}%` },
    ...(this.mode.isPhone() ? [] : [{ id: 'palette', label: 'Dock palette', icon: 'columns', hint: this.paletteDocked() ? 'on' : 'off', divider: true }]),
    { id: 'select-all', label: 'Select all', icon: 'layers', hint: 'Ctrl A', divider: true, disabled: this.isEmpty() },
    { id: 'clear', label: 'Clear canvas…', icon: 'trash', danger: true, disabled: this.isEmpty() },
  ]);

  constructor() {
    // Selection made elsewhere (readiness pill, outline, drawer close) reaches the library through here.
    effect(() => {
      const ids = [...this.store.selection()];
      const edge = this.store.selectedEdge();
      const flow = this.flow();
      if (!flow) {
        return;
      }
      const current = flow.getSelection();
      const same = sameSet(current.fNodeIds, ids) && sameSet(current.fConnectionIds, edge ? [edge] : []);
      if (!same) {
        flow.select(ids, edge ? [edge] : [], false);
      }
    });
    effect(() => {
      if (this.ui.fitRequests() > 0) {
        untracked(() => setTimeout(() => this.fit(), 60));
      }
    });
  }

  // --- gestures ----------------------------------------------------------------

  protected onCreateConnection(event: FCreateConnectionEvent): void {
    if (this.readonly() || !event.targetId) {
      return;
    }
    const source = connectorId.parse(event.sourceId);
    const target = connectorId.parse(event.targetId);
    if (!source || !target || source.direction !== 'out' || target.direction !== 'in' || source.nodeId === target.nodeId) {
      return;
    }
    const out = this.store.node(source.nodeId);
    const inp = this.store.node(target.nodeId);
    const outType = out && nodeSpec(out.type).outputs.find((p) => p.key === source.port)?.type;
    const inType = inp && nodeSpec(inp.type).inputs.find((p) => p.key === target.port)?.type;
    if (!outType || outType !== inType) {
      return;
    }
    this.store.dispatch(commands.connect({ sourceNode: source.nodeId, sourcePort: source.port, targetNode: target.nodeId, targetPort: target.port }));
  }

  protected onReassignConnection(event: FReassignConnectionEvent): void {
    if (this.readonly()) {
      return;
    }
    this.store.dispatch(commands.removeEdges([event.connectionId]));
    if (event.nextSourceId && event.nextTargetId) {
      this.onCreateConnection(new FCreateConnectionEvent(event.nextSourceId, event.nextTargetId, event.dropPosition));
    }
  }

  protected onMoveNodes(event: FMoveNodesEvent): void {
    const moves = event.nodes.map((n) => ({ id: n.id, position: { x: Math.round(n.position.x), y: Math.round(n.position.y) } }));
    if (moves.length && !this.readonly()) {
      this.store.dispatch(commands.moveNodes(moves));
    }
  }

  protected onCreateNode(event: FCreateNodeEvent<string>): void {
    if (this.readonly()) {
      return;
    }
    const at = event.dropPosition ?? { x: event.externalItemRect.x, y: event.externalItemRect.y };
    const position = { x: Math.round(at.x), y: Math.round(at.y) };
    if (event.data.startsWith(PRESET_PREFIX)) {
      const preset = this.presets.presets().find((p) => p.id === event.data.slice(PRESET_PREFIX.length));
      if (preset) {
        this.ui.addNode(preset.nodeType, position, preset.values);
      }
      return;
    }
    this.ui.addNode(event.data, position);
  }

  protected onSelectionChange(event: FSelectionChangeEvent): void {
    this.store.select(event.nodeIds);
    this.store.selectedEdge.set(event.connectionIds[0] ?? null);
  }

  protected onDeleteSelected(event: FDeleteSelectedEvent): void {
    this.deleteIds(event.nodeIds, event.connectionIds);
  }

  private deleteIds(nodeIds: readonly string[], edgeIds: readonly string[]): void {
    if (this.readonly()) {
      return;
    }
    const recipe = this.store.recipe();
    const nodes = nodeIds.filter((id) => recipe.nodes.some((n) => n.id === id));
    const edges = edgeIds.filter((id) => recipe.edges.some((e) => e.id === id));
    if (nodes.length) {
      this.store.dispatch(commands.removeNodes(nodes));
    }
    if (edges.length) {
      this.store.dispatch(commands.removeEdges(edges));
    }
  }

  protected onKey(event: KeyboardEvent): void {
    const target = event.target as HTMLElement | null;
    const typing = !!target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable);
    if (typing || this.readonly()) {
      return;
    }
    const key = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && key === 'a') {
      event.preventDefault();
      this.selectAll();
    } else if ((event.ctrlKey || event.metaKey) && key === 'd') {
      event.preventDefault();
      [...this.store.selection()].forEach((id) => this.store.dispatch(commands.duplicateNode(id)));
    } else if (event.key === 'Delete' || event.key === 'Backspace') {
      const edge = this.store.selectedEdge();
      this.deleteIds([...this.store.selection()], edge ? [edge] : []);
    }
  }

  // --- toolbar -------------------------------------------------------------------

  protected onMenu(id: string): void {
    switch (id) {
      case 'layout':
        this.store.dispatch(commands.autoLayout());
        this.ui.requestFit();
        break;
      case 'minimap':
        this.disclosure.toggle(MINIMAP_KEY);
        break;
      case 'zoom-in':
        this.zoomTo(this.canvas().getScale() * 1.25);
        break;
      case 'zoom-out':
        this.zoomTo(this.canvas().getScale() / 1.25);
        break;
      case 'zoom-reset':
        this.canvas().resetScaleAndCenter(true);
        this.syncZoom();
        break;
      case 'palette':
        this.disclosure.toggle(PALETTE_KEY);
        break;
      case 'select-all':
        this.selectAll();
        break;
      case 'clear':
        if (window.confirm('Remove every node and connection from the canvas? (Undo is available.)')) {
          this.store.dispatch(commands.clearCanvas());
        }
        break;
    }
  }

  private selectAll(): void {
    this.store.select(this.store.recipe().nodes.map((n) => n.id));
  }

  // --- viewport -------------------------------------------------------------------

  protected onNodesRendered(): void {
    if (!this.fitted && !this.isEmpty()) {
      this.fitted = true;
      this.canvas().fitToScreen({ x: 60, y: 60 }, false);
      this.syncZoom();
    }
  }

  protected fit(): void {
    if (this.isEmpty()) {
      return;
    }
    this.canvas().fitToScreen({ x: 60, y: 60 }, true);
    this.syncZoom();
  }

  protected onCanvasChange(): void {
    this.syncZoom();
  }

  /** Zooms about the middle of the visible canvas and repaints, which `setScale` alone does not. */
  private zoomTo(scale: number): void {
    const canvas = this.canvas();
    const next = Math.min(Math.max(scale, this.MIN_ZOOM), this.MAX_ZOOM);
    if (Math.abs(next - canvas.getScale()) < 0.0001) {
      return;
    }
    const rect = this.host.nativeElement.getBoundingClientRect();
    canvas.setScale(next, { x: rect.width / 2, y: rect.height / 2 });
    canvas.redrawWithAnimation();
    canvas.emitCanvasChangeEvent();
    this.syncZoom();
  }

  private syncZoom(): void {
    this.zoom.set(Math.round(this.canvas().getScale() * 100));
  }

  protected problemsOf(node: RecipeNode) {
    return this.store.problemsByNode().get(node.id) ?? [];
  }
}

function sameSet(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((x) => b.includes(x));
}
