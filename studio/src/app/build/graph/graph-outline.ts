import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { nodeSpec } from '@shared/catalog/node-catalog';
import { FAMILY_ACCENT, type NodeFamily } from '@shared/catalog/types';
import type { RecipeNode } from '@shared/recipe.model';
import { RecipeStore } from '../../core/recipe-store';
import { Icon } from '../../ui/icon';
import { GraphUi } from './graph-ui';
import { summaryLines } from './node-summary';

const FLOW_ORDER: readonly NodeFamily[] = ['data', 'model', 'objective', 'optimization', 'run'];

interface Row {
  readonly node: RecipeNode;
  readonly label: string;
  readonly icon: string;
  readonly accent: string;
  readonly summary: string;
  readonly status: 'blocker' | 'warning' | null;
  /** "→ Training Plan": where this node's output goes. */
  readonly feeds: string;
}

/** Phone: the graph as an outline in flow order (Data → Model → Losses → Optimization → Run). */
@Component({
  selector: 'app-graph-outline',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  template: `
    <div class="bar">
      <button type="button" class="btn btn--sm" (click)="ui.addOpen.set(true)"><app-icon name="plus" [size]="14" /> Add</button>
      <span class="spacer"></span>
      <button type="button" class="btn btn--ghost btn--sm" (click)="ui.canvasOnPhone.set(true)" [disabled]="rows().length === 0"><app-icon name="grid" [size]="14" /> View graph</button>
    </div>
    @for (r of rows(); track r.node.id) {
      <button type="button" class="row" [class]="'row accent-' + r.accent" [class.row--off]="r.node.disabled" [class.row--active]="selectedId() === r.node.id" (click)="ui.openInspector(r.node.id)">
        <app-icon class="row__icon" [name]="r.icon" [size]="16" />
        <span class="row__body">
          <span class="row__label">{{ r.label }}@if (r.node.disabled) {<span class="row__off">off</span>}</span>
          <span class="row__summary">{{ r.summary }}</span>
          @if (r.feeds) {
            <span class="row__feeds">→ {{ r.feeds }}</span>
          }
        </span>
        @if (r.status; as s) {
          <app-icon class="row__status" [class.row__status--warn]="s === 'warning'" name="alert-triangle" [size]="15" />
        }
        <app-icon class="row__chevron" name="chevron-right" [size]="14" />
      </button>
    } @empty {
      <p class="empty">No nodes yet — add one with +, or start from a template.</p>
    }
  `,
  styles: `
    :host { display: block; flex: 1; min-height: 0; overflow: auto; padding: 8px 12px 24px; }
    .bar { display: flex; align-items: center; gap: 8px; padding: 4px 0 10px; }
    .bar .btn { min-height: 36px; }
    .row { position: relative; display: flex; align-items: center; gap: 10px; width: 100%; min-height: 56px; padding: 8px 10px 8px 14px; margin-bottom: 6px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--panel); color: var(--text); text-align: left; }
    .row::before { content: ''; position: absolute; left: 0; top: 8px; bottom: 8px; width: 3px; border-radius: 0 2px 2px 0; background: var(--accent-node); }
    .row--active { border-color: var(--selection); }
    .row--off { opacity: 0.6; }
    .row__icon { flex: none; color: var(--text-muted); }
    .row__body { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 1px; }
    .row__label { font-weight: 600; }
    .row__off { margin-left: 6px; padding: 0 5px; border-radius: 999px; border: 1px solid var(--border-strong); color: var(--text-muted); font-size: var(--fs-xs); font-weight: 400; }
    .row__summary { color: var(--text-muted); font-size: var(--fs-sm); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .row__feeds { color: var(--text-faint); font-size: var(--fs-xs); }
    .row__status { flex: none; color: var(--danger); }
    .row__status--warn { color: var(--warn); }
    .row__chevron { flex: none; color: var(--text-faint); }
    .empty { padding: 24px 12px; color: var(--text-muted); text-align: center; }
  `,
})
export class GraphOutline {
  protected readonly ui = inject(GraphUi);
  private readonly store = inject(RecipeStore);

  protected readonly selectedId = computed(() => this.store.selectedNode()?.id ?? null);

  protected readonly rows = computed<Row[]>(() => {
    const recipe = this.store.recipe();
    const phaseId = this.store.activePhaseId();
    const byId = new Map(recipe.nodes.map((n) => [n.id, n]));
    const problems = this.store.problemsByNode();
    return FLOW_ORDER.flatMap((family) =>
      recipe.nodes
        .filter((n) => nodeSpec(n.type).family === family)
        .map((node) => {
          const spec = nodeSpec(node.type);
          const own = problems.get(node.id) ?? [];
          const target = recipe.edges.find((e) => e.sourceNode === node.id);
          const feedsNode = target ? byId.get(target.targetNode) : undefined;
          return {
            node,
            label: node.title || spec.label,
            icon: spec.icon,
            accent: FAMILY_ACCENT[spec.family],
            summary: summaryLines(node, recipe, phaseId).join(' · '),
            status: own.some((p) => p.severity === 'blocker') ? 'blocker' : own.some((p) => p.severity === 'warning') ? 'warning' : null,
            feeds: feedsNode ? feedsNode.title || nodeSpec(feedsNode.type).label : '',
          };
        }),
    );
  });
}
