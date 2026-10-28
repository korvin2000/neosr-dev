import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import { FFlowModule } from '@foblex/flow';
import { FAMILY_ACCENT, portVisible, type NodeSpec, type PortSpec } from '@shared/catalog/types';
import { nodeValues } from '@shared/compile/graph';
import type { Problem } from '@shared/compile/problems';
import type { RecipeNode } from '@shared/recipe.model';
import { RecipeStore } from '../../core/recipe-store';
import { Icon } from '../../ui/icon';
import { Tooltip } from '../../ui/tooltip';
import { connectorId } from './graph-ui';
import { summaryLines } from './node-summary';

interface PortRow {
  readonly port: PortSpec;
  readonly id: string;
  readonly colourClass: string;
}

/**
 * A summary card: title, a 3 px family stripe, two or three value lines, ports on the border and
 * a status icon only when something is wrong. Editing happens in the drawer, never on the card.
 * The host carries Foblex's `fNode`; the whole card is the drag handle.
 */
@Component({
  selector: 'app-graph-node',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FFlowModule, Icon, Tooltip],
  templateUrl: './graph-node.html',
  styleUrl: './graph-node.scss',
  host: {
    tabindex: '0',
    role: 'button',
    '[attr.aria-label]': 'title()',
    '(keydown.enter)': 'activate()',
    '(keydown.space)': 'activate(); $event.preventDefault()',
    '[class]': 'accentClass()',
    '[class.is-disabled]': 'node().disabled',
    '[class.has-blocker]': 'status() === "blocker"',
  },
})
export class GraphNode {
  private readonly store = inject(RecipeStore);

  readonly node = input.required<RecipeNode>();
  readonly spec = input.required<NodeSpec>();
  readonly problems = input<readonly Problem[]>([]);
  readonly phaseId = input<string | null>(null);
  readonly readonly = input(false);

  protected readonly accentClass = computed(() => `accent-${FAMILY_ACCENT[this.spec().family]}`);

  /** Keyboard twin of the click: select the node (which opens the inspector). */
  protected activate(): void {
    if (!this.readonly()) {
      this.store.selectOne(this.node().id);
    }
  }
  protected readonly title = computed(() => this.node().title || this.spec().label);
  protected readonly lines = computed(() => summaryLines(this.node(), this.store.recipe(), this.phaseId()).slice(0, 3));

  protected readonly status = computed<'blocker' | 'warning' | null>(() => {
    const problems = this.problems();
    return problems.some((p) => p.severity === 'blocker') ? 'blocker' : problems.some((p) => p.severity === 'warning') ? 'warning' : null;
  });

  protected readonly statusText = computed(() => {
    const worst = this.status();
    const first = this.problems().find((p) => p.severity === worst);
    const n = this.problems().filter((p) => p.severity !== 'info').length;
    return first ? (n > 1 ? `${first.message} (+${n - 1} more)` : first.message) : '';
  });

  protected readonly inputs = computed<PortRow[]>(() => {
    const node = this.node();
    const values = nodeValues(node, this.spec());
    const wired = new Set(this.store.recipe().edges.filter((e) => e.targetNode === node.id).map((e) => e.targetPort));
    return this.spec()
      .inputs.filter((port) => portVisible(port, values) || wired.has(port.key))
      .map((port) => ({ port, id: connectorId.input(node.id, port.key), colourClass: `port-${port.type}` }));
  });

  protected readonly outputs = computed<PortRow[]>(() => this.spec().outputs.map((port) => ({ port, id: connectorId.output(this.node().id, port.key), colourClass: `port-${port.type}` })));
}
