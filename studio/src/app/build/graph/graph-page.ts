import { ChangeDetectionStrategy, Component, computed, effect, inject, input, untracked } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { DisclosureState } from '../../core/disclosure-state.service';
import { RecipeStore } from '../../core/recipe-store';
import { UiModeService } from '../../core/ui-mode.service';
import { Drawer } from '../../ui/drawer';
import { Icon } from '../../ui/icon';
import { GraphCanvas, PALETTE_KEY } from './graph-canvas';
import { GraphOutline } from './graph-outline';
import { GraphUi } from './graph-ui';
import { NodeDrawer } from './node-drawer';
import { NodePalette } from './node-palette';
import { PhaseStrip } from './phase-strip';

/**
 * Build › Graph: summary cards on the canvas, the section editor in a drawer on selection (docked
 * on wide screens), a + sheet for adding. Phones get an outline list instead of the canvas.
 * `?node=<id>&field=<key>` (from the readiness pill) selects the node and focuses the field.
 */
@Component({
  selector: 'app-graph-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [GraphCanvas, GraphOutline, NodeDrawer, NodePalette, PhaseStrip, Drawer, Icon],
  providers: [GraphUi],
  templateUrl: './graph-page.html',
  styleUrl: './graph-page.scss',
})
export class GraphPage {
  protected readonly ui = inject(GraphUi);
  protected readonly mode = inject(UiModeService);
  private readonly store = inject(RecipeStore);
  private readonly disclosure = inject(DisclosureState);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);

  /** Query params bound by the router (`withComponentInputBinding`). */
  readonly node = input<string>();
  readonly field = input<string>();

  protected readonly selectedId = computed(() => this.store.selectedNode()?.id ?? null);
  protected readonly multiPhase = computed(() => this.store.recipe().phases.length > 1);
  protected readonly docked = computed(() => this.mode.isWide() && !!this.selectedId());
  protected readonly paletteDocked = computed(() => !this.mode.isPhone() && this.disclosure.isOpen(PALETTE_KEY));
  protected readonly focusField = computed(() => {
    const f = this.ui.focus();
    return f && f.nodeId === this.selectedId() ? f.key : null;
  });

  constructor() {
    effect(() => {
      const nodeId = this.node();
      const field = this.field();
      if (!nodeId) {
        return;
      }
      untracked(() => {
        if (this.store.node(nodeId)) {
          this.ui.openInspector(nodeId, field ?? null);
        }
        // Clear the params so the same problem can be focused again later.
        void this.router.navigate([], { relativeTo: this.route, queryParams: {}, replaceUrl: true });
      });
    });
  }

  protected closeDrawer(): void {
    this.store.selectOne(null);
  }
}
