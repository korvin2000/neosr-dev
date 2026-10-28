import { ChangeDetectionStrategy, Component, ElementRef, computed, effect, inject, input, signal, untracked } from '@angular/core';
import { ARCH_BY_TYPE } from '@shared/catalog/archs';
import { DISCRIMINATORS } from '@shared/catalog/discriminators';
import { LOSSES } from '@shared/catalog/losses';
import { NODE_TYPES, nodeSpec } from '@shared/catalog/node-catalog';
import type { FieldSpec, NodeSpec } from '@shared/catalog/types';
import { lossRows, nodeValues, phaseNodeValues, scaleOf, valDatasets, type ValDatasetRow } from '@shared/compile/graph';
import type { Problem } from '@shared/compile/problems';
import { totalIters, type LossRow, type Phase, type RecipeNode } from '@shared/recipe.model';
import { SECTIONS, baselineValue, isChanged, partFields, partNode, type SectionDef, type SectionPart } from '@shared/sections';
import * as commands from '../core/commands';
import { DisclosureState } from '../core/disclosure-state.service';
import { RecipeStore } from '../core/recipe-store';
import { SettingsService } from '../core/settings.service';
import { Disclosure } from './disclosure';
import { ArchSelect } from './arch-select';
import { ArchParamsEditor } from './editors/arch-params-editor';
import { LossEditor } from './editors/loss-editor';
import { PhasesEditor } from './editors/phases-editor';
import { ValDatasetsEditor } from './editors/val-datasets-editor';
import { FieldRow, type PhaseState } from './field-row';
import { Icon } from './icon';
import { Tooltip } from './tooltip';

interface FieldView {
  readonly field: FieldSpec;
  readonly value: unknown;
  readonly readout: string;
  readonly changed: boolean;
  readonly baseline: unknown;
  readonly problems: Problem[];
  readonly suggestion: { label: string; value: unknown; why?: string } | null;
  readonly phase: PhaseState | null;
}

interface PartView {
  readonly part: SectionPart;
  readonly spec: NodeSpec;
  readonly title: string | null;
  readonly node: RecipeNode | undefined;
  readonly basics: FieldView[];
  readonly more: FieldView[];
  readonly moreKey: string;
  readonly moreChanged: number;
  readonly nodeProblems: Problem[];
}

/**
 * The one inspector. Renders a section (Build list, Review "More ›") or a single node (graph
 * drawer, TOML "Edit"), always as Basics plus a remembered "More options" fold. Every edit is a
 * store command; nothing here holds recipe state.
 */
@Component({
  selector: 'app-section-editor',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FieldRow, Disclosure, Icon, Tooltip, LossEditor, ValDatasetsEditor, ArchParamsEditor, PhasesEditor, ArchSelect],
  templateUrl: './section-editor.html',
  styleUrl: './section-editor.scss',
})
export class SectionEditor {
  /** Section mode. */
  readonly section = input<SectionDef | null>(null);
  /** Single-node mode: every part across sections that edits this node. */
  readonly nodeId = input<string | null>(null);
  /** Scroll to and flash this field once. */
  readonly focusField = input<string | null>(null);

  protected readonly store = inject(RecipeStore);
  private readonly settings = inject(SettingsService);
  private readonly disclosure = inject(DisclosureState);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);

  protected readonly showKeys = computed(() => this.settings.settings().showKeys);
  protected readonly phases = computed(() => this.store.recipe().phases);
  protected readonly multiPhase = computed(() => this.phases().length > 1);
  protected readonly activePhase = computed<Phase | null>(() => this.phases().find((p) => p.id === this.store.activePhaseId()) ?? null);

  private readonly parts = computed<SectionPart[]>(() => {
    const section = this.section();
    if (section) {
      return [...section.parts];
    }
    const id = this.nodeId();
    const node = id ? this.store.node(id) : undefined;
    if (!node) {
      return [];
    }
    return SECTIONS.flatMap((s) => s.parts.filter((p) => p.nodeType === node.type));
  });

  /** Whether this editor shows anything phase-aware (and so a phase selector). */
  protected readonly phaseAware = computed(() => this.multiPhase() && this.parts().some((p) => p.custom?.includes('losses') || nodeSpec(p.nodeType).fields.some((f) => f.phaseAware)));

  protected readonly views = computed<PartView[]>(() => {
    const recipe = this.store.recipe();
    const ctx = this.store.context();
    const problems = this.store.problems();
    const phase = this.activePhase();
    const sectionId = this.section()?.id ?? 'node';
    const parts = this.parts().filter((p) => !p.when || p.when(ctx, recipe));
    const multi = parts.length > 1;
    return parts.map((part) => {
      const spec = nodeSpec(part.nodeType);
      const node = this.nodeId() ? this.store.node(this.nodeId()!) : partNode(part, recipe, ctx);
      const { basics, more } = partFields(part, node);
      const toView = (field: FieldSpec): FieldView => {
        const shared = node ? nodeValues(node, spec) : {};
        const overridden = !!phase && !!node && field.phaseAware === true && node.phaseValues[phase.id]?.[field.key] !== undefined;
        const value = node ? (phase && field.phaseAware ? phaseNodeValues(node, phase.id, spec)[field.key] : shared[field.key]) : field.default;
        const fieldProblems = node ? problems.filter((p) => p.nodeId === node.id && p.fieldKey === field.key && (!p.phaseId || !phase || p.phaseId === phase.id)) : [];
        const hint = fieldProblems.map((p) => p.hint ?? '').find((h) => /^Use \d/.test(h));
        const suggestion = hint ? { label: hint.match(/\d+/)![0], value: Number(hint.match(/\d+/)![0]), why: fieldProblems[0].message.replace(/\.$/, '') } : null;
        return {
          field,
          value,
          readout: this.readout(field, node),
          changed: !!node && !phase && isChanged(recipe, node, field.key),
          baseline: node ? baselineValue(recipe, node, field.key) : field.default,
          problems: fieldProblems,
          suggestion,
          phase: phase && field.phaseAware ? { phaseId: phase.id, phaseName: phase.name, overridden } : null,
        };
      };
      const basicViews = basics.map(toView);
      const moreViews = more.map(toView);
      return {
        part,
        spec,
        title: multi || part.title ? (part.title ?? spec.label) : null,
        node,
        basics: basicViews,
        more: moreViews,
        moreKey: `more:${sectionId}:${part.nodeType}`,
        moreChanged: moreViews.filter((v) => v.changed || v.problems.length).length,
        nodeProblems: node ? problems.filter((p) => p.nodeId === node.id && !p.fieldKey) : [],
      };
    });
  });

  protected readonly lossRowsOf = (node: RecipeNode) => lossRows(node);
  protected readonly valRowsOf = (node: RecipeNode) => valDatasets(node);
  protected readonly archParamsOf = (node: RecipeNode) => {
    const type = String(nodeValues(node)['type']);
    return node.type === NODE_TYPES.generator ? (ARCH_BY_TYPE.get(type)?.params ?? []) : (DISCRIMINATORS.find((d) => d.type === type)?.params ?? []);
  };
  protected readonly paramsOf = (node: RecipeNode) => (nodeValues(node)['params'] as Record<string, unknown>) ?? {};
  /** Whether a given loss node has an active gan row in any phase (the switch state for that node). */
  protected readonly ganOn = (node: RecipeNode) => this.phases().some((p) => lossRows(node).some((r) => r.type === 'gan_loss' && (r.schedule[p.id]?.active ?? r.enabled)));
  protected readonly hasDiscriminator = computed(() => !!this.store.context().dmodel);
  protected readonly scale = computed(() => scaleOf(this.store.context().model));
  protected readonly isArchField = (v: PartView, f: FieldView) => v.part.nodeType === NODE_TYPES.generator && f.field.key === 'type';

  private readonly flashed = signal<string | null>(null);

  constructor() {
    // One-shot: depends on the focus request only, never on the views it opens.
    effect(() => {
      const key = this.focusField();
      if (key) {
        this.flashed.set(key);
        untracked(() => this.openMoreFor(key));
        setTimeout(() => this.scrollTo(key), 80);
      }
    });
  }

  // --- edits -------------------------------------------------------------------------------

  protected set(view: PartView, f: FieldView, value: unknown): void {
    if (!view.node) {
      return;
    }
    if (f.phase && f.phase.overridden) {
      this.store.setPhaseValue(view.node.id, f.phase.phaseId, f.field.key, value);
    } else if (f.phase && !f.phase.overridden) {
      // Editing an inherited value in phase view creates the override explicitly.
      this.store.setPhaseValue(view.node.id, f.phase.phaseId, f.field.key, value);
    } else {
      this.store.setValue(view.node.id, f.field.key, value);
    }
  }

  protected patch(view: PartView, patch: Record<string, unknown>): void {
    if (view.node) {
      this.store.setValues(view.node.id, patch, `${view.spec.label}: preset values`);
    }
  }

  protected reset(view: PartView, f: FieldView): void {
    if (view.node) {
      this.store.setValue(view.node.id, f.field.key, f.baseline);
    }
  }

  protected toggleOverride(view: PartView, f: FieldView, on: boolean): void {
    if (!view.node || !f.phase) {
      return;
    }
    const shared = nodeValues(view.node)[f.field.key];
    this.store.setPhaseValue(view.node.id, f.phase.phaseId, f.field.key, on ? shared : undefined);
  }

  protected addNode(part: SectionPart): void {
    const count = this.store.recipe().nodes.length;
    this.store.dispatch(commands.addNodeWired(part.nodeType, { x: 120 + (count % 4) * 320, y: 80 + Math.floor(count / 4) * 220 }));
  }

  protected setLosses(node: RecipeNode, rows: LossRow[]): void {
    this.store.dispatch(commands.setLossRows(node.id, rows));
  }

  protected setValRows(node: RecipeNode, rows: ValDatasetRow[]): void {
    this.store.setValues(node.id, { datasets: rows }, 'Validation sets');
  }

  protected setParams(node: RecipeNode, params: Record<string, unknown>): void {
    this.store.setValues(node.id, { params }, 'Architecture options');
  }

  protected setPhases(phases: Phase[]): void {
    this.store.dispatch(commands.setPhases(phases));
  }

  /** The GAN switch adds or removes the whole bundle (gan row, Discriminator, Optimizer D) in one undo step. */
  protected toggleGan(node: RecipeNode, on: boolean): void {
    this.store.dispatch(commands.setGanBundle(node.id, on));
  }

  protected selectPhase(id: string | null): void {
    this.store.activePhaseId.set(id);
  }

  protected isMoreOpen(view: PartView): boolean {
    return this.disclosure.isOpen(view.moreKey, this.settings.settings().openSections);
  }

  protected setMoreOpen(view: PartView, open: boolean): void {
    this.disclosure.set(view.moreKey, open);
  }

  protected readonly lossLabel = (type: string) => LOSSES.find((l) => l.type === type)?.label ?? type;

  private readout(field: FieldSpec, node: RecipeNode | undefined): string {
    if (field.widget.kind !== 'readout' || !node) {
      return '';
    }
    const v = nodeValues(node);
    switch (field.widget.compute) {
      case 'gt_crop':
        return String(Number(v['patch_size'] ?? 0) * scaleOf(this.store.context().model));
      case 'effective_batch':
        return String(Number(v['batch_size'] ?? 0) * Number(v['accumulate'] ?? 1));
      case 'total_iter':
        return totalIters(this.store.recipe().phases).toLocaleString();
      default:
        return '';
    }
  }

  private openMoreFor(key: string): void {
    for (const view of this.views()) {
      if (view.more.some((f) => f.field.key === key)) {
        this.disclosure.set(view.moreKey, true);
      }
    }
  }

  private scrollTo(key: string): void {
    const el = this.host.nativeElement.querySelector<HTMLElement>(`[data-field="${key}"]`);
    if (el) {
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      el.classList.add('flash');
      setTimeout(() => el.classList.remove('flash'), 1600);
      el.querySelector<HTMLElement>('app-field-widget input, app-field-widget select, app-field-widget [type="checkbox"], .current')?.focus({ preventScroll: true });
    }
  }
}
