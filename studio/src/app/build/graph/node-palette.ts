import { ChangeDetectionStrategy, Component, computed, inject, output, signal } from '@angular/core';
import { FFlowModule } from '@foblex/flow';
import { NODE_CATALOG } from '@shared/catalog/node-catalog';
import { FAMILY_ACCENT, FAMILY_LABEL, type NodeFamily, type NodeSpec } from '@shared/catalog/types';
import { TEMPLATE_INFOS, type TemplateInfo } from '@shared/templates';
import { PresetService, type NodePreset } from '../../core/preset.service';
import { RecipeStore } from '../../core/recipe-store';
import { Icon } from '../../ui/icon';
import { ToastService } from '../../ui/toast.service';
import { Tooltip } from '../../ui/tooltip';
import { GraphUi, PRESET_PREFIX } from './graph-ui';

type Tab = 'nodes' | 'presets' | 'templates';

const FAMILY_ORDER: readonly NodeFamily[] = ['data', 'model', 'objective', 'optimization', 'run'];

interface NodeEntry {
  readonly spec: NodeSpec;
  /** Singleton that already exists on the canvas. */
  readonly disabled: boolean;
}

/**
 * The "Add" content: node catalog, saved presets and recipe templates with one search box. Lives
 * in the + sheet, or as a docked left column. Click adds (drag onto the canvas also works);
 * `added` lets the sheet close itself.
 */
@Component({
  selector: 'app-node-palette',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FFlowModule, Icon, Tooltip],
  templateUrl: './node-palette.html',
  styleUrl: './node-palette.scss',
})
export class NodePalette {
  readonly added = output<void>();

  private readonly store = inject(RecipeStore);
  private readonly ui = inject(GraphUi);
  private readonly presets = inject(PresetService);
  private readonly toasts = inject(ToastService);

  protected readonly tab = signal<Tab>('nodes');
  protected readonly query = signal('');
  protected readonly presetPrefix = PRESET_PREFIX;

  protected readonly nodeGroups = computed(() => {
    const q = this.query().trim().toLowerCase();
    const present = new Set(this.store.recipe().nodes.map((n) => n.type));
    return FAMILY_ORDER.map((family) => ({
      family,
      label: FAMILY_LABEL[family],
      accent: FAMILY_ACCENT[family],
      entries: NODE_CATALOG.filter((s) => s.family === family)
        .filter((s) => !q || s.label.toLowerCase().includes(q) || s.description.toLowerCase().includes(q) || s.id.includes(q))
        .map<NodeEntry>((spec) => ({ spec, disabled: !!spec.singleton && present.has(spec.id) })),
    })).filter((g) => g.entries.length > 0);
  });

  protected readonly presetGroups = computed(() => {
    const q = this.query().trim().toLowerCase();
    const groups = new Map<string, NodePreset[]>();
    for (const p of this.presets.presets()) {
      if (q && !p.name.toLowerCase().includes(q) && !p.description.toLowerCase().includes(q) && !p.nodeLabel.toLowerCase().includes(q)) {
        continue;
      }
      groups.set(p.nodeLabel, [...(groups.get(p.nodeLabel) ?? []), p]);
    }
    return [...groups.entries()].map(([label, entries]) => ({
      label,
      accent: FAMILY_ACCENT[NODE_CATALOG.find((s) => s.id === entries[0].nodeType)?.family ?? 'run'],
      entries,
    }));
  });

  protected readonly filteredTemplates = computed(() => {
    const q = this.query().trim().toLowerCase();
    return TEMPLATE_INFOS.filter((t) => !q || t.label.toLowerCase().includes(q) || t.description.toLowerCase().includes(q) || t.tags.some((tag) => tag.includes(q)));
  });

  protected addNode(entry: NodeEntry): void {
    if (!entry.disabled && this.ui.addNode(entry.spec.id)) {
      this.added.emit();
    }
  }

  protected addPreset(preset: NodePreset): void {
    if (this.ui.addNode(preset.nodeType, undefined, preset.values)) {
      this.added.emit();
    }
  }

  protected isBuiltIn(preset: NodePreset): boolean {
    return this.presets.isBuiltIn(preset.id);
  }

  protected deletePreset(preset: NodePreset): void {
    if (window.confirm(`Delete preset "${preset.name}"?`)) {
      this.presets.remove(preset.id);
    }
  }

  protected loadTemplate(template: TemplateInfo): void {
    if (this.store.dirty() && !window.confirm(`Replace the current recipe with "${template.label}"? Unsaved changes are lost.`)) {
      return;
    }
    this.store.loadTemplate(template.id);
    this.ui.requestFit();
    this.toasts.show('ok', `Loaded template "${template.label}".`);
    this.added.emit();
  }

  protected describePreset(preset: NodePreset): string {
    const values = Object.entries(preset.values)
      .map(([k, v]) => `${k} = ${Array.isArray(v) ? `[${v.join(', ')}]` : String(v)}`)
      .join(' · ');
    return preset.description ? `${preset.description} — ${values}` : values;
  }

  protected templateMeta(t: TemplateInfo): string {
    return `×${t.scale} · ${t.arch} · ${t.objective} · ${t.phases} phase${t.phases === 1 ? '' : 's'} · ${(t.totalIter / 1000).toLocaleString()}k`;
  }

  protected onQuery(event: Event): void {
    this.query.set((event.target as HTMLInputElement).value);
  }
}
