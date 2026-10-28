import { ChangeDetectionStrategy, Component, computed, effect, inject, signal, untracked } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { ARCH_BY_TYPE } from '@shared/catalog/archs';
import type { ArchInfo } from '@shared/catalog/data-types';
import { lossRows, nodeValues, valDatasets } from '@shared/compile/graph';
import type { Problem } from '@shared/compile/problems';
import { lossRowsFor } from '@shared/recipe-builder';
import { newId, totalIters, type LossRow, type Recipe } from '@shared/recipe.model';
import { recipeChanges, sectionForField, type Change, type SectionId } from '@shared/sections';
import { ApiService, LaunchRejected } from '../core/api.service';
import * as commands from '../core/commands';
import { RecipeStore } from '../core/recipe-store';
import { RunStore } from '../core/run-store';
import { SettingsService } from '../core/settings.service';
import { copyText } from '../library/util';
import { Icon } from '../ui/icon';
import { Menu, type MenuItem } from '../ui/menu';
import { ToastService } from '../ui/toast.service';
import { Tooltip } from '../ui/tooltip';
import { LOOKS, fmtUpdates, lookOf } from './looks';
import { Popover } from './popover';
import { ScanService, type ScanRequest } from './scan.service';
import { fmtValue } from './util';

const ARCH_CHOICES = ['compact', 'span', 'realplksr', 'hat_s'];
const LENGTHS = [40_000, 150_000, 300_000, 500_000];
const VAL_FREQS = [500, 1000, 2500, 5000];
const STARTS = [
  { id: 'scratch', label: 'From scratch', line: 'Random weights' },
  { id: 'pretrain', label: 'From weights', line: 'A net_g_*.pth of the same architecture' },
  { id: 'resume', label: 'Resume a state', line: 'Continue an interrupted run exactly' },
] as const;
const OPTIM_D = { type: 'adan_sf', lr: 0.0001, betas: [0.98, 0.92, 0.99], weight_decay: 0.01, warmup_steps: 600 };

/**
 * Review: the whole plan in one sentence whose tokens are the decisions left to make. Everything
 * else is one click away in Build. Launch is the one primary action.
 */
@Component({
  selector: 'app-review-step',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, Icon, Menu, Popover, Tooltip],
  templateUrl: './review-step.html',
  styleUrl: './review-step.scss',
})
export class ReviewStep {
  protected readonly store = inject(RecipeStore);
  private readonly api = inject(ApiService);
  private readonly settings = inject(SettingsService);
  private readonly runs = inject(RunStore);
  private readonly toasts = inject(ToastService);
  private readonly router = inject(Router);
  private readonly scans = inject(ScanService);

  private readonly ctx = this.store.context;
  private readonly recipe = this.store.recipe;
  protected readonly derived = this.store.derived;

  // --- the sentence ------------------------------------------------------------------------------

  protected readonly archType = computed(() => (this.ctx().model ? String(nodeValues(this.ctx().model!)['type']) : ''));
  protected readonly archLabel = computed(() => ARCH_BY_TYPE.get(this.archType())?.label ?? this.archType() ?? 'a model');
  protected readonly archChoices: readonly ArchInfo[] = ARCH_CHOICES.map((t) => ARCH_BY_TYPE.get(t)).filter((a): a is ArchInfo => !!a);

  private readonly trainReq = computed<ScanRequest | null>(() => {
    const data = this.ctx().data;
    if (!data) {
      return null;
    }
    const v = nodeValues(data);
    const otf = v['type'] === 'otf';
    const gt = String(v['dataroot_gt'] ?? '').trim();
    const lq = String(v['dataroot_lq'] ?? '').trim();
    return gt && (otf || lq) ? { gt, lq: otf ? undefined : lq, scale: this.derived().scale, patch: this.derived().patch } : null;
  });
  protected readonly pairsText = computed(() => {
    const req = this.trainReq();
    const report = req ? this.scans.state(req)?.report : undefined;
    if (!report) {
      return 'your images';
    }
    return req!.lq === undefined ? `${report.gtCount.toLocaleString()} images` : `${report.pairs.toLocaleString()} pairs`;
  });

  protected readonly lastPhase = computed(() => this.recipe().phases.at(-1));
  protected readonly look = computed(() => lookOf(lossRows(this.ctx().losses), this.lastPhase()?.id ?? ''));
  protected readonly lookText = computed(() => (this.look() ? `${this.look()!.label.toLowerCase()} losses` : 'custom losses'));
  protected readonly looks = LOOKS;

  protected readonly total = computed(() => totalIters(this.recipe().phases));
  protected readonly lengths = LENGTHS;
  protected readonly customLength = signal('');

  protected readonly startMode = computed(() => (this.ctx().init ? String(nodeValues(this.ctx().init!)['mode']) : 'scratch'));
  protected readonly startText = computed(() => ({ pretrain: 'from weights', resume: 'resuming a state' })[this.startMode()] ?? 'from scratch');
  protected readonly starts = STARTS;
  protected readonly startPath = computed(() => {
    const init = this.ctx().init;
    const key = this.startMode() === 'resume' ? 'resume_state' : 'pretrain_network_g';
    return init ? String(nodeValues(init)[key] ?? '') : '';
  });

  private readonly valReq = computed<ScanRequest | null>(() => {
    const row = valDatasets(this.ctx().val)[0];
    const path = row ? (row.type === 'paired' ? row.dataroot_gt : row.dataroot_lq).trim() : '';
    return path ? { gt: path, scale: this.derived().scale, patch: this.derived().patch } : null;
  });
  protected readonly valFreq = computed(() => (this.ctx().val ? Number(nodeValues(this.ctx().val!)['val_freq']) : 0));
  protected readonly valText = computed(() => {
    if (!this.ctx().val) {
      return 'without validation';
    }
    const req = this.valReq();
    const n = req ? this.scans.state(req)?.report?.gtCount : undefined;
    // The mock scan cannot count a validation folder honestly; say how often, not how many.
    void n;
    return `validating every ${fmtUpdates(this.valFreq())}`;
  });
  protected readonly valFreqs = VAL_FREQS;
  protected readonly ckpt = computed(() => {
    const plan = this.ctx().plan;
    return fmtUpdates(plan ? Number(nodeValues(plan)['save_checkpoint_freq'] ?? 1000) : 1000);
  });

  // --- below the sentence ---------------------------------------------------------------------

  protected readonly fitsOpen = signal(false);
  protected readonly changes = computed(() => recipeChanges(this.recipe(), this.ctx()));
  protected readonly changesOpen = signal(false);
  protected readonly toCheck = computed(() => this.store.problems().filter((p) => p.severity !== 'info').sort((a, b) => (a.severity === b.severity ? 0 : a.severity === 'blocker' ? -1 : 1)));
  protected readonly checkOpen = signal(false);
  protected readonly blockers = computed(() => this.toCheck().filter((p) => p.severity === 'blocker'));
  protected readonly runsText = computed(() => {
    const phases = this.recipe().phases;
    if (phases.length < 2) {
      return '';
    }
    const second = phases[1];
    const how = second.init === 'continue' ? 'continues the state of phase 1' : 'fine-tunes from phase 1 weights';
    return `${phases.length} sequential runs — phase 2 ${how}`;
  });
  protected readonly launching = signal(false);
  protected readonly launchLabel = computed(() => {
    const b = this.blockers().length;
    if (b) {
      return `Launch · ${b} to fix`;
    }
    const n = this.store.compiled().phases.length;
    return n > 1 ? `Launch ${n} runs` : 'Launch';
  });
  protected readonly menu: readonly MenuItem[] = [
    { id: 'toml', label: 'Show TOML', icon: 'code' },
    { id: 'command', label: 'Copy command', icon: 'terminal' },
    { id: 'save', label: 'Save to library', icon: 'save', divider: true },
    { id: 'template', label: 'Save as template', icon: 'bookmark' },
  ];
  protected readonly fmtValue = fmtValue;
  protected readonly fmtUpdates = fmtUpdates;

  constructor() {
    // The counts in the sentence come from the same checks the Data step runs; after a reload they run again here.
    effect(() => {
      const reqs = [this.trainReq(), this.valReq()];
      untracked(() => reqs.forEach((req) => req && this.scans.ensure(req)));
    });
  }

  // --- edits ------------------------------------------------------------------------------------

  protected rename(name: string): void {
    const trimmed = name.trim();
    if (!trimmed || trimmed === this.recipe().name) {
      return;
    }
    const plan = this.ctx().plan;
    const runName = trimmed.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'run';
    this.store.dispatch({
      label: 'Rename recipe',
      apply: (r) => {
        const named = commands.setName(trimmed).apply(r);
        return plan ? commands.setValues(plan.id, { name: runName }).apply(named) : named;
      },
    });
  }

  protected setArch(arch: ArchInfo): void {
    const model = this.ctx().model;
    if (!model) {
      return;
    }
    this.store.setValues(model.id, { type: arch.type }, `Model: ${arch.label}`);
    // Window transformers need a patch that is a multiple of the window; the template patch is.
    const data = this.ctx().data;
    if (arch.windowSize && data && Number(nodeValues(data)['patch_size']) % arch.windowSize !== 0) {
      this.store.setValue(data.id, 'patch_size', arch.templatePatch);
    }
  }

  protected setLook(id: (typeof LOOKS)[number]['id']): void {
    const rows = lossRows(this.ctx().losses);
    const target = lossRowsFor(id);
    const phases = this.recipe().phases;
    let next: LossRow[];
    if (phases.length <= 1) {
      next = target;
    } else {
      // A curriculum keeps its earlier phases; the look applies to the last one through the schedule.
      const last = phases[phases.length - 1].id;
      next = rows.map((r) => {
        const t = target.find((x) => x.type === r.type);
        return { ...r, schedule: { ...r.schedule, [last]: { active: !!t, weight: t?.weight ?? r.weight } } };
      });
      for (const t of target) {
        if (!rows.some((r) => r.type === t.type)) {
          next.push({ ...t, enabled: false, schedule: { [last]: { active: true, weight: t.weight } } });
        }
      }
    }
    const gan = phases.some((p) => next.some((r) => r.type === 'gan_loss' && (r.schedule[p.id]?.active ?? r.enabled)));
    const ctx = this.ctx();
    this.store.dispatch(
      commands.applyDraftCommand(
        {
          losses: next,
          discriminator: gan ? (ctx.dmodel ? {} : { type: 'metagan' }) : null,
          optimD: gan ? (ctx.optimD ? {} : { ...OPTIM_D }) : null,
        },
        `Look: ${LOOKS.find((l) => l.id === id)?.label}`,
      ),
    );
  }

  /** A new total scales every phase proportionally (rounded to hundreds), so a curriculum keeps its shape. */
  protected setLength(total: number): void {
    const phases = this.recipe().phases;
    if (!phases.length || !Number.isFinite(total) || total < 1) {
      return;
    }
    const current = totalIters(phases);
    const scaled = phases.map((p) => ({ ...p, iters: Math.max(100, Math.round((p.iters / current) * total / 100) * 100) }));
    const drift = Math.round(total) - totalIters(scaled);
    const last = scaled[scaled.length - 1];
    scaled[scaled.length - 1] = { ...last, iters: Math.max(1, last.iters + drift) };
    this.store.dispatch(commands.setPhases(scaled, 'Training length'));
  }

  protected setCustomLength(): void {
    const raw = this.customLength().replace(/[,\s_]/g, '').toLowerCase();
    const m = /^(\d+(?:\.\d+)?)(k|m)?$/.exec(raw);
    if (!m) {
      return;
    }
    const n = Math.round(Number(m[1]) * (m[2] === 'k' ? 1000 : m[2] === 'm' ? 1_000_000 : 1));
    if (n > 0) {
      this.setLength(n);
      this.customLength.set('');
    }
  }

  protected setStart(mode: string): void {
    const init = this.ctx().init;
    if (init) {
      this.store.setValues(init.id, { mode }, 'Start from');
    } else {
      this.store.dispatch(commands.applyDraftCommand({ init: { mode } }, 'Start from'));
    }
  }

  protected setStartPath(value: string): void {
    const init = this.ctx().init;
    if (init) {
      this.store.setValue(init.id, this.startMode() === 'resume' ? 'resume_state' : 'pretrain_network_g', value.trim());
    }
  }

  protected setValFreq(n: number): void {
    const val = this.ctx().val;
    if (val) {
      this.store.setValue(val.id, 'val_freq', n);
    }
  }

  protected setDataValue(key: 'patch_size' | 'batch_size', raw: string): void {
    const n = Math.round(Number(raw));
    const data = this.ctx().data;
    if (data && Number.isFinite(n) && n > 0) {
      this.store.setValue(data.id, key, n);
    }
  }

  protected revert(change: Change): void {
    if (change.kind === 'phases') {
      const preset = this.recipe().meta.presetPhases;
      if (preset) {
        this.store.dispatch(commands.setPhases(preset, 'Reset training length'));
      }
      return;
    }
    if (change.kind === 'override' && change.phaseId) {
      const baseline = this.recipe().meta.presetOverrides?.[change.nodeType]?.[change.phaseId]?.[change.key];
      this.store.setPhaseValue(change.nodeId, change.phaseId, change.key, baseline);
      return;
    }
    this.store.setValue(change.nodeId, change.key, change.from);
  }

  protected more(section: SectionId): void {
    void this.router.navigate(['/build/list'], { queryParams: { section } });
  }

  protected focusProblem(problem: Problem): void {
    this.store.activePhaseId.set(problem.phaseId ?? null);
    const node = problem.nodeId ? this.store.node(problem.nodeId) : undefined;
    const wired = node && Object.values(this.ctx()).some((n) => n?.id === node.id);
    if (node && !wired) {
      this.store.selectOne(node.id);
      void this.router.navigate(['/build/graph'], { queryParams: { node: node.id, field: problem.fieldKey ?? null } });
      return;
    }
    const section = node ? sectionForField(node.type, problem.fieldKey) : undefined;
    void this.router.navigate(['/build/list'], { queryParams: { section: section?.id ?? null, node: problem.nodeId ?? null, field: problem.fieldKey ?? null } });
  }

  // --- actions -----------------------------------------------------------------------------------

  protected async launch(): Promise<void> {
    if (this.blockers().length) {
      this.focusProblem(this.blockers()[0]);
      return;
    }
    if (this.launching()) {
      return;
    }
    this.launching.set(true);
    try {
      const run = await this.api.launch(this.recipe(), this.settings.settings().platform, this.settings.settings().simSpeed);
      await this.runs.refresh().catch(() => undefined);
      await this.router.navigate(['/runs', run.id]);
    } catch (error) {
      if (error instanceof LaunchRejected) {
        this.toasts.show('danger', error.problems[0]?.message ?? 'The backend rejected the recipe.');
      } else {
        this.toasts.show('danger', 'Could not launch: is the backend running? (npm run api)');
      }
    } finally {
      this.launching.set(false);
    }
  }

  protected async onMenu(id: string): Promise<void> {
    switch (id) {
      case 'toml':
        await this.router.navigate(['/build/toml']);
        break;
      case 'command': {
        const command = this.store.compiled().phases[0]?.command ?? '';
        const ok = command ? await copyText(command) : false;
        this.toasts.show(ok ? 'ok' : 'warn', ok ? 'Command copied.' : 'Clipboard not available.');
        break;
      }
      case 'save':
        await this.save();
        break;
      case 'template':
        await this.saveAsTemplate();
        break;
    }
  }

  private async save(): Promise<void> {
    this.store.saving.set(true);
    const hash = this.store.hash();
    try {
      await this.api.saveRecipe(this.recipe());
      this.store.markSaved(hash);
      this.toasts.show('ok', `Saved "${this.recipe().name}".`);
    } catch {
      this.toasts.show('danger', 'Could not save: is the backend running? (npm run api)');
    } finally {
      this.store.saving.set(false);
    }
  }

  private async saveAsTemplate(): Promise<void> {
    const name = window.prompt('Template name', `${this.recipe().name} template`);
    if (!name?.trim()) {
      return;
    }
    const base = this.recipe();
    const copy: Recipe = { ...base, id: newId('r'), name: name.trim(), meta: { ...base.meta, tags: [...new Set([...base.meta.tags, 'template'])] } };
    try {
      await this.api.saveRecipe(copy);
      this.toasts.show('ok', `Template "${copy.name}" saved to the library.`);
    } catch {
      this.toasts.show('danger', 'Could not save: is the backend running?');
    }
  }
}
