import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, DestroyRef, computed, effect, inject, signal, untracked } from '@angular/core';
import { RouterLink } from '@angular/router';
import { DEGRADATION_TEMPLATE } from '@shared/catalog/degradations';
import { NODE_TYPES, nodeSpec } from '@shared/catalog/node-catalog';
import { defaultValues, type FieldSpec } from '@shared/catalog/types';
import { nodeValues, valDatasets, type ValDatasetRow } from '@shared/compile/graph';
import type { RecipeDraft } from '@shared/recipe-builder';
import { newId } from '@shared/recipe.model';
import { baselineValue, isChanged } from '@shared/sections';
import * as commands from '../core/commands';
import { RecipeStore } from '../core/recipe-store';
import { FieldRow } from '../ui/field-row';
import { Icon } from '../ui/icon';
import { Popover } from './popover';
import { ScanService, scanIssues, type ScanRequest } from './scan.service';
import { fieldOf, nodeOf, problemsOf } from './util';

const DATA = NODE_TYPES.trainDataset;
const SCAN_DELAY_MS = 600;
const SCALES = [1, 2, 3, 4, 8] as const;

type ValMode = 'paired' | 'input' | 'skip';

const VAL_GT: FieldSpec = {
  key: 'val_gt',
  label: 'Validation high-res',
  help: 'A small folder (8–10 varied images) scored every validation. Keep it out of the training set.',
  tomlKey: 'datasets.val.dataroot_gt',
  widget: { kind: 'path', mode: 'directory' },
  default: '',
  required: true,
  level: 'essentials',
};
const VAL_LQ: FieldSpec = { ...VAL_GT, key: 'val_lq', label: 'Validation low-res', help: 'The matching low-res images, same file names.', tomlKey: 'datasets.val.dataroot_lq' };
const VAL_INPUT: FieldSpec = { ...VAL_LQ, key: 'val_input', label: 'Input images', help: 'Real low-res inputs, upscaled and saved for inspection. No ground truth, so no metrics.' };

/**
 * Data: where the images are and how to validate. Five controls, one result line; the check runs
 * by itself when a folder changes. Patch and batch are not asked here — the preset has them.
 */
@Component({
  selector: 'app-data-step',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, DecimalPipe, FieldRow, Icon, Popover],
  templateUrl: './data-step.html',
  styleUrl: './data-step.scss',
})
export class DataStep {
  private readonly store = inject(RecipeStore);
  private readonly scans = inject(ScanService);

  protected readonly data = computed(() => nodeOf(this.store.context(), this.store.recipe(), DATA));
  private readonly values = computed(() => (this.data() ? nodeValues(this.data()!) : defaultValues(nodeSpec(DATA))));
  protected readonly otf = computed(() => this.values()['type'] === 'otf');
  protected readonly gt = computed(() => String(this.values()['dataroot_gt'] ?? ''));
  protected readonly lq = computed(() => String(this.values()['dataroot_lq'] ?? ''));
  protected readonly scale = computed(() => this.store.derived().scale);

  protected readonly gtField = fieldOf(DATA, 'dataroot_gt', { label: 'High-res folder' });
  protected readonly lqField = fieldOf(DATA, 'dataroot_lq', { label: 'Low-res folder' });
  protected readonly valGtField = VAL_GT;
  protected readonly valLqField = VAL_LQ;
  protected readonly valInputField = VAL_INPUT;

  protected readonly val = computed(() => this.store.context().val);
  private readonly valRow = computed<ValDatasetRow | undefined>(() => valDatasets(this.val())[0]);
  protected readonly valMode = computed<ValMode>(() => (!this.val() ? 'skip' : this.valRow()?.type === 'single' ? 'input' : 'paired'));
  protected readonly valGt = computed(() => this.valRow()?.dataroot_gt ?? '');
  protected readonly valLq = computed(() => this.valRow()?.dataroot_lq ?? '');
  protected readonly valProblems = computed(() => problemsOf(this.store.problems(), this.val()?.id, 'datasets'));

  // --- the check ------------------------------------------------------------------------------

  private readonly scanReq = computed<ScanRequest | null>(() => {
    const gt = this.gt().trim();
    if (!gt || (!this.otf() && !this.lq().trim())) {
      return null;
    }
    return { gt, lq: this.otf() ? undefined : this.lq().trim(), scale: this.scale(), patch: this.store.derived().patch };
  });
  protected readonly scan = computed(() => {
    const req = this.scanReq();
    return req ? this.scans.state(req) : undefined;
  });
  protected readonly report = computed(() => (this.scan()?.status === 'done' ? this.scan()!.report! : null));
  protected readonly issues = computed(() => (this.report() ? scanIssues(this.report()!) : []));
  protected readonly blockerCount = computed(() => this.issues().filter((i) => i.severity === 'blocker').length);
  /** Scale is "detected" when a paired check succeeded; otherwise the user picks it. */
  protected readonly detected = computed(() => !this.otf() && !!this.report());
  protected readonly moreScales = signal(false);
  protected readonly scales = computed(() => (this.moreScales() || this.scale() === 3 || this.scale() === 8 ? SCALES : SCALES.filter((s) => s !== 3 && s !== 8)));
  protected readonly allScales = SCALES;

  constructor() {
    const destroy = inject(DestroyRef);
    let timer: ReturnType<typeof setTimeout> | null = null;
    effect(() => {
      const req = this.scanReq();
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (req && !untracked(() => this.scans.state(req))) {
        timer = setTimeout(() => this.scans.ensure(req), SCAN_DELAY_MS);
      }
    });
    destroy.onDestroy(() => {
      if (timer) {
        clearTimeout(timer);
      }
    });
  }

  // --- reads for the template -------------------------------------------------------------------

  protected changed(key: string): boolean {
    const node = this.data();
    return !!node && isChanged(this.store.recipe(), node, key);
  }

  protected baseline(key: string): unknown {
    const node = this.data();
    return node ? baselineValue(this.store.recipe(), node, key) : '';
  }

  protected problems(key: string) {
    return problemsOf(this.store.problems(), this.data()?.id, key);
  }

  // --- writes -------------------------------------------------------------------------------

  private apply(draft: RecipeDraft): void {
    this.store.dispatch(commands.applyDraftCommand(draft, 'Setup: data'));
  }

  protected setSource(otf: boolean): void {
    if (otf === this.otf()) {
      return;
    }
    // Switching to OTF adds the shipped degradation table once; the user's own table is never overwritten.
    this.apply({ dataset: { type: otf ? 'otf' : 'paired' }, degradation: otf ? (this.store.context().degradation ? {} : { ...DEGRADATION_TEMPLATE }) : null });
  }

  protected setFolder(key: 'dataroot_gt' | 'dataroot_lq', value: unknown): void {
    this.apply({ dataset: { [key]: String(value ?? '').trim() } });
  }

  protected reset(key: 'dataroot_gt' | 'dataroot_lq'): void {
    this.apply({ dataset: { [key]: this.baseline(key) } });
  }

  protected setScale(scale: number): void {
    this.apply({ generator: { scale: String(scale) } });
  }

  protected setValMode(mode: ValMode): void {
    if (mode === this.valMode()) {
      return;
    }
    if (mode === 'skip') {
      this.apply({ validation: null });
      return;
    }
    const rows = valDatasets(this.val());
    const first: ValDatasetRow = rows[0] ?? { id: newId('v'), name: 'val', type: 'paired', dataroot_gt: '', dataroot_lq: '' };
    this.apply({ validation: { datasets: [{ ...first, type: mode === 'paired' ? 'paired' : 'single' }, ...rows.slice(1)] } });
  }

  protected setValFolder(key: 'dataroot_gt' | 'dataroot_lq', value: unknown): void {
    const rows = valDatasets(this.val());
    const first: ValDatasetRow = rows[0] ?? { id: newId('v'), name: 'val', type: 'paired', dataroot_gt: '', dataroot_lq: '' };
    this.apply({ validation: { datasets: [{ ...first, [key]: String(value ?? '').trim() }, ...rows.slice(1)] } });
  }
}
