import { ChangeDetectionStrategy, Component, computed, inject, output, signal } from '@angular/core';
import { Router } from '@angular/router';
import { importToml, type ImportResult } from '@shared/compile/import-toml';
import { emptyRecipe, snapshotPreset } from '@shared/recipe.model';
import { RecipeStore } from '../core/recipe-store';
import { readPickedFile } from '../library/util';
import { Drawer } from '../ui/drawer';
import { Icon } from '../ui/icon';

const PRESERVED_SHOWN = 10;

/** Import a NeoSR TOML: paste or pick a file, see what it becomes, open it in Build. */
@Component({
  selector: 'app-import-sheet',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Drawer, Icon],
  template: `
    <app-drawer [open]="true" title="Import TOML" [width]="440" (closed)="closed.emit()">
      <div class="body">
        <label class="btn btn--sm pick">
          <app-icon name="upload" [size]="13" /> Choose a .toml file
          <input type="file" accept=".toml,text/plain" hidden (change)="pickFile($event)" />
        </label>
        <textarea class="textarea mono" rows="12" spellcheck="false" placeholder="…or paste a NeoSR train_*.toml here" aria-label="TOML text" [value]="text()" (input)="setText($any($event.target).value)"></textarea>
        @if (error(); as e) {
          <p class="err"><app-icon name="alert-triangle" [size]="13" /> {{ e }}</p>
        }
        @if (result(); as r) {
          <div class="preview">
            <p>{{ r.recipe.nodes.length }} nodes · {{ r.recipe.phases.length }} phase{{ r.recipe.phases.length === 1 ? '' : 's' }}@if (r.preserved.length) { · {{ r.preserved.length }} settings kept as native extras }</p>
            @for (w of r.warnings; track $index) {
              <p class="warn"><app-icon name="alert-triangle" [size]="13" /> {{ w }}</p>
            }
            @if (r.preserved.length) {
              <p class="faint paths">{{ shownPreserved().join(' · ') }}@if (r.preserved.length > shownPreserved().length) { · +{{ r.preserved.length - shownPreserved().length }} more }</p>
            }
          </div>
          <button type="button" class="btn btn--primary" (click)="apply(r)"><app-icon name="layers" [size]="14" /> Open in Build</button>
        }
      </div>
    </app-drawer>
  `,
  styles: `
    .body { display: flex; flex-direction: column; gap: 10px; }
    .pick { align-self: flex-start; cursor: pointer; }
    .textarea { font-size: var(--fs-sm); line-height: 1.5; }
    .err, .warn { display: flex; gap: 6px; align-items: flex-start; font-size: var(--fs-sm); }
    .err { color: var(--danger); }
    .warn { color: var(--warn); }
    .preview { display: flex; flex-direction: column; gap: 4px; padding: 8px 10px; border-radius: var(--radius); background: var(--panel-sunken); font-size: var(--fs-sm); }
    .paths { font-family: var(--font-mono); font-size: var(--fs-xs); overflow-wrap: anywhere; }
    .btn--primary { align-self: flex-start; }
  `,
})
export class ImportSheet {
  readonly closed = output<void>();
  private readonly store = inject(RecipeStore);
  private readonly router = inject(Router);

  protected readonly text = signal('');
  protected readonly name = signal('imported');
  protected readonly result = signal<ImportResult | null>(null);
  protected readonly error = signal<string | null>(null);
  protected readonly shownPreserved = computed(() => this.result()?.preserved.slice(0, PRESERVED_SHOWN) ?? []);

  protected setText(text: string): void {
    this.text.set(text);
    this.preview();
  }

  protected async pickFile(event: Event): Promise<void> {
    const file = await readPickedFile(event);
    if (file) {
      this.name.set(file.name.replace(/\.toml$/i, ''));
      this.setText(file.text);
    }
  }

  private preview(): void {
    if (!this.text().trim()) {
      this.result.set(null);
      this.error.set(null);
      return;
    }
    try {
      this.result.set(importToml(this.text(), emptyRecipe(this.name()), this.name()));
      this.error.set(null);
    } catch (e) {
      this.result.set(null);
      this.error.set(`Not valid TOML: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  protected apply(r: ImportResult): void {
    if (this.store.dirty() && !confirm(`Replace the current draft "${this.store.recipe().name}"? Unsaved changes will be lost.`)) {
      return;
    }
    // The import is its own baseline: "changed from preset" starts empty, not "everything differs from the catalog".
    this.store.load(snapshotPreset(r.recipe, 'imported TOML'));
    this.closed.emit();
    void this.router.navigate(['/build']);
  }
}
