import { ChangeDetectionStrategy, Component, computed, input, output, signal } from '@angular/core';
import { importToml, type ImportResult } from '@shared/compile/import-toml';
import { emptyRecipe } from '@shared/recipe.model';
import { Icon } from '../../ui/icon';
import { readPickedFile } from '../../library/util';

const PRESERVED_SHOWN = 14;

/**
 * Transactional TOML import: paste or pick a file, preview what the importer made of it (nodes,
 * warnings, keys preserved as native extras), then let the host decide what "apply" means.
 */
@Component({
  selector: 'app-toml-import',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon],
  template: `
    <div class="row">
      <input class="input input--name" type="text" [value]="name()" (input)="name.set($any($event.target).value)" aria-label="Recipe name for the import" placeholder="name" />
      <label class="btn btn--sm" title="Pick a .toml file">
        <app-icon name="upload" [size]="13" /> File…
        <input type="file" accept=".toml,text/plain" hidden (change)="pickFile($event)" />
      </label>
    </div>
    <textarea
      class="textarea mono"
      [rows]="rows()"
      spellcheck="false"
      placeholder="Paste a NeoSR train_*.toml here"
      aria-label="TOML text"
      [value]="text()"
      (input)="setText($any($event.target).value)"
    ></textarea>
    <div class="row">
      <button type="button" class="btn btn--sm" [disabled]="!text().trim()" (click)="preview()" title="Parse the TOML without changing anything">
        <app-icon name="eye" [size]="13" /> Preview
      </button>
      <span class="spacer"></span>
      @if (result(); as r) {
        <button type="button" class="btn btn--primary btn--sm" (click)="apply.emit(r)" [title]="applyLabel()">
          <app-icon name="import" [size]="13" /> {{ applyLabel() }}
        </button>
      }
    </div>
    @if (error(); as e) {
      <p class="err"><app-icon name="alert-triangle" [size]="13" /> {{ e }}</p>
    }
    @if (result(); as r) {
      <div class="preview">
        <div class="row stats">
          <span><b>{{ r.recipe.nodes.length }}</b> nodes</span>
          <span><b>{{ r.recipe.phases.length }}</b> phase{{ r.recipe.phases.length === 1 ? '' : 's' }}</span>
          <span><b>{{ r.preserved.length }}</b> preserved</span>
          <span><b>{{ r.warnings.length }}</b> warning{{ r.warnings.length === 1 ? '' : 's' }}</span>
        </div>
        @for (w of r.warnings; track $index) {
          <p class="warn">{{ w }}</p>
        }
        @if (r.preserved.length) {
          <p class="faint label">Kept verbatim as native extras</p>
          <div class="paths">
            @for (p of shownPreserved(); track p) {
              <span class="path mono">{{ p }}</span>
            }
            @if (r.preserved.length > shownPreserved().length) {
              <span class="faint">+{{ r.preserved.length - shownPreserved().length }} more</span>
            }
          </div>
        }
      </div>
    }
  `,
  styles: `
    :host { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
    .input--name { flex: 1; }
    .textarea { font-size: var(--fs-sm); line-height: 1.5; }
    .err { color: var(--danger); font-size: var(--fs-sm); display: flex; gap: 6px; align-items: flex-start; }
    .preview { display: flex; flex-direction: column; gap: 6px; padding: 8px 10px; border-radius: var(--radius); background: var(--panel-sunken); border: 1px solid var(--border-soft); font-size: var(--fs-sm); }
    .stats { flex-wrap: wrap; gap: 12px; color: var(--text-muted); }
    .stats b { color: var(--text); }
    .warn { color: var(--warn); }
    .label { font-size: var(--fs-xs); text-transform: uppercase; letter-spacing: 0.04em; }
    .paths { display: flex; flex-wrap: wrap; gap: 4px; }
    .path { padding: 1px 6px; border-radius: var(--radius-sm); background: var(--panel-raised); font-size: var(--fs-xs); color: var(--text-muted); }
  `,
})
export class TomlImport {
  readonly applyLabel = input('Open in Graph');
  readonly rows = input(8);
  readonly apply = output<ImportResult>();

  protected readonly text = signal('');
  protected readonly name = signal('imported');
  protected readonly result = signal<ImportResult | null>(null);
  protected readonly error = signal<string | null>(null);
  protected readonly shownPreserved = computed(() => this.result()?.preserved.slice(0, PRESERVED_SHOWN) ?? []);

  protected setText(text: string): void {
    this.text.set(text);
    this.result.set(null);
    this.error.set(null);
  }

  protected async pickFile(event: Event): Promise<void> {
    const file = await readPickedFile(event);
    if (file) {
      this.name.set(file.name.replace(/\.toml$/i, ''));
      this.setText(file.text);
      this.preview();
    }
  }

  protected preview(): void {
    const name = this.name().trim() || 'imported';
    try {
      this.result.set(importToml(this.text(), emptyRecipe(name), name));
      this.error.set(null);
    } catch (e) {
      this.result.set(null);
      this.error.set(`Not valid TOML: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
}
