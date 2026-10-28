import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import type { TomlLine, TomlOrigin } from '@shared/compile/toml-writer';

interface Token {
  readonly text: string;
  readonly cls: string;
}

interface LineView {
  readonly index: number;
  readonly line: TomlLine;
  readonly tokens: readonly Token[];
  readonly highlighted: boolean;
}

/**
 * Syntax-highlighted TOML with a source map: hovering or clicking a line reports which node and
 * field produced it, and lines from `highlightNodeId` are emphasised.
 */
@Component({
  selector: 'app-toml-code',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <pre class="code" [class.code--wrap]="wrap()"><code>@for (l of views(); track l.index) {<span class="line" [class.line--hl]="l.highlighted" [class.line--origin]="!!l.line.origin" [attr.data-node]="l.line.origin?.nodeId ?? null" (mouseenter)="originHover.emit(l.line.origin ?? null)" (mouseleave)="originHover.emit(null)" (click)="l.line.origin && originClick.emit(l.line.origin)"><span class="ln">{{ l.index + 1 }}</span>@for (t of l.tokens; track $index) {<span [class]="t.cls">{{ t.text }}</span>}@if (editable() && l.line.kind === 'header' && l.line.origin) {<button type="button" class="edit" (click)="originClick.emit(l.line.origin); $event.stopPropagation()">Edit</button>}
</span>}</code></pre>
  `,
  styles: `
    :host { display: block; min-height: 0; }
    .code { margin: 0; padding: 10px 0; background: var(--panel-sunken); border-radius: var(--radius); font-family: var(--font-mono); font-size: var(--fs-sm); line-height: 1.55; overflow: auto; tab-size: 2; }
    .code--wrap { white-space: pre-wrap; }
    .line { display: block; padding: 0 12px 0 0; }
    .line--origin { cursor: pointer; }
    .line--origin:hover { background: rgba(255, 255, 255, 0.04); }
    .line--hl { background: var(--accent-soft); }
    .ln { display: inline-block; width: 40px; padding-right: 12px; text-align: right; color: var(--text-faint); user-select: none; }
    .t-header { color: #8fbdff; font-weight: 600; }
    .t-comment { color: var(--text-faint); font-style: italic; }
    .t-key { color: #cfd5df; }
    .t-eq { color: var(--text-faint); }
    .t-str { color: #6cc5a0; }
    .t-num { color: #e3a86d; }
    .t-bool { color: #c9a6ff; }
    .t-punct { color: var(--text-muted); }
    .edit { display: inline-block; margin-left: 12px; min-height: 20px; line-height: 20px; padding: 0 8px; vertical-align: baseline; border-radius: var(--radius-sm); color: var(--text-faint); font-family: var(--font); font-size: var(--fs-xs); }
    .edit:hover, .edit:focus-visible { color: var(--accent-text); background: var(--panel-raised); }
  `,
})
export class TomlCode {
  readonly lines = input.required<readonly TomlLine[]>();
  readonly highlightNodeId = input<string | null>(null);
  readonly wrap = input(false);
  /** Render an "Edit" button on section headers that have an origin. */
  readonly editable = input(false);
  readonly originHover = output<TomlOrigin | null>();
  readonly originClick = output<TomlOrigin>();

  protected readonly views = computed<LineView[]>(() => {
    const hl = this.highlightNodeId();
    return this.lines().map((line, index) => ({
      index,
      line,
      tokens: tokenize(line),
      highlighted: !!hl && line.origin?.nodeId === hl,
    }));
  });
}

function tokenize(line: TomlLine): Token[] {
  switch (line.kind) {
    case 'header':
      return [{ text: line.text, cls: 't-header' }];
    case 'comment':
      return [{ text: line.text, cls: 't-comment' }];
    case 'blank':
      return [{ text: '', cls: '' }];
    default: {
      const eq = line.text.indexOf(' = ');
      if (eq < 0) {
        return [{ text: line.text, cls: 't-key' }];
      }
      const key = line.text.slice(0, eq);
      const value = line.text.slice(eq + 3);
      return [{ text: key, cls: 't-key' }, { text: ' = ', cls: 't-eq' }, ...tokenizeValue(value)];
    }
  }
}

function tokenizeValue(value: string): Token[] {
  const tokens: Token[] = [];
  const re = /("(?:[^"\\]|\\.)*"|'[^']*')|(\btrue\b|\bfalse\b)|(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)|([\[\],{}=])|(\s+)|([^\s\[\],{}="']+)/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(value))) {
    if (m[1]) {
      tokens.push({ text: m[1], cls: 't-str' });
    } else if (m[2]) {
      tokens.push({ text: m[2], cls: 't-bool' });
    } else if (m[3]) {
      tokens.push({ text: m[3], cls: 't-num' });
    } else if (m[4]) {
      tokens.push({ text: m[4], cls: 't-punct' });
    } else {
      tokens.push({ text: m[0], cls: 't-key' });
    }
  }
  return tokens;
}
