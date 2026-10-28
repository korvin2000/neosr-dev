/**
 * A small TOML writer with a source map. NeoSR configs are flat enough (tables, arrays of
 * scalars, scalars) that a purpose-built writer beats a generic one: section order follows the
 * NeoSR convention, Windows paths come out as literal strings, and every line remembers which
 * node and field produced it, so the TOML view can highlight the node under the cursor.
 */

export interface TomlOrigin {
  readonly nodeId: string;
  readonly fieldKey?: string;
}

export interface TomlLine {
  readonly text: string;
  readonly origin?: TomlOrigin;
  readonly kind: 'header' | 'pair' | 'comment' | 'blank';
}

/** A value plus where it came from. Plain values are allowed too. */
export interface Annotated {
  readonly __value: unknown;
  readonly __origin?: TomlOrigin;
  /** Print integral numbers with a decimal point (`1.0`), as NeoSR templates write float keys. */
  readonly __float?: boolean;
}

export function annotate(value: unknown, origin?: TomlOrigin, float = false): Annotated {
  return { __value: value, __origin: origin, __float: float };
}

function isAnnotated(value: unknown): value is Annotated {
  return typeof value === 'object' && value !== null && '__value' in value;
}

/** A table section in output order. `comment` is printed above the header. */
export interface TomlSection {
  readonly path: string;
  readonly comment?: string;
  readonly origin?: TomlOrigin;
  readonly entries: Readonly<Record<string, unknown>>;
}

export interface TomlDoc {
  /** Top-level scalar keys, printed first. */
  readonly root: Readonly<Record<string, unknown>>;
  readonly sections: readonly TomlSection[];
}

export function writeToml(doc: TomlDoc): TomlLine[] {
  const lines: TomlLine[] = [];
  for (const [key, value] of Object.entries(doc.root)) {
    const line = pairLine(key, value);
    if (line) {
      lines.push(line);
    }
  }
  for (const section of doc.sections) {
    const entries = Object.entries(section.entries).filter(([, v]) => isPresent(unwrap(v)));
    if (entries.length === 0) {
      continue;
    }
    lines.push({ text: '', kind: 'blank' });
    if (section.comment) {
      lines.push({ text: `# ${section.comment}`, kind: 'comment' });
    }
    lines.push({ text: `[${section.path}]`, kind: 'header', origin: section.origin });
    for (const [key, value] of entries) {
      const line = pairLine(key, value);
      if (line) {
        lines.push(line);
      }
    }
  }
  return lines;
}

export function tomlText(lines: readonly TomlLine[]): string {
  return lines.map((l) => l.text).join('\n') + '\n';
}

export function unwrap(value: unknown): unknown {
  return isAnnotated(value) ? value.__value : value;
}

function isPresent(value: unknown): boolean {
  return value !== undefined && value !== null && value !== '';
}

function pairLine(key: string, raw: unknown): TomlLine | null {
  const origin = isAnnotated(raw) ? raw.__origin : undefined;
  const float = isAnnotated(raw) ? raw.__float === true : false;
  const value = unwrap(raw);
  if (!isPresent(value)) {
    return null;
  }
  return { text: `${formatKey(key)} = ${formatValue(value, float)}`, kind: 'pair', origin };
}

function formatKey(key: string): string {
  return /^[A-Za-z0-9_-]+$/.test(key) ? key : JSON.stringify(key);
}

export function formatValue(value: unknown, float = false): string {
  if (typeof value === 'string') {
    return formatString(value);
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  if (typeof value === 'number') {
    return float && Number.isInteger(value) ? `${value}.0` : formatNumber(value);
  }
  if (Array.isArray(value)) {
    return `[ ${value.map((v) => formatValue(v, float)).join(', ')} ]`;
  }
  if (typeof value === 'object' && value !== null) {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => isPresent(v))
      .map(([k, v]) => `${formatKey(k)} = ${formatValue(v)}`);
    return `{ ${entries.join(', ')} }`;
  }
  return JSON.stringify(String(value));
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return value > 0 ? 'inf' : value < 0 ? '-inf' : 'nan';
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  // Small learning rates read better in scientific notation, exactly as the templates write them.
  if (Math.abs(value) < 0.001) {
    return value.toExponential().replace('e+', 'e');
  }
  return String(value);
}

function formatString(value: string): string {
  // Paths with backslashes: literal strings, which is what the NeoSR wiki recommends on Windows.
  if (value.includes('\\') && !value.includes("'")) {
    return `'${value}'`;
  }
  return JSON.stringify(value);
}
