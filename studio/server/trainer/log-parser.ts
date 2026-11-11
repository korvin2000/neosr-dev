/**
 * Pure parser of `train.py`'s console output (the "neosr" logger's stream handler on stderr, plus
 * `config error:` and argparse lines) into the run event bodies the simulator emits. The formats
 * are the trainer's own: `neosr/utils/logger.py` (MessageLogger, stream handler), `train.py`
 * (`train`, `main`) and `neosr/models/base.py` (`log_metrics`, `save`, `save_best`). Feed it whole
 * lines (no `\r`/`\n`) in stream order; `native` is the trainer's update counter.
 */
import type { RunEventBody } from '../../shared/run.model';

export interface ParsedEvent {
  /** The trainer's (native) update counter when the line was logged. */
  readonly native: number;
  readonly body: RunEventBody;
}

type Level = 'info' | 'warn' | 'error';

const ANSI = /\x1b\[[0-9;]*m/g;
/** Stream handler (`%d-%m-%Y %I:%M %p |`) or file handler (`asctime`) record header. */
const HEADER = /^(?:\d{2}-\d{2}-\d{4} \d{2}:\d{2}(?: [^|]*)? \||\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (DEBUG|INFO|WARNING|ERROR|CRITICAL): ?(.*)$/;
const PROGRESS = /^\[ epoch:\s*(\d+) \] \[ iter:\s*([\d,]+) \]/;
const FIELD = /\[ ([^:\]]+): ([^\]]*?) \]/g;
const PERFORMANCE = /^([\d.]+) it\/s(?:, data (\d+)%)?/;
const MEM = /^([\d.]+) \/ ([\d.]+) GiB$/;
const ETA = /^(?:(\d+) days?, )?(\d+):(\d{2}):(\d{2})$/;
const STARTING = /^Starting at epoch \d+, update (\d+)\.$/;
const SAVING = /^saving update (\d+)$/;
const EXPORT_SIZE = /^one export set is ([\d.]+) MiB/;
const STOP_REQUESTED = /^stop requested: saving update (\d+)$/;
const END = /^(End of training|Stopped) at update (\d+)\./;
const NOTHING = /^nothing to do: update (\d+) >= total_iter/;
const BEST = /^best \S+ on \S+: \S+ @ (\d+), saved (\S+)$/;
const VALIDATION = /^Validation (.+)$/;
const METRIC = /^\s*# (\S+): (\S+?)(?:\.{8} Best:.*)?$/;
const NETWORK_D = /^\[\w+\] network_d: /;
const CONFIG_ERROR = /^config error: (.*)$/;
const ARGPARSE_ERROR = /^neosr: error: (.*)$/;
const TRACEBACK = /^Traceback \(most recent call last\):$/;
/** tqdm bars (validation `pbar`), redrawn with `\r`. */
const TQDM = /\d+%\|.*\| *\d+\/\d+/;
const SERIES = /^(l|out|grad_norm|clipped|skipped)_([a-z])(?:_(.+))?$/;

export class TrainerLogParser {
  /** Last update the trainer reported. */
  native = 0;
  /** How the trainer said it ended (`End of training`, `nothing to do` / `Stopped`). */
  outcome?: 'completed' | 'stopped';
  /** The most specific error seen: `config error:`, argparse, the traceback's last line, an ERROR record. */
  error?: string;

  private level: Level = 'info';
  private hasD = false;
  private exportMb = 0;
  private val?: { readonly dataset: string; readonly metrics: Record<string, number> };
  /** `saving update N` is logged before `save()`: the checkpoint is reported once the next line shows it returned. */
  private pendingSave?: number;
  private traceback = false;

  /** `native`: the update the process starts from (a resume), until it reports one. */
  constructor(
    private readonly phaseIndex = 0,
    native = 0,
  ) {
    this.native = native;
  }

  push(raw: string): ParsedEvent[] {
    const line = raw.replace(ANSI, '').trimEnd();
    const out: ParsedEvent[] = [];
    if (TQDM.test(line)) {
      return out;
    }
    const header = HEADER.exec(line);
    if (header) {
      const level = levelOf(header[1] as string);
      const message = header[2] as string;
      const size = EXPORT_SIZE.exec(message);
      if (size) {
        this.exportMb = Number(size[1]);
      }
      this.closeBlocks(out, level !== 'error');
      this.level = level;
      this.emit(out, { type: 'log', level, line: message });
      this.record(message, level, out);
      return out;
    }
    if (!line.trim()) {
      if (this.val && Object.keys(this.val.metrics).length) {
        this.flushValidation(out);
      }
      return out;
    }
    this.continuation(line, out);
    return out;
  }

  /** End of the stream: closes open blocks; a save not confirmed by a later line counts only when the process exited cleanly. */
  end(cleanExit: boolean): ParsedEvent[] {
    const out: ParsedEvent[] = [];
    this.closeBlocks(out, cleanExit);
    return out;
  }

  private record(message: string, level: Level, out: ParsedEvent[]): void {
    let m: RegExpExecArray | null;
    if ((m = PROGRESS.exec(message))) {
      this.progress(Number((m[2] as string).replaceAll(',', '')), message.slice(m[0].length), out);
    } else if ((m = STARTING.exec(message))) {
      this.native = Number(m[1]);
      this.emit(out, { type: 'status', status: 'running' });
    } else if ((m = SAVING.exec(message))) {
      this.native = Number(m[1]);
      this.pendingSave = this.native;
    } else if ((m = STOP_REQUESTED.exec(message))) {
      this.native = Number(m[1]);
      this.emit(out, { type: 'status', status: 'saving', message: 'stop requested' });
    } else if ((m = END.exec(message))) {
      this.native = Number(m[2]);
      this.outcome = m[1] === 'Stopped' ? 'stopped' : 'completed';
      this.emit(out, { type: 'checkpoint', files: this.files(this.native, true), sizeMb: this.exportMb, state: true });
    } else if ((m = NOTHING.exec(message))) {
      this.native = Number(m[1]);
      this.outcome = 'completed';
      this.emit(out, { type: 'progress', itersPerSec: 0, etaSec: 0, phaseIndex: this.phaseIndex, nativeIter: this.native }); // the restored update
    } else if ((m = BEST.exec(message))) {
      this.emit(out, { type: 'checkpoint', files: [m[2] as string], sizeMb: 0, state: false });
    } else if ((m = VALIDATION.exec(message))) {
      this.val = { dataset: m[1] as string, metrics: {} };
    } else if (NETWORK_D.test(message)) {
      this.hasD = true;
    } else if (level === 'warn') {
      this.emit(out, { type: 'warning', code: 'TRAINER_WARNING', message });
    } else if (level === 'error') {
      this.error = message;
    }
  }

  private continuation(line: string, out: ParsedEvent[]): void {
    const metric = this.val ? METRIC.exec(line) : null;
    if (this.val && metric) {
      const value = Number(metric[2]);
      if (Number.isFinite(value)) {
        this.val.metrics[metric[1] as string] = value;
      }
      this.emit(out, { type: 'log', level: this.level, line: line.trim() });
      return;
    }
    let m: RegExpExecArray | null;
    if ((m = CONFIG_ERROR.exec(line) ?? ARGPARSE_ERROR.exec(line))) {
      this.error = m[1];
      this.level = 'error';
    } else if (TRACEBACK.test(line)) {
      this.traceback = true;
      this.level = 'error';
    } else if (this.traceback && !/^\s/.test(line)) {
      this.error = line; // the exception line is the last unindented one
    }
    this.emit(out, { type: 'log', level: this.level, line });
  }

  private progress(native: number, rest: string, out: ParsedEvent[]): void {
    this.native = native;
    let itersPerSec = 0;
    let etaSec = 0;
    const scalars: [string, number][] = [];
    let dataWait: number | undefined;
    let mem: RegExpExecArray | null = null;
    for (const [, key, value] of rest.matchAll(FIELD)) {
      if (key === 'performance') {
        const p = PERFORMANCE.exec(value as string);
        itersPerSec = p ? Number(p[1]) : 0;
        dataWait = p?.[2] === undefined ? undefined : Number(p[2]);
      } else if (key === 'mem') {
        mem = MEM.exec(value as string); // CUDA runs: the window's peak allocated / device total
      } else if (key === 'lr' || key === 'lr_d') {
        scalars.push([key === 'lr' ? 'lr/g' : 'lr/d', Number((value as string).split(' ')[0])]);
      } else if (key === 'eta') {
        etaSec = parseEta(value as string);
      } else {
        scalars.push([seriesOf(key as string), Number(value)]);
      }
    }
    this.emit(out, { type: 'progress', itersPerSec, etaSec, phaseIndex: this.phaseIndex, nativeIter: native });
    if (mem) {
      // GPU utilisation is not in the log: null; the data share rides on the system event
      this.emit(out, { type: 'system', gpuUtil: null, vramUsedGb: Number(mem[1]), vramTotalGb: Number(mem[2]), dataWaitPct: dataWait ?? 0 });
    } else if (dataWait !== undefined) {
      scalars.unshift(['sys/data_wait', dataWait]);
    }
    for (const [series, value] of scalars) {
      if (Number.isFinite(value)) {
        this.emit(out, { type: 'scalar', series, value });
      }
      if (series.startsWith('skipped/') && value > 0) {
        const net = series.slice('skipped/'.length);
        this.emit(out, { type: 'warning', code: 'NONFINITE', message: `${net}: ${value} update(s) skipped in the last window (non-finite gradient norm)` });
      }
    }
  }

  private closeBlocks(out: ParsedEvent[], keepSave: boolean): void {
    if (this.val) {
      this.flushValidation(out);
    }
    if (this.pendingSave !== undefined && keepSave) {
      this.emit(out, { type: 'checkpoint', files: this.files(this.pendingSave, false), sizeMb: this.exportMb, state: true });
    }
    this.pendingSave = undefined;
    this.traceback = false;
  }

  private flushValidation(out: ParsedEvent[]): void {
    const val = this.val;
    this.val = undefined;
    if (val && Object.keys(val.metrics).length) {
      this.emit(out, { type: 'validation', dataset: val.dataset, metrics: val.metrics, images: [] });
    }
  }

  /** `BaseModel.save`: the exports, then `<update>.state`; `final` also writes `*_latest.pth`. */
  private files(update: number, final: boolean): string[] {
    const nets = this.hasD ? ['net_g', 'net_d'] : ['net_g'];
    return [...nets.map((n) => `${n}_${update}.pth`), ...(final ? nets.map((n) => `${n}_latest.pth`) : []), `${update}.state`];
  }

  private emit(out: ParsedEvent[], body: RunEventBody): void {
    out.push({ native: this.native, body });
  }
}

function levelOf(name: string): Level {
  return name === 'WARNING' ? 'warn' : name === 'ERROR' || name === 'CRITICAL' ? 'error' : 'info';
}

/** `str(datetime.timedelta)`: `0:21:30`, `1 day, 2:03:04`. */
export function parseEta(text: string): number {
  const m = ETA.exec(text.trim());
  if (!m) {
    return 0;
  }
  const [, days, h, min, s] = m;
  return Number(days ?? 0) * 86_400 + Number(h) * 3600 + Number(min) * 60 + Number(s);
}

/** Trainer log keys to the simulator's series: `l_g_pix` → `loss/g/pix`, `grad_norm_g` → `grad_norm/g`, `out_d_real` → `out/d/real`. */
export function seriesOf(key: string): string {
  if (key.startsWith('nonfinite_')) {
    return `nonfinite/${seriesOf(key.slice('nonfinite_'.length))}`;
  }
  const m = SERIES.exec(key);
  if (!m) {
    return `train/${key}`;
  }
  const head = m[1] === 'l' ? 'loss' : (m[1] as string);
  return m[3] ? `${head}/${m[2]}/${m[3]}` : `${head}/${m[2]}`;
}
