import { ChangeDetectionStrategy, Component, computed, inject, input, signal } from '@angular/core';
import type { TomlLine } from '@shared/compile/toml-writer';
import { semanticHash } from '@shared/recipe.model';
import type { RunDetail } from '@shared/run.model';
import { Icon } from '../ui/icon';
import { TomlCode } from '../ui/toml-code';
import { ToastService } from '../ui/toast.service';
import { Tooltip } from '../ui/tooltip';
import { copyText, downloadText, joinPath, weightsOf } from './run-format';

/** The run's config as frozen at launch: commands, per-phase TOML, expected artifacts and the convert command. */
@Component({
  selector: 'app-run-recipe-panel',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Icon, TomlCode, Tooltip],
  templateUrl: './run-recipe-panel.html',
  styleUrl: './run-recipe-panel.scss',
})
export class RunRecipePanel {
  readonly run = input.required<RunDetail>();
  readonly platform = input<'windows' | 'linux'>('windows');

  private readonly toasts = inject(ToastService);
  protected readonly tab = signal(0);

  protected readonly hash = computed(() => semanticHash(this.run().recipe));
  protected readonly commands = computed(() => this.run().commands.join('\n'));
  protected readonly tomls = computed(() => this.run().tomls.map((t) => ({ name: t.name, lines: toTomlLines(t.text) })));
  protected readonly activeToml = computed(() => this.tomls()[Math.min(this.tab(), this.tomls().length - 1)] ?? null);
  protected readonly activeText = computed(() => this.run().tomls[Math.min(this.tab(), this.run().tomls.length - 1)] ?? null);

  protected readonly artifacts = computed(() => {
    const p = this.platform();
    return this.run().phases.map((phase, i) => ({
      phase: `${i + 1}. ${phase.name}`,
      reached: i <= this.run().phaseIndex,
      files: [
        joinPath(p, 'experiments', phase.runName, 'models', 'net_g_latest.pth'),
        ...(phase.hasGan ? [joinPath(p, 'experiments', phase.runName, 'models', 'net_d_latest.pth')] : []),
        joinPath(p, 'experiments', phase.runName, 'training_states', '<iter>.state'),
      ],
    }));
  });

  /** `neosr-convert` for the generator of the last phase the run reached. */
  protected readonly convert = computed(() => {
    const run = this.run();
    return run.status === 'completed' || run.status === 'stopped' ? (weightsOf(run, this.platform())?.convert ?? null) : null;
  });

  protected download(): void {
    const toml = this.activeText();
    if (toml) {
      downloadText(toml.name, toml.text);
    }
  }

  protected async copy(text: string, what: string): Promise<void> {
    const ok = await copyText(text);
    this.toasts.show(ok ? 'ok' : 'warn', ok ? `Copied ${what}.` : 'Clipboard not available in this browser context.');
  }
}

export function toTomlLines(text: string): TomlLine[] {
  return text.split('\n').map((line) => {
    const t = line.trim();
    const kind: TomlLine['kind'] = !t ? 'blank' : t.startsWith('[') ? 'header' : t.startsWith('#') ? 'comment' : 'pair';
    return { text: line.replace(/\r$/, ''), kind };
  });
}
