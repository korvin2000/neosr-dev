import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { ARCHS } from '@shared/catalog/archs';
import { LOSSES } from '@shared/catalog/losses';
import { DisclosureState } from '../core/disclosure-state.service';
import { EnvironmentService } from '../core/environment.service';
import { SettingsService, type StudioSettings } from '../core/settings.service';
import { Disclosure } from '../ui/disclosure';
import { Icon } from '../ui/icon';
import { Menu, type MenuItem } from '../ui/menu';
import { PageHeader } from '../ui/page-header';
import { ToastService } from '../ui/toast.service';

const SIM_MIN = 50;
const SIM_MAX = 20_000;
const LOCAL_KEYS = ['neosr-studio.draft.v1', 'neosr-studio.node-presets.v1'];

type PathKey = 'neosrRoot' | 'experimentsRoot' | 'optionsDir';
type PrefKey = 'showKeys' | 'openSections' | 'inspectorDocked';

/** Settings: three disclosures (Environment · Preferences · Prototype) and a folded About. */
@Component({
  selector: 'app-settings-page',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [Disclosure, Icon, Menu, PageHeader],
  templateUrl: './settings-page.html',
  styleUrl: './settings-page.scss',
})
export class SettingsPage {
  private readonly toasts = inject(ToastService);
  private readonly disclosure = inject(DisclosureState);
  protected readonly env = inject(EnvironmentService);
  protected readonly service = inject(SettingsService);
  protected readonly s = this.service.settings;

  protected readonly platforms: readonly StudioSettings['platform'][] = ['windows', 'linux'];
  protected readonly paths: readonly { key: PathKey; label: string; help: string }[] = [
    { key: 'neosrRoot', label: 'NeoSR root', help: 'Launch commands run from here.' },
    { key: 'experimentsRoot', label: 'Experiments root', help: 'Where NeoSR writes models, states and logs.' },
    { key: 'optionsDir', label: 'Options folder', help: 'Where compiled .toml files belong.' },
  ];
  protected readonly prefs: readonly { key: PrefKey; label: string; help: string }[] = [
    { key: 'showKeys', label: 'Show NeoSR keys', help: 'The TOML key under every setting label.' },
    { key: 'openSections', label: 'Open sections by default', help: 'Build shows every section expanded instead of its summary.' },
    { key: 'inspectorDocked', label: 'Inspector docked', help: 'Beside the graph on wide screens instead of over it.' },
  ];
  protected readonly simMin = SIM_MIN;
  protected readonly simMax = SIM_MAX;
  protected readonly archCount = ARCHS.length;
  protected readonly lossCount = LOSSES.length;
  protected readonly menu: readonly MenuItem[] = [{ id: 'reset', label: 'Reset to defaults', icon: 'undo' }];

  protected readonly envSummary = computed(() => `${this.s().platform} · ${this.s().neosrRoot}`);
  protected readonly prefSummary = computed(() => {
    const s = this.s();
    return [s.showKeys ? 'keys shown' : 'keys hidden', s.openSections ? 'sections open' : 'sections folded', s.inspectorDocked ? 'inspector docked' : 'inspector floating'].join(' · ');
  });
  protected readonly protoSummary = computed(() => `simulated training · ${this.s().simSpeed.toLocaleString()} updates/s`);
  protected readonly gpu = computed(() => {
    const info = this.env.info();
    return info?.gpus[0] ? `${info.gpus[0].name} · ${info.gpus[0].vramGb} GB` : this.env.offline() ? 'backend offline' : '…';
  });

  protected isOpen(key: string): boolean {
    return this.disclosure.isOpen(`settings:${key}`);
  }

  protected setOpen(key: string, open: boolean): void {
    this.disclosure.set(`settings:${key}`, open);
  }

  protected patch(patch: Partial<StudioSettings>): void {
    this.service.patch(patch);
  }

  protected setPath(key: PathKey, event: Event): void {
    const value = (event.target as HTMLInputElement).value.trim();
    if (value) {
      this.service.patch({ [key]: value });
    }
  }

  protected setPref(key: PrefKey, event: Event): void {
    this.service.patch({ [key]: (event.target as HTMLInputElement).checked });
  }

  protected setSpeed(event: Event): void {
    const input = event.target as HTMLInputElement;
    const n = Math.round(Number(input.value));
    const clamped = Number.isFinite(n) ? Math.min(SIM_MAX, Math.max(SIM_MIN, n)) : this.s().simSpeed;
    input.value = String(clamped);
    this.service.patch({ simSpeed: clamped });
  }

  protected onMenu(id: string): void {
    if (id === 'reset') {
      this.service.reset();
      this.toasts.show('info', 'Settings reset to defaults.');
    }
  }

  protected clearLocal(): void {
    if (!confirm('Clear the local draft and your saved node presets? Saved recipes on the backend are kept. The page will reload.')) {
      return;
    }
    for (const key of LOCAL_KEYS) {
      try {
        localStorage.removeItem(key);
      } catch {
        // nothing stored, nothing to clear
      }
    }
    location.reload();
  }
}
