import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

/**
 * Inline stroke icons on a 24×24 grid (a small, consistent subset in the style of Lucide). One
 * component instead of an icon font keeps the bundle small and lets the colour follow `currentColor`.
 */
const ICONS: Record<string, string> = {
  plus: 'M12 5v14M5 12h14',
  minus: 'M5 12h14',
  close: 'M18 6L6 18M6 6l12 12',
  check: 'M20 6L9 17l-5-5',
  chevron: 'M6 9l6 6 6-6',
  'chevron-right': 'M9 6l6 6-6 6',
  'chevron-left': 'M15 6l-6 6 6 6',
  search: 'M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4-4',
  sliders: 'M4 6h10M18 6h2M4 12h2M10 12h10M4 18h12M20 18h0M14 4v4M6 10v4M16 16v4',
  play: 'M7 4l12 8-12 8z',
  stop: 'M6 6h12v12H6z',
  pause: 'M8 5v14M16 5v14',
  undo: 'M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3',
  redo: 'M15 14l5-5-5-5M20 9H10a6 6 0 0 0 0 12h3',
  save: 'M5 3h11l3 3v15H5zM8 3v6h8V3M8 21v-7h8v7',
  folder: 'M3 6a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z',
  file: 'M6 3h8l5 5v13H6zM14 3v5h5',
  database: 'M12 3c5 0 8 1.3 8 3s-3 3-8 3-8-1.3-8-3 3-3 8-3zM4 6v12c0 1.7 3 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3 3 8 3s8-1.3 8-3',
  cpu: 'M8 8h8v8H8zM5 5h14v14H5zM9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3',
  shield: 'M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z',
  download: 'M12 4v12M6 11l6 6 6-6M4 20h16',
  upload: 'M12 16V4M6 9l6-6 6 6M4 20h16',
  target: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 11a1 1 0 1 0 0 2 1 1 0 0 0 0-2',
  'trending-down': 'M3 7l7 7 4-4 7 7M15 17h6v-6',
  gpu: 'M3 7h18v10H3zM7 11h4v2H7zM13 11h4v2h-4zM8 17v3M16 17v3',
  grid: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
  layout: 'M3 4h18v16H3zM3 10h18M9 10v10',
  list: 'M8 6h12M8 12h12M8 18h12M4 6h0M4 12h0M4 18h0',
  image: 'M4 5h16v14H4zM4 15l5-5 4 4 3-3 4 4M15 9h0',
  'check-image': 'M4 5h16v14H4zM4 15l5-5 4 4 3-3 4 4M9 9l1.5 1.5L13 8',
  chart: 'M4 20V10M10 20V4M16 20v-8M22 20H2',
  activity: 'M2 12h4l3-8 4 16 3-8h6',
  'alert-triangle': 'M12 3l10 18H2zM12 9v5M12 17h0',
  info: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 11v5M12 8h0',
  copy: 'M9 9h11v11H9zM5 15H4V4h11v1',
  trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6',
  edit: 'M4 20h4l11-11-4-4L4 16zM13 7l4 4',
  bookmark: 'M6 3h12v18l-6-4-6 4z',
  fit: 'M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5',
  zap: 'M13 2L4 14h7l-1 8 9-12h-7z',
  eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6',
  layers: 'M12 3l9 5-9 5-9-5zM3 13l9 5 9-5M3 17l9 5 9-5',
  wand: 'M4 20l10-10M12 4v2M18 10h2M17 5l-1.5 1.5M6.5 6.5L8 8M15 15l-1 1M20 14l-1.5-1.5',
  book: 'M4 4h12a3 3 0 0 1 3 3v13H7a3 3 0 0 1-3-3zM4 17a3 3 0 0 1 3-3h12',
  home: 'M3 11l9-8 9 8v10h-6v-6H9v6H3z',
  compare: 'M4 4h7v16H4zM13 4h7v16h-7zM9 9l-2 3 2 3M15 9l2 3-2 3',
  refresh: 'M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5',
  link: 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1',
  power: 'M12 3v9M6.3 7A8 8 0 1 0 17.7 7',
  more: 'M5 12h0M12 12h0M19 12h0',
  external: 'M14 4h6v6M20 4l-9 9M19 14v6H4V5h6',
  waves: 'M2 8c2-2 4-2 6 0s4 2 6 0 4-2 6 0 2 0 2 0M2 14c2-2 4-2 6 0s4 2 6 0 4-2 6 0 2 0 2 0M2 20c2-2 4-2 6 0s4 2 6 0 4-2 6 0 2 0 2 0',
  package: 'M3 8l9-5 9 5v8l-9 5-9-5zM3 8l9 5 9-5M12 13v8',
  clock: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7v5l3 2',
  flag: 'M5 21V4h12l-2 4 2 4H5',
  tag: 'M3 12V3h9l9 9-9 9zM8 8h0',
  terminal: 'M4 5h16v14H4zM7 9l3 3-3 3M12 15h5',
  code: 'M8 7l-5 5 5 5M16 7l5 5-5 5M13 4l-2 16',
  timeline: 'M3 12h18M7 12V8M12 12v-6M17 12V9M7 12v4M12 12v3M17 12v6',
  drag: 'M9 5h0M15 5h0M9 12h0M15 12h0M9 19h0M15 19h0',
  dot: 'M12 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4',
  wrench: 'M14 4a5 5 0 0 0-5 6L3 16l5 5 6-6a5 5 0 0 0 6-5l-3 3-3-3z',
  rocket: 'M5 15c-1 1-1 4-1 4s3 0 4-1M9 15l-3-3 2-4c2-3 6-5 10-5-1 4-2 8-5 10l-4 2zM14 10h0',
  columns: 'M4 4h16v16H4zM12 4v16',
  arrow: 'M5 12h14M13 6l6 6-6 6',
  'arrow-left': 'M19 12H5M11 6l-6 6 6 6',
  settings: 'M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM4 12l1.5-.5.5-1.2-.8-1.4 1.4-1.4 1.4.8 1.2-.5L10 6h4l.5 1.5 1.2.5 1.4-.8 1.4 1.4-.8 1.4.5 1.2L20 12l-1.5.5-.5 1.2.8 1.4-1.4 1.4-1.4-.8-1.2.5L14 18h-4l-.5-1.5-1.2-.5-1.4.8-1.4-1.4.8-1.4-.5-1.2z',
  import: 'M12 3v12M8 11l4 4 4-4M4 17v3h16v-3',
  fork: 'M7 4a2 2 0 1 0 0 4 2 2 0 0 0 0-4zM17 4a2 2 0 1 0 0 4 2 2 0 0 0 0-4zM12 16a2 2 0 1 0 0 4 2 2 0 0 0 0-4zM7 8v2a3 3 0 0 0 3 3h4a3 3 0 0 0 3-3V8M12 13v3',
  help: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .8-1 1.7M12 17h0',
  sun: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5',
};

@Component({
  selector: 'app-icon',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <svg
      [attr.width]="size()"
      [attr.height]="size()"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      [attr.stroke-width]="weight()"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path [attr.d]="path()" />
    </svg>
  `,
  styles: `
    :host {
      display: inline-flex;
      flex: none;
      line-height: 0;
      vertical-align: middle;
    }
  `,
})
export class Icon {
  readonly name = input.required<string>();
  readonly size = input(16);
  readonly weight = input(1.8);
  protected readonly path = computed(() => ICONS[this.name()] ?? ICONS['help']);
}

export const ICON_NAMES = Object.keys(ICONS);
