/**
 * Procedural validation images. Each scene is drawn in a 256×256 user space; `gt` is crisp, `lq`
 * is the scene at 64×64 with blur, grain and 8-px block artefacts, and `sr` sharpens with training
 * progress.
 */

export const IMAGE_IDS = ['building', 'foliage', 'portrait', 'text'] as const;
export type ImageId = (typeof IMAGE_IDS)[number];
export type ImageKind = 'lq' | 'sr' | 'gt';

const SIZE = 256;
const LQ_SIZE = 64;

interface Scene {
  readonly defs: string;
  readonly body: string;
}

export function isImageId(value: string): value is ImageId {
  return (IMAGE_IDS as readonly string[]).includes(value);
}

export function isImageKind(value: unknown): value is ImageKind {
  return value === 'lq' || value === 'sr' || value === 'gt';
}

/** `progress` is iter / totalIter of the run, clamped to [0, 1]. */
export function renderValidationImage(id: ImageId, kind: ImageKind, progress: number): string {
  const scene = SCENES[id]();
  if (kind === 'gt') {
    return svg(SIZE, scene.defs, scene.body);
  }
  if (kind === 'lq') {
    return lqSvg(scene);
  }
  const p = Math.min(1, Math.max(0, progress));
  const blur = lerp(1.4, 0.15, p);
  const grain = lerp(0.5, 0.05, p);
  const filter = degradeFilter('sr', blur, grain, SIZE, 0.8);
  return svg(SIZE, scene.defs + filter, `<g filter="url(#sr)">${scene.body}</g>`);
}

function lqSvg(scene: Scene): string {
  const filter = degradeFilter('lq', 1.6, 0.6, LQ_SIZE, 0.9);
  const blocks =
    '<pattern id="blocks" width="16" height="16" patternUnits="userSpaceOnUse">' +
    '<rect width="8" height="8" fill="#000"/><rect x="8" y="8" width="8" height="8" fill="#000"/></pattern>';
  const body =
    `<g filter="url(#lq)"><g transform="scale(${LQ_SIZE / SIZE})">${scene.body}</g></g>` +
    `<rect width="${LQ_SIZE}" height="${LQ_SIZE}" fill="url(#blocks)" opacity="0.06"/>`;
  return svg(LQ_SIZE, scene.defs + filter + blocks, body, 'crispEdges');
}

function svg(viewBox: number, defs: string, body: string, shapeRendering?: string): string {
  const rendering = shapeRendering ? ` shape-rendering="${shapeRendering}"` : '';
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${SIZE}" height="${SIZE}" viewBox="0 0 ${viewBox} ${viewBox}"${rendering}>` +
    `<defs>${defs}</defs>${body}</svg>`
  );
}

/** Blur plus zero-mean grey grain (arithmetic composite: grain·k2 + image − k2/2). */
function degradeFilter(id: string, blur: number, grain: number, region: number, frequency: number): string {
  const k2 = round(grain * 0.5);
  return (
    `<filter id="${id}" filterUnits="userSpaceOnUse" x="0" y="0" width="${region}" height="${region}" color-interpolation-filters="sRGB">` +
    `<feGaussianBlur in="SourceGraphic" stdDeviation="${round(blur)}" edgeMode="duplicate" result="blur"/>` +
    `<feTurbulence type="fractalNoise" baseFrequency="${frequency}" numOctaves="2" seed="7" result="grain"/>` +
    '<feColorMatrix in="grain" type="saturate" values="0" result="mono"/>' +
    `<feComposite in="mono" in2="blur" operator="arithmetic" k1="0" k2="${k2}" k3="1" k4="${round(-k2 / 2)}"/>` +
    '</filter>'
  );
}

const SCENES: Readonly<Record<ImageId, () => Scene>> = {
  building: buildingScene,
  foliage: foliageScene,
  portrait: portraitScene,
  text: textScene,
};

function buildingScene(): Scene {
  const defs =
    '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6fa8dc"/><stop offset="1" stop-color="#d9e8f5"/></linearGradient>';
  const towers = [
    { x: 12, y: 70, w: 70, h: 186, color: '#8a8f98' },
    { x: 92, y: 36, w: 80, h: 220, color: '#6d7480' },
    { x: 182, y: 96, w: 62, h: 160, color: '#9c8f84' },
  ];
  const parts = towers.map((t) => `<rect x="${t.x}" y="${t.y}" width="${t.w}" height="${t.h}" fill="${t.color}"/>${windows(t)}`);
  const sign =
    '<rect x="100" y="52" width="64" height="22" fill="#1f2a36"/>' +
    '<text x="132" y="68" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#ffd966" text-anchor="middle">NeoSR</text>';
  return { defs, body: `<rect width="${SIZE}" height="${SIZE}" fill="url(#sky)"/>${parts.join('')}${sign}` };
}

function windows(t: { x: number; y: number; w: number; h: number }): string {
  const cells: string[] = [];
  for (let y = t.y + 12; y < t.y + t.h - 14; y += 16) {
    for (let x = t.x + 6; x < t.x + t.w - 10; x += 14) {
      const lit = (x * 7 + y * 13) % 5 === 0;
      cells.push(`<rect x="${x}" y="${y}" width="8" height="10" fill="${lit ? '#ffe599' : '#2b3440'}"/>`);
    }
  }
  return cells.join('');
}

function foliageScene(): Scene {
  const defs =
    '<filter id="leaves" x="0" y="0" width="100%" height="100%">' +
    '<feTurbulence type="turbulence" baseFrequency="0.045" numOctaves="4" seed="11"/>' +
    '<feColorMatrix type="matrix" values="0.2 0 0 0 0.05  0.9 0.4 0 0 0.25  0.1 0 0 0 0.04  0 0 0 0 1"/>' +
    '</filter>';
  const stems = [40, 96, 150, 210].map((x, i) => `<path d="M${x} 256 Q ${x + 18 - i * 9} 170 ${x - 6 + i * 5} 96" stroke="#3d2b1f" stroke-width="5" fill="none"/>`);
  return { defs, body: `<rect width="${SIZE}" height="${SIZE}" filter="url(#leaves)"/>${stems.join('')}` };
}

function portraitScene(): Scene {
  const defs =
    '<radialGradient id="skin" cx="0.45" cy="0.4" r="0.65"><stop offset="0" stop-color="#f6d2b8"/><stop offset="0.7" stop-color="#d9a07e"/><stop offset="1" stop-color="#a86f52"/></radialGradient>' +
    '<linearGradient id="backdrop" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4a6670"/><stop offset="1" stop-color="#1d2b30"/></linearGradient>';
  const body =
    `<rect width="${SIZE}" height="${SIZE}" fill="url(#backdrop)"/>` +
    '<ellipse cx="128" cy="270" rx="100" ry="70" fill="#7b3f3f"/>' +
    '<ellipse cx="128" cy="104" rx="70" ry="78" fill="#3b2418"/>' +
    '<circle cx="128" cy="124" r="62" fill="url(#skin)"/>' +
    '<circle cx="106" cy="114" r="8" fill="#fff"/><circle cx="150" cy="114" r="8" fill="#fff"/>' +
    '<circle cx="107" cy="115" r="4" fill="#3a5a40"/><circle cx="151" cy="115" r="4" fill="#3a5a40"/>' +
    '<path d="M128 122 L122 144 L132 144" stroke="#a86f52" stroke-width="3" fill="none"/>' +
    '<path d="M108 160 Q 128 174 148 160" stroke="#9b3b3b" stroke-width="5" fill="none" stroke-linecap="round"/>';
  return { defs, body };
}

function textScene(): Scene {
  const lines = [
    { size: 26, weight: 700, text: 'NeoSR Studio' },
    { size: 16, weight: 400, text: 'The quick brown fox' },
    { size: 16, weight: 400, text: 'jumps over the lazy dog.' },
    { size: 12, weight: 400, text: 'Super-resolution keeps edges' },
    { size: 12, weight: 400, text: 'and thin strokes legible.' },
    { size: 10, weight: 400, text: '0123456789 ABCDEFGHIJKLM' },
    { size: 8, weight: 400, text: 'nopqrstuvwxyz · tiny glyphs are hard' },
  ];
  let y = 40;
  const texts = lines.map((line) => {
    const out = `<text x="16" y="${y}" font-family="Georgia, serif" font-size="${line.size}" font-weight="${line.weight}" fill="#111">${line.text}</text>`;
    y += line.size + 14;
    return out;
  });
  return { defs: '', body: `<rect width="${SIZE}" height="${SIZE}" fill="#fbfaf6"/>${texts.join('')}` };
}

function lerp(from: number, to: number, t: number): number {
  return from + (to - from) * t;
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
