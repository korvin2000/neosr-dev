/** Deterministic randomness: mulberry32 plus coordinate-seeded draws, so values never depend on timing. */

export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** FNV-1a, 32 bit. */
export function hashString(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

/** A generator for one coordinate (seed, iteration, channel, …): same keys, same numbers. */
export function rngAt(seed: number, ...keys: readonly (number | string)[]): Rng {
  let h = seed >>> 0;
  for (const key of keys) {
    const k = typeof key === 'number' ? key >>> 0 : hashString(key);
    h = Math.imul(h ^ k, 0x9e3779b1) >>> 0;
    h ^= h >>> 16;
  }
  return mulberry32(h);
}

export function between(rng: Rng, min: number, max: number): number {
  return min + (max - min) * rng();
}

export function intBetween(rng: Rng, min: number, max: number): number {
  return Math.floor(between(rng, min, max + 1));
}

/** Symmetric noise in [-amount, amount]. */
export function noise(rng: Rng, amount: number): number {
  return (rng() * 2 - 1) * amount;
}

export function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length)] as T;
}
