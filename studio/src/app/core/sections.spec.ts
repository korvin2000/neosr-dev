import { describe, expect, it } from 'vitest';
import { NODE_TYPES } from '@shared/catalog/node-catalog';
import { planContext } from '@shared/compile/graph';
import { SECTIONS, findSetting, partFields, partNode, recipeChanges, sectionOf } from '@shared/sections';
import { buildTemplate } from '@shared/templates';
import * as commands from './commands';

describe('sections', () => {
  it('summarises every section of a template in plain words', () => {
    const recipe = buildTemplate('4x-curriculum');
    const ctx = planContext(recipe);
    const summaries = Object.fromEntries(SECTIONS.map((s) => [s.id, s.summary(recipe, ctx)]));
    expect(summaries['data']).toContain('Paired images');
    expect(summaries['model']).toContain('RealPLKSR');
    expect(summaries['losses']).toMatch(/\d term/);
    expect(summaries['plan']).toContain('2 phases');
    expect(summaries['optimizer']).toContain('adan_sf');
    expect(summaries['validation']).toContain('every 1k');
    expect(summaries['runtime']).toContain('AMP bf16');
  });

  it('keeps Basics short and puts the rest behind More options', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    for (const section of SECTIONS) {
      for (const part of section.parts) {
        const { basics } = partFields(part, partNode(part, recipe));
        expect(basics.length, `${section.id}/${part.nodeType}`).toBeLessThanOrEqual(5);
      }
    }
  });

  it('reports changes against the preset snapshot, not the catalog defaults', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    expect(recipeChanges(recipe)).toEqual([]);
    const data = recipe.nodes.find((n) => n.type === NODE_TYPES.trainDataset)!;
    const changed = commands.setValues(data.id, { patch_size: 96 }).apply(recipe);
    const changes = recipeChanges(changed);
    expect(changes).toHaveLength(1);
    expect(changes[0]).toMatchObject({ section: 'data', key: 'patch_size', from: 64, to: 96 });
  });

  it('finds a setting by its NeoSR key and says where it lives', () => {
    const recipe = buildTemplate('4x-span-fidelity');
    const hits = findSetting('val_freq', recipe);
    expect(hits[0]?.section.id).toBe('validation');
    expect(hits[0]?.field.key).toBe('val_freq');
    expect(hits[0]?.inMore).toBe(false);
    const more = findSetting('keep_states', recipe);
    expect(more[0]?.inMore).toBe(true);
  });

  it('shows the discriminator parts only when a GAN row is active', () => {
    const fidelity = buildTemplate('4x-span-fidelity');
    const gan = buildTemplate('4x-neosr-gan');
    const model = sectionOf('model');
    const dPart = model.parts.find((p) => p.nodeType === NODE_TYPES.discriminator)!;
    expect(dPart.when!(planContext(fidelity), fidelity)).toBe(false);
    expect(dPart.when!(planContext(gan), gan)).toBe(true);
  });
});
