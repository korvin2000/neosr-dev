/** Recipes: in-memory map backed by one JSON file per recipe. */
import fs from 'node:fs';
import path from 'node:path';
import { nodeValues, planContext, scaleOf } from '../../shared/compile/graph';
import { newId, type Recipe, type RecipeMeta } from '../../shared/recipe.model';
import type { RecipeSummary } from '../api-types';
import { DATA_DIR } from '../config';
import { assertSafeId, badRequest, isRecord } from '../http';

export class RecipeStore {
  private readonly recipes = new Map<string, Recipe>();

  constructor(private readonly dir: string) {}

  load(): void {
    fs.mkdirSync(this.dir, { recursive: true });
    for (const file of fs.readdirSync(this.dir).filter((f) => f.endsWith('.json'))) {
      try {
        const recipe = parseRecipe(JSON.parse(fs.readFileSync(path.join(this.dir, file), 'utf8')));
        this.recipes.set(recipe.id, recipe);
      } catch (error) {
        console.warn(`[recipes] skipped ${file}: ${(error as Error).message}`);
      }
    }
  }

  list(): RecipeSummary[] {
    return [...this.recipes.values()].map(summarize).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  get(id: string): Recipe | undefined {
    return this.recipes.get(id);
  }

  save(recipe: Recipe): Recipe {
    assertSafeId(recipe.id);
    fs.writeFileSync(this.file(recipe.id), JSON.stringify(recipe, null, 2));
    this.recipes.set(recipe.id, recipe);
    return recipe;
  }

  remove(id: string): boolean {
    if (!this.recipes.delete(id)) {
      return false;
    }
    fs.rmSync(this.file(id), { force: true });
    return true;
  }

  private file(id: string): string {
    return path.join(this.dir, `${assertSafeId(id)}.json`);
  }
}

/** Structural check of a client recipe; fills a missing id and meta. */
export function parseRecipe(value: unknown): Recipe {
  if (!isRecord(value) || value['format'] !== 'neosr-studio-recipe') {
    throw badRequest('Body is not a neosr-studio-recipe');
  }
  for (const key of ['nodes', 'edges', 'phases'] as const) {
    if (!Array.isArray(value[key])) {
      throw badRequest(`Recipe.${key} must be an array`);
    }
  }
  if (typeof value['name'] !== 'string') {
    throw badRequest('Recipe.name must be a string');
  }
  const id = typeof value['id'] === 'string' && value['id'] ? value['id'] : newId('r');
  const now = new Date().toISOString();
  const meta: RecipeMeta = { createdAt: now, updatedAt: now, tags: [], description: '', ...(isRecord(value['meta']) ? value['meta'] : {}) };
  const nativeExtras = isRecord(value['nativeExtras']) ? value['nativeExtras'] : {};
  return { ...(value as unknown as Recipe), id: assertSafeId(id), meta, nativeExtras };
}

function summarize(recipe: Recipe): RecipeSummary {
  const { arch, scale } = archOf(recipe);
  return {
    id: recipe.id,
    name: recipe.name,
    updatedAt: recipe.meta.updatedAt,
    templateId: recipe.meta.templateId,
    phases: recipe.phases.length,
    arch,
    scale,
    nodeCount: recipe.nodes.length,
    description: recipe.meta.description,
  };
}

function archOf(recipe: Recipe): { arch: string; scale: number } {
  try {
    const model = planContext(recipe).model;
    return { arch: model ? String(nodeValues(model)['type'] ?? '') : '', scale: scaleOf(model) };
  } catch {
    return { arch: '', scale: 4 };
  }
}

export const recipeStore = new RecipeStore(path.join(DATA_DIR, 'recipes'));
