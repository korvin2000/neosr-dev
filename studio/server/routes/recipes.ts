import { Router } from 'express';
import { notFound } from '../http';
import { parseRecipe, recipeStore } from '../store/recipe-store';

export const recipesRouter = Router();

recipesRouter.get('/recipes', (_req, res) => {
  res.json(recipeStore.list());
});

recipesRouter.get('/recipes/:id', (req, res) => {
  const recipe = recipeStore.get(req.params.id);
  if (!recipe) {
    throw notFound(`Recipe ${req.params.id}`);
  }
  res.json(recipe);
});

recipesRouter.post('/recipes', (req, res) => {
  res.status(201).json(recipeStore.save(parseRecipe(req.body)));
});

recipesRouter.put('/recipes/:id', (req, res) => {
  const recipe = parseRecipe(req.body);
  const updated = { ...recipe, id: req.params.id, meta: { ...recipe.meta, updatedAt: new Date().toISOString() } };
  res.json(recipeStore.save(updated));
});

recipesRouter.delete('/recipes/:id', (req, res) => {
  if (!recipeStore.remove(req.params.id)) {
    throw notFound(`Recipe ${req.params.id}`);
  }
  res.status(204).end();
});
