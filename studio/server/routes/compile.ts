import { Router } from 'express';
import { compileRecipe } from '../../shared/compile/compile';
import { importToml } from '../../shared/compile/import-toml';
import { emptyRecipe } from '../../shared/recipe.model';
import { badRequest, bodyOf, optionalString, platformOf } from '../http';
import { parseRecipe } from '../store/recipe-store';
import { hostPlatform } from './environment';

export const compileRouter = Router();

compileRouter.post('/compile', (req, res) => {
  const body = bodyOf(req.body);
  const recipe = parseRecipe(body['recipe']);
  try {
    res.json(compileRecipe(recipe, { platform: platformOf(body['platform']) ?? hostPlatform() }));
  } catch (error) {
    throw badRequest(`Cannot compile recipe: ${(error as Error).message}`);
  }
});

compileRouter.post('/import', (req, res) => {
  const body = bodyOf(req.body);
  const text = optionalString(body['text']);
  if (text === undefined) {
    throw badRequest('Body needs { text: string }');
  }
  const name = optionalString(body['name']);
  try {
    res.json(importToml(text, emptyRecipe(name), name));
  } catch (error) {
    throw badRequest(`Cannot import TOML: ${(error as Error).message}`);
  }
});
