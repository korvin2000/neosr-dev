/**
 * Dev smoke check: build every template, compile it, and round-trip a shipped NeoSR TOML.
 * Run with `npx tsx tools/smoke-compile.ts`.
 */
import { readFileSync } from 'node:fs';
import { compileRecipe } from '../shared/compile/compile';
import { importToml } from '../shared/compile/import-toml';
import { emptyRecipe } from '../shared/recipe.model';
import { TEMPLATE_INFOS, buildTemplate } from '../shared/templates';

let failures = 0;
for (const info of TEMPLATE_INFOS) {
  const recipe = buildTemplate(info.id);
  const result = compileRecipe(recipe, { platform: 'windows' });
  const blockers = result.problems.filter((p) => p.severity === 'blocker');
  const line = `${info.id.padEnd(24)} nodes=${recipe.nodes.length} edges=${recipe.edges.length} phases=${result.phases.length} blockers=${blockers.length} warnings=${result.problems.filter((p) => p.severity === 'warning').length}`;
  console.log(line);
  for (const b of blockers) {
    console.log(`   BLOCKER: ${b.message}`);
    failures++;
  }
}

const showId = process.argv[2] ?? '4x-curriculum';
const show = compileRecipe(buildTemplate(showId), { platform: 'windows' });
for (const phase of show.phases) {
  console.log(`\n# ---- ${phase.fileName} (${phase.command}) native ${phase.nativeStart}..${phase.nativeEnd}`);
  console.log(phase.text);
}

const tomlPath = process.argv[3] ?? '../options/train_asid.toml';
const text = readFileSync(new URL(tomlPath, `file://${process.cwd()}/`), 'utf8');
const imported = importToml(text, emptyRecipe('import'));
console.log(`\n# ---- imported ${tomlPath}: nodes=${imported.recipe.nodes.length} preserved=[${imported.preserved.join(', ')}] warnings=${imported.warnings.length}`);
for (const w of imported.warnings) {
  console.log(`   warn: ${w}`);
}
const re = compileRecipe(imported.recipe, { platform: 'windows' });
console.log(re.phases[0]?.text ?? '(no phases)');
for (const p of re.problems.filter((p) => p.severity !== 'info')) {
  console.log(`   ${p.severity}: ${p.message}`);
}
if (failures) {
  console.error(`\n${failures} blocker(s) in templates`);
  process.exit(1);
}
