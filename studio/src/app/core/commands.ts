/**
 * Commands: pure `Recipe → Recipe` transformations with a label for the history. Every edit from
 * every view (wizard field, node drag, inspector, timeline, TOML import) is a command, so undo is
 * uniform and the semantic hash only moves when a command changes meaning.
 */
import { NODE_TYPES, nodeSpec } from '@shared/catalog/node-catalog';
import { defaultValues } from '@shared/catalog/types';
import { lossRows, planContext } from '@shared/compile/graph';
import { applyDraft, autoLayout as layoutRecipe, newLossRow, type RecipeDraft } from '@shared/recipe-builder';
import { newId, newNode, type LossRow, type Phase, type Point, type Recipe, type RecipeEdge, type RecipeNode } from '@shared/recipe.model';

export interface Command {
  readonly label: string;
  apply(recipe: Recipe): Recipe;
}

type Values = Record<string, unknown>;

function touch(recipe: Recipe): Recipe {
  return { ...recipe, meta: { ...recipe.meta, updatedAt: new Date().toISOString() } };
}

function updateNode(recipe: Recipe, id: string, fn: (node: RecipeNode) => RecipeNode): Recipe {
  return touch({ ...recipe, nodes: recipe.nodes.map((n) => (n.id === id ? fn(n) : n)) });
}

export function addNode(type: string, position: Point, values: Values = {}): Command {
  return {
    label: `Add ${nodeSpec(type).label}`,
    apply: (recipe) => touch({ ...recipe, nodes: [...recipe.nodes, newNode(type, position, { ...defaultValues(nodeSpec(type)), ...values })] }),
  };
}

export function removeNodes(ids: readonly string[]): Command {
  const set = new Set(ids);
  return {
    label: ids.length === 1 ? 'Delete node' : `Delete ${ids.length} nodes`,
    apply: (recipe) =>
      touch({
        ...recipe,
        nodes: recipe.nodes.filter((n) => !set.has(n.id)),
        edges: recipe.edges.filter((e) => !set.has(e.sourceNode) && !set.has(e.targetNode)),
      }),
  };
}

export function moveNodes(moves: readonly { id: string; position: Point }[]): Command {
  const byId = new Map(moves.map((m) => [m.id, m.position]));
  return {
    label: 'Move',
    apply: (recipe) => ({ ...recipe, nodes: recipe.nodes.map((n) => (byId.has(n.id) ? { ...n, position: byId.get(n.id)! } : n)) }),
  };
}

export function setValues(nodeId: string, patch: Values, label = 'Edit'): Command {
  return {
    label,
    apply: (recipe) =>
      updateNode(recipe, nodeId, (n) => {
        const spec = nodeSpec(n.type);
        const defaults = defaultValues(spec);
        const values: Values = { ...n.values, ...patch };
        // A changed architecture type invalidates its constructor options (FieldSpec.resets).
        for (const field of spec.fields) {
          if (field.resets && field.key in patch && patch[field.key] !== n.values[field.key]) {
            for (const key of field.resets) {
              if (!(key in patch)) {
                values[key] = defaults[key];
              }
            }
          }
        }
        return { ...n, values };
      }),
  };
}

/** `undefined` in the patch removes the override (the phase falls back to the shared value). */
export function setPhaseValues(nodeId: string, phaseId: string, patch: Values, label = 'Edit phase'): Command {
  return {
    label,
    apply: (recipe) =>
      updateNode(recipe, nodeId, (n) => {
        const current = { ...(n.phaseValues[phaseId] ?? {}) };
        for (const [k, v] of Object.entries(patch)) {
          if (v === undefined) {
            delete current[k];
          } else {
            current[k] = v;
          }
        }
        const phaseValues = { ...n.phaseValues };
        if (Object.keys(current).length) {
          phaseValues[phaseId] = current;
        } else {
          delete phaseValues[phaseId];
        }
        return { ...n, phaseValues };
      }),
  };
}

export function setNodeFlags(nodeId: string, flags: Partial<Pick<RecipeNode, 'collapsed' | 'disabled' | 'title'>>): Command {
  const label = flags.disabled !== undefined ? (flags.disabled ? 'Disable node' : 'Enable node') : flags.title !== undefined ? 'Rename node' : 'Collapse';
  return {
    label,
    apply: (recipe) => {
      const next = updateNode(recipe, nodeId, (n) => ({ ...n, ...flags }));
      // Collapsing is presentation only; do not bump updatedAt for it.
      return flags.collapsed !== undefined && Object.keys(flags).length === 1 ? { ...next, meta: recipe.meta } : next;
    },
  };
}

export function connect(edge: Omit<RecipeEdge, 'id'>): Command {
  return {
    label: 'Connect',
    apply: (recipe) => {
      // An input port holds one wire: connecting replaces whatever was there.
      const edges = recipe.edges.filter((e) => !(e.targetNode === edge.targetNode && e.targetPort === edge.targetPort));
      return touch({ ...recipe, edges: [...edges, { ...edge, id: newId('e') }] });
    },
  };
}

export function removeEdges(ids: readonly string[]): Command {
  const set = new Set(ids);
  return {
    label: 'Disconnect',
    apply: (recipe) => touch({ ...recipe, edges: recipe.edges.filter((e) => !set.has(e.id)) }),
  };
}

export function disconnectNode(nodeId: string): Command {
  return {
    label: 'Disconnect node',
    apply: (recipe) => touch({ ...recipe, edges: recipe.edges.filter((e) => e.sourceNode !== nodeId && e.targetNode !== nodeId) }),
  };
}

export function duplicateNode(nodeId: string): Command {
  return {
    label: 'Duplicate',
    apply: (recipe) => {
      const source = recipe.nodes.find((n) => n.id === nodeId);
      if (!source || nodeSpec(source.type).singleton) {
        return recipe;
      }
      const copy: RecipeNode = { ...source, id: newId(), position: { x: source.position.x + 40, y: source.position.y + 40 } };
      return touch({ ...recipe, nodes: [...recipe.nodes, copy] });
    },
  };
}

export function setName(name: string): Command {
  return { label: 'Rename recipe', apply: (recipe) => touch({ ...recipe, name }) };
}

export function setPhases(phases: readonly Phase[], label = 'Edit phases'): Command {
  return { label, apply: (recipe) => touch({ ...recipe, phases }) };
}

export function updatePhase(id: string, patch: Partial<Phase>): Command {
  return {
    label: 'Edit phase',
    apply: (recipe) => touch({ ...recipe, phases: recipe.phases.map((p) => (p.id === id ? { ...p, ...patch } : p)) }),
  };
}

export function addPhase(phase: Phase, afterId?: string): Command {
  return {
    label: 'Add phase',
    apply: (recipe) => {
      const index = afterId ? recipe.phases.findIndex((p) => p.id === afterId) : recipe.phases.length - 1;
      const phases = [...recipe.phases];
      phases.splice(index + 1, 0, phase);
      return touch({ ...recipe, phases });
    },
  };
}

export function removePhase(id: string): Command {
  return {
    label: 'Remove phase',
    apply: (recipe) => {
      if (recipe.phases.length <= 1) {
        return recipe;
      }
      const phases = recipe.phases.filter((p) => p.id !== id).map((p, i) => (i === 0 ? { ...p, init: 'scratch' as const } : p));
      // Drop per-phase overrides that pointed at the removed phase.
      const nodes = recipe.nodes.map((n) => {
        if (!(id in n.phaseValues)) {
          return n;
        }
        const phaseValues = { ...n.phaseValues };
        delete phaseValues[id];
        return { ...n, phaseValues };
      });
      return touch({ ...recipe, phases, nodes });
    },
  };
}

export function setExtras(extras: Record<string, unknown>): Command {
  return { label: 'Edit native extras', apply: (recipe) => touch({ ...recipe, nativeExtras: extras }) };
}

export function applyDraftCommand(draft: RecipeDraft, label = 'Wizard'): Command {
  return { label, apply: (recipe) => applyDraft(recipe, draft) };
}

export function autoLayout(): Command {
  return { label: 'Auto layout', apply: (recipe) => ({ ...layoutRecipe(recipe), meta: recipe.meta }) };
}

export function replaceRecipe(next: Recipe, label = 'Load'): Command {
  return { label, apply: () => next };
}

export function clearCanvas(): Command {
  return { label: 'Clear', apply: (recipe) => touch({ ...recipe, nodes: [], edges: [] }) };
}

/** Add a node and, when it has an obvious home, wire it into the Training Plan. */
export function addNodeWired(type: string, position: Point, values: Values = {}): Command {
  return {
    label: `Add ${nodeSpec(type).label}`,
    apply: (recipe) => {
      const node = newNode(type, position, { ...defaultValues(nodeSpec(type)), ...values });
      let next = touch({ ...recipe, nodes: [...recipe.nodes, node] });
      const plan = next.nodes.find((n) => n.type === NODE_TYPES.plan && !n.disabled);
      const spec = nodeSpec(type);
      const out = spec.outputs[0];
      if (!plan || !out || type === NODE_TYPES.plan) {
        return next;
      }
      // Degradations feed the training dataset, not the plan.
      if (out.type === 'Degradation') {
        const data = planContext(next).data;
        const taken = data && next.edges.some((e) => e.targetNode === data.id && e.targetPort === 'degradation');
        if (data && !taken) {
          next = { ...next, edges: [...next.edges, { id: newId('e'), sourceNode: node.id, sourcePort: out.key, targetNode: data.id, targetPort: 'degradation' }] };
        }
        return next;
      }
      const planSpec = nodeSpec(NODE_TYPES.plan);
      const port = planSpec.inputs.find((p) => p.type === out.type);
      const taken = port && next.edges.some((e) => e.targetNode === plan.id && e.targetPort === port.key);
      if (port && !taken) {
        next = { ...next, edges: [...next.edges, { id: newId('e'), sourceNode: node.id, sourcePort: out.key, targetNode: plan.id, targetPort: port.key }] };
      }
      return next;
    },
  };
}

/** The plan a loss node feeds, if any. */
function planFedBy(recipe: Recipe, lossNodeId: string): RecipeNode | undefined {
  const edge = recipe.edges.find((e) => e.sourceNode === lossNodeId && e.targetPort === 'losses');
  return edge ? recipe.nodes.find((n) => n.id === edge.targetNode && n.type === NODE_TYPES.plan) : undefined;
}

function hasActiveGan(rows: readonly LossRow[], phases: readonly Phase[]): boolean {
  return rows.some((r) => r.type === 'gan_loss' && phases.some((p) => r.schedule[p.id]?.active ?? r.enabled));
}

/**
 * Bring the discriminator bundle in line with the loss rows of `lossNodeId`: with an active
 * gan_loss the plan needs a Discriminator and an Optimizer D; without one they are removed.
 * Only the plan this loss node feeds is touched, so a disconnected duplicate changes nothing.
 */
function reconcileGan(recipe: Recipe, lossNodeId: string): Recipe {
  const plan = planFedBy(recipe, lossNodeId);
  if (!plan) {
    return recipe;
  }
  const node = recipe.nodes.find((n) => n.id === lossNodeId);
  const wantGan = !!node && hasActiveGan(lossRows(node), recipe.phases);
  const feeding = (port: string) => {
    const edge = recipe.edges.find((e) => e.targetNode === plan.id && e.targetPort === port);
    return edge ? recipe.nodes.find((n) => n.id === edge.sourceNode) : undefined;
  };
  let next = recipe;
  const dmodel = feeding('discriminator');
  const optimD = feeding('optimD');
  if (wantGan) {
    const count = next.nodes.length;
    if (!dmodel) {
      next = addNodeWired(NODE_TYPES.discriminator, { x: 720, y: 60 + count * 24 }).apply(next);
    }
    if (!optimD) {
      next = addNodeWired(NODE_TYPES.optimizerD, { x: 720, y: 300 + count * 24 }).apply(next);
    }
  } else {
    const victims = [dmodel, optimD].filter((n): n is RecipeNode => !!n).map((n) => n.id);
    if (victims.length) {
      next = removeNodes(victims).apply(next);
    }
  }
  return next;
}

/** Replace the loss rows and keep the discriminator bundle consistent, as one undo step. */
export function setLossRows(lossNodeId: string, rows: readonly LossRow[], label = 'Losses'): Command {
  return {
    label,
    apply: (recipe) => reconcileGan(setValues(lossNodeId, { losses: rows }, label).apply(recipe), lossNodeId),
  };
}

/**
 * The GAN switch: on adds (or re-enables, in every phase) a gan_loss row plus Discriminator and
 * Optimizer D; off removes the rows and the two nodes. One command, one undo.
 */
export function setGanBundle(lossNodeId: string, on: boolean): Command {
  return {
    label: on ? 'Enable GAN' : 'Disable GAN',
    apply: (recipe) => {
      const node = recipe.nodes.find((n) => n.id === lossNodeId);
      if (!node) {
        return recipe;
      }
      const rows = lossRows(node);
      let next: LossRow[];
      if (on) {
        const activeEverywhere = (r: LossRow): LossRow => ({
          ...r,
          enabled: true,
          schedule: Object.fromEntries(Object.entries(r.schedule).map(([id, s]) => [id, { ...s, active: true }])),
        });
        next = rows.some((r) => r.type === 'gan_loss')
          ? rows.map((r) => (r.type === 'gan_loss' ? activeEverywhere(r) : r))
          : [...rows, { ...newLossRow('gan_loss', rows), params: { gan_type: 'bce', wavelet_init: 80000 } }];
      } else {
        next = rows.filter((r) => r.type !== 'gan_loss');
      }
      return reconcileGan(setValues(lossNodeId, { losses: next }, 'GAN').apply(recipe), lossNodeId);
    },
  };
}
