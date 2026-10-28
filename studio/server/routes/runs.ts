import { Router, type Request, type Response } from 'express';
import type { RunEvent } from '../../shared/run.model';
import { badRequest, bodyOf, notFound, optionalNumber, platformOf } from '../http';
import { isImageId, isImageKind, renderValidationImage } from '../sim/images';
import { parseRecipe } from '../store/recipe-store';
import { runStore } from '../store/run-store';
import { hostPlatform } from './environment';

const HEARTBEAT_MS = 15_000;

export const runsRouter = Router();

runsRouter.get('/runs', (_req, res) => {
  res.json(runStore.list());
});

runsRouter.post('/runs', (req, res) => {
  const body = bodyOf(req.body);
  const recipe = parseRecipe(body['recipe']);
  const platform = platformOf(body['platform']) ?? hostPlatform();
  res.status(201).json(runStore.create(recipe, platform, optionalNumber(body['speed'])));
});

runsRouter.get('/runs/:id', (req, res) => {
  res.json(runStore.snapshot(req.params.id));
});

runsRouter.get('/runs/:id/events', (req, res) => {
  streamEvents(req, res, req.params.id);
});

runsRouter.post('/runs/:id/stop', (req, res) => {
  res.json(runStore.stop(req.params.id, bodyOf(req.body)['force'] === true));
});

runsRouter.post('/runs/:id/resume', (req, res) => {
  res.json(runStore.resume(req.params.id));
});

runsRouter.post('/runs/:id/chaos', (req, res) => {
  const kind = bodyOf(req.body)['kind'];
  if (kind !== 'disconnect' && kind !== 'nan') {
    throw badRequest("Body needs { kind: 'disconnect' | 'nan' }");
  }
  res.json(runStore.chaos(req.params.id, kind));
});

runsRouter.delete('/runs/:id', (req, res) => {
  runStore.remove(req.params.id);
  res.status(204).end();
});

runsRouter.get('/runs/:id/images/:imageId', (req, res) => {
  const { imageId } = req.params;
  if (!isImageId(imageId)) {
    throw notFound(`Image ${imageId}`);
  }
  const kind = req.query['kind'] ?? 'sr';
  if (!isImageKind(kind)) {
    throw badRequest("kind must be 'lq', 'sr' or 'gt'");
  }
  const clock = runStore.clock(req.params.id);
  const iter = optionalNumber(req.query['iter']) ?? clock.iter;
  const progress = clock.totalIter > 0 ? iter / clock.totalIter : 0;
  res.set({ 'Content-Type': 'image/svg+xml', 'Cache-Control': 'public, max-age=3600' }).send(renderValidationImage(imageId, kind, progress));
});

/** SSE: replays events after `?after=` (or `Last-Event-ID` on an EventSource reconnect), then streams live. */
function streamEvents(req: Request, res: Response, id: string): void {
  const after = Math.max(optionalNumber(req.query['after']) ?? 0, optionalNumber(req.get('Last-Event-ID')) ?? 0);
  const backlog = runStore.eventsAfter(id, after);
  res.status(200).set({
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.flushHeaders();
  res.write('retry: 3000\n\n');
  const send = (events: readonly RunEvent[]) => {
    res.write(events.map(sseMessage).join(''));
  };
  send(backlog);
  const unsubscribe = runStore.subscribe(id, { events: send, end: () => res.end() });
  const heartbeat = setInterval(() => res.write(': heartbeat\n\n'), HEARTBEAT_MS);
  req.on('close', () => {
    clearInterval(heartbeat);
    unsubscribe();
  });
}

function sseMessage(event: RunEvent): string {
  return `id: ${event.seq}\nevent: run\ndata: ${JSON.stringify(event)}\n\n`;
}
