/** NeoSR Studio prototype backend: see server/API.md. */
import express, { type ErrorRequestHandler, type RequestHandler } from 'express';
import { NEOSR_PYTHON, NEOSR_ROOT, PORT, TRAINER_MODE } from './config';
import { HttpError, isRecord } from './http';
import { compileRouter } from './routes/compile';
import { environmentRouter, startEnvironmentProbe } from './routes/environment';
import { libraryRouter } from './routes/library';
import { recipesRouter } from './routes/recipes';
import { runsRouter } from './routes/runs';
import { recipeStore } from './store/recipe-store';
import { runStore } from './store/run-store';

/** How long a backend shutdown waits for live trainers to save and exit. */
const SHUTDOWN_MS = 120_000;
/** A second shutdown signal exits at once only this long after the first. */
const ESCAPE_AFTER_MS = 1000;

const cors: RequestHandler = (req, res, next) => {
  res.set({
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Last-Event-ID',
  });
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }
  next();
};

const apiNotFound: RequestHandler = (req) => {
  throw new HttpError(404, `No route for ${req.method} ${req.originalUrl}`);
};

const errorHandler: ErrorRequestHandler = (error: unknown, _req, res, _next) => {
  void _next;
  const status = statusOf(error);
  if (status >= 500) {
    console.error(error);
  }
  if (res.headersSent) {
    res.end();
    return;
  }
  const message = error instanceof Error ? error.message : String(error);
  const problems = error instanceof HttpError ? error.problems : undefined;
  res.status(status).json(problems ? { error: message, problems } : { error: message });
};

/** HttpError status, or the 4xx status body-parser attaches to malformed JSON. */
function statusOf(error: unknown): number {
  if (error instanceof HttpError) {
    return error.status;
  }
  const status = isRecord(error) ? Number(error['status'] ?? error['statusCode']) : NaN;
  return status >= 400 && status < 500 ? status : 500;
}

function createApp(): express.Express {
  const app = express();
  app.disable('x-powered-by');
  app.use(cors);
  app.use(express.json({ limit: '10mb' }));
  app.use('/api', environmentRouter, recipesRouter, compileRouter, runsRouter, libraryRouter);
  app.use('/api', apiNotFound);
  app.use(errorHandler);
  return app;
}

function main(): void {
  recipeStore.load();
  runStore.load();
  runStore.seedIfEmpty();
  startEnvironmentProbe();
  const server = createApp().listen(PORT, () => {
    const mode = TRAINER_MODE === 'real' ? `real trainer: ${NEOSR_PYTHON} train.py in ${NEOSR_ROOT}` : 'simulated';
    console.log(`[api] NeoSR Studio backend (${mode}) listening on http://localhost:${PORT}`);
  });
  let closingAt: number | undefined;
  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.on(signal, () => {
      if (closingAt !== undefined) {
        // one Ctrl-C can arrive twice (terminal group + a relaying npm/tsx): only a later one means "leave now"
        if (Date.now() - closingAt > ESCAPE_AFTER_MS) {
          process.exit(1);
        }
        return;
      }
      closingAt = Date.now();
      server.close();
      // live trainers save at their next update boundary; their final status is persisted before exit
      void runStore.shutdown(SHUTDOWN_MS).finally(() => process.exit(0));
    });
  }
}

main();
