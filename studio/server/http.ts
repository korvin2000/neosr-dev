import type { Problem } from '../shared/compile/problems';

/** An error the central handler turns into a JSON response with this status. */
export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly problems?: readonly Problem[],
  ) {
    super(message);
  }
}

export function notFound(what: string): HttpError {
  return new HttpError(404, `${what} not found`);
}

export function badRequest(message: string, problems?: readonly Problem[]): HttpError {
  return new HttpError(400, message, problems);
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** The body as an object; Express leaves it undefined without a JSON content type. */
export function bodyOf(body: unknown): Record<string, unknown> {
  return isRecord(body) ? body : {};
}

export function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

export function optionalNumber(value: unknown): number | undefined {
  const n = typeof value === 'string' && value.trim() !== '' ? Number(value) : value;
  return typeof n === 'number' && Number.isFinite(n) ? n : undefined;
}

export function platformOf(value: unknown): 'windows' | 'linux' | undefined {
  return value === 'windows' || value === 'linux' ? value : undefined;
}

/** Ids become file names, so only a safe alphabet is accepted. */
export function assertSafeId(id: string): string {
  if (!/^[A-Za-z0-9_-]{1,80}$/.test(id)) {
    throw badRequest(`Invalid id: ${id}`);
  }
  return id;
}
