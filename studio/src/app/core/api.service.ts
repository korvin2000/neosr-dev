import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import type { CompileResult } from '@shared/compile/compile';
import type { Problem } from '@shared/compile/problems';
import type { Recipe } from '@shared/recipe.model';
import type { EnvironmentInfo, RunDetail, RunSnapshot, RunSummary } from '@shared/run.model';

export interface RecipeSummary {
  readonly id: string;
  readonly name: string;
  readonly updatedAt: string;
  readonly templateId?: string;
  readonly phases: number;
  readonly arch: string;
  readonly scale: number;
  readonly nodeCount: number;
  readonly description: string;
}

export interface ModelFile {
  readonly file: string;
  readonly path: string;
  readonly arch: string;
  readonly scale: number;
  readonly iter: number;
  readonly sizeMb: number;
  readonly runName: string;
  readonly variant: 'ema' | 'eval' | 'state';
  readonly updatedAt: string;
}

export interface DatasetInfo {
  readonly id: string;
  readonly name: string;
  readonly gtPath: string;
  readonly lqPath?: string;
  readonly images: number;
  readonly scale?: number;
  readonly minWidth: number;
  readonly minHeight: number;
  readonly format: string;
  readonly note: string;
}

export interface ScanReport {
  readonly gtCount: number;
  readonly lqCount: number;
  readonly pairs: number;
  readonly unmatchedGt: readonly string[];
  readonly unmatchedLq: readonly string[];
  readonly scaleMismatch: readonly { file: string; gt: [number, number]; lq: [number, number] }[];
  readonly belowPatch: readonly { file: string; w: number; h: number }[];
  readonly channels: { rgb: number; rgba: number; gray: number };
  readonly overlapWithVal: readonly string[];
  readonly samples: readonly { name: string; w: number; h: number }[];
  readonly durationMs: number;
}

export class LaunchRejected extends Error {
  constructor(readonly problems: readonly Problem[]) {
    super('Launch rejected: the recipe has blockers.');
  }
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);

  environment(): Promise<EnvironmentInfo> {
    return firstValueFrom(this.http.get<EnvironmentInfo>('/api/environment'));
  }

  listRecipes(): Promise<RecipeSummary[]> {
    return firstValueFrom(this.http.get<RecipeSummary[]>('/api/recipes'));
  }

  getRecipe(id: string): Promise<Recipe> {
    return firstValueFrom(this.http.get<Recipe>(`/api/recipes/${encodeURIComponent(id)}`));
  }

  saveRecipe(recipe: Recipe): Promise<Recipe> {
    return firstValueFrom(this.http.put<Recipe>(`/api/recipes/${encodeURIComponent(recipe.id)}`, recipe));
  }

  deleteRecipe(id: string): Promise<void> {
    return firstValueFrom(this.http.delete<void>(`/api/recipes/${encodeURIComponent(id)}`));
  }

  compile(recipe: Recipe, platform: 'windows' | 'linux'): Promise<CompileResult> {
    return firstValueFrom(this.http.post<CompileResult>('/api/compile', { recipe, platform }));
  }

  listRuns(): Promise<RunSummary[]> {
    return firstValueFrom(this.http.get<RunSummary[]>('/api/runs'));
  }

  getRun(id: string): Promise<RunSnapshot> {
    return firstValueFrom(this.http.get<RunSnapshot>(`/api/runs/${encodeURIComponent(id)}`));
  }

  async launch(recipe: Recipe, platform: 'windows' | 'linux', speed: number): Promise<RunDetail> {
    try {
      return await firstValueFrom(this.http.post<RunDetail>('/api/runs', { recipe, platform, speed }));
    } catch (error) {
      if (error instanceof HttpErrorResponse && error.status === 400 && Array.isArray(error.error?.problems)) {
        throw new LaunchRejected(error.error.problems as Problem[]);
      }
      throw error;
    }
  }

  stopRun(id: string, force = false): Promise<RunSummary> {
    return firstValueFrom(this.http.post<RunSummary>(`/api/runs/${encodeURIComponent(id)}/stop`, { force }));
  }

  resumeRun(id: string): Promise<RunSummary> {
    return firstValueFrom(this.http.post<RunSummary>(`/api/runs/${encodeURIComponent(id)}/resume`, {}));
  }

  chaos(id: string, kind: 'disconnect' | 'nan'): Promise<RunSummary> {
    return firstValueFrom(this.http.post<RunSummary>(`/api/runs/${encodeURIComponent(id)}/chaos`, { kind }));
  }

  deleteRun(id: string): Promise<void> {
    return firstValueFrom(this.http.delete<void>(`/api/runs/${encodeURIComponent(id)}`));
  }

  eventsUrl(id: string, after: number): string {
    return `/api/runs/${encodeURIComponent(id)}/events?after=${after}`;
  }

  listModels(): Promise<ModelFile[]> {
    return firstValueFrom(this.http.get<ModelFile[]>('/api/library/models'));
  }

  listDatasets(): Promise<DatasetInfo[]> {
    return firstValueFrom(this.http.get<DatasetInfo[]>('/api/library/datasets'));
  }

  scanDataset(req: { gt: string; lq?: string; scale: number; patch: number }): Promise<ScanReport> {
    return firstValueFrom(this.http.post<ScanReport>('/api/datasets/scan', req));
  }
}
