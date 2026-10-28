import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', loadComponent: () => import('./start/start-page').then((m) => m.StartPage), title: 'NeoSR Studio' },
  {
    path: 'setup',
    loadComponent: () => import('./setup/setup-page').then((m) => m.SetupPage),
    title: 'Setup · NeoSR Studio',
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'data' },
      { path: 'data', loadComponent: () => import('./setup/data-step').then((m) => m.DataStep) },
      { path: 'review', loadComponent: () => import('./setup/review-step').then((m) => m.ReviewStep) },
    ],
  },
  {
    path: 'build',
    loadComponent: () => import('./build/build-page').then((m) => m.BuildPage),
    title: 'Build · NeoSR Studio',
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'list' },
      { path: 'list', loadComponent: () => import('./build/list/list-page').then((m) => m.ListPage) },
      { path: 'graph', loadComponent: () => import('./build/graph/graph-page').then((m) => m.GraphPage) },
      { path: 'toml', loadComponent: () => import('./build/toml/toml-page').then((m) => m.TomlPage) },
      { path: 'wizard', redirectTo: '/setup' },
      { path: 'wizard/:step', redirectTo: '/setup' },
    ],
  },
  { path: 'runs', loadComponent: () => import('./runs/runs-page').then((m) => m.RunsPage), title: 'Runs · NeoSR Studio' },
  { path: 'runs/:id', loadComponent: () => import('./runs/cockpit-page').then((m) => m.CockpitPage), title: 'Run · NeoSR Studio' },
  { path: 'compare', loadComponent: () => import('./compare/compare-page').then((m) => m.ComparePage), title: 'Compare · NeoSR Studio' },
  { path: 'library', loadComponent: () => import('./library/library-page').then((m) => m.LibraryPage), title: 'Library · NeoSR Studio' },
  { path: 'library/:tab', loadComponent: () => import('./library/library-page').then((m) => m.LibraryPage), title: 'Library · NeoSR Studio' },
  { path: 'settings', loadComponent: () => import('./settings/settings-page').then((m) => m.SettingsPage), title: 'Settings · NeoSR Studio' },
  { path: '**', redirectTo: '' },
];
