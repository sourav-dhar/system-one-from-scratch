// Typed access to the data exported by `scripts/site_data.py export`.
// The Python side validates everything (schema, sources, checked numbers); these types mirror its output.

export type Provenance = 'independent' | 'self-reported' | 'code' | 'ours' | 'inferred';

export interface Source { key: string; at: string | null }
export interface Fact { text: string; provenance: Provenance[]; sources: Source[] }
export interface SpecItem { label: string; value: string; provenance: Provenance; sources: Source[] }
export interface BenchRow {
  label: string; rank: string | null; readout: string;
  accuracy: number | null; accuracy_ci: [number, number] | null; brier: number | null; p50_ms: number | null;
  measured_on: string; provenance: Provenance; sources: Source[]; harness: string; focus: boolean; child: boolean;
}
export interface BenchTable {
  suite: string; title: string; caption: string;
  reference_line: number | null; reference_label: string | null; notes: Fact[]; rows: BenchRow[];
}
export interface Segment { kind: 'keep' | 'del' | 'ins'; text: string }
export interface Correction { date: string; commit: string; where: string; segments: Segment[]; provenance: Provenance; sources: Source[] }
export interface Coverage { element: string; source: Source; status: 'covered' | 'planned' | 'n/a'; section: string | null; reason: string | null }
export interface Reference { key: string; title: string; url: string; kind: 'arxiv' | 'link' | 'repo' }
export interface ModelPage {
  id: string; name: string; version: string | null; full_name: string; tier: 'A' | 'B' | 'C'; family: string;
  note: string; eyebrow: string[]; dek: string; spec: SpecItem[];
  summary: { what: Fact[]; why: Fact[]; weakness: Fact[] };
  benchmarks: BenchTable[]; corrections: Correction[]; coverage: Coverage[];
  claims: { text: string; sources: Source[]; verified_by: string; date: string }[];
  references: Reference[];
}
export interface RegistryEntry {
  id: string; name: string; tier: 'A' | 'B' | 'C'; family: string; note: string | null;
  status: 'planned' | 'draft' | 'published'; has_page: boolean;
}

const modules = import.meta.glob<{ default: ModelPage }>('../data/models/*.json', { eager: true });
export const models: ModelPage[] = Object.values(modules).map((m) => m.default);

import registryJson from '../data/registry.json';
export const registry = registryJson as RegistryEntry[];

export const PROVENANCE_LABEL: Record<Provenance, string> = {
  independent: 'independent measurement',
  'self-reported': "authors' own result",
  code: "authors' code or config",
  ours: 'our run',
  inferred: 'our inference',
};

export const FAMILY_LABEL: Record<string, string> = {
  biencoder: 'bi-encoder',
  'late-interaction': 'late interaction',
  'decoder-readout': 'decoder readout',
  encoder: 'encoder',
  'encoder-decoder': 'encoder–decoder',
  looped: 'looped decoder',
  'frozen-reader': 'frozen reader',
  generative: 'generative LLM',
  undisclosed: 'undisclosed',
  ours: 'in development',
};

/** A site-internal URL that respects the configured `base`. */
export function href(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const clean = path.replace(/^\//, '');
  return `${base}/${clean}`;
}

/** Roman numerals for section numbers (the cinema look shows them as intertitles). */
export function roman(n: number): string {
  const table: [number, string][] = [[10, 'X'], [9, 'IX'], [5, 'V'], [4, 'IV'], [1, 'I']];
  let out = '';
  for (const [v, s] of table) while (n >= v) { out += s; n -= v; }
  return out;
}
