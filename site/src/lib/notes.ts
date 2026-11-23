// Numbered source notes for a model page. Sources are numbered in reading order, once each, and every fact
// shows its markers ([1, 4]) instead of a wall of links; the page's last section lists the numbered sources.
import type { ModelPage, Source } from './data';

export interface Notes {
  num: (s: Source) => number;
  order: Source[];
}

const keyOf = (s: Source) => `${s.key}|${s.at ?? ''}`;

export function pageNotes(m: ModelPage & { primer?: { sources: Source[] }[] }): Notes {
  const order: Source[] = [];
  const index = new Map<string, number>();
  const add = (s: Source) => {
    const k = keyOf(s);
    if (!index.has(k)) { order.push(s); index.set(k, order.length); }
  };
  // Reading order of the page template.
  m.dek_sources.forEach(add);
  m.spec.forEach((s) => s.sources.forEach(add));
  (m.primer ?? []).forEach((t) => t.sources.forEach(add));
  [...m.summary.what, ...m.summary.why, ...m.summary.weakness].forEach((f) => f.sources.forEach(add));
  m.benchmarks.forEach((t) => {
    t.caption_sources.forEach(add);
    t.rows.forEach((r) => r.sources.forEach(add));
    t.notes.forEach((n) => n.sources.forEach(add));
  });
  m.corrections.forEach((c) => c.sources.forEach(add));
  m.coverage.forEach((c) => add(c.source));
  m.claims.forEach((c) => c.sources.forEach(add));
  return {
    order,
    num: (s: Source) => {
      const n = index.get(keyOf(s));
      if (n === undefined) throw new Error(`source ${keyOf(s)} was not numbered; add it to pageNotes()`);
      return n;
    },
  };
}
