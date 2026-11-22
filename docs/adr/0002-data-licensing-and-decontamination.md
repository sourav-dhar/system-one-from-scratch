# ADR 0002: Training data, licensing and decontamination

- **Status:** accepted, 2026-11-22
- **Evidence:** [`docs/research/survey.md`](../research/survey.md) §6; [`notes/raw/04-training-data-and-benchmarks.md`](../research/notes/raw/04-training-data-and-benchmarks.md)

## Context

The strongest typed-decision training sets have restrictive or missing licences:
- DecisionBank's card says "do not use commercially";
- kev-suites carries no licence;
- the NVIDIA Nemotron source text allows internal training only.

Benchmark sources overlap heavily with training mixtures:
- System1-Arena has no train split, and many of its sources are test splits of datasets present in DecisionBank, kev-suites, embed-nemotron and ADP v2;
- OpenDecisions shares generators with DecisionBank's `j11_generator` rows.

## Decision

1. **Research/portfolio licensing.** Non-commercial and unlicensed public datasets may be used for research runs. Every released checkpoint's model card lists each training source and its licence. No dataset is redistributed: the repo stores loaders, hashes and manifests only.
2. **An evaluation registry is built before any training run.** It holds normalized (lowercased, whitespace-collapsed) and hashed states, questions and options from every benchmark we report:
   - System1-Arena;
   - OpenDecisions;
   - typed-decisions test;
   - JevBench public;
   - Jeva suites;
   - our tricky-Q&A held-out chunks.
3. **Training rows are removed** on an exact hash match, a 13-gram overlap, or a MinHash Jaccard > 0.8 against the registry.
4. **Source-level exclusion.** A whole source dataset is excluded when its train and test splits are not verifiably disjoint from an evaluation source. Examples: ChaosNLI, Civil Comments, and CLINC/Banking77 *test* items. DecisionBank `j11_generator` rows are dropped whenever OpenDecisions is reported.
5. **Calibration is never fitted on a benchmark.** Temperatures are fitted on our own held-out *datasets*: DecisionBank by `group_id`, Nemotron by `doc_id`.
6. **Teacher labels** come from open models (e.g. Qwen-family reasoning models or Inkling) so we never depend on closed-model terms. No Jev outputs are used for training.

## Consequences

- Reported numbers are defensible as clean, at some cost in usable training data.
- The registry and filters are code with tests, run in CI on small fixtures.
- A commercial-safe variant remains possible by filtering on DecisionBank's `source_license` field and swapping sources.
