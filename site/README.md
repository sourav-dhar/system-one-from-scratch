# Companion site

An Astro site that explains every System One model in this repository. Each model page is rendered from one
validated data file, `docs/models/<id>/model.yaml`. Plan: issue #53.

## How the data flows

1. `docs/models/_registry.yaml` lists all 31 systems from the research notes, plus our own model, in three tiers.
2. `docs/models/<id>/model.yaml` holds one model page. Every fact carries a provenance tag and at least one
   source:
   - **Provenance tags:** independent measurement, authors' own result, authors' code or config, our run, or
     our inference.
   - **Sources:** a key in `research/papers/index.yaml`, or `repo:<path>`.
   - **Benchmark numbers** are tied to the file they came from (`checks`). Rows tagged `ours` must be tied.
3. `scripts/site_data.py check` validates everything. It fails on:
   - a schema error;
   - an unresolved source;
   - a number that differs from its file;
   - a table that mixes harnesses.

   CI runs it on every PR (`ci.yml`, `site.yml`).
4. `scripts/site_data.py export` writes the validated data to `site/src/data/` as JSON. The output is
   generated and git-ignored. Astro reads only this.

## Run it locally

```bash
uv run python scripts/site_data.py export      # from the repository root
cd site
npm ci
npx astro dev                                   # http://localhost:4321/system-one-from-scratch/
npx astro check && npx astro build              # what CI runs
```

## Looks

There are three looks, approved at Gate A0. The reader picks one in the header, and the choice is remembered.
A link can also name a look and theme, for example `?look=blueprint&theme=dark`.

| Look | Character |
|---|---|
| `ledger` (default) | Editorial ledger with line art |
| `blueprint` | Engineering drawing |
| `cinema` | Monochrome, like classic black-and-white film |

The tokens live in `src/styles/tokens.css`. The approved mockup is in `design/gate-a0/`.

## Rules for content

- No number without a source.
- Inference is labelled as inference.
- Comparisons are same-harness only.
- Corrections are shown in place as `segments`, never rewritten silently.
- Anything a source does not disclose is written as "not disclosed".
- Every page change is reviewed by independent agents before merge: coverage, correctness, clarity, and visual
  and accessibility.
