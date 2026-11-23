# Gate A0: the approved look bake-off

These are the sources of the mockup the owner reviewed on 2026-11-23. It is one CLM sample page in three
looks, each in light and dark.

**Decision:** keep all three looks as reader options, with **ledger** as the default for the site and for the
generated READMEs.

**Files:**
- `template.html`: the page, with the three looks as token sets.
- `build.py`: generates the two data-driven SVGs from our recorded results and writes `clm-bakeoff.html` next
  to it. These are the axonometric architecture drawing and the cosine-gap ruler.

Run `build.py` from the repository root with the `data` extra installed:
`uv run python site/design/gate-a0/build.py`. It reads the pinned S1-Arena RouterBench items and our CLM
responses.

The mockup went through two independent fact-check rounds. Their fixes are folded into the site's
`docs/models/clm/model.yaml`. The architecture drawing, the request player and the cosine ruler are ported
into the site as components in the next phase-A PRs.
