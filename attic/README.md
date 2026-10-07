# attic/ — quarantined non-qualifying corpus artifacts

Nothing in this directory is imported, collected, tested, or packaged.

`attic/` sits at the repository root, outside the Hatch wheel packages
(`src/fx1`, `src/quant_fund`), so it is excluded from distributions
automatically. It is also outside the pytest `testpaths`
(`tests/unit`, `tests/property`, `tests/regression`, `tests/end_to_end`,
`tests/formal`), so nothing here is collected.

## Why these files were quarantined

The wave-generation corpus in `src/quant_fund/models/` grew to ~10,386
modules, of which the overwhelming majority were **template copies**: modules
sharing one normalized-AST shape whose entire computation is boolean plumbing
over constant arguments — `X_ok(a, b) -> a and b`, `X_aux(aux) -> aux`, and a
`_bench_X` returning `float(sum(checks) / len(checks))` over constant
`True`/`False` checks. Per `docs/FX1_CAPABILITY_PROGRESS.md`, template copies
do **not** count toward the 1,000,000 distinct independently implemented
capabilities requirement, and leaving them in the live tree distorted every
corpus-size and coverage measurement.

The qualification ruleset lives in
`src/quant_fund/models/canon_qualification.py` (`RULESET`, pinned by
`ruleset_hash`). A file here was classified `NON_QUALIFYING_TEMPLATE` — all
five of: shared near-duplicate AST shape, no substantive bench, zero control
flow, effective body lines under the pinned threshold, and zero data-carrying
parameters — and its whole import-connected component had no live importers
outside the moved set.

Files that were classified non-qualifying but had live importers were **not**
moved; they remain in the live tree and are listed in
`quality/canon_qualification_audit.json` under `outstanding_quarantine` with
the blocking importer named. Owner-guarded modules (`garch*`, `vol*`, `har*`,
`__init__` in `src/quant_fund/models/`) were never candidates.

`benches_w*.py` adapter wiring and `tests/unit/models/test_w*.py` /
`tests/unit/research/test_benches_w*.py` wave tests were quarantined alongside
the modules they wired or exercised, moving only when every target they
reference moved with them.

## Layout

Paths mirror the live tree, so a quarantined module's provenance is obvious:

    attic/src/quant_fund/models/<module>.py
    attic/src/quant_fund/research/benches_w<NNN>.py
    attic/tests/unit/models/test_w<NNN>.py
    attic/tests/unit/research/test_benches_w<NNN>.py

Registered family names were **not** erased: `OPTIONAL_BENCHMARK_FAMILIES` in
`quant_fund.research.catalog.registry` is the append-only accepted set and
stays byte-stable so archived receipts containing retired families still
verify. Families backed only by quarantined modules are listed in
`RETIRED_BENCHMARK_FAMILIES` (with reason and last receipt schema version);
the live emission set is `LIVE_OPTIONAL_BENCHMARK_FAMILIES`.

## Reproducing the audit that produced this quarantine

```bash
uv run python scripts/canon_qualify.py           # regenerate the audit JSON
uv run python scripts/canon_qualify.py --check   # verify it is not stale
```

The audit writes `quality/canon_qualification_audit.json`: per-file verdicts,
family ids and waves, before/after (live vs attic) counts, the pinned
`ruleset_hash`, and the import-graph quarantine plan (component ids, import
edges, and `outstanding_quarantine` with blocking importers). Two runs over
the same tree are byte-identical; `tests/unit/models/test_canon_qualification.py`
pins the ruleset hash and fails if a `NON_QUALIFYING_TEMPLATE` module is
reintroduced into the live tree.

SYNTHETIC-labelled modules in this archive are correctness fixtures for a
generator. They were never market evidence and are not evidence now.
