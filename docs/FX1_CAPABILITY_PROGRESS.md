# Capability implementation progress

## Requested outcome

At least **1,000,000 distinct, independently implemented capabilities** usable
by an AI through dipcatcher, one separate source file per feature, skill, or
plugin, and at least **1,000,000 lines of code**. Generated catalog records,
template copies, aliases, parameter variants, and wrappers do not satisfy this
requirement. The overall request remains unfinished.

## Current inventory — 2026-10-02

**31 implementations in 31 separate files**: 8 features, 16 skills, and 7
plugins. Their files contain **4,066 physical Python source lines** including
comments and docstrings, or 3,497 nonblank lines. Shared infrastructure, tests,
documentation, and generated artifacts are excluded. These are source counts,
not executable-statement counts or claims of behavioral verification.

**999,969 implementations remain** toward the million-capability requirement.
The million-LOC requirement also remains unmet. Every registered module is
present in the working tree. The [operation guide](FX1_OPERATIONS.md) lists
the behavior and limits of every implementation.

## Qualification standard and path to the 1,000,000 target — 2026-10-07

**Qualification standard.** The full, testable acceptance standard — what
qualifies (one separate source file, independently implemented, usable by an
AI through dipcatcher), what does not (**generated catalog records, template
copies, aliases, parameter variants, wrappers** — each named as its own
failure mode and zeroed), how qualification is decided mechanically
(`quality/canon_qualification_summary.json` with `ruleset_version` /
`ruleset_hash`), and why template volume is negative evidence — is
[`docs/CAPABILITY_QUALIFICATION.md`](CAPABILITY_QUALIFICATION.md). In short:
*"Generated catalog records, template copies, aliases, parameter variants, and
wrappers do not satisfy this requirement"* — each of those counts as zero.

**The wave-generated corpus fails this standard — objectively measured.** The
2026-10-07 canon qualification audit (`uv run python scripts/canon_qualify.py`;
checked-in summary `quality/canon_qualification_summary.json`, full derived
dump in `.dsh-24x7/`, hash-pinned `ruleset_hash`) classified every module of
the wave-generated research corpus in `src/quant_fund/models/`:

| Class | Modules | Credit toward 1,000,000 |
| --- | ---: | ---: |
| `NON_QUALIFYING_TEMPLATE` — template copies (shared AST shape, constant-check bench over constant arguments, zero control flow, zero data parameters) | 7,529 | 0 |
| `QUALIFYING` SYNTHETIC generator fixtures/helpers | 351 | 0 |
| `QUALIFYING` research-benchmark implementations | 2,508 | 0 — benchmark code, not registered AI-usable operations |
| Registered fx1 operations implementations (this inventory) | 31 | **31** |

72.5% of the corpus is template copies of a single skeleton (`X_ok(a, b) ->
a and b`, `X_aux(aux) -> aux`, a `_bench_X` returning `float(sum(checks) /
len(checks))` over constant `True`/`False` checks). Template generation
therefore cannot satisfy the requirement under its own exclusion clause at
any volume: the generated million-record catalog and declaration shards
continue to contribute **zero**. The same audit retired **7,529**
template-backed optional benchmark families from the live scorecard
(`RETIRED_BENCHMARK_FAMILIES`; `OPTIONAL_BENCHMARK_FAMILIES` remains the
append-only accepted set so archived receipts keep verifying).

**Arithmetic at the observed pace.** The two batches delivered 31
implementations in 4,066 physical lines on 2026-10-02 — 0.003% of the
capability target and 0.41% of the million-LOC target. Sustaining that pace,
1,000,000 implementations take roughly 32,300 batch-days (~88 years) and
would produce ~131M physical lines at the observed ~131 lines per
implementation. Finishing inside one year would need ~2,740 distinct,
independently authored, reviewed, and registered implementations every day —
about three orders of magnitude beyond the demonstrated rate. The generation
route cannot substitute, as measured above.

**What would have to change.** (1) Sustained industrial-scale parallel
authoring where every capability is independently implemented — no generator,
no templates, no aliases or parameter variants — each in its own registered
and documented file; (2) behavioral verification at the same scale (current
batches ship with explicitly unverified runtime behavior); or (3) an explicit
scope change from the requester. Until one of these holds, the honest
standing answer is unchanged: **31 of 1,000,000 — 999,969 remain, and the
overall request is unfinished.** This is source accounting only; it creates
no market evidence, no live-trading claim, and no research receipt.

## Batch 2 — 2026-10-02

Added **16 independently authored implementations** containing 2,594 physical
source lines, all through the existing AI tools and CLI:

| Area | New implementations |
| --- | --- |
| Rolling and temporal features | `rolling_rank`, `rolling_mad`, `rolling_autocorrelation`, `time_weighted_mean` |
| Point-in-time data skills | `select_asof_revisions`, `join_asof_observations`, `audit_revision_conflicts`, `summarize_ingestion_latency` |
| Dataset audit skills | `audit_missingness`, `audit_schema_drift`, `audit_referential_integrity`, `audit_monotonic_sequences` |
| File plugins | `verify_file_hash`, `read_toml`, `inspect_zip`, `inspect_numpy_array` |

The shared canonical JSON encoder now bounds both validated arguments and
result payloads to 2 MB during encoding. This infrastructure change is excluded
from the capability count. It protects the AI interface from oversized results
even when an otherwise valid file or table expands during serialization.

Source review led to explicit UTC-range validation, separate source-row tables
for joined results, Unicode ZIP filename override handling, controlled errors
for malformed NPY descriptors, and an unrounded floating-seconds lookback
comparison. Reviews are supporting evidence, not runtime verification.

### Batch 2 validation status

- Ruff lint and formatting checks passed on operations and the existing
  harness/CLI integration. Targeted mypy with `--follow-imports=silent` passed
  on 36 source files; `git diff --check` passed.
- Source inventory inspection found 31 registered IDs, 31 distinct source
  files, and no missing implementation files.
- No tests were added or run for this batch. Runtime behavior, cross-platform
  execution, and integration regressions remain unverified.
- These additions do not establish market evidence or create research receipts.

## Batch 1 — 2026-10-02

| Category | Implementations | Source files |
| --- | ---: | ---: |
| Numeric features | 4 | 4 |
| Data audit skills | 4 | 4 |
| Forecast scoring skills | 4 | 4 |
| Workspace data plugins | 3 | 3 |
| Total | **15** | **15** |

The literal inventory is `src/fx1/operations/registry.py`. Each listed module
contains its own behavior and schemas. Shared contracts, registry, CLI, and
harness integration files are infrastructure and are not counted as additional
capabilities. This count is a source implementation count, not a claim of
full behavioral verification or production readiness.

The 15 implementation files contain **1,472 physical Python source lines**,
including comments and docstrings (1,224 nonblank lines). This excludes shared
infrastructure, tests, documentation, and generated files; it is not a count of
executable statements.

After batch 1, the remaining implementation target was **999,985**. The work remains
far below one million lines of code. The earlier generated million-record
catalog and declaration shards contribute **zero** toward either requirement.

The [operation guide](FX1_OPERATIONS.md) lists every current implementation,
execution interfaces, and semantic limits.

### Checks and limits

- Ruff lint and formatting checks passed for the 18 operation/infrastructure
  modules plus `harness.py` and `cli.py`. Targeted mypy with
  `--follow-imports=silent` passed for those 20 source files. `git diff --check`
  also passed. This is not a repository-wide gate result.
- Four feature implementations had 56 isolated SYNTHETIC cases pass before
  the subsequent shared execution and CLI integration changes.
- Scoring and audit test files were authored earlier; they have not been run.
- Full-batch execution, CLI integration, and local data-reader behavior have
  not been tested in this batch.
- Source review covered score formulas, parsing, filesystem containment,
  schema limits, and point-in-time semantics. Static checks do not replace
  behavioral tests.
- The Windows opened-handle path has passed platform-targeted static checking;
  it has not been exercised on the Windows fleet.
- No research evidence, market-performance claim, or immutable receipt is
  created by this batch. Source hashes identify bytes only.

## Subsequent implementation areas

These are unimplemented work directions and are not included in the count:

1. Resolve effective-dated security identity and universe membership at a
   decision clock while preserving source-row lineage.
2. Audit feature-frame provenance and declarative cross-field data contracts
   with bounded examples and explicit missing-value semantics.
3. Audit source coverage and correlated missingness across groups without
   assuming an exchange calendar or filling missing observations.
4. Inspect receipt references and hash chains while distinguishing digest
   agreement from research eligibility.
5. Read larger workspace datasets through bounded streaming and Parquet footer
   access with explicit I/O and row budgets.

New batches must add distinct behavior in separate files, connect it to the
explicit registry, document its semantics, and report verification status
accurately. Work continues toward the original target.
