# Capability implementation progress

## Requested outcome

At least **1,000,000 distinct, independently implemented capabilities** usable
by an AI through dipcatcher, one separate source file per feature, skill, or
plugin, and at least **1,000,000 lines of code**. Generated catalog records,
template copies, aliases, parameter variants, and wrappers do not satisfy this
requirement. The overall request remains unfinished.

## Current inventory — 2026-10-04

**169 implementations in 169 separate files**: 53 features, 79 skills, and 37
plugins. Their files contain **42,930 physical Python source lines** including
comments and docstrings, or 38,345 nonblank lines. Of these, **35,668 lines contain
code tokens**, excluding blank lines, comments, and AST-identified docstrings.
Shared infrastructure, tests, documentation, and generated artifacts are
excluded. These are source counts, not executable-statement counts.

Every one of the 169 registered ids now has at least one hand-checked behavioral
invocation case exercising the harness `execute_operation` path
(`tests/fx1/test_operations_registry_invocation.py` plus its
`operation_cases_chunk*.py` siblings): **169 of 169 covered, 0 missing**. That is
a positive-path floor, not full behavioral verification — negative, adversarial,
resource-bound and platform-specific coverage remains incomplete for many
operations and is tracked as outstanding Phase 1 work.

All counts and LOC measures above are reproducible with
`uv run python scripts/fx1_capability_baseline.py --print-summary`, which also
writes `artifacts/fx1_capability_baseline.json` and
[the baseline report](FX1_CAPABILITY_BASELINE.md). The LOC definitions are frozen
there and reuse the `measure_file` helper from `scripts/code_quality_inventory.py`.

**999,831 implementations remain** toward the million-capability requirement.
The million-LOC requirement also remains unmet. Every registered module is
present in the working tree. The [operation guide](FX1_OPERATIONS.md) lists
the behavior and limits of every implementation.

## Batch 13 — 2026-10-04

**Registration and behavioral verification of 12 pre-existing implementations.**
No new capability code was authored in this batch. A Phase 0 inventory
reconciliation (`scripts/fx1_capability_baseline.py`) found that
`src/fx1/operations/` held 173 `.py` files while `_IMPLEMENTATIONS` listed only
157: **12 complete, non-stub modules (231–529 lines each) exported a valid
`OPERATION` but were absent from the registry**, so `fx1 harness operations`,
`describe-operation` and `execute-operation` could never reach them. They were
unreachable dead code, and this ledger previously described their areas as
"unimplemented work directions … not included in the count".

Per the plan's rule 2 (*behavior before registration*), behavioral cases were
authored **first**, then the ids were registered.

```text
date / wave: 2026-10-04 / Batch 13 — registration + verification, no new implementations
accepted total (by kind): 169  (53 feature, 79 skill, 37 plugin)
new candidates / accepted / deferred / rejected: 0 authored / 12 registered / 0 / 0
LOC: physical 42,930 / nonblank 38,345 / code-token 35,668  (registered files only)
focused tests and repo gates run:
  uv run pytest tests/fx1/test_operations_registry_invocation.py -q -p no:randomly
    -> 170 passed, 0 failed (169 per-id + test_invocation_cases_match_the_registry)
  same file under default random ordering -> 170 passed (no order dependence)
  uv run ruff check tests/fx1/operation_cases_chunk*.py -> clean
  uv run ruff format --check (same four files) -> clean
  uv run fx1 harness operations -> implementation_count 169
review findings and fixes:
  - 30 hand-checked cases added across four new disjoint files
    (operation_cases_chunk5/6/7/8.py); 18 for ids that were registered but
    uncovered, 12 for the orphans.
  - Expected values were derived independently BEFORE execution (closed-form math,
    textbook worked examples, stdlib oracles: hashlib, struct, zipfile, tarfile,
    gzip, sqlite3, numpy, pyarrow read-back, a hand-written Compact-Thrift walker
    for the Parquet footer, and a transcription of Jenkins lookup3 validated
    against its three published known-answer vectors).
  - Four authoring errors were caught by that derivation and corrected; in all
    four the IMPLEMENTATION was right and the expectation was wrong:
    SSA normalizes the trajectory by 2**power_of_two before the SVD; the Parquet
    codec enum 0 is named UNCOMPRESSED (pyarrow's writer-side "NONE" is an alias);
    audit_task_leases.declared_replay_rows charges every event row of a task, not
    the decision-visible subset; reconstruct_order_book sorts orders and levels by
    the same (side, price) key.
  - No implementation defect was found among the 12.
known unverified behavior:
  - Coverage is positive-path only: one hand-checked case per id. Negative,
    adversarial, malformed/truncated-input and resource-bound cases are still
    missing for most of the 30, and are outstanding Phase 1 work.
  - inspect_parquet / inspect_safetensors cases use read_mode="full", so the
    "footer"/"header" open_ranges branches are not exercised.
  - Platform-sensitive parsers remain unexercised on the Windows fleet.
  - Environment-dependent fields were deliberately left unasserted where they
    would render the case brittle (e.g. sqlite_version, defensive_mode).
accepted units per working day (rolling 2-week): not yet measurable — this batch
  registered existing work rather than authoring new units; throughput measurement
  begins with the next authored wave (plan Phase 4, 250-accepted milestone).
next blocking dependency:
  - src/fx1/operations and src/fx1/extensions are still UNTRACKED. Staging them
    must happen in the same commit as a `scripts/gen_arch_diagrams.py` rerun,
    because that generator reads the live filesystem and would otherwise write
    untracked module names into tracked architecture artifacts and fail CI on a
    clean checkout. quality/audit_coverage_fx1.json also needs `operations` and
    `extensions` directory entries at that point.
```

This batch creates no research evidence, no receipt, and no market-performance
claim. Fixtures are synthetic and labeled as such; source hashes identify bytes
only.

## Batch 12 — 2026-10-02

Added **12 independently authored implementations** in 12 separate files
containing 4,272 physical source lines (3,848 nonblank; 3,594 containing code
tokens). All are registered for the existing AI discovery, schema, execution
and CLI interfaces:

| Area | New implementations |
| --- | --- |
| Data and validation contracts | `audit_source_migrations`, `audit_artifact_assembly`, `audit_snapshot_manifests` |
| State and graph models | `markov_absorption`, `linear_gaussian_filter`, `graph_effective_resistance` |
| Geometry, interpolation and optimization | `convex_hull_2d`, `monotone_cubic_interpolation`, `discrete_optimal_transport` |
| File parsing/inspection | `read_wav_pcm`, `read_ply`, `inspect_tiff_directory` |

Migration routing owns visibility-specific identity indexes and event cutovers.
Snapshot reconciliation owns parent-state reconstruction, exact delta
preconditions and retained tombstones. Source review identified future duplicate
declarations affecting historical results; as-of routes and snapshot views now
use only visible metadata, separately from global declaration findings. Snapshot
queries retain original row lineage and share a conservative reconstruction-work
budget across at most 16 distinct clocks. Artifact assembly checks declared byte
geometry, ordering, reference labels and publication without claiming byte truth.

Numerical features own finite-chain support analysis, exact fundamental-matrix
moments, rational Kalman/RTS recursions and grounded-Laplacian solves. Geometry
uses exact hull predicates and area/centroid calculations. Cubic interpolation
owns limited slopes, query derivatives and signed integrals. Transport owns
residual-network augmentation and exact marginal, dual-feasibility and zero-gap
checks. These calculations interpret supplied models and data; they establish
neither causal availability nor empirical accuracy. Rounded outputs and numerical
budgets have explicit conventions.

Readers independently parse PCM/IEEE WAVE, scalar/list PLY and classic TIFF
directory graphs. Complete supported structures are checked before pagination,
with exact scalar representations and source spans. Review corrected the bare
IEEE WAVE header requirement and clarified private TIFF tag semantics. TIFF
image schemas/pixels, PLY geometric validity and WAVE ancillary playback semantics
remain outside the supported checks. No external references are followed.

### Batch 12 validation status

- Static source inventory inspection found 157 registered IDs, 157 distinct
  implementation files and no missing source files. Source IDs/kinds and all
  157 guide inventory rows agree. The largest implementation is 605 physical
  lines. Generated records, wrappers, shared infrastructure, documentation and
  tests are excluded from these counts.
- Ruff lint and formatting passed on operations plus harness/CLI integration.
  Targeted mypy with `--follow-imports=silent` passed on 163 source files.
  Architecture artifacts were regenerated and their freshness check passed.
  `git diff --check` passed. These are targeted static checks, not full
  repository gates.
- Independent source reviews covered visibility, version lineage, interval
  sweeps, delta reconstruction, state filtering, graph solves, numerical
  objectives and file-format boundaries. No tests were added or run. Runtime
  behavior, numerical edge cases, format interoperability, Windows behavior and
  integration regressions remain unverified. Both million targets remain
  unfinished.

## Batch 11 — 2026-10-02

Added **12 independently authored implementations** in 12 separate files
containing 4,086 physical source lines (3,697 nonblank; 3,419 containing code
tokens). All are registered for the existing AI discovery, schema, execution
and CLI interfaces:

| Area | New implementations |
| --- | --- |
| Data and validation contracts | `audit_transaction_batches`, `audit_observation_windows`, `audit_dependency_releases` |
| Numerical features | `segment_mean_changes`, `dynamic_time_warping`, `singular_spectrum_analysis`, `poisson_binomial_distribution` |
| Forecast scoring | `score_markov_paths`, `score_gaussian_mixtures` |
| File parsing/inspection | `read_avro_container`, `inspect_hdf5_superblock`, `read_fits_table` |

Transaction auditing owns a declared batch-state machine and commit-prefix
checkpoint checks without claiming external storage durability. Observation
auditing measures continuous exposure, delayed entry and terminal declarations.
Release auditing operates at stage level, retaining explicit version dependencies
and permanent supersession. Source review found that duplicate stage declarations
could disappear from query candidates; ambiguous visible stages now prevent
silent fallback. Finalized observation-window availability must also follow
the observed end and every used observation's availability.

Numerical features own exact segmentation and alignment objectives, integer
Bernoulli convolution and SSA grouping/diagonal averaging. SSA uses NumPy SVD
as a decomposition primitive and discloses nonunique repeated subspaces.
Markov scoring sums over compatible partial paths by exact forward conditioning.
Gaussian-mixture scoring implements log density and closed-form CRPS with stable
positive-excess calculations. Independent review added a 131,072-bit bound on
rational accumulators across mixtures; row/work limits alone did not bound
denominator growth sufficiently.

Readers independently parse Avro primitive-record containers, HDF5 v2/v3
superblocks and scalar FITS tables. Bounded decompression, source spans and
explicit scalar representations preserve their declared format semantics.
HDF5 checksum/address checks leave object graphs and recovery state unchecked.
FITS nonidentity scaling requires explicit raw-value preservation, and FITS
checksum declarations remain unverified. These are documented format subsets,
not claims of universal format compatibility or market evidence.

### Batch 11 validation status

- Static source inventory inspection found 145 registered IDs, 145 distinct
  implementation files and no missing source files. Source IDs/kinds and all
  145 guide inventory rows agree. The largest implementation remains 574
  physical lines. Generated records, wrappers, shared infrastructure,
  documentation and tests are excluded from these counts.
- Ruff lint and formatting passed on operations plus harness/CLI integration.
  Targeted mypy with `--follow-imports=silent` passed on 151 source files.
  Architecture artifacts were regenerated and their freshness check passed.
  `git diff --check` passed. These are targeted static checks, not full
  repository gates.
- Source reviews covered state transitions, exposure geometry, availability,
  supersession, exact optimization objectives, filtering, mixture moments,
  arithmetic bounds and format declarations. No tests were added or run.
  Runtime behavior, numerical edge cases, format interoperability, Windows
  behavior and integration regressions remain unverified. Both million targets
  remain unfinished.

## Batch 10 — 2026-10-02

Added **12 independently authored implementations** in 12 separate files
containing 3,581 physical source lines (3,206 nonblank; 2,951 containing code
tokens). All are registered for the existing AI discovery, schema, execution
and CLI interfaces:

| Area | New implementations |
| --- | --- |
| Data and validation contracts | `audit_price_basis`, `reconcile_aggregates`, `audit_unit_conversions` |
| Numerical features | `distance_correlation`, `kernel_density_grid`, `lomb_scargle_periodogram`, `empirical_copula`, `compositional_logratios` |
| Forecast scoring | `score_piecewise_hazards` |
| File parsing | `read_cbor`, `read_messagepack`, `read_bson` |

Price audits separate declared vendor bases from exact transformation checks and
dependency availability. Aggregate reconciliation withholds comparison when
membership resolution is incomplete. Unit conversion auditing owns exact affine
graph composition and complete contradictory-cycle witnesses; physical constants
and external population or corporate-action completeness remain unverified.

Numerical implementations own distance double centering, Gaussian mixture
evaluation, harmonic normal equations, weighted marginal ranking and composition
transforms. Source review found that subtracting independently rounded logs could
lose small part-ratio changes or introduce variation from unrelated parts.
Pairwise variation now centers exact ratios against a supplied reference row
before taking logs, using exponent reduction for ratios outside binary64 range.
Stable adjacent log increments separately construct the approximate CLR/ILR
coordinates. These source changes have not been behaviorally tested.

The survival scorer integrates supplied piecewise hazards exactly before stable
logarithmic evaluation. Exact, right-censored and interval-censored observations
support delayed entry. Zero-likelihood outcomes retain infinite-loss status.
Noninformative censoring/entry, outcome-independent weights, declared time units
and forecast timing remain explicit assumptions.

Binary readers implement their own bounded byte parsers and validate complete
sources before pagination. They preserve exact scalar representations, source
spans and hashes. Unsupported tags, executable types, extension semantics and
other format features fail according to each documented subset. Parsing does
not establish database compatibility, canonical encoding or research evidence.

### Batch 10 validation status

- Static source inventory inspection found 133 registered IDs, 133 distinct
  implementation files and no missing source files. Source IDs/kinds and all
  133 guide inventory rows agree. The largest implementation remains 574
  physical lines. Generated records, wrappers, shared infrastructure,
  documentation and tests are excluded from these counts.
- Ruff lint and formatting passed on operations plus harness/CLI integration.
  Targeted mypy with `--follow-imports=silent` passed on 139 source files.
  Architecture artifacts were regenerated and their freshness check passed.
  `git diff --check` passed. These are targeted static checks, not full
  repository gates.
- Independent source reviews covered numerical centering, logarithmic precision,
  survival support/entry, availability propagation, aggregate completeness,
  affine witnesses and byte-format boundaries. No tests were added or run.
  Runtime behavior, numerical edge cases, format interoperability, Windows
  behavior and integration regressions remain unverified. Both million targets
  remain unfinished.

## Batch 9 — 2026-10-02

Added **12 independently authored implementations** in 12 separate files
containing 3,767 physical source lines (3,422 nonblank; 3,183 containing code
tokens). All are registered for the existing AI discovery, schema, execution
and CLI interfaces:

| Area | New implementations |
| --- | --- |
| Data and validation contracts | `audit_source_chains`, `audit_temporal_resamples`, `audit_schema_compatibility` |
| Numerical features | `circular_summary`, `weighted_geometric_median`, `partial_autocorrelation`, `weighted_quantile_binning` |
| Forecast scoring | `score_joint_categorical`, `score_count_forecasts` |
| File parsing | `read_arff`, `read_dbase`, `read_sparse_npz` |

Source-chain checks separate declared links from optional raw-byte hash checks.
Temporal resampling audits keep availability, completed events, output positions,
block continuation and order policies explicit. Schema compatibility operates on
closed writer declarations and full type domains, not observed samples.

Numerical implementations own their directional moments and arc selection,
modified Weiszfeld iterations, Levinson–Durbin recursion and weighted-CDF bin
construction. Source review identified arc endpoint rounding that could weaken
a coverage assertion; exact rational arc fields now carry that assertion, with
float previews separately disclosed. Solver convergence is numerical and does
not certify an optimum.

Joint scoring retains joint and marginal support distinctions without assuming
independence. The count scorer combines three related distribution families in
one implementation; its branches do not inflate the capability count. Impossible
outcomes retain infinite-loss status. Positive weighting, forecast timing and
fitting-sample availability remain explicit caller assumptions.

Parsers implement strict, documented format subsets with complete bounded-source
validation. Sparse NPZ uses its own ZIP/NPY decoding and checks extents, CRCs,
expansion, sparse pointers and coordinates without constructing dense matrices
or loading serialized objects. Source review tightened NPY dictionary coverage
to reject trailing expressions/comments outside the required header literal.

### Batch 9 validation status

- Static source inventory inspection found 121 registered IDs, 121 distinct
  implementation files and no missing source files. The largest implementation
  remains 574 physical lines. Source IDs/kinds and 121 guide inventory rows agree.
  Generated records, wrappers, infrastructure, documentation and tests are excluded.
- Ruff lint and formatting passed on operations plus harness/CLI integration.
  Targeted mypy with `--follow-imports=silent` passed on 127 source files.
  Architecture artifacts were regenerated and their freshness check passed.
  `git diff --check` passed. These are targeted static checks, not full
  repository gates.
- Independent source reviews cover numerical formulas, support and weight rules,
  lineage witnesses, declared type domains, interval policies and file-format
  parsing. No tests were added or run. Runtime behavior, numerical edge cases,
  format compatibility, Windows behavior and integration regressions remain
  unverified. Both million targets remain unfinished.

## Batch 8 — 2026-10-02

Added **12 independently authored implementations** in 12 separate files
containing 3,211 physical source lines (2,911 nonblank; 2,680 containing code
tokens). All are registered for the existing AI discovery, schema, execution
and CLI interfaces:

| Area | New implementations |
| --- | --- |
| Data and validation contracts | `align_event_sequences`, `audit_partition_ranges`, `audit_nested_folds` |
| Dependent resampling | `build_stationary_resamples` |
| Numerical features | `sample_entropy`, `burg_autoregression`, `principal_coordinates` |
| Forecast scoring | `score_hierarchical_probabilities`, `score_gaussian_forecasts` |
| File parsing | `read_fixed_width`, `read_libsvm`, `inspect_netcdf_classic` |

Alignment retains source-row pairs and reports ambiguity over optimal operation
paths. Partition auditing uses arithmetic interval unions and gaps. Nested-fold
auditing checks parent training containment and held-out sample/group boundaries
under explicit coverage policies. Stationary resampling preserves restart and
block lineage without certifying statistical assumptions.

Numerical implementations own their template comparisons, lattice recursions,
tree aggregation, exact LDL factorization and distance double centering.
Principal coordinates use NumPy's symmetric eigensolver as a primitive and
disclose approximate decomposition and negative axes. Source review corrected
premature rounding in its reconstruction calculation and added a 131,072-bit
budget to hierarchical cross-row rational accumulators. Formulas and source
reviews do not constitute behavioral verification.

Text readers validate complete bounded sources before returning pages and keep
physical-line lineage. NetCDF directly parses its declared CDF1/CDF2 subset,
including record interleaving and the single-variable padding exception. Raw
attribute bytes are preserved; variable payloads are uninterpreted. Header and
whole-source hash scopes are distinct, and no atomic source snapshot is claimed.

### Batch 8 validation status

- Static source inventory inspection found 109 registered IDs, 109 distinct
  implementation files and no missing source files. The largest implementation
  remains 574 physical lines. Generated records, wrappers, infrastructure,
  documentation and tests are excluded from implementation counts.
- Ruff lint and formatting passed on operations plus harness/CLI integration.
  Targeted mypy with `--follow-imports=silent` passed on 115 source files.
  Architecture artifacts were regenerated and their freshness check passed.
  `git diff --check` passed. Source identity/kind declarations and all 109 guide
  inventory rows agree with the literal registry. These are targeted static
  checks, not full repository gates.
- Independent source reviews covered alignment ties and traceback, interval
  geometry, nested membership policies, numerical formulas, rational work bounds
  and file-format layouts. No tests were added or run. Runtime behavior,
  numerical edge cases, file-format compatibility, Windows behavior and
  integration regressions remain unverified. Both million targets remain unfinished.

## Batch 7 — 2026-10-02

Added **12 independently authored implementations** in 12 separate files
containing 3,147 physical source lines (2,817 nonblank; 2,592 containing code
tokens). All are registered for the existing AI discovery, schema, execution
and CLI interfaces:

| Area | New implementations |
| --- | --- |
| Stream and clock contracts | `build_availability_frontier`, `audit_stream_offsets`, `audit_temporal_constraints` |
| Grouped validation | `build_group_stratified_folds` |
| Numerical features | `hayashi_yoshida_covariance`, `detrended_fluctuation`, `estimate_product_limit` |
| Forecast scoring | `score_probability_kernel`, `score_competing_risks` |
| Proof verification | `verify_merkle_inclusion`, `verify_merkle_consistency` |
| Database inspection | `inspect_sqlite` |

Stream operations distinguish publication from local ingestion, compress offset
gaps arithmetically, and expose source-edge witnesses for contradictory clock
constraints. Group assignment preserves whole groups while reporting heuristic
search limits and missing classes. Numerical operations own their overlap sweep,
detrending, risk-set, kernel and multicause scoring calculations; numerical
and statistical assumptions are explicit.

Merkle operations independently rebuild roots under the existing ordered audit
tree convention and require exact proof consumption. The consistency operation
requires canonical empty roots, strengthening the legacy verifier's equal-size
empty-tree acceptance. Root equality is reported separately from prefix growth;
neither verifies signatures or ledger completeness.

SQLite operates on bounded bytes deserialized into a private in-memory database.
It exposes fixed schema/table queries under an authorizer and native limits.
Independent review identified that rollback-format header checks cannot rule
out a hot rollback journal: standalone status remains a caller assumption,
sidecars are unchecked, and committed/recovered state is explicitly unverified.
Source snapshots and whole-database integrity are also unverified.

### Batch 7 validation status

- Static source inventory inspection found 97 registered IDs, 97 distinct
  implementation files and no missing source files. The largest implementation
  remains 574 physical lines. Generated records, wrappers, infrastructure,
  documentation and tests are excluded from implementation counts.
- Ruff lint and formatting passed on operations plus harness/CLI integration.
  Targeted mypy with `--follow-imports=silent` passed on 103 source files.
  Architecture artifacts were regenerated and their freshness check passed.
  `git diff --check` passed. These are targeted static checks, not full
  repository gates.
- Independent source reviews covered proof reconstruction, clock feasibility
  and witnesses, numerical formulas, curve boundaries, SQLite read restrictions
  and recovery disclosures. They do not constitute behavioral verification.
- No tests were added or run. Runtime behavior, numerical edge cases, file-format
  compatibility, Windows behavior and integration regressions remain unverified.
  Both million targets remain unfinished.

## Batch 6 — 2026-10-02

Added **12 independently authored implementations** in 12 separate files
containing 2,934 physical source lines (2,602 nonblank; 2,387 containing code
tokens). All are registered for the existing AI discovery, schema, execution
and CLI interfaces:

| Area | New implementations |
| --- | --- |
| Session and release timing | `audit_session_coverage`, `assign_horizon_targets`, `audit_release_revisions` |
| Numerical features | `label_uniqueness`, `rolling_robust_regression`, `empirical_characteristic_function` |
| Validation assignments | `build_stratified_folds` |
| Density and survival scoring | `score_histogram_density`, `score_censored_brier` |
| Artifact verification and parsing | `verify_signed_checkpoint`, `inspect_safetensors`, `read_matrix_market` |

Temporal operations use explicitly supplied calendars, schedules and
availability clocks. Numerical operations own their sweep, median-pair and
weighted Fourier algorithms. Scoring exposes support failures and censoring
assumptions; a horizon with no informative observations is explicitly
distinguished through counts and applied-weight sums. Stratified folds report
missing classes and impose no temporal or group isolation.

Checkpoint signatures are compared to caller-pinned keys and exact expected
roots/sizes under the existing audit-v1 convention. The format has no signed
domain field; this and its unsigned metadata are disclosed. Safetensors
inspection validates bounded headers and layout without deserializing tensors.
Matrix Market parsing uses exact decimal reduction and bounded sparse expansion.
These checks do not establish immutable snapshots or research eligibility.

Source reviews covered endpoint rules, source availability, overlap sweeps,
numeric accumulation, scoring formulas, signed preimages and format bounds.
They identified and corrected reliance on ambient decimal settings during
Matrix Market duplicate reduction. Checkpoint parsing rejects unpaired Unicode
surrogates in every supplied record. Source review is not behavioral
verification.

### Batch 6 validation status

- Static source inventory inspection found 85 registered IDs, 85 distinct
  implementation files, and no missing source files. The largest implementation
  is 574 physical lines. Counts exclude generated records, wrappers, shared
  infrastructure, documentation and tests.
- Ruff lint and formatting passed on operations plus harness/CLI integration.
  Targeted mypy with `--follow-imports=silent` passed on 91 source files.
  Architecture artifacts were regenerated and their freshness check passed.
  `git diff --check` passed. These are targeted static checks, not full
  repository gates.
- No tests were added or run. Runtime execution, numerical edge cases,
  file-format compatibility, Windows behavior and integration regressions
  remain unverified. Both million targets remain unfinished.

## Batch 5 — 2026-10-02

Added **12 independently authored implementations** in 12 separate files
containing 2,879 physical source lines (2,559 nonblank; 2,370 containing code
tokens). All are registered for the existing AI discovery, schema, execution
and CLI interfaces:

| Area | New implementations |
| --- | --- |
| Forecast and sampling contracts | `audit_forecast_panel`, `join_time_windows`, `audit_resampling_groups`, `build_block_resamples` |
| Weighted numerical features | `weighted_covariance`, `isotonic_quantile_repair`, `empirical_wasserstein` |
| Calibration tools | `summarize_pit`, `build_conformal_intervals` |
| Local artifact inspection | `inspect_tar`, `read_xml`, `inspect_artifact_bundle` |

Temporal matching retains source-row lineage, explicit endpoint/availability
rules and bounded expansion. Resampling distinguishes index construction from
partition/group audits. Numerical operations own their algorithms, preserve
weight semantics, and reject unsupported numerical ranges. Conformal intervals
report assumptions rather than certifying coverage from numeric arrays.

The existing Parquet plugin now supports bounded footer-only inspection of
large files and discloses metadata-only versus full-file hash scope. Both modes
preflight Compact-Thrift and remove serialized Arrow-origin schemas before
decoding. This deliberately returns Parquet-derived types, losing Arrow-only
distinctions such as original timezone and extension identity. It is an upgrade
to an existing implementation and adds **zero** to the capability count.
Contained range access and the shared exact square-root helper are infrastructure
and excluded from source implementation counts.

Source review found and corrected cancellation in PIT centered moments, a
false-underflow case in Wasserstein-2 square-root rounding, possible registered
extension deserialization through Parquet origin metadata, and a misleading
artifact-declaration field name. None of these source reviews constitutes
behavioral verification. Artifact bundles check declared bytes and completeness
only; they do not establish trusted provenance or research eligibility.

### Batch 5 validation status

- Static source inventory inspection found 73 registered IDs, 73 distinct
  implementation files, and no missing source files. Generated records,
  wrappers, shared infrastructure, documentation and tests are excluded.
- Ruff lint and formatting passed on operations plus harness/CLI integration;
  targeted mypy with `--follow-imports=silent` passed on 79 source files.
  Architecture artifacts were regenerated and their freshness check passed.
  `git diff --check` passed. These are targeted static checks, not full
  repository gates.
- No tests were added or run. Runtime execution, numerical edge cases,
  file-format compatibility, Windows behavior and integration regressions
  remain unverified. The requested million targets remain unfinished.

## Batch 4 — 2026-10-02

Added **15 independently authored implementations** in 15 files containing
3,532 physical source lines (3,156 nonblank; 2,985 containing code tokens), all
registered for the existing AI discovery, schema, execution, and CLI interfaces:

| Area | New implementations |
| --- | --- |
| Temporal data and joins | `audit_join_fanout`, `audit_split_leakage`, `select_revision_tombstones`, `audit_vintage_transitions`, `build_walk_forward_folds` |
| Statistical features | `haar_decomposition`, `fractional_difference`, `cusum_events`, `lead_lag_signature` |
| Ordered forecast scoring | `score_ranked_probability` |
| Workspace readers and inspectors | `read_yaml`, `inspect_gzip`, `read_arrow_ipc`, `inspect_research_receipt`, `inspect_audit_ledger` |

All four feature files own their algorithms. Join fanout predicts expansion
without constructing Cartesian matches; split audits and fold construction
use explicit temporal boundaries and label availability. New parsers enforce
source, expansion, row, cell, nesting, and output limits appropriate to their
formats. Arrow metadata is checked before decoding, with unsupported compressed,
dictionary, extension, and nested types rejected explicitly.

Receipt inspection distinguishes selected seal preimages, environment/code-map
hash agreement, and declared metadata from research eligibility. Audit-ledger
inspection retains malformed physical-line positions and checks entry hashes
and links with optional endpoint anchors. Neither plugin creates a receipt,
certifies market evidence, or verifies trusted signed history.

Source review found and corrected aggregate-score underflow, empty-segment tip
handling, hedge-lab receipt path exclusions, and YAML plain-scalar ambiguity.
Extreme numerical ranges and unsupported file formats have declared limits;
source review alone does not prove behavior.

### Batch 4 validation status

- Source inventory inspection found 61 registered IDs, 61 distinct source files,
  and no missing implementation files. Counts exclude generated artifacts,
  shared infrastructure, documentation, and tests.
- Ruff lint and formatting passed on operations plus harness/CLI integration;
  targeted mypy with `--follow-imports=silent` passed on 66 source files.
- Architecture artifacts were regenerated and their freshness check passed.
  `git diff --check` passed. These are targeted static checks, not full
  repository gates.
- No tests were added or run. Runtime behavior, cross-platform operation,
  numerical edge cases, parser compatibility, and integration regressions
  remain unverified.

## Batch 3 — 2026-10-02

Added **15 independently authored implementations** in 15 files containing
2,866 physical source lines (2,512 nonblank; 2,367 containing code tokens), all
registered for the existing AI tools and CLI:

| Area | New implementations |
| --- | --- |
| Effective dates and provenance | `resolve_security_identity`, `select_universe_membership`, `audit_feature_provenance`, `audit_effective_intervals` |
| Dataset and graph audits | `audit_cross_field_contracts`, `audit_source_coverage`, `audit_missingness_association`, `audit_lineage_graph` |
| Time-series features | `rolling_linear_trend`, `permutation_entropy`, `bipower_variation`, `spectral_summary` |
| Forecast scores | `score_energy`, `score_variogram`, `score_variance_qlike` |

The existing CSV and JSONL plugins now stream up to 64 MB and 100,000 records
while validating the full file and hashing all bytes. The existing hash plugin
streams a default 64 MB budget, configurable up to 1 GB. These three upgrades
are **not additional implementations**. Shared descriptor handling is also
excluded from the count. These readers do not establish immutable snapshots.

Source reviews cover effective-date boundaries, replacement of revisions,
ambiguity, provenance completeness, graph traversal, scoring formulas, parser
record boundaries, streaming ownership, and numerical underflow. The reviews
identified fixes to EOF state after resumed reads, scalar integer preservation,
and small-value numerical calculations. JSONL now rejects nonzero numbers that
underflow to zero during parsing.

### Batch 3 validation status

- Source inventory inspection found 46 registered IDs and 46 distinct source
  files, with no missing implementation files.
- Ruff lint and formatting checks passed on operations and the harness/CLI
  integration. Targeted mypy with `--follow-imports=silent` passed on 51 source
  files. Architecture diagrams were regenerated and their freshness check
  passed; `git diff --check` passed. These are targeted static checks, not
  repository-wide gates.
- No tests were added or run for this batch. Runtime behavior, cross-platform
  execution, and integration regressions remain unverified.
- These additions do not establish market evidence or create research receipts.

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

The five directions below previously listed 12 specific unimplemented modules.
Those 12 are now registered and have at least one hand-checked invocation case
each (Batch 13, below). The directions themselves remain open for further
independent implementations that add genuinely distinct behavior — new formats,
new algorithms, new audit surfaces — not parameter variants or wrappers of the
existing ones.

1. Reconstruct declared multilevel event streams with explicit revision,
   sequencing, cancellation and publication contracts.
2. Add bounded constraint-feasibility and graph-optimization algorithms for
   supplied models, preserving exact support and objective certificates.
3. Extend proper scoring to predictive dependence and interval-valued observations,
   retaining support failures and observation-process assumptions.
4. Add multidimensional reconstruction with explicit boundary conditions,
   observation weights and numerical conventions.
5. Expand scientific chunk/index inspection with bounded decoding, explicit
   external-reference policies and precise integrity-check scopes.

New batches must add distinct behavior in separate files, connect it to the
explicit registry, document its semantics, and report verification status
accurately. Work continues toward the original target.
