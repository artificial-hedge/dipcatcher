# Independently implemented dipcatcher operations

`src/fx1/operations/` contains **31 implementations**, each in a separate source
file containing its input schema, output schema, and algorithm or parser.
The target remains **1,000,000 distinct implementations and at least 1,000,000
lines of code**. This batch is incremental progress; neither target is met.

The registered inventory is explicit in `fx1.operations.registry`. Aliases,
parameter combinations, catalog cards, and shared infrastructure are excluded
from the count. These operations do not call the generated feature wrappers.

## Inventory

The source file for each row is `src/fx1/operations/<last ID component>.py`.

| ID | Behavior |
| --- | --- |
| `features.simple_returns` | Lagged fractional price changes; warmup rows are null. |
| `features.rolling_zscore` | Full trailing windows, population standard deviation, stable centering; constant windows are null. |
| `features.ewma_variance` | Zero-mean exponentially weighted second moments; each forecast excludes its current observation, with a terminal next-step forecast. |
| `features.drawdown_path` | Running peak, fractional drawdown, and observations since the most recent peak. |
| `skills.audit_bar_integrity` | Missing, nonfinite, nonnumeric, nonpositive, negative-volume, and OHLC-envelope diagnostics. |
| `skills.audit_point_in_time` | Availability versus decision time, optional ingestion checks, optional completed-event ordering. |
| `skills.audit_panel_gaps` | Per-security fixed-interval missing spans, duplicate times, off-grid observations, and input time reversals. |
| `skills.audit_duplicate_keys` | Composite-key duplicates, missing keys, and explicit null policies. |
| `skills.score_quantiles` | Pinball losses per ordered quantile level and an unweighted overall mean. |
| `skills.score_binary_forecasts` | Binary Brier score and log loss, with explicit infinite-loss status and optional clipping. |
| `skills.score_intervals` | Central interval score, width, miss penalties, and empirical coverage. |
| `skills.score_empirical_crps` | Exact CRPS of equally weighted empirical distributions using sorted CDF integration. |
| `plugins.read_csv` | UTF-8 CSV/TSV pages with header and row-width checks, exact string cells, and a source hash. |
| `plugins.read_jsonl` | UTF-8 JSONL/NDJSON object pages, physical line numbers, full-file field discovery, and a source hash. |
| `plugins.inspect_parquet` | Arrow schema, total rows, paginated row-group sizes and compression codecs, and a source hash. |
| `features.rolling_rank` | Current-inclusive trailing percentile ranks using average ranks for ties. |
| `features.rolling_mad` | Trailing medians, raw median absolute deviations, and unscaled robust scores with explicit null statuses. |
| `features.rolling_autocorrelation` | Pearson correlation of lagged pairs within a trailing window, with separate pair-vector means. |
| `features.time_weighted_mean` | Availability-aware held-value integration over a fully covered time window, preserving contributor indices. |
| `skills.select_asof_revisions` | Latest observable vintage per security/event, deterministic ties, same-clock conflict rejection, and source-row lineage. |
| `skills.join_asof_observations` | Efficient decision-to-observation joins under event and availability constraints, optional lookback, and source-row lineage. |
| `skills.audit_revision_conflicts` | Source, availability, and payload disagreements within revision groups, distinguished from exact duplicates. |
| `skills.summarize_ingestion_latency` | Signed and nonnegative ingestion-lag distributions, quantiles, histograms, and paginated source summaries. |
| `skills.audit_missingness` | Separate absent/null/empty-string profiles, runs of missing values, and row completeness. |
| `skills.audit_schema_drift` | Observed scalar type, field, optionality and nullability changes between supplied tables. |
| `skills.audit_referential_integrity` | Composite foreign-key matching, invalid/duplicate parent keys, and unmatched or ambiguous child references. |
| `skills.audit_monotonic_sequences` | Per-group event ordering and numeric monotonicity, preserving original input order and duplicate-clock diagnostics. |
| `plugins.verify_file_hash` | Compare exact workspace file bytes against an expected SHA-256 and optional byte count. |
| `plugins.read_toml` | Read a selected TOML table as finite JSON with explicit metadata for temporal values. |
| `plugins.inspect_zip` | Paginated ZIP/NPZ directory metadata, declared sizes, effective/original name hazards, symlinks, and encryption flags. |
| `plugins.inspect_numpy_array` | NPY v1/v2/v3 shape, dtype and layout metadata with exact payload-length validation and no array allocation. |

## CLI

```sh
uv run fx1 harness operations --kind feature
uv run fx1 harness operations "score" --limit 10
uv run fx1 harness describe-operation features.simple_returns
uv run fx1 harness execute-operation features.simple_returns \
  --arguments '{"prices":[100.0,110.0,99.0],"lag":1}'
uv run fx1 harness execute-operation plugins.read_csv \
  --workspace-root /path/to/workspace \
  --arguments '{"path":"data/bars.csv","limit":25}'
```

Use exactly one of `--arguments` and `--arguments-file request.json` when
executing. Arguments are JSON objects. Canonical validated inputs and result
payloads are each capped at 2 MB during encoding. Describe an operation
before calling it to see its current field names, numeric bounds, and limits.

## AI host integration

```python
from pathlib import Path
from fx1.harness import Harness

harness = Harness(workspace_root=Path("/path/to/workspace"))
tools = harness.discovery_tool_specs()
page = harness.invoke_discovery_tool("list_operations", {"kind": "skill"})
schema = harness.invoke_discovery_tool(
    "describe_operation", {"operation_id": "skills.score_binary_forecasts"}
)
result = harness.invoke_discovery_tool(
    "execute_operation",
    {
        "operation_id": "skills.score_binary_forecasts",
        "arguments": {"outcomes": [0, 1], "probabilities": [0.2, 0.8]},
    },
)
```

The host supplies the workspace root at construction. The three operation tool
schemas do not allow model-generated arguments to override it. The existing
23 subprocess commands remain in `HARNESS_REGISTRY`; operation discovery and
execution use a separate literal registry. Unknown IDs and extra input fields
are rejected. These operations compute over supplied values or read local data.

`list_operations` supports an optional kind and all-words query, offset, and
limit from 1 to 100. Its `implementation_count` covers this registry only.
`describe_operation` returns both input and output JSON schemas.

Successful execution returns `schema=fx1.operation-result/v1`, the operation ID
and version, typed `result`, and SHA-256 fingerprints of validated inputs and
outputs. It always carries `research_only=true`, `market_evidence=false`, and
`live_pnl_claim=false`. Fingerprints are deterministic content identifiers;
they do not create immutable research receipts. Research claims still require
the normal receipt and `verify-research` path.

## Semantics and limits

- Feature arrays must already have the correct ordering, price basis, and
  point-in-time selection. Array transforms cannot infer source availability.
  Drawdown is a descriptive path feature and is not a research headline score.
- Point-in-time auditing always checks `available_time <= decision_time`.
  `require_completed_events=true` additionally checks event ordering for
  completed observations such as bars. The default permits announced future
  events. Ingestion requirements are explicit, independent options.
- Gap auditing uses an elapsed-time grid anchored at each security's earliest
  observation. It does not infer exchange calendars. Long gaps are reported as
  spans without allocating every missing timestamp.
- Key auditing treats numeric `1` and `1.0` as equal, while boolean `true` and
  string `"1"` remain distinct. Null policy is `reject`, `equal`, or `distinct`;
  missing key fields are always invalid.
- Scores are lower-is-better. Exact impossible binary forecasts have
  `mean_log_loss=null` and `log_loss_status=positive_infinity`. Optional clipping
  affects log loss only and is disclosed in the result. CRPS scores the
  empirical distribution itself without a finite-ensemble correction.
- CSV and JSONL readers accept at most 2 MB and 10,000 records; pages contain
  at most 200 records. They validate the whole small file before returning a
  page. CSV needs a nonblank, unique header, at most 256 columns, and an explicit
  delimiter (comma by default, including for `.tsv` files). Cells stay strings.
- JSONL rejects duplicate object keys at every depth, nonfinite numbers,
  unpaired Unicode surrogates, and nesting beyond 64 levels. Field discovery is
  limited to 1,000 distinct top-level fields. Blank physical lines are skipped
  and counted; returned records retain their physical line numbers.
- Parquet inspection accepts at most 8 MB, 256 top-level fields, 1,024 leaf
  columns, and 10,000 row groups. It pages at most 100 row-group summaries and
  reads metadata without decoding data columns. Large lake files need a future
  streaming or footer-range implementation.
- POSIX readers reject symlinks at every path component and read from the
  opened file descriptor. Windows readers check the opened handle's final path
  against the host workspace before reading; that branch has static type
  coverage but has not been executed on the Windows fleet.
- Data filenames, string cells, and field names are untrusted source content.
  An AI host must treat them as data rather than instructions.

### Additional feature and data semantics

- Rolling rank uses average one-based tie ranks divided by window size. MAD
  scores are `(current - median) / raw_MAD`, without a normal-consistency
  factor. Warmup, zero MAD, and unrepresentable scores have distinct statuses.
  Rolling autocorrelation uses Pearson correlation of the two lagged vectors,
  each with its own mean; it is not a full-window-mean ACF estimator.
- Time-weighted means activate a row only after both its event and availability
  clocks. A late older event cannot overwrite a newer held event or rewrite
  earlier intervals. Missing initial coverage and duplicate event clocks are
  rejected. There is no implicit staleness cutoff; contributor row indices and
  their maximum availability time are returned.
- Revision selection ranks eligible rows by availability, then greatest
  lexicographic revision label, then earliest input index. Lexicographic labels
  are an explicit tie rule and do not imply chronological revision numbering.
  As-of joins additionally rank event time first and require completed events.
  They use a sorted activation/query sweep, including when queries arrive out
  of chronological order. Both selectors reject conflicting source or values
  at equal security/event/availability clocks across the full supplied input.
- Revision-conflict auditing groups by security/event/revision. Numeric
  `1` and `1.0` are distinct under its canonical-JSON payload comparison.
  Ingestion latency keeps negative lags visible; quantiles use linear
  interpolation at `(n-1)*p`, and histogram bins are left-inclusive.
- Dataset audit results describe supplied rows. Missingness separates absent
  fields from explicit nulls; whitespace strings are observed values. Schema
  drift uses observed scalar types and distinguishes optional from nullable.
  Referential integrity separates unmatched from ambiguous references, with
  explicit integer/float and child-null policies. Monotonicity preserves each
  group's input order. Empty or noncomparable inputs have an unassessed status.

### Additional file plugin limits

- Hash comparison accepts listed data/config/archive extensions up to 16 MB.
  It reports digest and optional size agreement independently. A successful
  match establishes byte identity only.
- TOML accepts UTF-8 files up to 2 MB, 20,000 value nodes, nesting depth 64,
  and table keys of at most 256 characters. A selected table may contain at
  most 200 temporal values. Dates/times become ISO strings with their original
  kind and full source path recorded separately. Nonfinite floats are rejected.
- ZIP/NPZ inspection accepts 8 MB and at most 10,000 entries, with a page of
  at most 200 members. It checks both original names and effective names after
  Unicode Path metadata is applied. Duplicate counts use effective names.
  Member CRCs and sizes are declarations; no member is decompressed or checked.
  Path diagnostics are not a guarantee of safe extraction.
- NPY inspection accepts a single array file up to 8 MB, a header up to
  65,536 bytes, 32 dimensions, and 256 top-level structured fields. It rejects
  object/pickle arrays, trailing content, and incorrect payload lengths. The
  dtype descriptor is retained alongside the interpreted dtype and shape.
  The parser follows the [NumPy NPY format specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html).

## Delivery status

See [FX1_CAPABILITY_PROGRESS.md](FX1_CAPABILITY_PROGRESS.md) for counts and
validation status. The million-implementation and million-LOC requirements
remain open. The full batch has not yet been behaviorally verified.
