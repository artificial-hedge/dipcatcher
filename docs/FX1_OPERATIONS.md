# Independently implemented dipcatcher operations

`src/fx1/operations/` contains **169 implementations**, each in a separate source
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
| `plugins.inspect_parquet` | Parquet-derived schema, declared rows and row-group sizes/codecs, with full-file or bounded footer-only hashing. |
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
| `skills.resolve_security_identity` | Resolve ticker/exchange mappings using observable revisions and effective intervals, with explicit ambiguity and source-row lineage. |
| `skills.select_universe_membership` | Select historical inclusion/exclusion intervals at a decision clock, preserving revisions and missing/ambiguous states. |
| `skills.audit_feature_provenance` | Check feature dependencies, field versions, availability claims, duplicate sources, and missing provenance. |
| `skills.audit_effective_intervals` | Invalid intervals, exact overlap-pair counts, bounded overlap witnesses, and internal coverage gaps. |
| `skills.audit_cross_field_contracts` | Declarative scalar field comparisons with explicit numeric-type, missing, and null policies. |
| `skills.audit_source_coverage` | Expected source/security coverage at a decision clock, selected-row lineage, and optional staleness. |
| `skills.audit_missingness_association` | Exact missingness contingency tables, Jaccard similarity, phi, and conditional missingness rates. |
| `skills.audit_lineage_graph` | Duplicate declarations, unresolved dependencies, cyclic components, and topological ordering of valid graphs. |
| `features.rolling_linear_trend` | Trailing centered-index OLS slopes, midpoint intercepts, residual scale, and constant-window statuses. |
| `features.permutation_entropy` | Delayed ordinal-pattern frequencies and entropy with explicit tie policies and bounded pattern output. |
| `features.bipower_variation` | Adjacent absolute-return products, realized variation, and an explicit finite-sample correction option. |
| `features.spectral_summary` | Boxcar one-sided FFT density, positive-frequency peak and entropy, and paginated frequency bins. |
| `skills.score_energy` | Euclidean energy score of multivariate empirical ensemble forecasts, without a finite-ensemble correction. |
| `skills.score_variogram` | Weighted variogram score using unordered coordinate pairs and explicit exponent and weights. |
| `skills.score_variance_qlike` | Normalized QLIKE loss for strictly positive realized and predicted variance, without flooring. |
| `skills.audit_join_fanout` | Predict composite-key inner/left join cardinalities, unmatched rows, and many-to-many expansion without materializing matches. |
| `skills.audit_split_leakage` | Cross-split sample IDs and information-window overlaps, with explicit group boundaries, purge, and embargo guards. |
| `skills.select_revision_tombstones` | Select observable upsert/delete states, including resurrection, same-clock ambiguity, and row lineage. |
| `skills.audit_vintage_transitions` | Compare before/after snapshots for typed field changes, availability regressions, and revision reuse conflicts. |
| `features.haar_decomposition` | Orthonormal Haar coefficients, optional zero padding, energy accounting, and reconstruction diagnostics. |
| `features.fractional_difference` | Fixed-width binomial lag filter with explicit warmup, truncation, coefficients, and DC gain. |
| `features.cusum_events` | Two-sided cumulative-sum event detection using supplied target, drift, threshold, and reset policy. |
| `features.lead_lag_signature` | First- and second-order path signatures for an explicitly ordered lead-lag lift, with optional numeric time augmentation. |
| `plugins.read_yaml` | Bounded single-document YAML to finite JSON with duplicate-key and scalar-ambiguity checks. |
| `plugins.inspect_gzip` | Validate all gzip members, CRC/trailers, expansion limits, and compressed/decompressed byte hashes. |
| `plugins.read_arrow_ipc` | Bounded flat Arrow IPC file/stream pages with raw metadata preflight and explicit scalar encodings. |
| `skills.score_ranked_probability` | Proper ordered-category forecast loss, per-boundary contributions, and disclosed probability-mass normalization. |
| `skills.build_walk_forward_folds` | Scheduled chronological train/test indices with label availability, label-end purging, and a pre-test gap. |
| `plugins.inspect_research_receipt` | Inspect receipt declarations, selected hash preimages, environment/code-map fingerprints, and JSON Pointer values. |
| `plugins.inspect_audit_ledger` | Stream v1 entry-chain diagnostics with physical-line indexing, hash checks, pagination, and optional end anchors. |
| `skills.audit_forecast_panel` | Align model-specific forecasts to completed outcomes, with duplicate, coverage, target-clock and availability diagnostics. |
| `skills.join_time_windows` | Construct grouped interval-to-event matches with explicit endpoints, availability filtering, source indices and expansion limits. |
| `skills.audit_resampling_groups` | Audit supplied resample memberships for duplicate inclusion and prohibited sample/group overlap between declared partitions. |
| `features.weighted_covariance` | Exact centered moment accumulation for weighted means and population or reliability-corrected covariance. |
| `features.isotonic_quantile_repair` | Weighted adjacent-pool projection of crossed quantiles, retaining adjustment and rounding diagnostics. |
| `features.empirical_wasserstein` | Weighted one-dimensional empirical transport distances from sorted mass matching. |
| `skills.build_block_resamples` | Seeded circular moving-block index samples with explicit block and replicate boundaries. |
| `plugins.inspect_tar` | Strict USTAR/V7 archive scanning, checksums, member hashes and path observations without extraction. |
| `plugins.read_xml` | Bounded XML element pages with namespace, text/tail and structural lineage; rejects DTD/entity declarations. |
| `plugins.inspect_artifact_bundle` | Check a strict JSON manifest against bounded local artifact sizes and hashes, with separate completeness observations. |
| `skills.summarize_pit` | Uniform-reference PIT histograms, empirical-CDF discrepancies, endpoint counts and ordered lag correlations. |
| `skills.build_conformal_intervals` | Absolute-residual split-conformal intervals with finite-sample rank selection and explicit all-real outcomes. |
| `skills.audit_session_coverage` | Audit completed bars against observable supplied session segments, with partial-close policies and compressed missing spans. |
| `skills.assign_horizon_targets` | Map decision clocks to supplied session-close offsets under explicit calendar availability and coverage bounds. |
| `skills.audit_release_revisions` | Compare visible source publications to supplied release schedules, distinguishing initial timing, coverage and revision conflicts. |
| `features.label_uniqueness` | Integrate inverse concurrent-label counts over explicit per-security intervals using exact duration arithmetic. |
| `features.rolling_robust_regression` | Trailing Theil–Sen median pair slopes and joint median intercepts with explicit warmup and equal-predictor statuses. |
| `features.empirical_characteristic_function` | Weighted empirical complex Fourier averages at supplied angular frequencies with bounded phases and exact accumulation. |
| `skills.build_stratified_folds` | Seeded class-balanced validation assignments with stable sample identity ordering and explicit small-class policies. |
| `plugins.verify_signed_checkpoint` | Verify audit-v1 Ed25519 checkpoints against supplied key identities and exact root/size anchors. |
| `plugins.inspect_safetensors` | Check bounded tensor headers, dimensions and payload layout without allocating tensors; distinguish header and full-file hashes. |
| `plugins.read_matrix_market` | Parse coordinate matrices with exact decimal duplicate reduction, sparse symmetry expansion and paginated entries. |
| `skills.score_histogram_density` | Logarithmic density loss for piecewise-uniform forecasts with explicit support, normalization and observation weights. |
| `skills.score_censored_brier` | IPCW survival Brier scores from supplied censoring curves, with explicit event ties, support and horizon integration. |
| `skills.build_availability_frontier` | Sweep publication and ingestion readiness to report completed-event frontiers, staleness and supplied-record backlogs. |
| `skills.audit_stream_offsets` | Compare supplied stream offsets to inclusive expected ranges, with compressed gaps, duplicates, hash conflicts and ordering checks. |
| `skills.audit_temporal_constraints` | Solve integer min/max clock differences with optional timestamp pins and source-edge witnesses for contradictory cycles. |
| `skills.build_group_stratified_folds` | Assign whole groups using exact class/sample balance objectives and bounded improving local moves. |
| `features.hayashi_yoshida_covariance` | Sum exact increment products over overlapping asynchronous observation intervals without interpolation. |
| `features.detrended_fluctuation` | Compute DFA1 scale fluctuations from integrated demeaned profiles and local linear residuals, with a descriptive scaling fit. |
| `skills.verify_merkle_inclusion` | Rebuild a pinned audit Merkle root from an indexed leaf and exactly consumed ordered sibling path. |
| `skills.verify_merkle_consistency` | Independently rebuild two pinned audit roots from a minimal prefix proof, with explicit empty/equal-size rules. |
| `plugins.inspect_sqlite` | Inspect bounded SQLite main-file images in memory using fixed schema/table queries and explicit scalar encodings. |
| `features.estimate_product_limit` | Estimate event or censoring survival curves with tie-aware risk sets, supported-time evaluations and restricted-mean integration. |
| `skills.score_probability_kernel` | Evaluate weighted empirical forecasts with a fixed Gaussian kernel, diagonal-inclusive pair masses and numerical diagnostics. |
| `skills.score_competing_risks` | Score terminal-cause cumulative incidence and event-free probabilities with IPCW multicategory Brier curves. |
| `skills.align_event_sequences` | Globally align ordered streams using explicit key/time eligibility and integer costs, retaining source pairs and optimal-path ambiguity. |
| `skills.audit_partition_ranges` | Audit half-open integer partition coverage with exact union sizes, compressed gaps, overlap witnesses and reused partition IDs. |
| `skills.audit_nested_folds` | Check supplied nested memberships for parent containment, held-out sample/group leakage, duplicates and explicit coverage policies. |
| `skills.build_stationary_resamples` | Construct seeded circular stationary-bootstrap indices with exact restart probabilities and observed block boundaries. |
| `features.sample_entropy` | Count matched and forward-matched templates using exact Chebyshev comparisons and explicit zero-match statuses. |
| `features.burg_autoregression` | Fit a bounded Burg lattice with explicit centering, reflection coefficients, prediction energies and stopping conditions. |
| `plugins.read_fixed_width` | Parse exact-width UTF-8 records through declared code-point spans, retaining exact cells and physical line/byte lineage. |
| `plugins.read_libsvm` | Read a bounded sparse LIBSVM subset with original numeric tokens, explicit zeros and independent row/coordinate pages. |
| `plugins.inspect_netcdf_classic` | Parse CDF1/CDF2 metadata and check fixed/record extents without allocating or decoding variable arrays. |
| `skills.score_hierarchical_probabilities` | Aggregate leaf probability masses through a supplied class tree and compute positive weighted binary Brier losses by node. |
| `skills.score_gaussian_forecasts` | Score multivariate Gaussian densities with exact positive-definiteness, determinant and Mahalanobis calculations before logarithms. |
| `features.principal_coordinates` | Embed symmetric dissimilarities using exact double centering and a numerical eigensolver, with negative-axis and reconstruction diagnostics. |
| `skills.audit_source_chains` | Audit declared source-separated sequence/hash-predecessor graphs, anchors, forks, conflicts and cycles, with optional raw payload hashes. |
| `skills.audit_temporal_resamples` | Check supplied resampling memberships against decision origins, availability clocks and explicit block-order policies. |
| `skills.audit_schema_compatibility` | Check directional flat writer-to-reader compatibility over declared type, presence, null, decimal, timestamp and physical-unit domains. |
| `features.circular_summary` | Summarize weighted circular direction, stable pairwise dispersion and the shortest arc containing positive-weight support. |
| `features.weighted_geometric_median` | Fit a weighted Euclidean geometric median through bounded modified Weiszfeld iterations with coincidence and convergence diagnostics. |
| `features.partial_autocorrelation` | Compute biased autocovariances and exact bounded Levinson–Durbin PACF/AR stages with explicit centering and degeneracy. |
| `plugins.read_arff` | Parse bounded numeric/nominal/string ARFF records with explicit sparse defaults, exact cells and source-line lineage. |
| `plugins.read_dbase` | Read a strict single-file dBASE III table subset with declared text encoding, deletion markers and exact cell encodings. |
| `plugins.read_sparse_npz` | Validate bounded CSR/CSC/COO NumPy archives and return sparse entries without dense allocation or object loading. |
| `features.weighted_quantile_binning` | Fit tie-preserving bins from an exact weighted empirical CDF and encode separate queries with fixed cuts. |
| `skills.score_joint_categorical` | Score joint categorical masses and their marginals, preserving zero-support losses and observed dependence diagnostics. |
| `skills.score_count_forecasts` | Score supplied Poisson, binomial and negative-binomial count laws in one implementation with explicit impossible-outcome losses. |
| `skills.audit_price_basis` | Reconcile declared price transformations, factor metadata and dependency availability with exact decimal arithmetic. |
| `skills.reconcile_aggregates` | Reconcile supplied detail memberships with weighted sums, means or counts under explicit duplicate, null and completeness policies. |
| `skills.audit_unit_conversions` | Compose declared affine unit maps exactly, checking dimensions and retaining contradictory-cycle witnesses. |
| `features.distance_correlation` | Compute biased multivariate distance covariance and correlation with exact double centering of rounded Euclidean distances. |
| `features.kernel_density_grid` | Evaluate a Gaussian kernel mixture at supplied queries with explicit bandwidth, weights and stable density/tail logarithms. |
| `features.lomb_scargle_periodogram` | Fit irregular-time harmonic spectra on supplied frequencies with mean-model choices, rank checks and returned-coefficient diagnostics. |
| `plugins.read_cbor` | Parse a bounded CBOR subset into typed preorder nodes with exact scalar encodings, source spans and complete-source validation. |
| `plugins.read_messagepack` | Decode bounded MessagePack values, exact timestamps and optional opaque application extensions with source lineage. |
| `plugins.read_bson` | Parse one bounded BSON document with ordered fields, exact scalars and nested document-length checks. |
| `features.empirical_copula` | Construct weighted marginal pseudo-observations under explicit tie rules and query their empirical joint CDF. |
| `features.compositional_logratios` | Compute positive composition closure, CLR/Helmert balances, geometric centers and pairwise log-ratio variation. |
| `skills.score_piecewise_hazards` | Score supplied piecewise hazards through exact-event, right-censored and interval-censored likelihoods with entry conditioning. |
| `skills.audit_transaction_batches` | Audit explicit batch state transitions, membership/hash labels, commit-prefix checkpoints and durability-claim references. |
| `skills.audit_observation_windows` | Check continuous exposure coverage, delayed entry, finalized availability and terminal-event/right-censoring declarations. |
| `skills.audit_dependency_releases` | Audit stage-level version dependencies, availability and permanent declared supersession with decision-time queries. |
| `features.segment_mean_changes` | Minimize exact penalized segment squared errors with minimum lengths, source boundaries and deterministic ties. |
| `features.dynamic_time_warping` | Align multivariate sequences through exact squared-cost dynamic programming with explicit bands, steps and source paths. |
| `features.singular_spectrum_analysis` | Embed Hankel trajectories and reconstruct caller-selected SVD component groups by diagonal averaging. |
| `plugins.read_avro_container` | Parse flat primitive Avro object containers with bounded DEFLATE, schema/block validation and exact cell lineage. |
| `plugins.inspect_hdf5_superblock` | Inspect checksummed HDF5 v2/v3 superblocks and declared address bounds while leaving object graphs unverified. |
| `plugins.read_fits_table` | Parse bounded scalar FITS ASCII/binary tables with exact values, raw scaling/null metadata and byte offsets. |
| `features.poisson_binomial_distribution` | Convolve unequal independent Bernoulli probabilities into exact count masses and exact-CDF quantiles. |
| `skills.score_markov_paths` | Score partial or full state paths through exact Markov forward conditioning and explicit zero-support propagation. |
| `skills.score_gaussian_mixtures` | Compute log density, CRPS, PIT and component responsibilities for supplied heteroscedastic Gaussian mixtures. |
| `skills.audit_source_migrations` | Audit partition ownership and cutover coverage, routing visible event/key queries to exactly the prescribed source. |
| `skills.audit_artifact_assembly` | Check declared byte-part extents, ordinal concatenation, target coverage, hash labels and publication clocks. |
| `skills.audit_snapshot_manifests` | Reconcile complete snapshot inventories through parent deltas, retained tombstones and immutable version declarations. |
| `features.markov_absorption` | Resolve finite Markov closed classes, class-entry probabilities, transient occupation and conditional hitting times. |
| `features.linear_gaussian_filter` | Filter supplied time-varying Gaussian state models with missing coordinates and optional fixed-interval RTS smoothing. |
| `features.graph_effective_resistance` | Solve grounded conductance Laplacians for exact pair resistance, voltage previews and weighted spanning-tree mass. |
| `plugins.read_wav_pcm` | Parse bounded PCM/IEEE RIFF/WAVE samples with exact values, frame/channel lineage and full framing checks. |
| `plugins.read_ply` | Parse ASCII/binary PLY scalar/list records with exact values, byte spans and declared face-index checks. |
| `plugins.inspect_tiff_directory` | Inspect classic TIFF directory graphs, primitive metadata and declared strip/tile extents without decoding images. |
| `features.convex_hull_2d` | Construct an exact-predicate planar hull with vertex lineage, area/centroid and boundary-aware point queries. |
| `features.monotone_cubic_interpolation` | Reconstruct a shape-preserving Hermite curve with exact rational derivatives and signed integrals. |
| `features.discrete_optimal_transport` | Solve balanced discrete transport with exact residual-network flows and primal/dual certificates. |
| `features.linear_constraint_feasibility` | Fourier–Motzkin elimination over exact rationals; returns an independently checked witness or a normalized Farkas certificate. |
| `features.maximum_flow` | Exact directed maximum flow with per-source-edge lineage, conservation checks and an equal-capacity residual minimum cut. |
| `features.minimum_spanning_forest` | Stable Kruskal forest with exact weight totals, source-edge classifications and verified optimal exchanges/uniqueness. |
| `features.multilinear_grid_interpolation` | Exact corner weights and gradients on a supplied rectilinear grid; oriented box integration by separable hat functions with explicit clipping and knot conventions. |
| `features.set_function_attribution` | Exact subset inversion of a complete supplied coalition table into Shapley/Banzhaf attributions, pair interactions and monotonicity diagnostics; no causal or model-evaluation claim. |
| `features.thin_plate_spline` | Bounded planar thin-plate radial fit with an affine tail and supplied smoothing; exposes returned-coefficient residuals, query gradients and approximate-kernel limits. |
| `plugins.inspect_zarr_metadata` | Strict Zarr v2 primitive array metadata, exact fill declarations and bounded chunk geometry; codecs and chunk data are never loaded. |
| `plugins.read_nrrd` | Bounded inline raw/ASCII NRRD scalar rasters with explicit axis order, retained metadata declarations and exact paged source spans. |
| `plugins.read_stl` | Bounded explicit-format STL triangles with exact scalars, raw spans and limited exact geometry observations. |
| `skills.audit_resource_reservations` | Audit visible capacity trees, declared ownership, reservation and consumption subwindows and exact hierarchical overcommit without time-grid expansion. |
| `skills.audit_task_leases` | Audit declared per-task lease transitions, strict fencing epochs and historical worker access claims without asserting external enforcement. |
| `skills.reconstruct_order_book` | Replay an explicit integer order log into bounded FIFO/price snapshots, stopping at the first visible sequence or transition fault. |

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
- CSV and JSONL readers stream at most 64 MB and 100,000 records; pages contain
  at most 200 records. They parse the complete file before returning a page,
  hash every source byte, and reject physical lines over 1,000,000 characters.
  CSV needs a nonblank, unique header, at most 256 columns, and an explicit
  delimiter (comma by default, including for `.tsv` files). Cells stay strings.
  It also honors the host Python process's `csv.field_size_limit()`; the
  physical-line limit does not override that field limit. A CSV page is capped
  at 1,000,000 cell/key characters; JSONL pages at 1,000,000 source characters.
  Canonical result encoding may impose a tighter bound; reduce the page limit.
- JSONL rejects duplicate object keys at every depth, nonfinite numbers,
  nonzero numbers that underflow to zero, unpaired Unicode surrogates, and
  nesting beyond 64 levels. Field discovery is
  limited to 1,000 distinct top-level fields. Blank physical lines are skipped
  and counted; returned records retain their physical line numbers. LF separates
  records; Unicode line separators inside strings remain part of those strings.
  Decimal numbers use binary floating-point precision.
- Parquet inspection defaults to `read_mode=full`, which reads and hashes at
  most 8 MB. `read_mode=footer` accepts files up to 1 TB and reads only the
  4-byte header, 8-byte trailer, and serialized footer (default 2 MB, configurable
  up to 8 MB). The footer path returns `source_sha256=null` and a separate
  `footer_sha256` over serialized file metadata only, with its offset and length.
  It checks descriptor metadata for changes during those reads; this does not
  establish an immutable snapshot. Both modes require plaintext `PAR1` markers,
  cap schemas at 256 top-level fields/1,024 leaves and metadata at 10,000 row
  groups, and page at most 100 row-group summaries. Declared row counts and
  sizes are metadata observations; column pages, checksums, and external
  column-file references are not validated. The metadata-only envelope follows
  the [Parquet file layout](https://parquet.apache.org/docs/file-format/).
  Compact-Thrift preflight limits value visits, collection sizes, strings and
  nesting before parsing. It removes top-level `ARROW:schema` metadata before
  PyArrow can restore registered extension types; both modes report whether it
  was removed and that no serialized Arrow-origin schema was restored. Returned
  types derive from Parquet, so Arrow-only timezone, extension and certain list
  distinctions are lost. Full-file and footer hashes still cover original bytes.
  Encrypted/signed metadata and duplicate metadata keys are rejected.
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

- Hash comparison streams listed data/config/archive extensions with a default
  64 MB budget; an explicit `max_bytes` can raise the budget to 1,000,000,000 bytes.
  It reports digest and optional size agreement independently. A successful
  match establishes agreement with the bytes read. Readers do not lock files or
  establish immutable snapshots; callers must supply stable artifacts.
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

### Effective dates, provenance, and structural audits

- Identity and membership selection first choose the latest observable revision
  of a stable mapping/membership ID, then apply its half-open effective interval
  `[valid_from, valid_to)`. Revisions can replace an interval or identity.
  Equal-clock conflicting records produce ambiguity rather than an invented
  tie winner. Compatible ties use lexicographic revision labels and input index.
- Membership resolves overlapping active records by latest effective start.
  An expired temporary exclusion can expose an older still-active inclusion.
  Missing and ambiguous results remain explicit. No current-universe list is
  substituted for historical membership.
- Provenance auditing checks supplied dependency references and field versions.
  Unresolved or duplicate source IDs prevent a complete availability bound.
  Matching declared provenance does not establish source truth or receipt
  eligibility. Effective-interval auditing treats missing endpoints as unbounded,
  counts every overlap pair, and returns bounded witnesses and internal gaps.
- Cross-field contracts allow six scalar comparisons and no executable
  expressions. Missing/null policies are explicit; skipped comparisons are
  unassessed. Ordering requires numeric operands, while equality is type-aware.
  Source coverage requires both event and availability times at or before the
  decision clock and uses caller-declared expected pairs and staleness limits.
- Missingness association describes binary missingness indicators. Constant
  indicators have undefined phi; empty unions have undefined Jaccard similarity.
  These are descriptive statistics with no causal or predictive claim.
- Lineage edges point from dependency to dependent. Graph auditing reports
  strongly connected cyclic components, not every possible cycle. It separates
  cycle members from downstream blocked nodes and returns a topological order
  only for a nonempty graph without structural errors. It does not verify hashes
  or research receipts.

### Additional numerical conventions

- Rolling trend regresses on equally spaced sample indices. The intercept is
  at the window midpoint; residual scale uses `sqrt(RSS / (window - 2))`.
  Constant windows have undefined R². Fitting uses power-of-two normalization;
  dimensionless R² is retained even when dimensional outputs round to zero.
  Callers supply ordered, eligible values.
- Permutation entropy uses all valid delayed starts, order 2–7, and explicit
  stable/drop/reject tie handling. Normalization is by `log(order!)`. Its minimum
  pattern-count status is a caller threshold, not a statistical adequacy claim.
  See the [ordinal-pattern entropy reference](https://link.aps.org/accepted/10.1103/PhysRevE.91.023101).
- Bipower variation is `pi/2 * sum(abs(r[t]) * abs(r[t-1]))`. The default applies
  no finite-sample correction; `n_over_n_minus_1` is an explicit option. A single
  return has no adjacent pair and produces null bipower variation. Values are
  descriptive features, with no market-evidence or jump-significance claim.
  See [Barndorff-Nielsen and Shephard](https://public.econ.duke.edu/~get/browse/courses/883/Spr16/COURSE-MATERIALS/Z_Papers/BNSJFEC2004.pdf).
- Spectral density uses a boxcar window with no padding, optional mean removal,
  and one-sided power. Interior bins are doubled; DC and an even-length Nyquist
  bin are unchanged. Peak and entropy exclude DC. Callers guarantee even
  sampling; constant inputs have no positive-frequency summary. Coefficient
  powers retain binary exponents until reported values are formed, with explicit
  density-bin, total-power, and entropy-probability underflow diagnostics.
  These flags cannot detect information lost inside the float64 FFT itself.
  These follow
  [periodogram conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html).
- Energy scoring uses Euclidean distance: `mean(||X-y||) - mean(||X-X'||)/2`,
  including diagonal ensemble pairs. Coordinates are not rescaled. Variogram
  scoring sums `weight * (abs(y_i-y_j)^p - mean(abs(X_i-X_j)^p))²` over unordered
  pairs once. This is half the symmetric double-sum convention with equal
  weights; weights are not normalized and must be chosen independently of the
  verification outcome. Both score the empirical forecast distribution without
  a finite-ensemble correction. See [Scheuerer and Hamill](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf).
- QLIKE uses `realized / forecast - log(realized / forecast) - 1`, where both
  inputs are strictly positive variances in the same units. It uses a series
  near equality to reduce cancellation and does not floor zero targets. See
  equation 24 in [Patton's robust volatility evaluation paper](https://public.econ.duke.edu/~ap172/Patton_robust_JoE_forthcoming.pdf).

### Join, split, and revision semantics

- Join fanout uses key multiplicities to predict inner and left joins without
  allocating Cartesian matches. Numeric equality and null handling are explicit.
  Invalid keys withhold full-table predictions; valid-subset counts remain
  separately named. Relationship constraints apply only to matched key groups.
- Split auditing treats observation and label intervals as separate half-open
  ranges. Interval comparisons stay within `group_id`; sample IDs are global.
  Optional purge and embargo spans surround validation/test information extents
  and are checked against training intervals. Pair work is bounded independently
  of the number of reported findings.
- Tombstone selection requires both revision and availability clocks at or
  before the decision. Latest revision time wins, followed by availability.
  A later upsert can resurrect a deleted key; no eligible event means missing.
  Contradictory winning-clock rows produce ambiguity or an explicit error.
  Snapshot transition auditing reports absence as removal from that snapshot,
  with no inferred deletion event. Scalar field comparison is type-sensitive.
- Walk-forward construction accepts an explicit fold schedule. Training
  decisions must precede `test_start - gap_microseconds`; label intervals must
  end by that cutoff and labels must be available by `test_start`. The three
  exclusion counters are disjoint, applied in gap/label-end/label-availability
  order. Test decisions use `[test_start, test_end)` and do not require labels
  to exist at prediction time. Every sample's feature sources must already be
  observable at its decision. Test intervals cannot overlap across folds;
  training sets may reuse earlier samples. Returned indices preserve input order.

### Additional transforms and ordinal scoring

- Haar analysis applies `(a+b, a-b) / sqrt(2)` repeatedly and returns details
  from finest to coarsest. Lengths must be powers of two unless right-zero-padding
  is selected. Reconstruction diagnostics use the returned rounded coefficients;
  this is a whole-block transform, so callers choose the information cutoff.
  See the [Haar filter-bank reference](https://labs.acme.byu.edu/Volume2/Wavelets/Wavelets.html).
- Fractional differencing uses `w[0]=1`, `w[k]=w[k-1]*(k-1-d)/k`, order in
  `[0,2]`, and a fixed width including lag zero. Warmup always spans the full
  requested width. Coefficients are not renormalized; the finite filter's DC
  gain is reported. It does not establish stationarity. The binomial expansion
  is described in the [fractional-differencing reference](https://pmc.ncbi.nlm.nih.gov/articles/PMC9554581/).
- CUSUM uses separate upward/downward accumulators and inclusive threshold
  crossing. Target, drift, and threshold are supplied in observation units;
  reset is either both directions or triggered directions. Event truncation
  does not alter state. There is no fitted threshold or false-alarm probability.
  See [NIST's CUSUM recurrence](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/cusum.htm).
- Lead-lag signatures move all lead coordinates, then all lag coordinates for
  each observation transition. The second level uses Chen's segment update
  `S2 += S1 tensor delta + delta tensor delta / 2` before updating `S1`.
  Optional numeric time is prepended and lifted with the other coordinates.
  Coordinate ordering, scaled coefficients, and physical-output underflow are
  explicit. See [Ni's lead-lag construction](https://arxiv.org/abs/1509.03346)
  and [Chen's identity and linear-segment signatures](https://arxiv.org/abs/1603.03788).
- Ranked probability loss sums squared cumulative errors over `K-1` ordered
  category boundaries. Normalization by `K-1` is optional. Accepted probability
  rows are divided by their sum; the mass tolerance, number changed, and largest
  input mass error are disclosed. Category spacing is not inferred. Boundary
  losses use tail mass directly to avoid subtracting a nearly-one CDF, and the
  aggregate mean is computed before rounding individual scores. See section 6
  of [Gneiting and Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

### YAML, compression, columnar data, and evidence inspection

- YAML accepts one UTF-8 document up to 500 KB, 20,000 scalar/container nodes
  including keys, and 32 collection levels. It rejects anchors, aliases, tags,
  duplicate keys, merge keys, and ambiguous plain scalars. Quote date-like,
  legacy-boolean, or numeric text when a string is intended. Keys must resolve
  to strings. Returned `value_node_count` excludes mapping keys.
- Gzip inspection reads at most 8 MB of compressed input and validates every
  concatenated member through its CRC and size trailer. The expanded-byte
  allowance defaults to 16 MB and can reach 64 MB across at most 100 members.
  Decompressed bytes are hashed in bounded chunks and discarded; trailing junk
  and zero padding are rejected. A `.tgz` file is checked as gzip, without tar
  extraction or inspection.
- Arrow IPC accepts little-endian uncompressed V4/V5 flat primitive columns.
  Raw framing and FlatBuffer metadata are checked before PyArrow decoding;
  compression, dictionaries, extensions, nested/view types, and tensors are
  rejected. Both supported file and stream forms require an explicit EOS marker.
  Limits are 8 MB source, 64 columns, 256 batches, 10,000 rows, 250,000 cells,
  and 16 MB summed data-buffer lengths. Metadata has separate bounds; 16 MB is
  not a total process-memory promise. All rows and columns are validated even
  when the returned page selects fewer. Integer/decimal values use strings,
  binary uses base64, and temporal values retain exact ticks with unit/zone
  metadata. An unspecified timestamp zone remains unspecified. See the
  [official Arrow format](https://arrow.apache.org/docs/format/Columnar.html).
- Receipt inspection reads strict finite JSON up to 2 MB. Select `receipt_body`
  to exclude only `receipt_sha256`, or `hedge_lab_body` to additionally exclude
  `receipt_path`, `artifact_path`, and `metadata_path`. The selected preimage and
  matching digest conventions are reported. An absent seal is not applicable;
  digest matches never certify schema, payload truth, or research eligibility.
  Up to 50 [JSON Pointer](https://www.rfc-editor.org/rfc/rfc6901) selections return
  values up to 16 KB each, otherwise their size/hash and an explicit status.
- Audit-ledger inspection streams up to 64 MB and 100,000 physical lines under
  the repository's v1 canonical ASCII-JSON entry convention. Malformed lines
  retain their position and break the next link's assessment. Pagination counts
  physical lines. Index zero requires the zero genesis hash; segments accept an
  explicit preceding anchor. Empty segments preserve that incoming tip. Optional
  final-hash/count anchors detect disagreement with the supplied endpoint.
  No signatures, Merkle checkpoints, trusted history, or research eligibility
  are verified, and readers do not lock or establish immutable snapshots.

### PIT and interval construction

- PIT diagnostics accept up to 50,000 supplied values in `[0,1]`, up to 100
  explicit histogram bins, and 20 distinct lags from 1 through 1,000. Bins are
  left-inclusive/right-exclusive except that the final bin includes one.
  Uniform-reference bin probabilities use bin widths, including unequal bins.
  KS and Cramer–von Mises values describe marginal empirical-CDF discrepancies;
  lagged Pearson correlations preserve input order and use separate pair means.
  Constant margins and fewer than two pairs return distinct null statuses.
  Centered moments use exact integer representations of supplied floats and
  square-root conversion uses exact midpoint comparisons to avoid premature
  underflow. Unrepresentable nonzero results are rejected.
  No p-values, independence finding, or calibration certification is produced.
- Split-conformal interval construction accepts up to 10,000 paired calibration
  predictions/outcomes and 1,000 new predictions. Its rank is
  `ceil((n+1)*(1-alpha))`, using exact arithmetic on the shortest decimal spelling
  of the supplied alpha. Residual ordering uses exact differences of supplied
  binary floats; emitted radii and endpoints round outward. Rank `n+1` returns
  `all_real_numbers` with null bounds and an explicit infinite-radius status.
  Finite endpoints outside representable float range are rejected. Reported
  calibration coverage is in-sample. The operation cannot establish independent
  predictor fitting or exchangeability of calibration/future scores, and makes
  no time-series coverage guarantee. See the
  [split-conformal construction](https://arxiv.org/abs/2107.07511).

### Forecast panels, window joins, and resampling contracts

- Forecast-panel audits identify forecasts by security, target, horizon,
  decision time and model version; realization keys omit model version so
  models can share outcomes. Horizon names are opaque labels and infer no
  exchange calendar. Explicit target times must follow decisions, completed
  outcomes must become available at or after their target times, and forecast
  sources must be available when the forecast is published and by the decision.
  Duplicate keys, even identical copies, block eligibility. The audit uses an
  explicit evaluation clock, reports source indices, and scores no values.
- Window joins match group IDs and event times using explicit open/closed
  endpoints, defaulting to `[start,end)`. Decision-time availability is enabled
  by default; known future events remain selectable if the supplied window
  includes them. Candidate pairs across every window are counted before the
  availability filter and capped at 250,000 by default, configurable to one
  million. Per-window retention defaults to the earliest 100 matches, with
  explicit truncation or error behavior. Output pair budgets apply to the
  requested page; counts and overflow checks still cover every window. Inputs
  permit 10,000 rows per side; pages permit 500 windows and 10,000 pairs.
- Resampling-group audits consume a unique sample-to-group catalog and supplied
  memberships with optional compressed `draw_count`. Explicit unordered pairs
  declare which partitions must be disjoint within each resample. Both repeated
  sample IDs and different samples from the same group are checked across those
  pairs. Repeated inclusion within a partition has a separate allow/reject
  policy; reuse across different resamples is permitted. Unknown samples remain
  violations. Inputs allow 10,000 catalog rows, 10,000 membership rows, 128
  partition constraints, and at most one million partition probes. No temporal
  purge or statistical independence is inferred.
- Circular block resampling returns indices only. A fresh local Python
  `Random(seed)` draws independent uniform block starts, preserves order modulo
  source length, and truncates the last block to the requested length. It reports
  starts, actual wrap counts, omitted-source counts and Python implementation/
  version. Repeatability is scoped to that RNG implementation/version. Limits
  are 10,000 source/sample indices, 200 replicates, and 100,000 total emitted
  indices. The circular seam introduces an artificial adjacency; stationarity,
  dependence and block-length assumptions are not established.

### Weighted numerical tools

- Weighted covariance accepts 2,048 observations and eight variables, with
  nonnegative reliability weights. Integer arithmetic accumulates centered
  moments of the supplied binary floats before final rounding. Population
  covariance uses normalized weights; `reliability_unbiased` divides by
  `1-sum(p_i**2)` and requires at least two positive weights. The correction's
  statistical interpretation assumes independent observations with a common
  mean/covariance and fixed weights. Frequency counts require a different
  estimator. Zero-weight observations do not affect the moments. Inputs are
  bounded by magnitude `1e100`; nonzero outputs that cannot be represented are
  rejected. See [NumPy's weighting convention](https://numpy.org/doc/stable/reference/generated/numpy.cov.html).
- Quantile repair applies its own weighted pool-adjacent-violators algorithm
  to rows aligned to strictly increasing supplied quantile levels. Positive
  weights are shared across rows; level spacing does not become a weight.
  Exact integer pool comparisons precede output rounding. Separate diagnostics
  report exact projection changes and changes in returned values; the adjustment
  objective uses those returned values. It accepts up to 512 rows, 256 levels,
  and 16,384 cells and rejects unrepresentable nonzero outputs. This projection
  makes no calibration or score-improvement claim. See the
  [weighted isotonic objective](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.isotonic_regression.html).
- Empirical Wasserstein distance independently normalizes nonnegative weights
  on two supplied one-dimensional supports, drops zero-weight entries, coalesces
  equal locations and matches sorted masses. It computes p=1 or p=2 transport
  costs with exact integer masses and support differences before output rounding.
  At most 2,048 values per side and 200 transport-plan entries are supported.
  A p=2 cost can underflow even when its square root is representable; the result
  then reports a null scalar cost and explicit binary mantissa/exponent. Tiny
  plan masses use the same convention. Nonzero unrepresentable final distances
  are rejected. This is a descriptive distribution distance, without a
  significance or calibration claim.

### Archive and XML inspection

- Tar inspection reads up to 32 MB of uncompressed USTAR/V7 data, with 1,000
  members and 16 MB per payload. It validates unsigned header checksums,
  nonnegative octal fields, payload extents, zero member padding, and two zero
  end blocks. Additional padding must consist of zero 512-byte blocks. PAX,
  GNU/sparse/long-name extensions, compression and concatenated archives are
  rejected. All members are scanned before pagination; source hashes include
  padding and regular/contiguous-member hashes cover payload bytes only. Paths
  and link targets are observations; no extraction occurs.
- XML inspection accepts a UTF-8 XML 1.0 document up to 1 MB and validates the
  full document before returning at most 100 element rows. Rows preserve
  preorder IDs, parent/sibling indices, expanded namespace URI/local names,
  attributes, text and tail; prefixes, CDATA boundaries, comments and processing
  instructions are not preserved. Comments and instructions are counted.
  DTD/custom/external entity declarations are rejected; predefined and numeric
  character references remain supported. No schema or XInclude resolution is
  performed. Limits include 5,000 elements, 64 levels, 20,000 attributes, 500,000
  text characters, and a 1 MB node page; individual fields have tighter bounds.
- Artifact bundles use strict UTF-8 JSON manifests with exactly `version: 1`
  and an `artifacts` array of `{path,size,sha256}` declarations. Paths are relative
  to the workspace, require portable spelling, and must be unique even under
  casefolding. The manifest is limited to 256 KB/200 entries; artifact reads to
  16 MB each and 64 MB plus one over-limit probe byte in aggregate. Every
  declaration receives a status, including missing, refused, mismatched and
  budget-exhausted reads. Hashes are withheld for incomplete reads. An empty
  manifest has no completeness evidence. Binding hashes cover sorted-path
  compact UTF-8 JSON declarations/observations; completeness covers listed
  artifacts only. The operation does not establish authenticity, provenance,
  research correctness, or an atomic multi-file snapshot.

### Session calendars and release schedules

- Session coverage uses caller-supplied, disjoint `(open,close]` segments per
  group. Expected bars lie on an elapsed-time grid from each segment's open;
  partial closing intervals have explicit reject/include/drop policies. Only
  schedules and bars observable at `asof` contribute to coverage. Completed
  bars cannot be available before their event time. Duplicate bars fill a slot
  once, and gaps are returned as compressed spans. Limits are 5,000 sessions,
  10,000 total segments, 10,000 bars, 500 returned sessions and 200 diagnostics.
  No exchange calendar or exceptional closure is inferred.
- Horizon assignment consumes up to 64 calendar snapshots with supplied close
  clocks, session IDs, availability and declared coverage. Offset one selects
  the first close strictly after a decision, or at/after under an explicit
  alternative. Larger offsets advance through supplied closes. Unavailable,
  missing and insufficient calendars return separate statuses. Decisions
  outside declared coverage remain unresolved. Limits are 20,000 total closes,
  10,000 decisions, 32 distinct positive offsets, 100,000 assignments and a
  1,000-row page. Calendar completeness is a caller assertion.
- Release audits compare the earliest supplied visible publication from each
  expected source to an observable release schedule and explicit early/late
  tolerances. Later revisions do not become late initial publications. The
  audit separately reports due-source coverage, source/event identity errors,
  exact duplicate rows, conflicting revision reuse and ambiguous first-clock
  payloads. Scalar payload comparison preserves types. An unavailable schedule
  cannot authorize a publication match. Limits are 10,000 schedule rows and
  observations each, 16 sources per release, 20,000 expected source pairs,
  500 returned source assessments and 200 diagnostics. Supplied observations
  do not prove the true first publication time.

### Label overlap, robust slopes, and stratified folds

- Label uniqueness integrates `1 / concurrency` over each positive-duration
  half-open interval, divides by that label's duration, and keeps securities
  separate. A sweep over exact microsecond boundaries avoids enumerating all
  active label pairs. Duplicate intervals are separate labels; touching
  endpoints do not overlap. Raw weights are in `(0,1]` and are not normalized.
  Exact rational numerators/denominators accompany rounded values. Limits are
  512 labels, 128 securities and 366 days per label. The result measures supplied
  interval overlap; it does not establish statistical independence or causal
  feature availability.
- Robust rolling regression computes the median of every distinct-predictor
  pair slope in a trailing window, then the median of `y - slope*x` using the
  unrounded slope. Even-sized medians average their central pair. Warmup and
  all-equal-predictor windows return explicit null statuses. Limits are 2,048
  paired rows, windows of 2–256 and 250,000 candidate pairs across all windows.
  Exact rational intermediates precede output rounding; unrepresentable
  nonzero outputs are rejected. These are the joint-intercept
  [Theil–Sen conventions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html).
- The empirical characteristic function returns weighted averages of
  `cos(t*x) + i*sin(t*x)` at supplied angular frequencies. Nonnegative weights
  are normalized, zero weights excluded, and duplicate frequencies preserved.
  Phase multiplication is exact before conversion for platform trigonometric
  functions; accumulation is exact over their rounded outputs. Limits are
  2,048 observations, 128 frequencies, 131,072 pairs and `abs(t*x) <= 1e6`
  radians. The returned modulus uses the emitted components; tiny excursions
  above one from rounding are disclosed. This describes an empirical
  distribution without fitting a forecast or establishing independence.
- Stratified folds shuffle lexically ordered sample IDs within each class
  using a fresh local seeded Python RNG, then allocate class quotas with
  remainders assigned to currently smallest folds. Both per-class and total
  validation sizes differ by at most one. Small classes either cause an error
  or return explicit missing-class diagnostics. Assignments retain source
  indices and are invariant to input-row reordering by stable IDs. Limits are
  2,048 samples, 128 classes and 2–20 folds. RNG implementation/version are
  reported. These folds impose no temporal, group or label-availability guards.

### Checkpoint signatures and tensor formats

- Signed-checkpoint inspection supports the existing `quant_fund.audit` v1
  Ed25519 convention. Required caller inputs select a trusted key identity and
  exact expected tree size/root. Signature validity, embedded-key agreement,
  key-ID agreement and root/size agreement are separate observations. The
  signed preimage has no domain field, so `expected_domain` selects a convention
  and `domain_cryptographically_bound` remains false. Embedded public-key and
  Sigstore declarations are unsigned metadata. Limits are 2 MB source, 1,000
  JSONL records, 16,384 bytes per record and 16 supplied keys. Every record
  receives strict flat-JSON checks; only the selected record receives checkpoint
  schema/signature checks. No ledger completeness, consistency proof, timestamp
  freshness, external identity trust or research eligibility is established.
- Safetensors inspection reads a bounded header by default, supporting files
  up to 1 TB without loading tensor payloads. Header limits default to 1 MB and
  can reach 8 MB. Optional full mode reads/hashes at most 16 MB. Strict JSON,
  dtype bit widths, dimensions and byte offsets must account for the entire
  payload without holes, overlaps or trailing bytes. Empty/scalar tensors are
  supported; packed tensors must end on byte boundaries. Limits are 10,000
  tensors, 32 dimensions and a 200-tensor page. Header hashes include original
  padding; full-file hashes are absent in header mode. Payload values are not
  decoded, and file metadata checks do not establish an immutable snapshot.
  See the [safetensors format](https://github.com/huggingface/safetensors#format).
- Matrix Market inspection accepts ASCII coordinate `.mtx` files with real,
  integer, complex or pattern entries. Symmetry declarations require square
  lower-triangle input, with explicit skew/Hermitian diagonal rules. Numeric
  duplicates either fail or sum exactly before sparse symmetry expansion;
  pattern duplicates fail. Input indices are one-based; returned entries are
  zero-based and sorted by row/column. Numerical components use decimal strings
  and explicit zeros remain present. Limits are 8 MB, 100,000 coordinate records,
  200,000 expanded entries and 200 returned entries. Restricted numeric spelling
  and exponent bounds permit exact 700-digit arithmetic in an explicit decimal
  context. No dense arrays are allocated. See the
  [Matrix Market format](https://math.nist.gov/MatrixMarket/formats.html).

### Density and censored-outcome scoring

- Histogram density scoring accepts common increasing bin edges and a mass row
  for each observation. Each nonnegative mass row is normalized to one, then
  divided by its bin widths to define a piecewise-uniform density. Bins are
  left-inclusive/right-exclusive except for the final included endpoint.
  Outside-support and zero-mass observations retain explicit infinite log-loss
  statuses. Zero observation weights exclude rows from the mean, and counts
  still describe every row. Natural-log density loss can be negative and is
  coordinate-unit dependent. Masses, widths and weighted aggregation use exact
  rational arithmetic; logarithms are floating approximations. Limits are
  2,000 rows, 256 bins, 100,000 mass cells and 500 returned details. Ordinary
  proper-score interpretation requires outcome-independent observation weights.
- Censored Brier scoring uses a supplied right-continuous censoring-survival
  curve and its declared valid-through horizon. Observed events at or before
  the horizon use `G(event_time-)`; known survivors use `G(horizon)`; rows censored
  by the horizon contribute zero. The divisor remains the full subject count.
  Event ties follow the declared `event_time <= censor_time` observation
  convention. Required zero censoring probabilities cause an error. Supplied
  survival forecasts and censoring curves must be nonincreasing. Results show
  informative/event/survivor counts, weight sums and separate score components;
  optional multi-horizon integration is trapezoidal over the supplied range.
  A horizon with no informative rows has zero contributions and a zero score;
  its zero informative count and weight sum distinguish it from predictive
  accuracy on observed outcomes.
  Limits are 2,000 subjects, 200 horizons and 100,000 scoring cells. Noninformative
  censoring, valid curve estimation and held-out prediction assumptions are
  not established by this calculation; left truncation and competing risks are
  unsupported. See the [IPCW Brier definition](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html).

### Survival curves and multivariate kernel scoring

- Product-limit estimation accepts an unweighted cohort of up to 5,000 subjects
  entering at time zero. Event survival uses `S *= (n-d)/n`; reverse censoring
  survival removes tied observed events before applying the censoring update.
  Exact rational products retain risk-set counts, event/censor counts and left
  and right curve values. Results include a 500-row curve page, up to 1,000
  supplied evaluation times, the first median crossing, and optional exact
  restricted-mean integration from zero within observed follow-up. Evaluation
  beyond the last observed duration returns an explicit unavailable status.
  No left truncation, conditional censoring model, uncertainty interval or
  independence finding is provided. See the
  [product-limit definition](https://www.itl.nist.gov/div898/handbook/apr/section2/apr215.htm)
  and [reverse tie convention](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.nonparametric.kaplan_meier_estimator.html).
- Gaussian-kernel scoring evaluates `1 + E[k(X,X')] - 2*E[k(X,y)]` for supplied
  weighted empirical forecasts and fixed positive coordinate bandwidths. Pairs
  include the diagonal with empirical mass weights; there is no finite-ensemble
  correction. Computing with `1-k` via `expm1` avoids subtracting nearly-unit
  kernels. Squared scaled distances and accumulation of rounded kernel terms
  use exact arithmetic. Saturated dissimilarities and negative numerical scores
  are disclosed, without clamping. Limits are 128 observations, 256 members,
  16 coordinates and 250,000 scalar distance-work units. Bandwidth selection
  must be independent of verification outcomes for the proper-score
  interpretation. See the [kernel-score definition](https://scoringrules.readthedocs.io/en/latest/theory.html).
- Competing-risk scoring accepts mutually exclusive terminal causes and
  cumulative cause probabilities on a shared horizon grid. Cause probabilities
  must be nondecreasing over time and sum to at most one using exact comparisons;
  the remaining mass is event-free probability. The score sums IPCW squared
  errors across every cause and the event-free state, using all subjects as
  the denominator. With one cause, this is twice the usual survival Brier score.
  Any observed cause contributes an event with weight `1/G(T-)`; a known
  survivor uses `1/G(t)`, and censoring by the horizon contributes zero. Cause
  components/counts, informative counts and weight diagnostics are retained.
  Limits are 1,000 subjects, 16 causes, 100 horizons and 100,000 state cells.
  The supplied censoring curve and its valid-through horizon receive shape
  checks, without fitting or assumption verification. A horizon with no
  informative rows has zero contribution, not evidence of accurate forecasts.
  Integration is trapezoidal over supplied horizons. See the
  [cause-specific IPCW loss](https://arxiv.org/abs/2106.12948).

### Grouped validation and asynchronous numerical features

- Group-stratified folds keep each supplied group intact. Seeded greedy
  assignment and bounded improving single-group moves minimize a documented
  sum of normalized class imbalance and weighted sample imbalance. Objective
  comparisons are exact. The first groups seed nonempty folds, and moves retain
  nonempty folds. Results report missing classes in both validation and training
  complements, classes represented in fewer groups than folds, pass limits and
  accepted moves. No global optimum or general feasibility result is claimed.
  Limits are 2,048 samples, 128 groups, 32 classes, 2–20 folds and 20 local passes.
  This supplies group isolation; temporal and availability constraints are not
  enforced. See [group-stratified validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- Hayashi–Yoshida covariance uses observed increments over `(previous,current]`
  intervals and sums products for interval pairs with strictly positive overlap.
  Merely touching intervals do not contribute. Both paths must have observed
  values at common initial and final clocks; interior clocks may differ. A
  linear sweep avoids constructing a Cartesian grid. Raw positive prices produce
  price-increment products; log-price inputs must already contain logarithms.
  There is no interpolation, demeaning, duration weighting or annualization.
  Exact integer accumulation precedes output rounding. Limits are 4,096 points
  per path. The caller establishes availability and sampling assumptions. See
  [Hayashi and Yoshida's overlap estimator](https://www.ism.ac.jp/editsec/aism/pdf/060_2_0367.pdf).
- First-order detrended fluctuation analysis forms an integrated demeaned
  profile, splits it into nonoverlapping boxes from the beginning, discards each
  scale's tail, and removes a local least-squares intercept and trend. Each
  eligible scale needs two boxes. Fluctuation is the square root of retained
  mean squared residuals; products and local fits are exact before conversion.
  An unweighted log-log regression uses positive eligible scales and reports
  constant/insufficient cases explicitly. Limits are 4,096 values, 32 increasing
  scales of at least four points and 131,072 input-scale cells. Samples must be
  equally spaced and ordered. The fitted exponent does not establish long
  memory, stationarity or a Hurst interpretation. See the
  [DFA definition](https://physionet.org/content/dfa/1.0.0/).

### Stream progress and offset coverage

- Availability frontiers distinguish publication readiness, defined as
  `max(event_time, available_time)`, from local readiness that also requires
  ingestion. Two activation sweeps answer decision queries with latest completed
  event times, earliest supporting readiness, row lineage, staleness and pending
  ingestion counts. Equal event/readiness ties retain the earliest input row.
  Publication and ingestion lag can expose negative declared lags. Counts
  include duplicate supplied records, and progress does not imply completeness
  of earlier events. Limits are 10,000 observations and decisions each, with
  a 500-decision page; the default convention requires publication and ingestion.
- Stream-offset audits use separate `(source_id, stream_id, partition_id)`
  identities and caller-declared inclusive offset ranges. Missing ranges are
  compressed without allocating absent rows. Identical duplicates have an
  explicit allow/reject policy; reused offsets with conflicting event clocks
  or supplied payload hashes always violate the contract. Optional input-offset
  and offset-to-event ordering checks report actual row witnesses. Hashes are
  supplied labels, not verified payload bytes. Coverage and overall consistency
  remain separate results. Limits are 2,048 stream identities, 10,000 records,
  a 500-stream page and 200 diagnostics; signed 63-bit offset spans remain
  arithmetic ranges even when they contain few supplied records.
- Temporal-constraint audits express minimum and maximum differences between
  named clocks in integer microseconds. Bellman–Ford either returns one feasible
  assignment or a complete negative-cycle witness with source constraint/pin
  indices and a negative summed bound. Timestamp pins attach clocks to an epoch
  reference. Components without pins expose relative offsets only; anchored
  candidates are not estimates or uniqueness claims. Candidates outside the
  datetime range retain exact epoch integers and an explicit status. Inverted
  input bounds are accepted as contradictions to diagnose. Limits are 256
  clocks, 2,048 constraints, 256 pins, signed delays up to `10^18` microseconds
  and a preflight vertex-times-edge budget of one million by default, configurable
  to two million. No missing clock constraints are inferred.

### Merkle proofs and SQLite inspection

- Inclusion verification uses the audit tree's ordered SHA-256 convention:
  prefix `00` for leaves and `01` for ordered internal nodes. It derives sibling
  orientation and exact minimal path length from the caller-pinned tree size
  and index. Inputs supply either a leaf preimage as hex or an already prefixed
  leaf digest; digest-only mode cannot validate payload bytes. Empty trees have
  no inclusion proof. Limits are signed 63-bit tree sizes, 64 siblings and a
  65,536-byte leaf preimage. No signature or ledger completeness is verified.
- Consistency verification independently reconstructs old and new audit roots
  from the minimal prefix proof, consuming each node exactly once. Both roots
  and sizes are required caller anchors. Equal sizes require equal roots and
  an empty proof and establish equality only. Zero-size roots must equal
  SHA-256 of the empty string; this is stricter than the existing verifier's
  equal-size empty-tree handling. Growth from zero remains explicitly unsupported
  under the repository convention. Shrinking trees, missing/extra nodes and
  each root mismatch have separate outcomes. Limits are signed 63-bit sizes
  and 64 siblings. These checks use the [RFC 6962 construction](https://www.rfc-editor.org/rfc/rfc6962#section-2.1)
  and do not establish checkpoint identity, signatures or research eligibility.
- SQLite inspection accepts rollback-format main-file database images up to
  8 MB. A contained read is hashed and deserialized into a private in-memory
  database; SQLite never receives the source path. Header page size/count,
  journal format and change counters must agree. Fixed queries describe up to
  128 schema objects and optionally preview one ordinary table. Views, virtual
  and shadow tables, generated columns, extensions and arbitrary SQL cannot
  supply preview rows. A restrictive authorizer, query-only mode, untrusted
  schema mode, native limits and progress interruption bound supported work.
  Limits include 64 columns, 200 rows, 10,000 cells, 4,096 bytes per text/blob
  value and 512 KB encoded cell data. Integers use decimal strings and blobs
  use hex. Table scans are unordered. Progress callbacks are approximate,
  and native limits do not promise a hard memory or wall-clock cap. No sidecars
  are checked or included. Header checks cannot establish the absence of a hot
  rollback journal, so previewed bytes may contain changes that normal recovery
  would undo. Committed/recovered state, source snapshot consistency and
  whole-database integrity remain unverified. See the
  [SQLite deserialize API](https://www.sqlite.org/c3ref/deserialize.html) and
  [rollback recovery rules](https://www.sqlite.org/lockingv3.html).

### Event alignment, partition geometry and nested validation

- Event alignment consumes every supplied row through a match or one-sided gap.
  Matches require equal keys, an inclusive clock tolerance and an optional clock
  direction. Dynamic programming minimizes exact integer gap, match and elapsed
  microsecond costs. Backward traceback prefers match, then left-only, then
  right-only on ties. Optimal-path counts saturate at two; ambiguity includes
  different gap orderings even with identical matched pairs. Inputs must already
  be ordered by aware event time. Limits are 2,000 rows per stream, 250,000 DP
  cells by default (maximum one million), 500 returned steps and 100 ambiguity
  diagnostics. Alignment is an optimum under supplied costs, not authenticated
  event identity. The recurrence follows [global sequence alignment](https://ocw.mit.edu/courses/6-047-computational-biology-fall-2015/pages/open-textbook/).
- Partition ranges use explicit half-open expected integer coverage for each
  dataset/version/key. Empty or inverted partitions are diagnosed and excluded
  from geometry; duplicate partition IDs remain in the geometry. IDs must be
  unique across all keys within a dataset/version. Heap sweeps count exact
  overlapping row pairs and retain witnesses without constructing every pair;
  union lengths and maximal missing spans avoid enumerating integers. Expected
  coverage may be empty. Missing expected declarations are violations. Limits
  are 10,000 partitions, 2,048 identities, a 500-identity page and 200 diagnostics.
  These declarations do not authenticate files or stored records.
- Nested-fold audits take a unique sample/group catalog, declared outer and
  inner folds, and explicit membership rows. Inner train/validation samples must
  belong to parent training and cannot belong to outer test. Sample separation
  is mandatory; group separation, duplicate policies, unassigned coverage,
  nonempty partitions and child-fold requirements are explicit. Child coverage
  uses known outer training samples excluding outer test. Reuse across different
  folds is permitted, and temporal leakage is not assessed. Limits are 10,000
  catalog samples, 128 outer folds, 1,024 inner folds, 10,000 membership rows per
  level, and one million coverage checks by default (maximum five million).
  Diagnostic policies distinguish observations from actual violations.

### Dependent resampling and time-series features

- Stationary resampling draws the first source index uniformly, then restarts
  uniformly with probability `1 / expected_block_length` or continues circularly.
  The restart probability is the exact rational interpretation of the supplied
  binary float; integer RNG draws implement its Bernoulli decisions. Restart
  boundaries remain visible even when the new index equals the continuation.
  Observed block lengths include a right-censored final block. Seed, Python RNG
  version, wrap and draw counts are reported. Limits are 10,000 source/output
  indices per replicate, 200 replicates, 100,000 total output indices and expected
  block lengths from one to one million. Stationarity, dependence and block-length
  suitability remain caller assumptions. See [Politis and Romano (1994)](https://doi.org/10.1080/01621459.1994.10476870).
- Sample entropy uses the same `N-m` eligible starts for both length-`m` and
  length-`m+1` templates. It counts unordered distinct pairs with exact binary
  input comparisons under a supplied absolute Chebyshev tolerance and inclusive
  or strict threshold. Overlapping templates are included; self-matches are
  excluded. The output distinguishes insufficient templates, no base matches,
  zero forward matches (infinite entropy), and finite `log(B/A)`. It does not
  infer a standard-deviation tolerance, add pseudocounts or impose temporal
  exclusion. Limits are 2,048 observations, embedding dimension 32 and two million
  candidate coordinate comparisons. See the [sample-entropy definition](https://www.wavemetrics.com/sites/default/files/2018-10/04RichmanMoorman_SampleEntropyChapter.pdf).
- Burg fitting applies an explicit demean/none policy, then its own forward and
  backward error lattice. Stage reflection ratios, coefficient updates and energy
  products use exact arithmetic over the current rounded error state; common
  scaling keeps subsequent error vectors bounded. It returns AR-polynomial and
  prediction coefficients, intercept, stage energies and fitted/requested orders.
  Zero input energy, singular stages, exact lattice prediction and reflection
  rounding to a unit boundary are distinct stopping outcomes. Limits are 4,096
  values, order 64 below sample count and 131,072 sample-order cells. Nonzero
  conversion underflow and oversized exact energy arithmetic fail explicitly.
  Lattice prediction is not a claim of exact prediction of original observations,
  stationarity, forecast timing or model selection. See the [Burg recursion](https://pyspectrum.readthedocs.io/en/latest/_modules/spectrum/burg.html).

### Hierarchical and Gaussian scoring; principal coordinates

- Hierarchical probabilities accept one connected rooted tree with at least two
  leaves. Every forecast supplies all leaf masses, which are normalized before
  upward aggregation. Each nonroot node receives a strictly positive fixed
  weight; the score is the weighted mean of its binary descendant-event Brier
  loss. Positive leaf weights make this construction strictly proper when the
  hierarchy and weights are fixed independently of outcomes. These assumptions
  are not verified. Per-node probability, realized count, loss and weighted
  contribution expose the construction. Internal forecasts are derived from
  leaves. Limits are 256 nodes, 128 leaves, depth 64, 1,000 observations and
  100,000 node-observation cells. Exact cross-row accumulators and final fractions
  have a 131,072-bit numerator/denominator budget. No hierarchy, calibration or
  node weights are fitted. See [proper binary scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
- Gaussian scoring requires an exactly symmetric positive-definite covariance
  for every supplied mean/outcome pair. An exact rational unpivoted LDL factorization
  establishes positive pivots, the determinant and the triangular-solve quadratic
  form. Logarithms use scaled integers or `log1p` near one; output conversion is
  finite and rejects nonzero underflow. The loss includes the dimensional normalizing
  constant and can be negative because it scores a density with coordinate units.
  No inverse, covariance repair, jitter, parameter fitting or forecast-timing
  certification is applied. Limits are 128 forecasts, 12 dimensions and 100,000
  observation-times-dimension-cubed work units. See the [Gaussian density formula](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html).
- Principal coordinates normalize a supplied symmetric nonnegative dissimilarity
  matrix by its largest distance and double-center squared distances exactly.
  NumPy's approximate symmetric eigensolver supplies up to 32 positive axes above
  an explicit relative threshold. Negative and near-zero axes are reported;
  distances are not repaired. Coordinates restore original units, and their
  rounded returned values determine reconstruction stress and maximum distance
  error. Eigenvalues retain normalized squared-distance units. Axis signs use
  deterministic largest-component pivots, but repeated eigenspaces may rotate
  across platforms. Limits are 128 points and relative eigenvalue tolerance
  between `1e-15` and `1e-3`. All-zero distances have an explicit empty embedding.
  No triangle inequality, Euclidean validity, latent dimension or scientific
  interpretation is certified. See [principal coordinate analysis](https://scikit.bio/docs/latest/generated/skbio.stats.ordination.pcoa.html).

### Fixed-width, sparse text and NetCDF readers

- Fixed-width records use zero-based half-open Unicode code-point spans, not byte,
  grapheme or screen-column positions. Data and optional observed-header records
  must exactly match the declared width. String cells are untrimmed; gaps between
  spans are ignored. Physical skip lines, a column-zero comment prefix and empty
  line policy apply in that order. Every line receives encoding and length checks.
  LF/CRLF are accepted; bare CR, initial UTF-8 BOM and NUL are rejected. Limits
  are 4 MB, 100,000 physical lines, 16,384 code points per line, 50,000 records,
  64 columns, two million parsed cells and a 512 KB encoded row/header page.
  The full source is validated and hashed before returning line/byte lineage.
- LIBSVM parsing accepts a numeric label and strictly increasing positive
  `index:value` coordinates, preserving original decimal tokens and explicit zero
  entries. Token conversions must be finite binary64 without nonzero underflow;
  output strings preserve their original precision. No task meaning or dense
  dimension is inferred; callers may supply a maximum feature count. Index zero,
  precomputed kernels, repeated/reordered indices, qid/cost extensions and multiple
  labels are unsupported. ASCII LF/CRLF records have explicit hash-comment and
  blank-line policies. Limits are 8 MB, 50,000 rows, 10,000 coordinates per row,
  250,000 total coordinates and indices through one billion. Row and coordinate
  pages are independent; the full bounded source is parsed and hashed.
- NetCDF inspection parses a strict CDF1/CDF2 subset directly: dimensions, raw
  attributes, variable shapes, fixed extents and ordered interleaved records.
  Numeric attributes retain original bits and decimal/hex-float string encodings;
  character attributes remain raw bytes. The lone BYTE/CHAR/SHORT record-variable
  exception uses an unpadded record stride despite padded declared vsize.
  Zero-record offsets may describe future slabs; declared data extents and EOF
  must satisfy the supported layout. CDF5, HDF5, compression, streaming counts,
  oversized vsize sentinels and trailing/preallocated records are unsupported.
  Header mode accepts files through one TB while reading at most two MB; full
  mode is limited to eight MB and adds a whole-source hash. Header hashes exclude
  gaps and payload. Limits include 256 dimensions/variables, rank 16, 512 total
  attributes and 128 KB aggregate raw attributes. No arrays are allocated, variable
  payloads decoded, CF conventions applied or atomic snapshot guaranteed. See the
  [NetCDF classic format specification](https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html).

### Joint and count distributions; weighted discretization

- Joint categorical scoring accepts two through six axes with two through 32
  labels each. A shared table of up to 2,048 unique tuples carries forecast mass;
  undeclared Cartesian tuples have zero mass without materializing the full grid.
  Each row is normalized exactly, then scored by joint multicategory Brier and
  log loss and marginalized for separate axis scores. Brier is summed over
  categories without division. Zero realized support yields infinite log loss
  and a null aggregate. The observed log joint/product-of-marginals ratio is a
  dependence diagnostic, not an information estimate or proper headline score.
  Limits are 1,000 outcomes, 200,000 state-observation-axis cells and 131,072 bits
  per exact cross-row accumulator. No independence assumption, pseudocount,
  clipping or forecast-timing verification is applied. See [categorical proper
  scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).
- Count scoring combines Poisson, binomial and negative-binomial laws in one
  implementation. Parameters are supplied predictions. Negative binomial uses
  mean `mu` and shape `r`, giving variance `mu + mu²/r`. Direct log losses avoid
  forming tiny probability masses; bounded log-ratio sums replace large
  log-gamma differences for combinatorial coefficients. Poisson uses `lgamma`.
  Elementary functions remain approximate. Zero-mean and endpoint-probability
  laws retain their deterministic support. An impossible observation makes the
  positive-weight aggregate infinite, represented by null and explicit status.
  Limits are 1,000 rows, one million counts/trials, 200,000 coefficient terms,
  positive means/probabilities of at least `1e-100`, means at most one million,
  and negative-binomial shape from `1e-6` through `1e6`. Weights are positive;
  their independence from outcomes is not verified. See [Poisson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html),
  [binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html),
  and [negative-binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html)
  mass-function conventions.
- Quantile binning fits inverse weighted empirical-CDF boundaries at `j/B` using
  exact mass comparisons. Duplicate cut values collapse, and a cut at maximum
  positive-weight support is discarded. Bins are lower-open, upper-closed with
  unbounded outer intervals. Ties are preserved, so equal mass and equal row
  counts are not guaranteed. Zero-weight fitting rows receive IDs and contribute
  to row counts but do not affect cuts. Separate queries reuse the fitted cuts.
  Every target boundary, effective cut, bin mass and source-row assignment is
  reported. Limits are 10,000 fitting rows, 10,000 queries, two through 128
  requested bins, and positive weights from `1e-100` through `1e100` (zero is
  allowed). Fitting-sample availability and representativeness are unverified.
  See the [weighted empirical-CDF convention](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html).

### Source chains and schema domains

- Source-chain auditing separates sources and treats sequence numbers and
  predecessor hashes as declarations. Each source starts at zero or continues
  after one caller anchor. Links must name the immediately previous sequence;
  reused sequences/hash labels, forks, duplicate metadata, missing spans and
  deterministic DFS cycle witnesses are reported. Missing tails are not inferred.
  Optional raw payload bytes are SHA-256 checked, but their preimage excludes
  sequence/source/predecessor metadata, so matching bytes do not authenticate
  links or anchors. Limits are 10,000 records, 1,024 sources and 262,144 supplied
  payload bytes, with at most 200 bounded diagnostics. Long cycle witnesses retain
  a prefix and closing edge with an omitted-edge count.
- Schema compatibility asks whether every row admitted by the closed writer
  declaration can satisfy the reader under supported lossless type promotions.
  Presence and nullability are distinct; optional reader fields may be absent,
  and extra writer fields use an explicit policy. Integer ranges must fit fully;
  integer-to-float promotions require complete exact-integer coverage. Decimal
  precision/scale promotions preserve integral and fractional capacities.
  Timestamp units and UTC/naive semantics must match because unrestricted int64
  tick rescaling can overflow or truncate. Physical-unit labels must match exactly.
  Limits are 512 fields per schema, decimal precision 76 and 200 diagnostics.
  Defaults, aliases, nested schemas and inferred unit conversions are unsupported.
  No rows are inspected or converted.

### Circular geometry, robust centers and partial autocorrelation

- Circular summaries reduce angles modulo 360 degrees or the supplied binary64
  `tau` period in radians. Weighted trigonometric moments describe the center
  and resultant; a pairwise half-angle identity preserves dispersion near a
  concentrated direction. The center is undefined at or below the supplied
  resultant tolerance. The shortest closed support arc complements the largest
  empty gap, with deterministic ties. Exact rational endpoints and width carry
  the coverage assertion; float endpoints are previews that disclose rounding,
  including canonicalization across the angle seam. Angular deviation is
  `sqrt(2 * circular_variance)` in the requested units, not logarithmic circular
  standard deviation. Limits are 256 angles, magnitude `1e12`, nonnegative
  weights through `1e100` and explicit nonzero-underflow rejection. Zero weights
  do not affect moments or the arc. Approximate trigonometric roundoff and small
  resultant clipping are disclosed. The arc is not a confidence interval.
- Geometric-median fitting minimizes the normalized weighted mean Euclidean
  distance with a modified Weiszfeld iteration. Coincident-point mass adjusts
  the update, allowing departure from a data point that is not a median.
  Initialization is either the weighted mean or first positive-weight point.
  Exact differences, bounded rational sums and correctly rounded distance roots
  support each update; iterates and gradients remain floating approximations.
  Diagnostics are recomputed at the returned point. Residual tolerance, iteration
  limit, rounded stagnation and objective increase have distinct statuses.
  Limits are 128 points, eight dimensions, 500 iterations, 200,000 work cells
  and 32,768-bit rational arithmetic. Numerical convergence does not certify
  global optimality or uniqueness. Dimension scaling is the caller's choice.
  See [Vardi and Zhang's modified iteration](https://www.pnas.org/doi/10.1073/pnas.97.4.1423).
- Partial autocorrelation uses caller-supplied evenly spaced observations with
  exact sample-mean or zero centering. Biased lag covariances divide by the same
  observation count at every lag. Exact bounded Levinson–Durbin stages return
  PACF, AR coefficients and Toeplitz innovation energies. These energies are
  not sample residual variances. Zero energy, exact singularity and a nonunit
  reflection rounding to a unit boundary have explicit statuses. Limits are
  4,096 observations, order 64 below sample count, 131,072 observation-lag cells
  and 65,536-bit fractions. Completed stages use exact internal coefficients;
  returned intercepts disclose their use of rounded coefficients. Equal spacing,
  stationarity and forecast timing remain unverified. See the [Levinson–Durbin
  convention](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.levinson_durbin.html).

### Temporal resampling membership

Temporal-resample audits receive source streams, replicate decision origins and
explicit output-position/block/source-row memberships. Every referenced source
must be available at the origin; completed-event checks are separate so announced
future events have an explicit policy. Duplicate and out-of-range positions still
receive point checks. Missing positions form compressed spans. Adjacency checks
use consecutive unambiguous positions and report the number of unassessed edges.
Block labels must form contiguous observed runs; continuation can be unchecked,
consecutive or circular. Optional circular event resets exempt within-block
ordering only; whole-replicate ordering remains independent. Source ordering is
checked for all supplied streams, including unused ones. Repeated source draws
are allowed. Limits are 64 streams, 10,000 source rows combined, 500 replicates,
20,000 memberships, 10,000 positions per replicate and 200 diagnostics. No RNG,
stationarity or resampling-inference claim is validated.

### ARFF and dBASE tables

- ARFF parsing supports numeric/real/integer, nominal and string fields in dense
  or sparse comma-separated UTF-8 records. Numeric lexemes remain exact strings
  after finite binary64-range checks. Sparse omitted numeric fields mean zero;
  nominal fields mean their first declared level. Sparse strings must be explicit
  because their implicit dictionary index is ambiguous. Unquoted `?` is missing;
  quoted `?` is a literal. Bounded single/double quoting and escapes, percent
  comments and LF/CRLF are supported. Dates, relational fields, instance weights,
  tab-separated records and multiline quotes are unsupported. Limits are 4 MB,
  128 fields, 50,000 rows, two million logical cells, a 256 KB encoded schema and
  a 512 KB page. All source rows are validated and hashed. See the [Weka ARFF
  specification](https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/).
- dBASE reading supports a strict version-`0x03` single-file III PLUS subset with
  character, numeric, date and logical fields. Header/descriptor lengths, reserved
  bytes, deletion flags, exact record extents and a final EOF marker are checked.
  The caller chooses an explicit single-byte codec. Character padding and raw
  field bytes are preserved; numeric cells use exact trimmed ASCII decimal
  strings. Dates must be valid, and logical/missing encodings are explicit.
  Deleted rows are validated even when excluded from the returned page. Pagination
  counts selected records while retaining physical record indexes and offsets.
  Limits are 8 MB, 100,000 records, 64 fields, 4,096 bytes per record, two million
  cells and a 512 KB encoded page. Memo/vendor extensions and sidecars are
  unsupported. Header update dates are observations, not freshness evidence.

### Sparse NumPy archives

Sparse NPZ reading supports explicit two-dimensional CSR, CSC and row/column COO
layouts. An independent ZIP parser checks local and central records, exact member
sets, offsets, sizes, CRCs and bounded stored/DEFLATE expansion. NumPy's local
ZIP64 size extra is supported; directory ZIP64, other extras, comments, links,
encryption, data descriptors and unexpected bytes are rejected. NPY 1.0/2.0
headers are inspected as bounded literal ASTs without evaluating them. Supported
members are scalar or one-dimensional with explicit portable numeric dtypes;
object, structured, native-endian, Unicode and unsupported float widths fail.

Shape, coordinate and compressed-pointer domains are checked without allocating
a dense matrix. Storage order, duplicates and explicit zeros are preserved.
Integers use decimal strings; finite real/complex components use exact hexadecimal
float strings plus original bytes. Sorted/unique/canonical observations are
separate. Limits are 8 MB source, five or six members, 16 MB per expanded member,
32 MB total expansion, 8,192-byte NPY headers, 250,000 entries, dimensions through
one billion, a compressed major axis through 100,000 and 1,000 returned entries.
Source hashes cover ZIP bytes; member hashes cover expanded NPY bytes. No pickle,
NumPy/SciPy object construction or extraction occurs. See [NumPy's format
specification](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
and [SciPy sparse NPZ storage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html).

### Price bases and aggregate reconciliation

- Price-basis auditing receives raw, split-adjusted, total-return or undeclared
  records with explicit basis versions, currencies and clocks. Each derivation
  declares `target = source * multiplier`. Exact decimal arithmetic checks that
  equation separately from source identity, factor endpoint metadata, event
  validity and availability. A topological traversal propagates dependency
  invalidity and availability. Missing references and cycles withhold a complete
  availability maximum; a complete maximum includes the record itself, ancestors
  and factors. Vendor records may remain `declared_only`; a matching equation
  does not establish a vendor's adjustment basis or corporate-action history.
  Limits are 5,000 records and 5,000 factors, one parent/factor per derivation,
  38 integral and 18 fractional decimal digits, and 200 returned diagnostics
  and record summaries. Every supplied factor is audited, including unused ones.
- Aggregate reconciliation receives parents, details and explicit memberships.
  Each parent declares expected detail IDs and a weighted sum, weighted mean or
  count. Arithmetic and tolerances are rational. Counts require unit weights;
  sums/means allow nonnegative weights. Repeated memberships, unexpected members
  and null values have explicit policies. Duplicate parent/detail IDs remain
  ambiguous even when contents match. Accepted contributions may yield a partial
  observed result; comparison with the supplied aggregate is withheld until
  memberships resolve. `fully_resolved` describes that resolution independently
  of numeric agreement. Empty sums/counts are zero; zero-weight means are
  undefined. Limits are 1,000 parents, 10,000 details, 20,000 memberships, 20,000
  expected-ID declarations and 200 parent results per page. External population
  completeness and availability are not inferred from arithmetic agreement.

### Declared affine unit conversions

Unit auditing receives dimensional exponent declarations and exact rational
maps `y = scale*x + offset`. Unit names carry no built-in constants. Scales must
be nonzero, endpoint dimensions must agree and duplicate unit names are ambiguous.
A breadth-first spanning forest assigns reference maps using lexical roots and
input edge order. Every usable edge is checked against those maps; disagreements
produce complete tree-path-plus-edge cycle witnesses with nonidentity composition.
Disagreement counts concern edges against this forest, not all possible cycles.

Composed queries resolve only within consistent connected components. The overall
`passed` flag describes the declarations; query outcomes have separate statuses.
Limits are 128 units, 16 dimensional axes per unit, 2,048 conversion edges,
1,000 queries, 100 findings, 20 retained cycles and 512 retained witness edges.
Normalized fractions have a caller-selected bit budget of 128–2,048, default
1,024; exceeding it fails execution. Queries page at 100 results. Physical
constants, nonlinear conversions and economic unit meanings remain unverified.

### Dependence, empirical ranks and positive compositions

- Distance correlation uses unweighted paired vectors and the biased V-statistic.
  Pairwise Euclidean differences begin exactly; rounded norms undergo exact
  integer double centering and matrix products. A direct fourth-root rounding
  step avoids premature underflow in correlation intermediates. Constant
  marginals yield an undefined correlation; positive physical moments that
  underflow are explicitly flagged. The operation accepts 2–128 paired rows,
  with 1–8 dimensions in each marginal. It provides no independence test or
  significance result. The convention follows [Székely, Rizzo and Bakirov,
  definitions 4 and 5](https://arxiv.org/abs/0803.4101).
- Empirical copula construction uses marginal positive-weight mass below a
  value, `L`, tied mass `E` and total mass `W`. Upper ranks are `(L+E)/W`;
  midpoint ranks are `(L+E/2)/W`. There is no `n+1` correction. Zero-weight rows
  receive transformations without contributing to fit. Joint CDF queries use
  exact rational ranks and inclusive comparisons, retaining bounded source-row
  witnesses. Float rank previews may not reproduce exact boundary comparisons.
  Limits are 2,048 observations, 2–16 dimensions, 512 queries and two million
  comparison cells. Tied margins are not certified as uniform. The [empirical
  CDF convention](https://copulae.readthedocs.io/en/latest/api_reference/copulae/empirical/)
  is combined with the explicit weight and tie policies above.
- Compositional log ratios require strictly positive parts, with no zero
  replacement. Closure uses exact sums. Stable adjacent log increments preserve
  small contrasts before CLR centering; ordered Helmert rows produce ILR
  balances. The reported basis and coordinates reconstruct compositions for
  error diagnostics. The geometric center uses mean CLR. Pairwise log-ratio
  variation divides by the observation count and centers each exact part ratio
  against the first row before taking logarithms, preserving small changes of
  large ratios. Binary exponent reduction avoids overflowing intermediate
  ratios. Limits are 1,000 rows, 2–32 parts, 20,000 row-part cells, 500,000
  squared-dimension work cells and parts in `[1e-100, 1e100]`. Logs, roots,
  exponentials and the returned basis remain approximate. Common part universes,
  fitting-block availability and statistical assumptions are not verified. See
  [CLR conventions](https://scikit.bio/docs/latest/generated/skbio.stats.composition.clr.html)
  and [Helmert matrices](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.helmert.html).

### Kernel density and irregular-time harmonic fits

- Gaussian KDE uses the caller's bandwidth as each kernel's standard deviation
  in input units; it does not fit bandwidth. Positive weights form a normalized
  mixture, and repeated query locations/order are retained. Log-sum-exp density
  and tail evaluation preserve logs when ordinary values underflow. Normal tails
  use `erfc` or a bounded Mills expansion; complements use `log1p`/`expm1`.
  Boundary rounding, density underflow and tiny quadratic terms are reported.
  Limits are 1,024 samples, 512 queries, 65,536 sample-query cells, bandwidth in
  `[1e-100, 1e100]`, and separation of at most one million bandwidths. No kernel
  is truncated. This is a fitted mixture, not a forecast score or calibration
  finding. See [Gaussian kernel mixtures](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
  and the [normal-tail asymptotic expansion](https://dlmf.nist.gov/7.12).
- Lomb–Scargle harmonic fits use supplied increasing timestamps and frequencies.
  The intercept is floating, fixed at the sample mean or zero. Exact cycle
  reduction precedes trigonometric evaluation; exact rational normal equations
  solve the rounded numerical design. Zero or insufficient Gram determinant
  produces an undefined fit. Power measures the fraction of baseline MSE removed
  by that solution. Separate residual MSE uses the rounded coefficients actually
  returned and may show negative improvement. Limits are 4,096 observations,
  512 frequencies, 131,072 observation-frequency cells, one million cycles over
  the time span and 65,536-bit arithmetic. No frequency-selection, false-alarm
  probability or significance claim is made. See [Zechmeister and Kürster's
  floating-mean formulation](https://arxiv.org/abs/0901.2573).

### Piecewise-hazard likelihood scoring

Hazards are supplied on right-continuous intervals starting at zero, with a
positive final rate on the infinite tail. Exact rational integration gives
cumulative hazard `H`. Entry time `e` conditions on survival to entry. Losses are
`H(t)-H(e)-log(h(t))` for exact events, `H(t)-H(e)` for right censoring, and
`H(l)-H(e)-log(1-exp(-(H(u)-H(l))))` for intervals `l < T <= u`. Exact events
at knots use the right-hand hazard. `expm1`/`log1p` stabilize interval terms.

Zero event hazard or zero interval mass yields infinite-loss status and a null
aggregate value; no finite-only aggregate is substituted. Negligible interval
tail underflow is disclosed, and execution fails if it would erase the entire
positive loss. Limits are 1,000 observations, 256 intervals and 100,000
row-interval cells. Positive times/rates lie in `[1e-100, 1e100]`; zero is
supported except for the final hazard. Weights are strictly positive. Exact-event
density losses may be negative and depend on the required time-unit declaration.
The observation-process likelihood is excluded; proper likelihood interpretation
requires noninformative censoring/entry and outcome-independent weighting.
Those assumptions and forecast timing remain unverified. See the [censored
likelihood identity](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html).

### CBOR, MessagePack and BSON readers

These independent byte parsers validate their entire supported input before
returning preorder node pages. Each caps input at 2 MB, node count at 20,000,
depth at 32, container membership at 10,000, ordinary string/binary payloads at
65,536 bytes, page size at 100 nodes and encoded pages at 512 KB. Original byte
spans and source SHA-256 are retained. Integers use exact decimal strings;
finite floats use hexadecimal values and original bits. Container and scalar
kinds distinguish semantic nulls. No object deserialization or code execution
occurs; format subsets and limits are explicit.

- [CBOR](https://www.rfc-editor.org/rfc/rfc8949.html) supports definite and
  indefinite strings, arrays and maps, unsigned/negative integers, finite
  binary16/32/64, booleans, null, undefined and simple-value codes. All tags are
  rejected. Map keys must be unique definite UTF-8 strings of at most 256 bytes.
  String chunks must be valid individually. Work is bounded at 50,000 headers.
  Non-shortest integer/length arguments are counted; deterministic serialization
  and float minimization are not certified.
- [MessagePack](https://github.com/msgpack/msgpack/blob/master/spec.md) supports
  core types, standard timestamp32/64/96 forms, and opt-in opaque application
  extensions. Reserved negative extension types are rejected. Timestamps retain
  exact seconds and bounded nanoseconds without calendar conversion. Maps require
  unique UTF-8 string keys of at most 256 bytes; at most 40,000 value/key headers
  are parsed. Non-minimal core encodings are accepted without a canonicality claim.
- [BSON 1.1](https://bsonspec.org/spec.html) supports ordered documents, sequentially
  indexed arrays, ordinary numeric/string scalars, ObjectIds, temporal fields,
  min/max keys, uncompiled regex and generic/UUID/MD5 binary bytes. Nested lengths
  and terminators are checked against their parent bounds. Executable/deprecated
  types, decimal128 and unsupported binary subtypes are rejected. Regex options
  are a sorted unique subset of `imsux`; regex syntax is not evaluated. Database
  insertion compatibility is not inferred from successful parsing.

### Heterogeneous Bernoulli counts

Poisson-binomial construction multiplies each supplied Bernoulli polynomial
`(1-p) + p*z` using integer coefficients and a common power-of-two denominator.
Unequal probabilities and exact zero/one trials are supported without odds
division, simulation or distributional approximation. Integer PMF numerators
and the shared denominator are authoritative; numerical PMF, CDF, strict tails
and logarithms have explicit underflow/boundary indicators. Quantiles use exact
CDF comparisons. Level zero selects the minimum positive-mass support point.

Limits are 256 trials, 256 quantile queries and an 8,192-bit common denominator,
checked before convolution. The operation emits the complete finite support.
Independence and forecast timing remain caller assumptions; this construction
does not establish calibration. See [the convolution approach](https://doi.org/10.1016/j.csda.2018.01.007).

### Markov path likelihoods

Markov scoring receives a fixed state universe, initial and transition masses,
and fully or partially observed paths. Each mass row is normalized exactly.
An observation is a set of possible states or null for unobserved. Exact forward
prediction, restriction and conditioning sum over compatible paths without
enumerating them. Scores sum negative log evidence probabilities. Full-state
and missing observations contribute zero; zero evidence gives infinite path
loss, after which conditional distributions remain undefined.

The weighted mean uses whole path scores without dividing by path length.
The transition model is time homogeneous over caller-defined discrete steps.
Coarsened-observation scoring requires a prespecified observation partition or
noninformative observation process; lengths, weights, state-model assumptions
and timing remain unverified. Limits are 2–16 states, 64 paths, 256 positions per
path, 1,024 total positions, 131,072 state-pair work cells and 16,384-bit exact
fractions. All paths are scored before returning at most 200 trace positions.
Probability preview underflows are counted. See [Markov path probabilities](https://data140.org/textbook/content/chapter-10/transitions/).

### Gaussian mixture forecast scores

Supplied normal mixtures may have different component means, standard deviations
and masses. The scorer normalizes masses and computes log density, CRPS, PIT and
posterior component responsibilities without fitting parameters. CRPS uses the
closed-form absolute-moment identity. Exact location differences and weighted
arithmetic preserve the atomic distance terms; Gaussian excess corrections use
scaled logarithms, `erfc` and a bounded Mills expansion. Combined standard
deviations and transcendental calculations remain approximate. Nonpositive
computed CRPS fails instead of being clipped. Underflow and PIT boundary rounding
are reported.

Limits are 256 forecasts, 32 components each, 32,768 component-pair work cells
and 131,072-bit rational accumulators, including aggregation across rows.
Positive scales, component masses and observation weights lie in
`[1e-100, 1e100]`; every observation/component or component-pair comparison is
within one million standard deviations. Zero component masses are supported.
The required measurement unit makes density-score interpretation explicit;
negative log densities may be negative. Component selection, observation weights
and forecast timing remain unverified. See [normal-mixture CRPS](https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_mixnorm.html).

### Segmentation and dynamic time warping

- Mean-change segmentation minimizes within-segment squared errors plus the
  supplied penalty per change. Exact integer prefix moments and rational dynamic
  programming enforce contiguous full coverage and minimum segment lengths.
  At each prefix, equal objectives prefer fewer segments, then an earlier start
  of the final segment; previously chosen prefix ties remain fixed. Returned
  means have a separate residual-error calculation. Limits are 256 observations,
  32,896 candidate segments and 16,384-bit fractions. This fits the complete
  block; no online detection or significance claim is made. See [optimal
  partitioning](https://www.lancaster.ac.uk/~romano/teaching/2425MATH337/3_multiple_changes.html).
- Dynamic time warping minimizes the sum of normalized, coordinate-weighted
  squared Euclidean cell costs. The path includes both sequences' first and last
  indices. Steps advance both sequences, the left sequence or the right sequence,
  in that tie priority. An optional absolute index band uses `abs(i-j)` without
  length rescaling. Integer arithmetic retains exact costs and ties; reconstructed
  source-index pairs are checked against the objective. Average path cost is
  descriptive and does not alter optimization. No feasible path has a separate
  status. Limits are 256 points per sequence, eight dimensions, 16,384 grid cells
  and 65,536 coordinate cells. Units, ordering and availability remain caller
  responsibilities. See [the DTW recurrence and endpoints](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html).

### Singular spectrum analysis

SSA forms a Hankel trajectory matrix after exact power-of-two normalization.
NumPy's thin SVD is the decomposition primitive; this implementation owns
eigentriple grouping, anti-diagonal averaging and reconstruction. Caller groups
must be disjoint, need not cover all components, and use zero-based descending
singular indices. The numerical-rank threshold is diagnostic and does not remove
selected components. Returned group series determine the selected reconstruction
and residuals. A separate reconstruction from all SVD components exposes
decomposition and rounding error.

Limits are 512 samples, a window through 64, 16,384 trajectory cells, 524,288
component-matrix work cells, 16 groups, 8,192 returned group samples and
16,384-bit rational post-processing. Unsupported normalization dynamic range
fails. Repeated singular subspaces may rotate, so splitting them between groups
does not define a unique or platform-reproducible basis. Singular-energy shares
are not unweighted time-series variance explained. The complete block is used;
spacing, point-in-time selection and signal/noise interpretations are unverified.
See [basic SSA](https://ssa.cf.ac.uk/zhigljavsky/changepoint/Methodology/node2.html)
and [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html).

### Transaction and observation declarations

- Transaction auditing treats input order as a contiguous log prefix and checks
  nondecreasing recorded clocks. Batches move from begin through writes to one
  commit or abort. Commit membership, hash labels, duplicate writes and optional
  write order are checked against explicit declarations. Checkpoints name the
  logged commit set through a sequence; their comparison is withheld when log
  order is invalid. A commit with membership errors remains a logged commit,
  separately from transaction validity. External durability claims receive only
  reference/clock checks. No storage durability, atomicity, isolation or payload
  authenticity is established. Limits are 2,000 batches, 20,000 events and
  expected members, 200 checkpoints with 20,000 total claimed IDs, 2,000 durability
  claims and 200 diagnostics. Batch and checkpoint results page independently.
- Observation-window auditing uses explicit delayed entry, observed and planned
  endpoints, exposure intervals and terminal-event/right-censoring declarations.
  Window availability publishes the finalized outcome and must follow the
  observed end and every used observation's availability, before the decision.
  Half-open exposure intervals are clipped for exact microsecond union coverage;
  out-of-window portions remain errors. Internal gaps, overlapping duration and
  multiplicity excess are distinct. Point events may occur at either observed
  endpoint. A declared terminal event must be the unique eligible event at the
  observed end; right censoring requires a reason and no eligible event. Zero
  duration is explicit. Limits are 5,000 windows, 10,000 linked observations,
  200 diagnostics and 200 window results per page. No calendar/grid is inferred,
  and absence of unreported events or independent censoring is not verified.

### Staged dependency releases

Release manifests specify exact component/version identities, a stage pipeline,
stage clocks and exact version/stage dependencies. A prior stage must be available
before the next starts. Dependencies must be observable, effective and not
superseded at consumer-stage start. Stage-level graph traversal preserves
interleaved workflows and propagates invalidity and availability. Artifact bytes
and real deployments are not checked.

Supersession is explicit and permanent after a declared terminal pipeline activates;
expiry of a replacement does not revive a retired version. Dependency-invalid
replacements do not silently restore older versions. Queries request an exact
version or require exactly one eligible version, without inferring version order.
Invalid or ambiguous observable stage declarations remain candidates, preventing
silent fallback. Complete availability maxima require valid resolved chains.
The overall `passed` flag concerns declarations; query outcomes have their own
statuses. Limits are 3,000 manifests, 16 stages, 10,000 milestones, 20,000
dependencies, 10,000 supersedes declarations, 1,000 queries and 500,000 candidate
version examinations. Diagnostics and query pages hold at most 200 results.

### Avro, HDF5 and FITS

- Avro reading supports flat primitive records in object containers with null
  or raw DEFLATE codecs. It implements metadata-map blocks, canonical zigzag
  integers, writer-schema validation, record decoding, block extents and sync
  checks. Unions, nested/named/logical types and unsupported schema attributes
  are rejected. Exact cells retain expanded-block encodings and row spans;
  absolute source offsets exist for uncompressed blocks. Every block is checked
  before pagination. Limits are 8 MB source, 32,768 schema bytes, 64 fields,
  50,000 records, one million cells, 1,000 blocks, 8 MB expansion per block,
  32 MB total expansion and a 512 KB page. Header-only empty containers are
  explicit. Sync markers provide framing checks, not authentication. See the
  [Avro specification](https://avro.apache.org/docs/1.12.0/specification/).
- HDF5 inspection supports checksummed superblock versions 2 and 3. Signature
  locations, field widths, lookup3 checksum, recorded base, absolute EOF and
  relative root/extension addresses are checked under a single-file convention.
  Relocated bases and versions 0/1 are unsupported. Header mode accepts files
  through 1 TB with at most 1,024 bytes read; full mode is bounded at 8 MB.
  Superblock and whole-source hashes have separate scopes. Extra physical bytes
  after declared EOF are reported. Object graphs, datasets, external mappings,
  recovery/committed state and atomic snapshots remain unverified. See the
  [HDF Group format specification](https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html).
- FITS reading requires an empty primary HDU followed by one scalar ASCII or
  binary table. It validates card order, dimensions, columns, row extents and
  padding across the complete source. Heaps, images, extra HDUs, vectors and
  compressed tables are unsupported. Exact values, raw cell bytes, null
  declarations and scaling metadata are preserved. Nonidentity scaling requires
  explicit `preserve_raw`; physical scaling is never applied. Declared FITS
  checksums remain unverified while source SHA-256 covers all input bytes.
  Limits are 8 MB, 1,024 cards per header, 64 columns, 100,000 rows, one million
  cells, 8,192 bytes per row, 256 bytes per field and a 512 KB page. Zero-row
  tables are explicit. See [FITS Standard 4.0](https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf).

### Migration routing and byte-assembly declarations

- Migration plans declare dataset identity, key namespace, half-open integer
  scope, old/new owners and an event-time cutover. Visible queries before that
  event cutover require the old source; later events require the new source.
  Decision time controls metadata visibility. Missing, invalid or competing
  candidates retain explicit statuses. Old-source coverage remains required
  for historical routing; new-source gaps are pending readiness information
  until cutover. Same-side overlap and dual-source coverage are distinct.
  Structural and duplicate-ID checks concern all supplied declarations,
  including future metadata. Coverage and candidate selection use separate
  visible identity/validity indexes; future duplicate IDs cannot veto a route.
  A resolved local query does not certify full-plan coverage or source records.
  Limits are 128 plans, 10,000 partitions, 1,000 queries and 500,000 candidate
  comparisons, with at most 200 findings and 200 query results per page.
- Byte assembly declares exact source object/version references and copy extents,
  target offsets, positive part lengths, ordinal concatenation and publication.
  Exact interval sweeps measure target gaps, overlap, excess and multiplicity
  without enumerating bytes. Source and part metadata must precede assembly
  publication; future publications remain pending at the decision. Hash labels
  are compared where byte identity is declared: whole-source parts, single-part
  assemblies and the empty-message hash. Multipart hashes are not combined into
  a claimed assembly digest. No file bytes, atomic publication or storage state
  are verified. Limits are 5,000 objects, 1,000 assemblies, 20,000 parts and
  expected slots, signed-64-bit size/offset declarations, 200 findings and 200
  assembly results per page. All supplied declarations are audited.

### Snapshot state reconstruction

Snapshot manifests declare complete inventories, one parent identity and at most
one change per object. Each delta asserts the exact parent version, with null
meaning absence. Deletes require live prior objects; replacement and resurrection
introduce a different version. Tombstones remain in subsequent inventories until
changed explicitly. Own graph traversal applies valid deltas, compares the full
derived inventory and propagates invalid or cyclic ancestry. Queries preserve
both the current inventory row and the root-entry/change row that introduced
the object state.

Repeated object/version identities must agree in state, size, hash label and
availability. Source-state and parent publication clocks cannot regress. Global
audit summaries concern every supplied declaration. Historical queries use
separate visible-manifest reconstructions so later identity conflicts do not
retroactively change earlier results. Limits are 500 snapshots, 20,000 inventory
entries, 20,000 changes, 1,000 queries and 16 distinct query clocks. Global and
historical reconstruction share a maximum 500,000 declared state-work budget,
conservatively charging the full inventory work for each requested clock;
diagnostics and query pages hold at most 200 rows. Root completeness is declared,
and storage bytes, atomicity and authenticated history remain unverified.

### Finite-state, Gaussian-state and conductance models

- Markov analysis normalizes supplied transition weights exactly, resolves
  communicating classes from positive support, and treats each closed class as
  an entry target. Own rational elimination computes transient occupation counts,
  closed-class probabilities and entry-time moments. Occupation includes the
  initial transient state and stops before first closed-class entry. A target
  with probability below one has infinite unconditional hitting time; conditional
  time is undefined for an unreachable target. Starting within a closed class
  gives zero entry time. Limits are 16 states, positive weights in [1e-6, 1e6]
  and 16,384-bit rational intermediates. Probability previews flag boundary
  rounding. The model is supplied, time homogeneous and unverified. See
  [Grinstead and Snell, chapter 11](https://math.dartmouth.edu/~prob/prob/prob.pdf).
- Linear Gaussian filtering accepts an explicit prior at state index -1 and
  time-varying transition, observation, offset and noise arrays. Every step first
  predicts, including step zero. Missing observations select the corresponding
  measurement covariance submatrix; completely missing steps only predict. Own
  rational matrix recursions and LDL solves require symmetric PSD initial/process
  covariance and PD measurement covariance. Optional RTS smoothing uses all
  observations and requires PD predicted covariances for its solves. Returned
  covariance matrices separately report exact rank and whether rounding preserved
  PSD. No jitter or repair is applied. Limits are 32 steps, dimensions through
  four, input magnitudes at most 1e6 and 32,768-bit rational intermediates.
  The caller supplies any irregular-time transition/noise scaling; independence,
  Gaussian assumptions and timing remain unverified. See
  [Särkkä, Bayesian Filtering and Smoothing](https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf).
- Graph resistance uses positive undirected conductances and own grounded
  Laplacian factorization. Parallel conductances add; self-loops are rejected.
  Each component grounds its smallest input vertex. Requested pairs return exact
  effective resistance and optional voltage previews; disconnected pairs have
  infinite resistance, and identical vertices have zero. Exact system residuals
  and pair energy are checked. Optional determinant-based tree mass sums products
  of conductances, including parallel-edge alternatives. Singletons have empty-tree
  mass one; a disconnected whole graph has tree mass zero. Limits are 20 vertices,
  128 edges, 200 queries, conductances in [1e-6, 1e6] and 12,000-bit rationals.
  No observed-network or causal claim follows. See
  [Spielman's electrical network notes](https://www.cs.yale.edu/homes/spielman/561/lect14-18.pdf).

### Exact transport, planar geometry and curve reconstruction

- Discrete transport normalizes each supplied mass vector separately and solves
  a complete bipartite transport problem with own integer residual-network
  augmentation. Reverse arcs can revise earlier assignments. Returned positive
  flows satisfy exact marginals; final potentials check dual feasibility,
  complementary slackness and zero objective gap. Arbitrary nonnegative costs
  need not be distances. Deterministic ties do not establish uniqueness. Limits
  are 16 source and 16 target masses, 2,048 augmentations, two million edge
  examinations and 12,000-bit integers. Budget exhaustion rejects the request.
  Exact rational outputs include flagged float previews. See the
  [MIT minimum-cost flow notes](https://courses.csail.mit.edu/6.854/21/Scribe/s9-minCostFlow/s9-minCostFlow.html).
- Planar hull construction uses exact orientation predicates, removes duplicate
  coordinates and collinear interior boundary points, and preserves source rows
  for each extreme vertex. Vertices run counterclockwise from the smallest
  lexicographic coordinate. Exact area and area centroid accompany an approximate
  perimeter formed from rounded square roots. A segment has zero area, no area
  centroid and perimeter twice its length; a point has zero perimeter. Query
  classification distinguishes boundary, interior and exterior. Limits are
  1,024 input points and 256 queries, with coordinate magnitudes at most 1e100.
  No coordinate system, statistical confidence region or timing is inferred.
  See the [CGAL hull algorithm overview](https://doc.cgal.org/latest/Convex_hull_2/index.html).
- Cubic reconstruction computes weighted harmonic interior slopes, limited
  endpoint slopes and normalized interval polynomials. Query values, first and
  second derivatives, and signed integrals use exact rational calculations
  before rounded output; nonzero underflow is flagged. At interior knots the
  second derivative comes from the right interval, and at the final knot from
  the left. Extrapolation and repeated x values are rejected. Limits are 256
  knots, 512 queries, 32 integral ranges and 12,000-bit rational intermediates.
  This uses the complete supplied grid and does not establish causal forecasts
  or data availability. The slope convention follows
  [Fritsch–Butland PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

### WAVE, PLY and TIFF inspection

- WAVE parsing independently reads little-endian RIFF chunks, PCM 8/16/24/32-bit
  samples and IEEE 32/64-bit samples. Supported extensible GUIDs require full-width
  valid bits and consistent speaker masks. Bare IEEE requires its extension-size
  field and a fact chunk; the fact count is reported with its agreement to stored
  frames. Exact integer/float-hex values retain raw bytes, channel/frame indices
  and byte offsets. No playback, resampling or normalization occurs. Nested audio
  containers and compressed/RF64/BW64 formats are unsupported. Limits are 8 MB,
  256 chunks, 32 channels, one million scalar samples and 200 frames per 512 KB
  page. Every stored sample is validated. Ancillary semantics remain opaque.
  See [Microsoft WAVEFORMATEX](https://learn.microsoft.com/en-us/windows/win32/api/mmreg/ns-mmreg-waveformatex).
- PLY reading supports version-one ASCII and both binary byte orders, bounded
  scalar/list properties, exact integers, decimal ASCII real tokens and binary
  float-hex values. Conventional face-index lists must refer to declared vertices;
  repeated indices remain explicit. No triangulation, manifold, winding or unit
  validity is claimed. ASCII real tokens retain decimal precision without binary
  rounding. All records are validated before pagination, with raw property spans.
  Limits are 8 MB, 64 KB headers, 32 element types, 256 total properties, 100,000
  records, one million property occurrences, two million scalar reads and 4,096
  items per list. Pages hold at most 200 records and 512 KB. See
  [Greg Turk's PLY reference archive](https://sites.cc.gatech.edu/projects/large_models/files/ply.tar.gz).
- TIFF inspection follows the classic directory graph, next/SubIFD links and
  explicit IFD-type values. It checks cycles, sorted tags, aligned extents and
  metadata overlap while allowing shared child directories. Known EXIF/GPS/
  Interoperability pointer conventions and BigTIFF are unsupported. Primitive
  observations preserve rational zero denominators and nonfinite floating bits;
  private tag semantics remain unverified. Strip/tile offset-count pairs receive
  bounds checks; image schema, decoding, payload overlap and full-file coverage
  remain unchecked. Limits are 8 MB, 128 directories, 2,048 tags, 1,024 graph
  edges, two MB of summed value bytes, 131,072 values and 4,096 payload references.
  All reachable supported metadata is checked before 100-tag/512 KB pages. See
  [Adobe TIFF 6.0](https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf).

## Delivery status

See [FX1_CAPABILITY_PROGRESS.md](FX1_CAPABILITY_PROGRESS.md) for counts and
validation status. The million-implementation and million-LOC requirements
remain open. The full batch has not yet been behaviorally verified.
