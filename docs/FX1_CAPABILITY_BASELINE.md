# FX-1 capability baseline — Phase 0 snapshot

**Generated:** 2026-10-04T13:10:11+00:00  
**Commit:** `main` @ `0e6f5a1b10`  
**Tree:** 6 staged · 15 modified · 26 untracked  
**Reproduce:** `uv run python scripts/fx1_capability_baseline.py --output-json artifacts/fx1_capability_baseline.json --output-markdown docs/FX1_CAPABILITY_BASELINE.md`

Phase 0 of `docs/ULTRA_INTENSE_CAPABILITY_PURSUIT_PLAN.md`. Read-only inventory;
no capability file was modified to produce this snapshot.

## Frozen counting rules

| Measure | Definition |
|---|---|
| `physical_lines` | len(text.splitlines()); comments and docstrings included |
| `nonblank_lines` | physical lines that are not whitespace-only |
| `code_token_lines` | lines carrying a real token, excluding blanks, comments and AST-identified docstrings (canonical tool's source_lines) |
| `accepted_capability` | module exporting OPERATION whose id is a key of registry._IMPLEMENTATIONS |
| `orphan` | module exporting OPERATION that is absent from _IMPLEMENTATIONS; unreachable via the harness and never counted as accepted |
| `loc_scope` | registered implementation files only; infrastructure, tests, documentation and generated artifacts excluded — matching the progress ledger's stated scope |
| `measurement_source` | scripts/code_quality_inventory.measure_file (reused, not reimplemented) |

## Inventory

`src/fx1/operations/` holds **173** `.py` files:

- **169** registered implementations (accepted candidates)
- **0** orphan modules — export `OPERATION`, absent from the registry
- **4** infrastructure: `__init__.py`, `_numeric.py`, `base.py`, `registry.py`

Registry `_IMPLEMENTATIONS` holds **169** ids: feature=53, plugin=37, skill=79.

## LOC (registered implementation files only)

Scope matches the progress ledger: shared infrastructure, tests, documentation and
generated artifacts excluded.

| Measure | Registered | Orphans | All operation modules |
|---|---:|---:|---:|
| physical lines | 42,930 | 0 | 42,930 |
| nonblank lines | 38,345 | 0 | 38,345 |
| code-token lines | 35,668 | 0 | 35,668 |
| files | 169 | 0 | 169 |

## Reconciliation — every mismatch classified

- Registry ids with no source file: **none** — every registered id resolves to a module.

- Source modules not in the registry: **none**.

### Doc claims vs measured

| Doc | Claim | Claimed | Measured | Verdict |
|---|---|---:|---:|---|
| `docs/FX1_CAPABILITIES.md`:8 | implementations | 169 | 169 | match |
| `docs/FX1_CAPABILITIES.md`:9 | separate files | 169 | 169 | match |
| `docs/FX1_CAPABILITY_PROGRESS.md`:15 | code-token lines | 35,668 | 35,668 | match |
| `docs/FX1_CAPABILITY_PROGRESS.md`:13 | features | 53 | 53 | match |
| `docs/FX1_CAPABILITY_PROGRESS.md`:15 | nonblank lines | 38,345 | 38,345 | match |
| `docs/FX1_CAPABILITY_PROGRESS.md`:14 | physical lines | 42,930 | 42,930 | match |
| `docs/FX1_CAPABILITY_PROGRESS.md`:13 | plugins | 37 | 37 | match |
| `docs/FX1_CAPABILITY_PROGRESS.md`:13 | separate files | 169 | 169 | match |
| `docs/FX1_CAPABILITY_PROGRESS.md`:13 | skills | 79 | 79 | match |
| `docs/FX1_OPERATIONS.md`:3 | implementations | 169 | 169 | match |

## Verification evidence (Phase 1 input)

- Registered ids **with** a hand-checked invocation case: **169**
- Registered ids **without** one: **0**
- Cases referencing ids that are not registered: **0**
- Gate: `tests/fx1/test_operations_registry_invocation.py::test_invocation_cases_match_the_registry`

## Verification matrix

Harness path per capability. `case` = a hand-checked invocation case exists.

| # | Operation id | Kind | Ver | Class | Case | phys | nonblank | code | Harness command |
|---:|---|---|---|---|:--:|---:|---:|---:|---|
| 1 | `skills.align_event_sequences` | skill | 1.0.0 | registered | yes | 285 | 256 | 241 | `fx1 harness execute-operation skills.align_event_sequences` |
| 2 | `skills.assign_horizon_targets` | skill | 1.0.0 | registered | yes | 238 | 208 | 196 | `fx1 harness execute-operation skills.assign_horizon_targets` |
| 3 | `skills.audit_artifact_assembly` | skill | 1.0.0 | registered | yes | 409 | 372 | 352 | `fx1 harness execute-operation skills.audit_artifact_assembly` |
| 4 | `skills.audit_bar_integrity` | skill | 1.0.0 | registered | yes | 131 | 110 | 107 | `fx1 harness execute-operation skills.audit_bar_integrity` |
| 5 | `skills.audit_cross_field_contracts` | skill | 1.0.0 | registered | yes | 252 | 219 | 211 | `fx1 harness execute-operation skills.audit_cross_field_contracts` |
| 6 | `skills.audit_dependency_releases` | skill | 1.0.0 | registered | yes | 560 | 518 | 489 | `fx1 harness execute-operation skills.audit_dependency_releases` |
| 7 | `skills.audit_duplicate_keys` | skill | 1.0.0 | registered | yes | 153 | 129 | 126 | `fx1 harness execute-operation skills.audit_duplicate_keys` |
| 8 | `skills.audit_effective_intervals` | skill | 1.0.0 | registered | yes | 214 | 189 | 183 | `fx1 harness execute-operation skills.audit_effective_intervals` |
| 9 | `skills.audit_feature_provenance` | skill | 1.0.0 | registered | yes | 338 | 300 | 294 | `fx1 harness execute-operation skills.audit_feature_provenance` |
| 10 | `skills.audit_forecast_panel` | skill | 1.0.0 | registered | yes | 353 | 317 | 306 | `fx1 harness execute-operation skills.audit_forecast_panel` |
| 11 | `skills.audit_join_fanout` | skill | 1.0.0 | registered | yes | 256 | 233 | 226 | `fx1 harness execute-operation skills.audit_join_fanout` |
| 12 | `skills.audit_lineage_graph` | skill | 1.0.0 | registered | yes | 290 | 259 | 249 | `fx1 harness execute-operation skills.audit_lineage_graph` |
| 13 | `skills.audit_missingness` | skill | 1.0.0 | registered | yes | 160 | 143 | 136 | `fx1 harness execute-operation skills.audit_missingness` |
| 14 | `skills.audit_missingness_association` | skill | 1.0.0 | registered | yes | 156 | 137 | 131 | `fx1 harness execute-operation skills.audit_missingness_association` |
| 15 | `skills.audit_monotonic_sequences` | skill | 1.0.0 | registered | yes | 233 | 208 | 201 | `fx1 harness execute-operation skills.audit_monotonic_sequences` |
| 16 | `skills.audit_nested_folds` | skill | 1.0.0 | registered | yes | 444 | 406 | 392 | `fx1 harness execute-operation skills.audit_nested_folds` |
| 17 | `skills.audit_observation_windows` | skill | 1.0.0 | registered | yes | 448 | 410 | 384 | `fx1 harness execute-operation skills.audit_observation_windows` |
| 18 | `skills.audit_panel_gaps` | skill | 1.0.0 | registered | yes | 167 | 145 | 139 | `fx1 harness execute-operation skills.audit_panel_gaps` |
| 19 | `skills.audit_partition_ranges` | skill | 1.0.0 | registered | yes | 372 | 341 | 329 | `fx1 harness execute-operation skills.audit_partition_ranges` |
| 20 | `skills.audit_point_in_time` | skill | 1.0.0 | registered | yes | 122 | 102 | 97 | `fx1 harness execute-operation skills.audit_point_in_time` |
| 21 | `skills.audit_price_basis` | skill | 1.0.0 | registered | yes | 452 | 409 | 384 | `fx1 harness execute-operation skills.audit_price_basis` |
| 22 | `skills.audit_referential_integrity` | skill | 1.0.0 | registered | yes | 265 | 241 | 235 | `fx1 harness execute-operation skills.audit_referential_integrity` |
| 23 | `skills.audit_release_revisions` | skill | 1.0.0 | registered | yes | 406 | 369 | 348 | `fx1 harness execute-operation skills.audit_release_revisions` |
| 24 | `skills.audit_resampling_groups` | skill | 1.0.0 | registered | yes | 324 | 290 | 276 | `fx1 harness execute-operation skills.audit_resampling_groups` |
| 25 | `skills.audit_resource_reservations` | skill | 1.0.0 | registered | yes | 529 | 481 | 453 | `fx1 harness execute-operation skills.audit_resource_reservations` |
| 26 | `skills.audit_revision_conflicts` | skill | 1.0.0 | registered | yes | 190 | 168 | 165 | `fx1 harness execute-operation skills.audit_revision_conflicts` |
| 27 | `skills.audit_schema_compatibility` | skill | 1.0.0 | registered | yes | 287 | 259 | 240 | `fx1 harness execute-operation skills.audit_schema_compatibility` |
| 28 | `skills.audit_schema_drift` | skill | 1.0.0 | registered | yes | 173 | 152 | 146 | `fx1 harness execute-operation skills.audit_schema_drift` |
| 29 | `skills.audit_session_coverage` | skill | 1.0.0 | registered | yes | 348 | 312 | 297 | `fx1 harness execute-operation skills.audit_session_coverage` |
| 30 | `skills.audit_snapshot_manifests` | skill | 1.0.0 | registered | yes | 605 | 549 | 519 | `fx1 harness execute-operation skills.audit_snapshot_manifests` |
| 31 | `skills.audit_source_chains` | skill | 1.0.0 | registered | yes | 431 | 401 | 385 | `fx1 harness execute-operation skills.audit_source_chains` |
| 32 | `skills.audit_source_coverage` | skill | 1.0.0 | registered | yes | 238 | 210 | 204 | `fx1 harness execute-operation skills.audit_source_coverage` |
| 33 | `skills.audit_source_migrations` | skill | 1.0.0 | registered | yes | 491 | 448 | 426 | `fx1 harness execute-operation skills.audit_source_migrations` |
| 34 | `skills.audit_split_leakage` | skill | 1.0.0 | registered | yes | 268 | 239 | 232 | `fx1 harness execute-operation skills.audit_split_leakage` |
| 35 | `skills.audit_stream_offsets` | skill | 1.0.0 | registered | yes | 407 | 373 | 361 | `fx1 harness execute-operation skills.audit_stream_offsets` |
| 36 | `skills.audit_task_leases` | skill | 1.0.0 | registered | yes | 465 | 416 | 390 | `fx1 harness execute-operation skills.audit_task_leases` |
| 37 | `skills.audit_temporal_constraints` | skill | 1.0.0 | registered | yes | 334 | 295 | 279 | `fx1 harness execute-operation skills.audit_temporal_constraints` |
| 38 | `skills.audit_temporal_resamples` | skill | 1.0.0 | registered | yes | 474 | 435 | 417 | `fx1 harness execute-operation skills.audit_temporal_resamples` |
| 39 | `skills.audit_transaction_batches` | skill | 1.0.0 | registered | yes | 464 | 421 | 399 | `fx1 harness execute-operation skills.audit_transaction_batches` |
| 40 | `skills.audit_unit_conversions` | skill | 1.0.0 | registered | yes | 442 | 389 | 369 | `fx1 harness execute-operation skills.audit_unit_conversions` |
| 41 | `skills.audit_vintage_transitions` | skill | 1.0.0 | registered | yes | 291 | 260 | 254 | `fx1 harness execute-operation skills.audit_vintage_transitions` |
| 42 | `features.bipower_variation` | feature | 1.0.0 | registered | yes | 95 | 79 | 66 | `fx1 harness execute-operation features.bipower_variation` |
| 43 | `skills.build_availability_frontier` | skill | 1.0.0 | registered | yes | 245 | 215 | 205 | `fx1 harness execute-operation skills.build_availability_frontier` |
| 44 | `skills.build_block_resamples` | skill | 1.0.0 | registered | yes | 134 | 117 | 98 | `fx1 harness execute-operation skills.build_block_resamples` |
| 45 | `skills.build_conformal_intervals` | skill | 1.0.0 | registered | yes | 151 | 132 | 120 | `fx1 harness execute-operation skills.build_conformal_intervals` |
| 46 | `skills.build_group_stratified_folds` | skill | 1.0.0 | registered | yes | 298 | 269 | 242 | `fx1 harness execute-operation skills.build_group_stratified_folds` |
| 47 | `skills.build_stationary_resamples` | skill | 1.0.0 | registered | yes | 149 | 132 | 109 | `fx1 harness execute-operation skills.build_stationary_resamples` |
| 48 | `skills.build_stratified_folds` | skill | 1.0.0 | registered | yes | 178 | 155 | 132 | `fx1 harness execute-operation skills.build_stratified_folds` |
| 49 | `skills.build_walk_forward_folds` | skill | 1.0.0 | registered | yes | 210 | 184 | 176 | `fx1 harness execute-operation skills.build_walk_forward_folds` |
| 50 | `features.burg_autoregression` | feature | 1.0.0 | registered | yes | 230 | 205 | 174 | `fx1 harness execute-operation features.burg_autoregression` |
| 51 | `features.circular_summary` | feature | 1.0.0 | registered | yes | 237 | 215 | 191 | `fx1 harness execute-operation features.circular_summary` |
| 52 | `features.compositional_logratios` | feature | 1.0.0 | registered | yes | 247 | 220 | 194 | `fx1 harness execute-operation features.compositional_logratios` |
| 53 | `features.convex_hull_2d` | feature | 1.0.0 | registered | yes | 198 | 170 | 161 | `fx1 harness execute-operation features.convex_hull_2d` |
| 54 | `features.cusum_events` | feature | 1.0.0 | registered | yes | 139 | 123 | 108 | `fx1 harness execute-operation features.cusum_events` |
| 55 | `features.detrended_fluctuation` | feature | 1.0.0 | registered | yes | 210 | 188 | 166 | `fx1 harness execute-operation features.detrended_fluctuation` |
| 56 | `features.discrete_optimal_transport` | feature | 1.0.0 | registered | yes | 269 | 240 | 229 | `fx1 harness execute-operation features.discrete_optimal_transport` |
| 57 | `features.distance_correlation` | feature | 1.0.0 | registered | yes | 203 | 178 | 159 | `fx1 harness execute-operation features.distance_correlation` |
| 58 | `features.drawdown_path` | feature | 1.0.0 | registered | yes | 55 | 42 | 36 | `fx1 harness execute-operation features.drawdown_path` |
| 59 | `features.dynamic_time_warping` | feature | 1.0.0 | registered | yes | 224 | 201 | 184 | `fx1 harness execute-operation features.dynamic_time_warping` |
| 60 | `features.empirical_characteristic_function` | feature | 1.0.0 | registered | yes | 166 | 142 | 122 | `fx1 harness execute-operation features.empirical_characteristic_function` |
| 61 | `features.empirical_copula` | feature | 1.0.0 | registered | yes | 186 | 166 | 151 | `fx1 harness execute-operation features.empirical_copula` |
| 62 | `features.empirical_wasserstein` | feature | 1.0.0 | registered | yes | 227 | 204 | 187 | `fx1 harness execute-operation features.empirical_wasserstein` |
| 63 | `features.estimate_product_limit` | feature | 1.0.0 | registered | yes | 208 | 184 | 168 | `fx1 harness execute-operation features.estimate_product_limit` |
| 64 | `features.ewma_variance` | feature | 1.0.0 | registered | yes | 53 | 40 | 33 | `fx1 harness execute-operation features.ewma_variance` |
| 65 | `features.fractional_difference` | feature | 1.0.0 | registered | yes | 136 | 118 | 96 | `fx1 harness execute-operation features.fractional_difference` |
| 66 | `features.graph_effective_resistance` | feature | 1.0.0 | registered | yes | 316 | 282 | 260 | `fx1 harness execute-operation features.graph_effective_resistance` |
| 67 | `features.haar_decomposition` | feature | 1.0.0 | registered | yes | 212 | 192 | 175 | `fx1 harness execute-operation features.haar_decomposition` |
| 68 | `features.hayashi_yoshida_covariance` | feature | 1.0.0 | registered | yes | 163 | 142 | 123 | `fx1 harness execute-operation features.hayashi_yoshida_covariance` |
| 69 | `plugins.inspect_artifact_bundle` | plugin | 1.0.0 | registered | yes | 309 | 274 | 249 | `fx1 harness execute-operation plugins.inspect_artifact_bundle` |
| 70 | `plugins.inspect_audit_ledger` | plugin | 1.0.0 | registered | yes | 298 | 266 | 258 | `fx1 harness execute-operation plugins.inspect_audit_ledger` |
| 71 | `plugins.inspect_gzip` | plugin | 1.0.0 | registered | yes | 131 | 116 | 107 | `fx1 harness execute-operation plugins.inspect_gzip` |
| 72 | `plugins.inspect_hdf5_superblock` | plugin | 1.0.0 | registered | yes | 236 | 214 | 193 | `fx1 harness execute-operation plugins.inspect_hdf5_superblock` |
| 73 | `plugins.inspect_netcdf_classic` | plugin | 1.0.0 | registered | yes | 465 | 433 | 405 | `fx1 harness execute-operation plugins.inspect_netcdf_classic` |
| 74 | `plugins.inspect_numpy_array` | plugin | 1.0.0 | registered | yes | 146 | 131 | 125 | `fx1 harness execute-operation plugins.inspect_numpy_array` |
| 75 | `plugins.inspect_parquet` | plugin | 1.1.0 | registered | yes | 439 | 397 | 374 | `fx1 harness execute-operation plugins.inspect_parquet` |
| 76 | `plugins.inspect_research_receipt` | plugin | 1.0.0 | registered | yes | 267 | 233 | 227 | `fx1 harness execute-operation plugins.inspect_research_receipt` |
| 77 | `plugins.inspect_safetensors` | plugin | 1.0.0 | registered | yes | 304 | 277 | 259 | `fx1 harness execute-operation plugins.inspect_safetensors` |
| 78 | `plugins.inspect_sqlite` | plugin | 1.0.0 | registered | yes | 484 | 438 | 402 | `fx1 harness execute-operation plugins.inspect_sqlite` |
| 79 | `plugins.inspect_tar` | plugin | 1.0.0 | registered | yes | 285 | 260 | 248 | `fx1 harness execute-operation plugins.inspect_tar` |
| 80 | `plugins.inspect_tiff_directory` | plugin | 1.0.0 | registered | yes | 429 | 394 | 362 | `fx1 harness execute-operation plugins.inspect_tiff_directory` |
| 81 | `plugins.inspect_zarr_metadata` | plugin | 1.0.0 | registered | yes | 367 | 329 | 307 | `fx1 harness execute-operation plugins.inspect_zarr_metadata` |
| 82 | `plugins.inspect_zip` | plugin | 1.0.0 | registered | yes | 171 | 154 | 147 | `fx1 harness execute-operation plugins.inspect_zip` |
| 83 | `features.isotonic_quantile_repair` | feature | 1.0.0 | registered | yes | 201 | 177 | 161 | `fx1 harness execute-operation features.isotonic_quantile_repair` |
| 84 | `skills.join_asof_observations` | skill | 1.0.0 | registered | yes | 240 | 209 | 203 | `fx1 harness execute-operation skills.join_asof_observations` |
| 85 | `skills.join_time_windows` | skill | 1.0.0 | registered | yes | 261 | 230 | 217 | `fx1 harness execute-operation skills.join_time_windows` |
| 86 | `features.kernel_density_grid` | feature | 1.0.0 | registered | yes | 237 | 211 | 192 | `fx1 harness execute-operation features.kernel_density_grid` |
| 87 | `features.label_uniqueness` | feature | 1.0.0 | registered | yes | 188 | 161 | 144 | `fx1 harness execute-operation features.label_uniqueness` |
| 88 | `features.lead_lag_signature` | feature | 1.0.0 | registered | yes | 194 | 173 | 148 | `fx1 harness execute-operation features.lead_lag_signature` |
| 89 | `features.linear_constraint_feasibility` | feature | 1.0.0 | registered | yes | 376 | 328 | 306 | `fx1 harness execute-operation features.linear_constraint_feasibility` |
| 90 | `features.linear_gaussian_filter` | feature | 1.0.0 | registered | yes | 406 | 355 | 327 | `fx1 harness execute-operation features.linear_gaussian_filter` |
| 91 | `features.lomb_scargle_periodogram` | feature | 1.0.0 | registered | yes | 289 | 261 | 238 | `fx1 harness execute-operation features.lomb_scargle_periodogram` |
| 92 | `features.markov_absorption` | feature | 1.0.0 | registered | yes | 265 | 238 | 218 | `fx1 harness execute-operation features.markov_absorption` |
| 93 | `features.maximum_flow` | feature | 1.0.0 | registered | yes | 231 | 206 | 189 | `fx1 harness execute-operation features.maximum_flow` |
| 94 | `features.minimum_spanning_forest` | feature | 1.0.0 | registered | yes | 272 | 240 | 222 | `fx1 harness execute-operation features.minimum_spanning_forest` |
| 95 | `features.monotone_cubic_interpolation` | feature | 1.0.0 | registered | yes | 211 | 182 | 174 | `fx1 harness execute-operation features.monotone_cubic_interpolation` |
| 96 | `features.multilinear_grid_interpolation` | feature | 1.0.0 | registered | yes | 260 | 228 | 213 | `fx1 harness execute-operation features.multilinear_grid_interpolation` |
| 97 | `features.partial_autocorrelation` | feature | 1.0.0 | registered | yes | 209 | 188 | 169 | `fx1 harness execute-operation features.partial_autocorrelation` |
| 98 | `features.permutation_entropy` | feature | 1.0.0 | registered | yes | 132 | 116 | 105 | `fx1 harness execute-operation features.permutation_entropy` |
| 99 | `features.poisson_binomial_distribution` | feature | 1.0.0 | registered | yes | 191 | 168 | 152 | `fx1 harness execute-operation features.poisson_binomial_distribution` |
| 100 | `features.principal_coordinates` | feature | 1.0.0 | registered | yes | 240 | 222 | 206 | `fx1 harness execute-operation features.principal_coordinates` |
| 101 | `plugins.read_arff` | plugin | 1.0.0 | registered | yes | 358 | 324 | 304 | `fx1 harness execute-operation plugins.read_arff` |
| 102 | `plugins.read_arrow_ipc` | plugin | 1.0.0 | registered | yes | 574 | 526 | 509 | `fx1 harness execute-operation plugins.read_arrow_ipc` |
| 103 | `plugins.read_avro_container` | plugin | 1.0.0 | registered | yes | 391 | 354 | 332 | `fx1 harness execute-operation plugins.read_avro_container` |
| 104 | `plugins.read_bson` | plugin | 1.0.0 | registered | yes | 301 | 275 | 251 | `fx1 harness execute-operation plugins.read_bson` |
| 105 | `plugins.read_cbor` | plugin | 1.0.0 | registered | yes | 287 | 261 | 238 | `fx1 harness execute-operation plugins.read_cbor` |
| 106 | `plugins.read_csv` | plugin | 1.1.0 | registered | yes | 125 | 109 | 103 | `fx1 harness execute-operation plugins.read_csv` |
| 107 | `plugins.read_dbase` | plugin | 1.0.0 | registered | yes | 259 | 236 | 215 | `fx1 harness execute-operation plugins.read_dbase` |
| 108 | `plugins.read_fits_table` | plugin | 1.0.0 | registered | yes | 517 | 474 | 441 | `fx1 harness execute-operation plugins.read_fits_table` |
| 109 | `plugins.read_fixed_width` | plugin | 1.0.0 | registered | yes | 210 | 189 | 172 | `fx1 harness execute-operation plugins.read_fixed_width` |
| 110 | `plugins.read_jsonl` | plugin | 1.1.0 | registered | yes | 162 | 141 | 132 | `fx1 harness execute-operation plugins.read_jsonl` |
| 111 | `plugins.read_libsvm` | plugin | 1.0.0 | registered | yes | 242 | 219 | 197 | `fx1 harness execute-operation plugins.read_libsvm` |
| 112 | `plugins.read_matrix_market` | plugin | 1.0.0 | registered | yes | 277 | 255 | 231 | `fx1 harness execute-operation plugins.read_matrix_market` |
| 113 | `plugins.read_messagepack` | plugin | 1.0.0 | registered | yes | 288 | 261 | 241 | `fx1 harness execute-operation plugins.read_messagepack` |
| 114 | `plugins.read_nrrd` | plugin | 1.0.0 | registered | yes | 445 | 413 | 386 | `fx1 harness execute-operation plugins.read_nrrd` |
| 115 | `plugins.read_ply` | plugin | 1.0.0 | registered | yes | 414 | 380 | 353 | `fx1 harness execute-operation plugins.read_ply` |
| 116 | `plugins.read_sparse_npz` | plugin | 1.0.0 | registered | yes | 518 | 480 | 453 | `fx1 harness execute-operation plugins.read_sparse_npz` |
| 117 | `plugins.read_stl` | plugin | 1.0.0 | registered | yes | 353 | 318 | 290 | `fx1 harness execute-operation plugins.read_stl` |
| 118 | `plugins.read_toml` | plugin | 1.0.0 | registered | yes | 116 | 100 | 97 | `fx1 harness execute-operation plugins.read_toml` |
| 119 | `plugins.read_wav_pcm` | plugin | 1.0.0 | registered | yes | 259 | 238 | 213 | `fx1 harness execute-operation plugins.read_wav_pcm` |
| 120 | `plugins.read_xml` | plugin | 1.0.0 | registered | yes | 281 | 250 | 226 | `fx1 harness execute-operation plugins.read_xml` |
| 121 | `plugins.read_yaml` | plugin | 1.0.0 | registered | yes | 189 | 169 | 160 | `fx1 harness execute-operation plugins.read_yaml` |
| 122 | `skills.reconcile_aggregates` | skill | 1.0.0 | registered | yes | 384 | 347 | 326 | `fx1 harness execute-operation skills.reconcile_aggregates` |
| 123 | `skills.reconstruct_order_book` | skill | 1.0.0 | registered | yes | 434 | 392 | 368 | `fx1 harness execute-operation skills.reconstruct_order_book` |
| 124 | `skills.resolve_security_identity` | skill | 1.0.0 | registered | yes | 249 | 215 | 208 | `fx1 harness execute-operation skills.resolve_security_identity` |
| 125 | `features.rolling_autocorrelation` | feature | 1.0.0 | registered | yes | 97 | 83 | 74 | `fx1 harness execute-operation features.rolling_autocorrelation` |
| 126 | `features.rolling_linear_trend` | feature | 1.0.0 | registered | yes | 114 | 101 | 86 | `fx1 harness execute-operation features.rolling_linear_trend` |
| 127 | `features.rolling_mad` | feature | 1.0.0 | registered | yes | 95 | 82 | 71 | `fx1 harness execute-operation features.rolling_mad` |
| 128 | `features.rolling_rank` | feature | 1.0.0 | registered | yes | 55 | 42 | 35 | `fx1 harness execute-operation features.rolling_rank` |
| 129 | `features.rolling_robust_regression` | feature | 1.0.0 | registered | yes | 158 | 136 | 119 | `fx1 harness execute-operation features.rolling_robust_regression` |
| 130 | `features.rolling_zscore` | feature | 1.0.0 | registered | yes | 66 | 52 | 42 | `fx1 harness execute-operation features.rolling_zscore` |
| 131 | `features.sample_entropy` | feature | 1.0.0 | registered | yes | 140 | 124 | 103 | `fx1 harness execute-operation features.sample_entropy` |
| 132 | `skills.score_binary_forecasts` | skill | 1.0.0 | registered | yes | 87 | 73 | 68 | `fx1 harness execute-operation skills.score_binary_forecasts` |
| 133 | `skills.score_censored_brier` | skill | 1.0.0 | registered | yes | 217 | 191 | 176 | `fx1 harness execute-operation skills.score_censored_brier` |
| 134 | `skills.score_competing_risks` | skill | 1.0.0 | registered | yes | 255 | 229 | 212 | `fx1 harness execute-operation skills.score_competing_risks` |
| 135 | `skills.score_count_forecasts` | skill | 1.0.0 | registered | yes | 247 | 213 | 191 | `fx1 harness execute-operation skills.score_count_forecasts` |
| 136 | `skills.score_empirical_crps` | skill | 1.0.0 | registered | yes | 76 | 62 | 54 | `fx1 harness execute-operation skills.score_empirical_crps` |
| 137 | `skills.score_energy` | skill | 1.0.0 | registered | yes | 94 | 80 | 73 | `fx1 harness execute-operation skills.score_energy` |
| 138 | `skills.score_gaussian_forecasts` | skill | 1.0.0 | registered | yes | 186 | 162 | 148 | `fx1 harness execute-operation skills.score_gaussian_forecasts` |
| 139 | `skills.score_gaussian_mixtures` | skill | 1.0.0 | registered | yes | 265 | 231 | 208 | `fx1 harness execute-operation skills.score_gaussian_mixtures` |
| 140 | `skills.score_hierarchical_probabilities` | skill | 1.0.0 | registered | yes | 248 | 222 | 204 | `fx1 harness execute-operation skills.score_hierarchical_probabilities` |
| 141 | `skills.score_histogram_density` | skill | 1.0.0 | registered | yes | 183 | 161 | 146 | `fx1 harness execute-operation skills.score_histogram_density` |
| 142 | `skills.score_intervals` | skill | 1.0.0 | registered | yes | 80 | 66 | 61 | `fx1 harness execute-operation skills.score_intervals` |
| 143 | `skills.score_joint_categorical` | skill | 1.0.0 | registered | yes | 261 | 233 | 218 | `fx1 harness execute-operation skills.score_joint_categorical` |
| 144 | `skills.score_markov_paths` | skill | 1.0.0 | registered | yes | 273 | 242 | 222 | `fx1 harness execute-operation skills.score_markov_paths` |
| 145 | `skills.score_piecewise_hazards` | skill | 1.0.0 | registered | yes | 265 | 228 | 208 | `fx1 harness execute-operation skills.score_piecewise_hazards` |
| 146 | `skills.score_probability_kernel` | skill | 1.0.0 | registered | yes | 192 | 168 | 151 | `fx1 harness execute-operation skills.score_probability_kernel` |
| 147 | `skills.score_quantiles` | skill | 1.0.0 | registered | yes | 75 | 61 | 56 | `fx1 harness execute-operation skills.score_quantiles` |
| 148 | `skills.score_ranked_probability` | skill | 1.0.0 | registered | yes | 130 | 114 | 106 | `fx1 harness execute-operation skills.score_ranked_probability` |
| 149 | `skills.score_variance_qlike` | skill | 1.0.0 | registered | yes | 76 | 62 | 53 | `fx1 harness execute-operation skills.score_variance_qlike` |
| 150 | `skills.score_variogram` | skill | 1.0.0 | registered | yes | 135 | 117 | 110 | `fx1 harness execute-operation skills.score_variogram` |
| 151 | `features.segment_mean_changes` | feature | 1.0.0 | registered | yes | 204 | 180 | 163 | `fx1 harness execute-operation features.segment_mean_changes` |
| 152 | `skills.select_asof_revisions` | skill | 1.0.0 | registered | yes | 184 | 159 | 151 | `fx1 harness execute-operation skills.select_asof_revisions` |
| 153 | `skills.select_revision_tombstones` | skill | 1.0.0 | registered | yes | 237 | 210 | 203 | `fx1 harness execute-operation skills.select_revision_tombstones` |
| 154 | `skills.select_universe_membership` | skill | 1.0.0 | registered | yes | 237 | 206 | 199 | `fx1 harness execute-operation skills.select_universe_membership` |
| 155 | `features.set_function_attribution` | feature | 1.0.0 | registered | yes | 229 | 201 | 186 | `fx1 harness execute-operation features.set_function_attribution` |
| 156 | `features.simple_returns` | feature | 1.0.0 | registered | yes | 46 | 33 | 27 | `fx1 harness execute-operation features.simple_returns` |
| 157 | `features.singular_spectrum_analysis` | feature | 1.0.0 | registered | yes | 313 | 284 | 252 | `fx1 harness execute-operation features.singular_spectrum_analysis` |
| 158 | `features.spectral_summary` | feature | 1.0.0 | registered | yes | 246 | 222 | 195 | `fx1 harness execute-operation features.spectral_summary` |
| 159 | `skills.summarize_ingestion_latency` | skill | 1.0.0 | registered | yes | 227 | 193 | 188 | `fx1 harness execute-operation skills.summarize_ingestion_latency` |
| 160 | `skills.summarize_pit` | skill | 1.0.0 | registered | yes | 177 | 152 | 144 | `fx1 harness execute-operation skills.summarize_pit` |
| 161 | `features.thin_plate_spline` | feature | 1.0.0 | registered | yes | 257 | 222 | 203 | `fx1 harness execute-operation features.thin_plate_spline` |
| 162 | `features.time_weighted_mean` | feature | 1.0.0 | registered | yes | 155 | 135 | 124 | `fx1 harness execute-operation features.time_weighted_mean` |
| 163 | `plugins.verify_file_hash` | plugin | 1.1.0 | registered | yes | 93 | 79 | 74 | `fx1 harness execute-operation plugins.verify_file_hash` |
| 164 | `skills.verify_merkle_consistency` | skill | 1.0.0 | registered | yes | 202 | 184 | 168 | `fx1 harness execute-operation skills.verify_merkle_consistency` |
| 165 | `skills.verify_merkle_inclusion` | skill | 1.0.0 | registered | yes | 149 | 132 | 115 | `fx1 harness execute-operation skills.verify_merkle_inclusion` |
| 166 | `plugins.verify_signed_checkpoint` | plugin | 1.0.0 | registered | yes | 271 | 235 | 217 | `fx1 harness execute-operation plugins.verify_signed_checkpoint` |
| 167 | `features.weighted_covariance` | feature | 1.0.0 | registered | yes | 176 | 156 | 138 | `fx1 harness execute-operation features.weighted_covariance` |
| 168 | `features.weighted_geometric_median` | feature | 1.0.0 | registered | yes | 294 | 266 | 242 | `fx1 harness execute-operation features.weighted_geometric_median` |
| 169 | `features.weighted_quantile_binning` | feature | 1.0.0 | registered | yes | 192 | 172 | 158 | `fx1 harness execute-operation features.weighted_quantile_binning` |

## Extension tree (generated bindings — not accepted capabilities)

- Hand-written core: `__init__.py`, `contracts.py`, `feature_catalog.py`, `naming.py`, `registry.py`
- Generated bindings: **89** across `features`=46, `plugins`=19, `skills`=24
- LOC: 1,368 physical / 1,115 nonblank / 990 code-token

Per the plan's independence rule, generated wrappers **do not count** toward the
accepted total. `docs/FX1_CAPABILITIES.md` already states this.

## Gate 0 status

**Gate 0 closed** — inventory reconciled, counting rules frozen.
