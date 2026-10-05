"""Invoke every registered fx-1 operation through the shipped registry.

Expected values are hand calculations for tiny synthetic inputs. They are not
market evidence and they are not snapshots copied from a previous run.
"""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from fx1.operations.registry import _IMPLEMENTATIONS, execute_operation

# Each case is invoked with execute_operation (get_operation + Operation.invoke).
# "expect" is a partial structural match of the operation result.
CASES: dict[str, dict[str, Any]] = {
    "features.bipower_variation": {
        "arguments": {"returns": [2.0, 2.0], "finite_sample_policy": "none"},
        "expect": {
            "observation_count": 2,
            "adjacent_pair_count": 1,
            "realized_variation": 8.0,
            "adjacent_absolute_product_sum": 4.0,
            "bipower_status": "finite",
            "applied_finite_sample_factor": 1.0,
        },
        "approx": True,
        "note": "RV=2^2+2^2=8; one adjacent product is 4; factor none is 1.",
    },
    "features.cusum_events": {
        "arguments": {
            "values": [0.0, 5.0],
            "target": 0.0,
            "drift": 0.0,
            "threshold": 3.0,
            "reset_policy": "both",
        },
        "expect": {
            "event_count": 1,
            "upward_event_count": 1,
            "downward_event_count": 0,
            "final_upward_accumulator": 0.0,
            "final_downward_accumulator": 0.0,
            "maximum_upward_before_reset": 5.0,
            "events": [{"row_index": 1, "direction": "upward", "observed_value": 5.0}],
        },
        "approx": False,
        "note": "Second row lifts the upward accumulator to 5 and resets both sides.",
    },
    "features.drawdown_path": {
        "arguments": {"prices": [100.0, 80.0]},
        "expect": {
            "high_water_marks": [100.0, 100.0],
            "drawdowns": [0.0, -0.2],
            "durations": [0, 1],
        },
        "approx": True,
        "note": "80/100-1 = -0.2 and one row has passed since the peak.",
    },
    "features.ewma_variance": {
        "arguments": {"returns": [2.0, -1.0], "decay": 0.5, "initial_variance": 4.0},
        "expect": {"forecast_variances": [4.0, 4.0], "next_variance": 2.5, "decay": 0.5},
        "approx": False,
        "note": "Forecasts exclude the current row; next is 0.5*4 + 0.5*1 = 2.5.",
    },
    "features.fractional_difference": {
        "arguments": {"values": [4.0, 7.0], "order": 1.0, "width": 2},
        "expect": {
            "values": [None, 3.0],
            "weights_newest_first": [1.0, -1.0],
            "coefficient_sum_dc_gain": 0.0,
            "warmup_count": 1,
            "computed_count": 1,
            "nonzero_weight_count": 2,
        },
        "approx": False,
        "note": "Order 1, width 2 is the first difference 7-4.",
    },
    "features.label_uniqueness": {
        "arguments": {
            "labels": [
                {
                    "security_id": "SYN",
                    "start": "2020-01-01T00:00:00Z",
                    "end": "2020-01-01T00:00:01Z",
                }
            ]
        },
        "expect": {
            "label_count": 1,
            "security_count": 1,
            "weights": [
                {
                    "source_index": 0,
                    "duration_microseconds": 1_000_000,
                    "uniqueness_numerator": 1,
                    "uniqueness_denominator": 1,
                    "uniqueness_weight": 1.0,
                }
            ],
            "securities": [
                {
                    "security_id": "SYN",
                    "label_count": 1,
                    "peak_concurrency": 1,
                    "covered_duration_microseconds": 1_000_000,
                }
            ],
        },
        "approx": False,
        "note": "One isolated one-second label has inverse concurrency 1.",
    },
    "features.permutation_entropy": {
        "arguments": {
            "values": [1.0, 2.0],
            "embedding_dimension": 2,
            "delay": 1,
            "tie_policy": "stable",
            "minimum_patterns": 1,
        },
        "expect": {
            "total_embeddings": 1,
            "used_embeddings": 1,
            "tied_embeddings": 0,
            "observed_pattern_count": 1,
            "possible_pattern_count": 2,
            "entropy_nats": 0.0,
            "normalized_entropy": 0.0,
            "sample_status": "meets_threshold",
            "patterns": [{"ordinal_positions": [0, 1], "count": 1, "probability": 1.0}],
        },
        "approx": False,
        "note": "The only length-2 pattern is strictly increasing, so entropy is zero.",
    },
    "features.rolling_autocorrelation": {
        "arguments": {"values": [1.0, 2.0, 3.0], "window": 3, "lag": 1, "minimum_pairs": 2},
        "expect": {
            "correlations": [None, None, 1.0],
            "status": ["warmup", "warmup", "finite"],
            "pairs_per_full_window": 2,
        },
        "approx": True,
        "note": "Lagged pairs (1,2) and (2,3) are perfectly aligned.",
    },
    "features.rolling_linear_trend": {
        "arguments": {"values": [1.0, 2.0, 3.0], "window": 3},
        "expect": {
            "slopes_per_observation": [None, None, 1.0],
            "centered_intercepts": [None, None, 2.0],
            "residual_scales": [None, None, 0.0],
            "r_squared": [None, None, 1.0],
            "status": ["warmup", "warmup", "finite"],
            "residual_degrees_of_freedom": 1,
        },
        "approx": False,
        "note": "Centered indices -1,0,1 fit y=x+2 with zero residual.",
    },
    "features.rolling_mad": {
        "arguments": {"values": [1.0, 2.0, 3.0], "window": 3},
        "expect": {
            "medians": [None, None, 2.0],
            "median_absolute_deviations": [None, None, 1.0],
            "robust_zscores": [None, None, 1.0],
            "zscore_status": ["warmup", "warmup", "finite"],
        },
        "approx": False,
        "note": "Median 2, raw MAD 1, current score (3-2)/1.",
    },
    "features.rolling_rank": {
        "arguments": {"values": [1.0, 3.0, 3.0], "window": 2},
        "expect": {"percentiles": [None, 1.0, 0.75], "window": 2},
        "approx": False,
        "note": "Unique max is 1; a two-way tie is (0 + 1.5) / 2 = 0.75.",
    },
    "features.rolling_zscore": {
        "arguments": {"values": [1.0, 3.0], "window": 2},
        "expect": {"zscores": [None, 1.0], "window": 2, "population_variance": True},
        "approx": False,
        "note": "Population z-score of 3 against {1, 3} is 1.",
    },
    "features.sample_entropy": {
        "arguments": {
            "values": [1.0, 1.0, 1.0],
            "embedding_length": 1,
            "tolerance": 0.0,
            "threshold_rule": "inclusive",
        },
        "expect": {
            "template_population": 2,
            "candidate_pairs": 1,
            "matching_m_pairs": 1,
            "matching_m_plus_one_pairs": 1,
            "conditional_match_probability": 1.0,
            "sample_entropy": 0.0,
            "status": "finite",
        },
        "approx": False,
        "note": "Identical samples match at both embedding lengths, so -log(1) is 0.",
    },
    "features.simple_returns": {
        "arguments": {"prices": [100.0, 110.0, 99.0], "lag": 1},
        "expect": {"returns": [None, 0.1, -0.1], "lag": 1},
        "approx": True,
        "note": "110/100-1 and 99/110-1.",
    },
    "features.time_weighted_mean": {
        "arguments": {
            "observations": [
                {
                    "event_time": "2020-01-01T00:00:00Z",
                    "available_time": "2020-01-01T00:00:00Z",
                    "value": 4.0,
                }
            ],
            "query_time": "2020-01-01T00:00:02Z",
            "lookback_seconds": 2,
        },
        "expect": {
            "time_weighted_mean": 4.0,
            "integral_value_seconds": 8.0,
            "covered_seconds": 2.0,
            "coverage_fraction": 1.0,
            "positive_duration_segments": 1,
            "initial_observation_index": 0,
            "contributing_observation_indices": [0],
            "rows_not_activated_before_query": 0,
        },
        "approx": False,
        "note": "One value is active for the entire 2-second lookback.",
    },
    "plugins.read_csv": {
        "arguments": {"path": "sample.csv", "delimiter": ",", "offset": 0, "limit": 10},
        "files": {"sample.csv": "a,b\n1,2\n"},
        "expect": {
            "columns": ["a", "b"],
            "rows": [{"a": "1", "b": "2"}],
            "total_rows": 1,
            "returned_rows": 1,
            "has_more": False,
            "blank_records_skipped": 0,
            "source_bytes": 8,
            "source_sha256": "492d5ea496056f1a6a6592241032fab764c321596317930b4fa0e1e8bc3b7470",
        },
        "approx": False,
        "note": "One data row under a two-column header; digest is SHA-256 of those 8 bytes.",
    },
    "plugins.read_jsonl": {
        "arguments": {"path": "sample.jsonl", "offset": 0, "limit": 10},
        "files": {"sample.jsonl": '{"n":1}\n'},
        "expect": {
            "columns": ["n"],
            "rows": [{"n": 1}],
            "physical_line_numbers": [1],
            "total_rows": 1,
            "returned_rows": 1,
            "blank_lines_skipped": 0,
            "source_bytes": 8,
            "source_sha256": "cedf74272c9fc8db5448283a93277e7e7eb7534b71df3bd8ab35fd9b1b73404c",
        },
        "approx": False,
        "note": "One JSON object on physical line 1.",
    },
    "plugins.read_toml": {
        "arguments": {"path": "sample.toml", "table_path": []},
        "files": {"sample.toml": "answer = 7\n"},
        "expect": {
            "data": {"answer": 7},
            "temporal_values": [],
            "source_bytes": 11,
            "source_sha256": "647aa9ac8244e521184f1dd115896099c2bfe3e8bb5092376315c37cb251ca28",
        },
        "approx": False,
        "note": "A single integer assignment is the whole table.",
    },
    "plugins.verify_file_hash": {
        "arguments": {
            "path": "sample.txt",
            "expected_sha256": "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb",
            "expected_bytes": 4,
        },
        "files": {"sample.txt": "abc\n"},
        "expect": {
            "digest_matches": True,
            "size_matches": True,
            "matches": True,
            "actual_bytes": 4,
            "actual_sha256": "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb",
        },
        "approx": False,
        "note": "SHA-256 of the four bytes a, b, c, newline.",
    },
    "skills.audit_duplicate_keys": {
        "arguments": {"rows": [{"k": "a"}, {"k": "a"}], "keys": ["k"], "null_policy": "reject"},
        "expect": {
            "passed": False,
            "row_count": 2,
            "comparable_rows": 2,
            "distinct_keys": 1,
            "duplicate_key_groups": 1,
            "duplicate_rows": 1,
            "invalid_key_rows": 0,
        },
        "approx": False,
        "note": "Two rows share the only composite key.",
    },
    "skills.audit_missingness": {
        "arguments": {
            "rows": [{"a": 1}, {"a": None}],
            "columns": ["a"],
            "empty_string_is_missing": False,
        },
        "expect": {
            "row_count": 2,
            "column_count": 1,
            "assessment": "observed",
            "no_missing_values": False,
            "complete_row_count": 1,
            "incomplete_row_count": 1,
            "effective_missing_cells": 1,
            "incomplete_row_indices": [1],
            "columns": [
                {"column": "a", "null_count": 1, "absent_count": 0, "observed_value_count": 1}
            ],
        },
        "approx": False,
        "note": "The second row's explicit null is the only missing cell.",
    },
    "skills.build_block_resamples": {
        "arguments": {
            "source_length": 1,
            "block_length": 1,
            "replicate_count": 1,
            "resample_length": 1,
            "seed": 0,
        },
        "expect": {
            "resample_length": 1,
            "total_index_count": 1,
            "total_block_draws": 1,
            "replicates": [
                {
                    "indices": [0],
                    "block_starts": [0],
                    "final_block_used_length": 1,
                    "blocks_wrapping_in_returned_indices": 0,
                    "unique_source_indices": 1,
                    "omitted_source_indices": 0,
                }
            ],
        },
        "approx": False,
        "note": "randrange(1) is 0, so the only circular block is index 0.",
    },
    "skills.build_conformal_intervals": {
        "arguments": {
            "calibration_predictions": [0.0],
            "calibration_outcomes": [0.0],
            "predictions": [1.0],
            "alpha": 0.5,
        },
        "expect": {
            "calibration_count": 1,
            "order_statistic_rank": 1,
            "radius": 0.0,
            "radius_status": "finite",
            "calibration_covered_count": 1,
            "calibration_coverage": 1.0,
            "intervals": [
                {
                    "prediction_index": 0,
                    "prediction": 1.0,
                    "lower": 1.0,
                    "upper": 1.0,
                    "status": "finite",
                }
            ],
        },
        "approx": False,
        "note": "Rank ceil(2*0.5)=1 selects the only residual, which is 0.",
    },
    "skills.build_stationary_resamples": {
        "arguments": {
            "source_length": 1,
            "expected_block_length": 1.0,
            "resample_length": 1,
            "replicate_count": 1,
            "seed": 0,
        },
        "expect": {
            "restart_probability_numerator": 1,
            "restart_probability_denominator": 1,
            "total_restart_decisions": 0,
            "total_uniform_source_draws": 1,
            "total_index_count": 1,
            "replicates": [{"indices": [0], "restart_count": 0, "unique_source_indices": 1}],
        },
        "approx": False,
        "note": "A length-1 resample draws one uniform index and makes no restart decision.",
    },
    "skills.score_binary_forecasts": {
        "arguments": {"outcomes": [1], "probabilities": [1.0]},
        "expect": {
            "observation_count": 1,
            "brier_score": 0.0,
            "mean_log_loss": 0.0,
            "log_loss_status": "finite",
            "impossible_event_count": 0,
        },
        "approx": False,
        "note": "A certain correct forecast has zero Brier and log loss.",
    },
    "skills.score_empirical_crps": {
        "arguments": {"outcomes": [0.0], "samples": [[0.0]]},
        "expect": {
            "observation_count": 1,
            "mean_crps": 0.0,
            "crps_by_observation": [0.0],
            "minimum_sample_count": 1,
        },
        "approx": False,
        "note": "A one-point ensemble at the outcome has CRPS 0.",
    },
    "skills.score_energy": {
        "arguments": {"outcomes": [[0.0]], "ensembles": [[[0.0], [0.0]]]},
        "expect": {
            "observation_count": 1,
            "dimension_count": 1,
            "mean_energy_score": 0.0,
            "energy_scores": [0.0],
            "minimum_ensemble_size": 2,
        },
        "approx": False,
        "note": "Every Euclidean distance in the ensemble is zero.",
    },
    "skills.score_intervals": {
        "arguments": {
            "outcomes": [1.0],
            "lower": [0.0],
            "upper": [2.0],
            "nominal_coverage": 0.5,
        },
        "expect": {
            "interval_score": 2.0,
            "mean_width": 2.0,
            "mean_lower_miss_penalty": 0.0,
            "mean_upper_miss_penalty": 0.0,
            "empirical_coverage": 1.0,
        },
        "approx": False,
        "note": "The outcome sits inside a width-2 interval, so the score is the width.",
    },
    "skills.score_quantiles": {
        "arguments": {"outcomes": [1.0], "levels": [0.5], "quantiles": [[1.0]]},
        "expect": {
            "observation_count": 1,
            "mean_pinball_by_level": [0.0],
            "mean_pinball": 0.0,
        },
        "approx": False,
        "note": "The median forecast equals the outcome.",
    },
    "skills.score_ranked_probability": {
        "arguments": {
            "ordered_categories": ["down", "up"],
            "outcomes": ["down"],
            "probabilities": [[1.0, 0.0]],
            "normalization": "none",
        },
        "expect": {
            "boundary_count": 1,
            "mean_ranked_probability_score": 0.0,
            "scores": [0.0],
            "mean_boundary_losses": [0.0],
            "renormalized_row_count": 0,
        },
        "approx": False,
        "note": "A point mass on the observed category has zero boundary error.",
    },
    "skills.score_variance_qlike": {
        "arguments": {"realized_variances": [1.0], "forecast_variances": [1.0]},
        "expect": {
            "observation_count": 1,
            "mean_qlike": 0.0,
            "minimum_qlike": 0.0,
            "maximum_qlike": 0.0,
            "qlike_by_observation": [0.0],
        },
        "approx": False,
        "note": "QLIKE is zero when realized and forecast variances are equal.",
    },
    "skills.score_variogram": {
        "arguments": {
            "outcomes": [[0.0, 0.0]],
            "ensembles": [[[0.0, 0.0]]],
            "power": 1.0,
        },
        "expect": {
            "observation_count": 1,
            "dimension_count": 2,
            "pair_count": 1,
            "mean_variogram_score": 0.0,
            "variogram_scores": [0.0],
            "total_pair_weight": 1.0,
        },
        "approx": False,
        "note": "The only coordinate pair has observed and predicted absolute gaps of 0.",
    },
    "features.distance_correlation": {
        "arguments": {"x": [[0.0], [1.0]], "y": [[0.0], [1.0]]},
        "expect": {
            "observation_count": 2,
            "distance_correlation": 1.0,
            "correlation_status": "defined",
            "x_constant": False,
            "y_constant": False,
            "maximum_distance_x": 1.0,
            "maximum_distance_y": 1.0,
        },
        "approx": True,
        "note": "Two paired nonconstant observations have biased distance correlation 1.",
    },
    "features.haar_decomposition": {
        "arguments": {"values": [4.0], "levels": 0},
        "expect": {
            "original_length": 1,
            "transformed_length": 1,
            "padded_count": 0,
            "levels": 0,
            "approximation": [4.0],
            "details_finest_first": [],
            "input_energy": 16.0,
            "maximum_reconstruction_error": 0.0,
            "reconstructed_prefix": [4.0],
        },
        "approx": False,
        "note": "Level zero is the identity, so energy is 4 squared.",
    },
    "features.isotonic_quantile_repair": {
        "arguments": {"quantile_levels": [0.5], "predictions": [[3.0]]},
        "expect": {
            "normalized_weights": [1.0],
            "repaired_predictions": [[3.0]],
            "absolute_adjustments": [[0.0]],
            "rows_with_crossings": 0,
            "rows_changed_after_rounding": 0,
            "total_changed_after_rounding": 0,
        },
        "approx": False,
        "note": "A single quantile cannot cross, so the projection is the input.",
    },
    "features.poisson_binomial_distribution": {
        "arguments": {"probabilities": [0.0]},
        "expect": {
            "trial_count": 1,
            "certain_successes": 0,
            "certain_failures": 1,
            "uncertain_trials": 0,
            "minimum_support": 0,
            "maximum_support": 0,
            "mean": 0.0,
            "variance": 0.0,
            "standard_deviation": 0.0,
            "common_denominator": "1",
        },
        "approx": False,
        "note": "A certain failure puts all mass on count 0.",
    },
    "features.segment_mean_changes": {
        "arguments": {"values": [2.0, 2.0], "penalty_per_change": 1.0, "minimum_segment_length": 1},
        "expect": {
            "segment_count": 1,
            "change_indices": [],
            "fitted_values": [2.0, 2.0],
            "minimum_squared_error": 0.0,
            "penalty_cost": 0.0,
            "objective": 0.0,
            "objective_exact_numerator": "0",
            "objective_exact_denominator": "1",
        },
        "approx": False,
        "note": "A constant series has one zero-error segment; a split would only add penalty.",
    },
    "skills.audit_bar_integrity": {
        "arguments": {
            "bars": [{"open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 0.0}]
        },
        "expect": {
            "row_count": 1,
            "valid_rows": 1,
            "invalid_rows": 0,
            "passed": True,
            "violation_count": 0,
            "violation_counts": {},
        },
        "approx": False,
        "note": "Close is inside the high-low envelope and volume is zero.",
    },
    "skills.audit_panel_gaps": {
        "arguments": {
            "observations": [
                {"security_id": "SYN", "event_time": "2020-01-01T00:00:00Z"},
                {"security_id": "SYN", "event_time": "2020-01-01T00:01:00Z"},
            ],
            "interval_seconds": 60,
        },
        "expect": {
            "security_count": 1,
            "passed": True,
            "expected_grid_points": 2,
            "observed_grid_points": 2,
            "missing_grid_points": 0,
            "duplicate_rows": 0,
            "off_grid_rows": 0,
        },
        "approx": False,
        "note": "Two stamps one interval apart fill the grid anchored at the first stamp.",
    },
    "skills.audit_point_in_time": {
        "arguments": {
            "observations": [
                {
                    "event_time": "2020-01-01T00:00:00Z",
                    "available_time": "2020-01-01T00:00:01Z",
                    "ingested_time": "2020-01-01T00:00:02Z",
                }
            ],
            "decision_time": "2020-01-01T00:00:02Z",
        },
        "expect": {
            "row_count": 1,
            "valid_rows": 1,
            "invalid_rows": 0,
            "passed": True,
            "violation_count": 0,
            "violation_counts": {},
        },
        "approx": False,
        "note": "Availability is at the decision clock and ingestion is not before availability.",
    },
    "features.convex_hull_2d": {
        "arguments": {"points": [[0.0, 0.0]]},
        "expect": {
            "input_count": 1,
            "distinct_point_count": 1,
            "affine_dimension": 0,
            "area": {"numerator": "0", "denominator": "1", "value": 0.0},
            "perimeter": 0.0,
            "vertices": [{"coordinates": [0.0, 0.0], "source_rows": [0]}],
            "queries": [],
        },
        "approx": False,
        "note": "One point is a 0-dimensional hull with area and perimeter zero.",
    },
    "features.discrete_optimal_transport": {
        "arguments": {"source_masses": [1.0], "target_masses": [1.0], "costs": [[0.0]]},
        "expect": {
            "source_probabilities": [{"numerator": "1", "denominator": "1", "value": 1.0}],
            "target_probabilities": [{"numerator": "1", "denominator": "1", "value": 1.0}],
            "minimum_cost": {"numerator": "0", "denominator": "1", "value": 0.0},
            "dual_objective": {"numerator": "0", "denominator": "1", "value": 0.0},
            "positive_flows": [
                {
                    "source_index": 0,
                    "target_index": 0,
                    "mass": {"numerator": "1", "denominator": "1", "value": 1.0},
                    "cost": 0.0,
                }
            ],
        },
        "approx": False,
        "note": "Unit masses and a zero cost move all mass on the only route at cost 0.",
    },
    "features.dynamic_time_warping": {
        "arguments": {"left": [[0.0]], "right": [[0.0]]},
        "expect": {
            "left_count": 1,
            "right_count": 1,
            "status": "aligned",
            "total_squared_cost": 0.0,
            "total_cost_exact_numerator": "0",
            "total_cost_exact_denominator": "1",
            "path_length": 1,
            "permitted_grid_cells": 1,
            "reachable_grid_cells": 1,
        },
        "approx": False,
        "note": "Identical one-point series have a single zero-cost alignment cell.",
    },
    "features.kernel_density_grid": {
        "arguments": {"samples": [0.0], "queries": [0.0], "bandwidth": 1.0},
        "expect": {
            "sample_count": 1,
            "query_count": 1,
            "queries": [
                {
                    "query_index": 0,
                    "location": 0.0,
                    "density": 0.3989422804014327,
                    "cdf": 0.5,
                    "survival_probability": 0.5,
                }
            ],
        },
        "approx": True,
        "note": "A unit-bandwidth Gaussian at its center has density 1/sqrt(2*pi) and CDF 1/2.",
    },
    "features.markov_absorption": {
        "arguments": {"states": ["stay"], "transition_weights": [[1.0]]},
        "expect": {
            "state_count": 1,
            "normalized_transition_matrix": [[1.0]],
            "closed_classes": [
                {
                    "class_index": 0,
                    "state_indices": [0],
                    "state_names": ["stay"],
                    "singleton_absorbing_state": True,
                }
            ],
            "transient_state_indices": [],
            "transient_state_names": [],
        },
        "approx": False,
        "note": "A single self-loop is one absorbing state and has no transient class.",
    },
    "features.weighted_covariance": {
        "arguments": {"observations": [[5.0]], "normalization": "population"},
        "expect": {
            "observation_count": 1,
            "variable_count": 1,
            "positive_weight_count": 1,
            "means": [5.0],
            "covariance": [[0.0]],
            "constant_columns": [0],
            "effective_sample_size": 1.0,
        },
        "approx": False,
        "note": "One observation has mean 5 and population covariance 0.",
    },
    "features.burg_autoregression": {
        "arguments": {"values": [1.0, 3.0], "order": 0, "centering": "demean"},
        "expect": {
            "observation_count": 2,
            "requested_order": 0,
            "fitted_order": 0,
            "centering_offset": 2.0,
            "constant_input": False,
            "ar_coefficients": [],
            "polynomial_coefficients": [1.0],
            "initial_innovation_variance": 1.0,
            "innovation_variance": 1.0,
            "fit_status": "fitted",
            "stages": [],
        },
        "approx": False,
        "note": "Order zero only reports the demeaned energy (1^2+1^2)/2.",
    },
    "features.circular_summary": {
        "arguments": {"angles": [0.0], "unit": "radians"},
        "expect": {
            "observation_count": 1,
            "positive_weight_count": 1,
            "distinct_positive_angles": 1,
            "mean_cosine": 1.0,
            "mean_sine": 0.0,
            "resultant_length": 1.0,
            "center_status": "defined",
            "mean_direction": 0.0,
        },
        "approx": True,
        "note": "A single angle at 0 has cosine 1, sine 0 and resultant length 1.",
    },
    "features.compositional_logratios": {
        "arguments": {"compositions": [[1.0, 1.0]]},
        "expect": {
            "observation_count": 1,
            "part_count": 2,
            "center_clr": [0.0, 0.0],
        },
        "approx": False,
        "note": "Equal positive parts close to one half and have centered log-ratio 0.",
    },
    "features.empirical_wasserstein": {
        "arguments": {"left_values": [0.0], "right_values": [0.0], "p": 1},
        "expect": {
            "p": 1,
            "distance": 0.0,
            "left_observation_count": 1,
            "right_observation_count": 1,
            "left_support_count": 1,
            "right_support_count": 1,
            "transport_step_count": 1,
        },
        "approx": False,
        "note": "Identical one-point measures have Wasserstein-1 distance 0.",
    },
    "features.hayashi_yoshida_covariance": {
        "arguments": {
            "x": [
                {"time": "2020-01-01T00:00:00Z", "value": 1.0},
                {"time": "2020-01-01T00:00:01Z", "value": 1.0},
            ],
            "y": [
                {"time": "2020-01-01T00:00:00Z", "value": 4.0},
                {"time": "2020-01-01T00:00:01Z", "value": 4.0},
            ],
            "value_semantics": "log_price",
        },
        "expect": {
            "covariance_sum": 0.0,
            "duration_microseconds": 1_000_000,
            "x_interval_count": 1,
            "y_interval_count": 1,
            "overlap_pair_count": 1,
            "zero_product_pair_count": 1,
        },
        "approx": False,
        "note": "Constant paths have zero increments, so the overlap product is zero.",
    },
    "features.monotone_cubic_interpolation": {
        "arguments": {"x": [0.0, 1.0], "y": [2.0, 4.0], "queries": [0.5]},
        "expect": {
            "knot_count": 2,
            "knot_slopes": [{"value": 2.0}, {"value": 2.0}],
            "evaluations": [
                {
                    "query_row": 0,
                    "interval_index": 0,
                    "value": {"value": 3.0},
                    "first_derivative": {"value": 2.0},
                    "second_derivative": {"value": 0.0},
                }
            ],
        },
        "approx": False,
        "note": "Two knots make a line of slope 2, so the midpoint is 3.",
    },
    "features.partial_autocorrelation": {
        "arguments": {"values": [1.0, 3.0], "order": 0, "centering": "demean"},
        "expect": {
            "observation_count": 2,
            "requested_order": 0,
            "fitted_order": 0,
            "constant_input": False,
            "autocovariances": [1.0],
            "autocorrelations": [1.0],
            "partial_autocorrelations": [1.0],
            "initial_variance": 1.0,
            "stages": [],
        },
        "approx": False,
        "note": "Order zero stops at the demeaned variance (1^2+1^2)/2 = 1.",
    },
    "features.rolling_robust_regression": {
        "arguments": {"x": [0.0, 1.0], "y": [0.0, 2.0], "window": 2},
        "expect": {
            "window": 2,
            "full_window_count": 1,
            "evaluated_candidate_pairs": 1,
            "estimates": [
                {"source_index": 0, "status": "warmup", "slope": None, "intercept": None},
                {
                    "source_index": 1,
                    "status": "estimated",
                    "usable_pair_count": 1,
                    "slope": 2.0,
                    "intercept": 0.0,
                },
            ],
        },
        "approx": False,
        "note": "The only pair slope is (2-0)/(1-0); both intercept residuals are 0.",
    },
    "features.spectral_summary": {
        "arguments": {"values": [1.0, 1.0], "sample_interval_seconds": 1.0, "detrend": "none"},
        "expect": {
            "observation_count": 2,
            "spectral_status": "constant_input",
            "integrated_positive_frequency_power": 0.0,
            "peak_frequency_hz": None,
        },
        "approx": False,
        "note": "A constant series has no positive-frequency power.",
    },
    "features.weighted_geometric_median": {
        "arguments": {
            "points": [[3.0]],
            "initialization": "first_positive_point",
            "residual_tolerance": 1e-9,
            "max_iterations": 5,
        },
        "expect": {
            "observation_count": 1,
            "dimension_count": 1,
            "median_estimate": [3.0],
            "initial_point": [3.0],
            "weighted_mean_distance": 0.0,
            "normalized_subgradient_residual": 0.0,
            "coincident_source_indexes": [0],
            "status": "residual_tolerance",
            "converged_numerically": True,
        },
        "approx": False,
        "note": "The geometric median of one point is that point and the residual is zero.",
    },
    "features.weighted_quantile_binning": {
        "arguments": {"values": [1.0], "requested_bins": 2},
        "expect": {
            "fitting_row_count": 1,
            "requested_bins": 2,
            "effective_bins": 1,
            "collapsed_boundary_count": 1,
            "cutpoints": [],
            "fitting_bin_ids": [0],
            "bins": [{"bin_id": 0, "fitting_row_count": 1, "fitting_weight": 1.0}],
        },
        "approx": False,
        "note": "The only candidate cut equals the maximum, so it collapses to one unbounded bin.",
    },
}


def _merge_sibling_cases() -> None:
    """Load hand-checked cases kept beside this module. Inline cases win."""
    directory = Path(__file__).resolve().parent
    for path in sorted(directory.glob("operation_cases_chunk*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load invocation cases from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for operation_id, case in module.CASES.items():
            CASES.setdefault(operation_id, case)


_merge_sibling_cases()


def _assert_matches(actual: Any, expected: Any, *, approx: bool) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            assert key in actual
            _assert_matches(actual[key], value, approx=approx)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            _assert_matches(got, want, approx=approx)
        return
    if approx and isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)
        return
    assert actual == expected


def _write_files(root: Path, files: dict[str, Any]) -> None:
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, dict):
            path.write_bytes(base64.b64decode(payload["base64"]))
        else:
            path.write_bytes(payload.encode("utf-8"))


@pytest.mark.parametrize("operation_id", sorted(_IMPLEMENTATIONS))
def test_registered_operation_is_invoked(operation_id: str, tmp_path: Path) -> None:
    """Every registered operation must run through execute_operation."""
    case = CASES[operation_id]
    workspace = tmp_path.resolve()
    _write_files(workspace, case.get("files", {}))
    envelope = execute_operation(operation_id, case["arguments"], workspace_root=workspace)
    assert envelope["schema"] == "fx1.operation-result/v1"
    assert envelope["operation_id"] == operation_id
    assert envelope["market_evidence"] is False
    assert envelope["research_only"] is True
    assert envelope["live_pnl_claim"] is False
    _assert_matches(envelope["result"], case["expect"], approx=bool(case["approx"]))


def test_invocation_cases_match_the_registry() -> None:
    """A newly registered operation fails this test until it has a real invocation."""
    assert set(CASES) == set(_IMPLEMENTATIONS)
