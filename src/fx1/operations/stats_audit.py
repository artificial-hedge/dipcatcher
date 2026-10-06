"""stats_audit — rolling/time-series feature-math contract battery.

The ``features.*`` operations are the descriptive transform layer of the
harness: every z-score, EWMA forecast, trend slope, and entropy figure a
skill or eval consumes is produced here, under causal (current-inclusive
trailing window) semantics where warmup and undefined quantities must be
explicit nulls — never quietly fabricated numbers. These probes pin the
formulas against hand-computed fixtures:

- *simple_returns / drawdown_path* — lagged fractional changes (first
  ``lag`` rows null), running high-water marks, nonpositive drawdowns,
  durations resetting on reattained peaks; nonpositive prices refused.
- *ewma_variance* — ``v[t+1] = d·v[t] + (1-d)·r²`` with each aligned
  forecast strictly pre-observation; ``next_variance`` continues the
  recursion; zero-mean assumption declared.
- *rolling_zscore / rolling_mad / rolling_rank* — population z-scores,
  median + raw MAD + unscaled robust scores with ``warmup`` /
  ``zero_mad`` / ``numeric_overflow`` statuses, and average-rank
  percentile ties with unique-max = 1 convention.
- *rolling_autocorrelation* — Pearson on separately centered lagged
  pair vectors, ``window - lag ≥ minimum_pairs`` admission, clamped to
  [-1, 1], ``zero_variance`` status.
- *rolling_linear_trend* — OLS against the centered index: slope,
  midpoint intercept, ``sqrt(RSS/(w-2))`` residual scale, R² clamped to
  [0,1] and undefined (null) on constant windows.
- *bipower_variation* — ``RV = Σr²``, ``BV = (π/2)·factor·Σ|r[i-1]||r[i]|``
  with the ``n/(n-1)`` correction named explicitly; a single return is
  ``insufficient_pairs`` with nulls, not a fake zero.
- *permutation_entropy* — delayed ordinal patterns, stable/drop/reject
  tie policies, ``H/log(m!)`` normalization, the
  ``no_embeddings → no_eligible_patterns → below_threshold →
  meets_threshold`` status ladder, bounded pattern reporting.
- *spectral_summary* — boxcar periodogram ``dt·|X|²/n`` with interior
  doubling (DC/Nyquist excepted), DC-excluded peak + entropy, constant
  input zeroing positive power instead of leaking FFT roundoff, and a
  Parseval invariant checked against a direct time-domain computation.

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect it found. In-process and deterministic: no served app, no
network, no workspace files (these transforms never touch the context).
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import UTC, datetime
from math import fsum
from pathlib import Path
from typing import Any

import pydantic

from fx1.operations import (
    bipower_variation,
    drawdown_path,
    ewma_variance,
    permutation_entropy,
    rolling_autocorrelation,
    rolling_linear_trend,
    rolling_mad,
    rolling_rank,
    rolling_zscore,
    simple_returns,
    spectral_summary,
    time_weighted_mean,
)
from fx1.operations.base import Operation, OperationContext

__all__ = ["stats_audit", "stats_audit_bench"]

_T_INIT = "2023-12-31T23:57:00+00:00"
_T_MID = "2023-12-31T23:59:00+00:00"
_T_LATE = "2023-12-31T23:59:30+00:00"


def _ctx() -> OperationContext:
    # These transforms never read the workspace; a real (validated) root
    # keeps the call shape honest anyway.
    return OperationContext(workspace_root=Path(tempfile.gettempdir()))


def _refuses(model: Any, **kwargs: Any) -> bool:
    try:
        model(**kwargs)
    except (pydantic.ValidationError, ValueError):
        return True
    return False


def _probe_descriptors() -> dict[str, bool]:
    out: dict[str, bool] = {}
    ops: tuple[Operation[Any, Any], ...] = (
        simple_returns.OPERATION,
        drawdown_path.OPERATION,
        ewma_variance.OPERATION,
        rolling_zscore.OPERATION,
        rolling_mad.OPERATION,
        rolling_rank.OPERATION,
        rolling_autocorrelation.OPERATION,
        rolling_linear_trend.OPERATION,
        bipower_variation.OPERATION,
        permutation_entropy.OPERATION,
        spectral_summary.OPERATION,
        time_weighted_mean.OPERATION,
    )
    out["op_all_features_kind"] = all(o.kind == "feature" for o in ops)
    out["op_ids_features_ns"] = all(o.id.startswith("features.") for o in ops)
    out["op_schemas_module_local"] = all(
        o.input_model.__module__ == o.handler.__module__ == o.output_model.__module__ for o in ops
    )
    out["op_describe_shape"] = all(
        d["id"] == o.id and d["description"].strip()
        for d, o in zip((o.describe() for o in ops), ops, strict=True)
    )
    return out


def _probe_simple_returns() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = simple_returns
    ctx = _ctx()
    r = mod.execute(mod.Input(prices=[100.0, 110.0, 121.0], lag=1), ctx)
    out["s_lag1_handcheck"] = (
        r.returns[0] is None
        and math.isclose(r.returns[1] or 0.0, 0.1)
        and math.isclose(r.returns[2] or 0.0, 0.1)
    )
    r2 = mod.execute(mod.Input(prices=[100.0, 90.0, 80.0, 120.0], lag=2), ctx)
    out["s_lag2_warmup"] = r2.returns[0] is None and r2.returns[1] is None
    out["s_lag2_handcheck"] = math.isclose(
        r2.returns[2] or 9.0, 80.0 / 100.0 - 1.0
    ) and math.isclose(r2.returns[3] or 0.0, 120.0 / 90.0 - 1.0)
    out["s_length_preserved"] = len(r2.returns) == 4
    r3 = mod.execute(mod.Input(prices=[100.0, 110.0], lag=5), ctx)
    out["s_lag_beyond_length_all_null"] = r3.returns == [None, None]
    out["s_lag_echoed"] = r2.lag == 2
    out["s_reject_zero_price"] = _refuses(mod.Input, prices=[0.0, 1.0], lag=1)
    out["s_reject_negative_price"] = _refuses(mod.Input, prices=[100.0, -1.0], lag=1)
    out["s_reject_lag_zero"] = _refuses(mod.Input, prices=[1.0, 2.0], lag=0)
    out["s_reject_nan"] = _refuses(mod.Input, prices=[float("nan"), 1.0], lag=1)
    out["s_extra_forbid"] = _refuses(mod.Input, prices=[1.0, 2.0], lag=1, bogus=1)
    return out


def _probe_drawdown() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = drawdown_path
    ctx = _ctx()
    r = mod.execute(mod.Input(prices=[100.0, 90.0, 80.0, 110.0, 105.0]), ctx)
    out["d_peak_tracks"] = r.high_water_marks == [100.0, 100.0, 100.0, 110.0, 110.0]
    out["d_drawdown_handcheck"] = math.isclose(r.drawdowns[1], -0.1) and math.isclose(
        r.drawdowns[4], 105.0 / 110.0 - 1.0
    )
    out["d_durations"] = r.durations == [0, 1, 2, 0, 1]
    out["d_all_nonpositive"] = all(d <= 0.0 for d in r.drawdowns)
    eq = mod.execute(mod.Input(prices=[100.0, 90.0, 100.0]), ctx)
    out["d_reattain_resets"] = eq.durations == [0, 1, 0] and math.isclose(eq.drawdowns[2], 0.0)
    out["d_reject_zero_price"] = _refuses(mod.Input, prices=[0.0])
    out["d_reject_nan"] = _refuses(mod.Input, prices=[float("nan")])
    out["d_extra_forbid"] = _refuses(mod.Input, prices=[1.0], bogus=1)
    return out


def _probe_ewma() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = ewma_variance
    ctx = _ctx()
    r = mod.execute(
        mod.Input(returns=[0.1, -0.1, 0.2], decay=0.9, initial_variance=0.0),
        ctx,
    )
    # forecasts are pre-observation: f0 = v0 = 0, f1 = .9*0 + .1*.01 = .001
    out["e_forecasts_pre_observation"] = math.isclose(r.forecast_variances[0], 0.0)
    v1 = 0.9 * 0.0 + 0.1 * 0.01
    v2 = 0.9 * v1 + 0.1 * 0.01
    v3 = 0.9 * v2 + 0.1 * 0.04
    out["e_recursion_handcheck"] = (
        math.isclose(r.forecast_variances[1], v1)
        and math.isclose(r.forecast_variances[2], v2)
        and math.isclose(r.next_variance, v3)
    )
    out["e_forecast_len"] = len(r.forecast_variances) == 3
    # same-row independence: change the last return, earlier forecasts identical
    r_b = mod.execute(
        mod.Input(returns=[0.1, -0.1, -0.5], decay=0.9, initial_variance=0.0),
        ctx,
    )
    out["e_same_row_excluded"] = r.forecast_variances[:2] == r_b.forecast_variances[:2]
    out["e_zero_mean_declared"] = r.zero_mean_assumption is True
    out["e_decay_echoed"] = math.isclose(r.decay, 0.9)
    seeded = mod.execute(
        mod.Input(returns=[0.0], decay=0.94, initial_variance=0.0025),
        ctx,
    )
    out["e_initial_prior_first"] = math.isclose(seeded.forecast_variances[0], 0.0025)
    out["e_reject_decay_one"] = _refuses(mod.Input, returns=[0.1], decay=1.0, initial_variance=0.0)
    out["e_reject_negative_init"] = _refuses(
        mod.Input, returns=[0.1], decay=0.9, initial_variance=-1.0
    )
    out["e_reject_nan"] = _refuses(
        mod.Input, returns=[float("nan")], decay=0.9, initial_variance=0.0
    )
    out["e_extra_forbid"] = _refuses(
        mod.Input, returns=[0.1], decay=0.9, initial_variance=0.0, bogus=1
    )
    return out


def _probe_rolling_zscore() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = rolling_zscore
    ctx = _ctx()
    r = mod.execute(mod.Input(values=[1.0, 2.0, 3.0, 4.0], window=3), ctx)
    out["z_warmup_nulls"] = r.zscores[0] is None and r.zscores[1] is None
    # window [1,2,3]: mean 2, pop var 2/3, z(3) = 1/sqrt(2/3)
    out["z_handcheck"] = math.isclose(
        r.zscores[2] or 0.0, 1.0 / math.sqrt(2.0 / 3.0), rel_tol=1e-12
    )
    const = mod.execute(mod.Input(values=[2.0, 2.0, 2.0, 2.0], window=3), ctx)
    out["z_constant_null"] = const.zscores[2] is None and const.zscores[3] is None
    out["z_population_declared"] = r.population_variance is True
    out["z_window_echoed"] = r.window == 3
    out["z_reject_window_one"] = _refuses(mod.Input, values=[1.0, 2.0], window=1)
    out["z_reject_nan"] = _refuses(mod.Input, values=[float("nan")], window=2)
    out["z_extra_forbid"] = _refuses(mod.Input, values=[1.0], window=2, bogus=1)
    return out


def _probe_rolling_mad() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = rolling_mad
    ctx = _ctx()
    r = mod.execute(mod.Input(values=[1.0, 2.0, 100.0], window=3), ctx)
    out["m_warmup_status"] = r.zscore_status[0] == "warmup" and r.robust_zscores[0] is None
    # window [1,2,100]: median 2, MAD median(1,0,98)=1, robust z = (100-2)/1
    out["m_handcheck"] = (
        math.isclose(r.medians[2] or 0.0, 2.0)
        and math.isclose(r.median_absolute_deviations[2] or 0.0, 1.0)
        and math.isclose(r.robust_zscores[2] or 0.0, 98.0)
    )
    const = mod.execute(mod.Input(values=[5.0, 5.0, 5.0], window=3), ctx)
    out["m_zero_mad_status"] = (
        const.zscore_status[2] == "zero_mad" and const.robust_zscores[2] is None
    )
    out["m_median_still_reported"] = (
        const.median_absolute_deviations[2] is not None
        and math.isclose(const.medians[2] or 0.0, 5.0)
        and math.isclose(const.median_absolute_deviations[2], 0.0)
    )
    one = mod.execute(mod.Input(values=[7.0, 9.0], window=1), ctx)
    out["m_window_one_zero_mad"] = one.zscore_status == ["zero_mad"] * 2
    out["m_no_consistency_factor"] = r.normal_consistency_factor_applied is False
    out["m_lengths_aligned"] = (
        len(r.medians)
        == len(r.median_absolute_deviations)
        == len(r.robust_zscores)
        == len(r.zscore_status)
        == 3
    )
    out["m_reject_nan"] = _refuses(mod.Input, values=[float("nan")], window=1)
    out["m_extra_forbid"] = _refuses(mod.Input, values=[1.0], window=1, bogus=1)
    return out


def _probe_rolling_rank() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = rolling_rank
    ctx = _ctx()
    r = mod.execute(mod.Input(values=[1.0, 2.0, 3.0, 0.5], window=3), ctx)
    out["r_warmup_nulls"] = r.percentiles[0] is None
    # [1,2,3]: current 3 → (2 + (1+1)/2)/3 = 1.0
    out["r_unique_max_one"] = math.isclose(r.percentiles[2] or 0.0, 1.0)
    # [2,3,0.5]: current 0.5 → (0 + 1)/3 = 1/3
    out["r_unique_min_floor"] = math.isclose(r.percentiles[3] or 0.0, 1.0 / 3.0)
    ties = mod.execute(mod.Input(values=[1.0, 2.0, 2.0, 2.0], window=4), ctx)
    # current 2: smaller=1, ties=3 → (1 + 4/2)/4 = 0.75
    out["r_ties_average"] = math.isclose(ties.percentiles[3] or 0.0, 0.75)
    out["r_convention_declared"] = ties.tie_convention == "average_one_based_rank_divided_by_window"
    out["r_bounds"] = all(p is None or 0.0 < p <= 1.0 for p in r.percentiles)
    out["r_reject_nan"] = _refuses(mod.Input, values=[float("nan")], window=1)
    out["r_extra_forbid"] = _refuses(mod.Input, values=[1.0], window=1, bogus=1)
    return out


def _probe_rolling_autocorr() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = rolling_autocorrelation
    ctx = _ctx()
    up = mod.execute(mod.Input(values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], window=6, lag=1), ctx)
    out["a_warmup_status"] = up.status[0] == "warmup"
    out["a_perfect_positive"] = math.isclose(up.correlations[5] or 0.0, 1.0, abs_tol=1e-9)
    alt = mod.execute(mod.Input(values=[1.0, 3.0, 1.0, 3.0, 1.0, 3.0], window=6, lag=1), ctx)
    out["a_alternating_negative"] = math.isclose(alt.correlations[5] or 0.0, -1.0, abs_tol=1e-9)
    const = mod.execute(mod.Input(values=[2.0, 2.0, 2.0, 2.0], window=4, lag=1), ctx)
    out["a_zero_variance_status"] = (
        const.status[3] == "zero_variance" and const.correlations[3] is None
    )
    out["a_pairs_reported"] = up.pairs_per_full_window == 5
    out["a_bounds_clamped"] = all(c is None or -1.0 <= c <= 1.0 for c in up.correlations)
    out["a_reject_pair_shortfall"] = _refuses(
        mod.Input, values=[1.0] * 6, window=4, lag=3, minimum_pairs=2
    )
    out["a_reject_lag_zero"] = _refuses(mod.Input, values=[1.0] * 6, window=4, lag=0)
    out["a_reject_window_two"] = _refuses(mod.Input, values=[1.0] * 6, window=2, lag=1)
    out["a_extra_forbid"] = _refuses(mod.Input, values=[1.0] * 6, window=4, lag=1, bogus=1)
    return out


def _probe_rolling_trend() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = rolling_linear_trend
    ctx = _ctx()
    r = mod.execute(mod.Input(values=[1.0, 2.0, 3.0, 4.0], window=4), ctx)
    out["t_warmup_status"] = r.status[0] == "warmup"
    out["t_slope_exact"] = math.isclose(r.slopes_per_observation[3] or 0.0, 1.0, abs_tol=1e-12)
    out["t_intercept_midpoint"] = math.isclose(r.centered_intercepts[3] or 0.0, 2.5, abs_tol=1e-12)
    out["t_residual_zero_perfect_fit"] = math.isclose(
        r.residual_scales[3] or 9.0, 0.0, abs_tol=1e-12
    )
    out["t_r2_perfect"] = math.isclose(r.r_squared[3] or 0.0, 1.0)
    const = mod.execute(mod.Input(values=[5.0, 5.0, 5.0, 5.0], window=4), ctx)
    out["t_constant_status"] = const.status[3] == "constant"
    out["t_constant_slope_zero"] = const.slopes_per_observation[3] is not None and math.isclose(
        const.slopes_per_observation[3], 0.0
    )
    out["t_constant_r2_null"] = const.r_squared[3] is None
    out["t_constant_intercept_level"] = math.isclose(const.centered_intercepts[3] or 0.0, 5.0)
    out["t_dof_reported"] = r.residual_degrees_of_freedom == 2
    out["t_r2_bounded"] = all(v is None or 0.0 <= v <= 1.0 for v in r.r_squared)
    out["t_reject_window_two"] = _refuses(mod.Input, values=[1.0, 2.0], window=2)
    out["t_reject_nan"] = _refuses(mod.Input, values=[float("nan")] * 4, window=3)
    out["t_extra_forbid"] = _refuses(mod.Input, values=[1.0, 2.0, 3.0], window=3, bogus=1)
    return out


def _probe_bipower() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = bipower_variation
    ctx = _ctx()
    r = mod.execute(mod.Input(returns=[0.01, -0.02, 0.03], finite_sample_policy="none"), ctx)
    out["b_rv_handcheck"] = math.isclose(
        r.realized_variation, 0.0001 + 0.0004 + 0.0009, rel_tol=1e-12
    )
    out["b_bv_handcheck"] = math.isclose(
        r.bipower_variation or 0.0,
        (math.pi / 2.0) * (0.01 * 0.02 + 0.02 * 0.03),
        rel_tol=1e-12,
    )
    out["b_factor_none_default"] = r.applied_finite_sample_factor is not None and math.isclose(
        r.applied_finite_sample_factor, 1.0
    )
    adj = mod.execute(
        mod.Input(returns=[0.01, -0.02, 0.03], finite_sample_policy="n_over_n_minus_1"),
        ctx,
    )
    out["b_factor_correction"] = math.isclose(
        adj.applied_finite_sample_factor or 0.0, 1.5
    ) and math.isclose(
        adj.bipower_variation or 0.0,
        (math.pi / 2.0) * 1.5 * (0.01 * 0.02 + 0.02 * 0.03),
        rel_tol=1e-12,
    )
    single = mod.execute(mod.Input(returns=[0.05]), ctx)
    out["b_insufficient_pairs"] = (
        single.bipower_status == "insufficient_pairs"
        and single.bipower_variation is None
        and single.adjacent_absolute_product_sum is None
        and single.applied_finite_sample_factor is None
    )
    out["b_rv_still_scored"] = math.isclose(single.realized_variation, 0.0025)
    out["b_pair_count"] = r.adjacent_pair_count == 2
    out["b_policy_echoed"] = adj.finite_sample_policy == "n_over_n_minus_1"
    out["b_reject_bad_policy"] = _refuses(
        mod.Input, returns=[0.1, 0.2], finite_sample_policy="auto"
    )
    out["b_reject_nan"] = _refuses(mod.Input, returns=[float("nan"), 0.1])
    out["b_extra_forbid"] = _refuses(mod.Input, returns=[0.1, 0.2], bogus=1)
    return out


def _probe_permutation_entropy() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = permutation_entropy
    ctx = _ctx()
    mono = mod.execute(mod.Input(values=[float(i) for i in range(30)], embedding_dimension=3), ctx)
    out["p_monotone_zero_entropy"] = (
        mono.entropy_nats is not None
        and mono.normalized_entropy is not None
        and math.isclose(mono.entropy_nats, 0.0)
        and math.isclose(mono.normalized_entropy, 0.0)
    )
    out["p_monotone_one_pattern"] = mono.observed_pattern_count == 1
    alt = mod.execute(
        mod.Input(values=[0.0, 1.0] * 15, embedding_dimension=2, minimum_patterns=1),
        ctx,
    )
    used = alt.used_embeddings
    # alternating m=2: every adjacent pair flips direction; embeddings 29
    up_count = sum(1 for p in alt.patterns if p.ordinal_positions == [0, 1])
    counts = {tuple(p.ordinal_positions): p.count for p in alt.patterns}
    expect = -fsum((c / used) * math.log(c / used) for c in counts.values())
    out["p_entropy_handcheck"] = (
        used == 29
        and len(counts) == 2
        and math.isclose(alt.entropy_nats or -1.0, expect, rel_tol=1e-12)
        and math.isclose(alt.normalized_entropy or -1.0, expect / math.log(2), rel_tol=1e-9)
        and up_count == 1
    )
    out["p_status_meets"] = alt.sample_status == "meets_threshold"
    const = mod.execute(mod.Input(values=[3.0] * 10, embedding_dimension=3), ctx)
    out["p_stable_ties_single_pattern"] = (
        const.tied_embeddings == const.total_embeddings
        and const.observed_pattern_count == 1
        and const.patterns[0].ordinal_positions == [0, 1, 2]
    )
    drop = mod.execute(
        mod.Input(values=[3.0] * 10, embedding_dimension=3, tie_policy="drop"),
        ctx,
    )
    out["p_drop_no_eligible"] = (
        drop.sample_status == "no_eligible_patterns"
        and drop.entropy_nats is None
        and drop.discarded_embeddings == drop.total_embeddings
    )
    try:
        mod.execute(
            mod.Input(values=[3.0] * 10, embedding_dimension=3, tie_policy="reject"),
            ctx,
        )
        out["p_reject_raises"] = False
    except ValueError:
        out["p_reject_raises"] = True
    tiny = mod.execute(mod.Input(values=[1.0, 2.0], embedding_dimension=4, delay=1), ctx)
    out["p_no_embeddings_status"] = (
        tiny.sample_status == "no_embeddings" and tiny.entropy_nats is None
    )
    below = mod.execute(
        mod.Input(
            values=[0.0, 1.0] * 5,
            embedding_dimension=2,
            minimum_patterns=100,
        ),
        ctx,
    )
    out["p_below_threshold_status"] = below.sample_status == "below_threshold"
    limited = mod.execute(
        mod.Input(
            values=[0.0, 1.0] * 15,
            embedding_dimension=2,
            minimum_patterns=1,
            pattern_limit=1,
        ),
        ctx,
    )
    out["p_pattern_limit_truncates"] = (
        len(limited.patterns) == 1 and limited.omitted_pattern_count == 1
    )
    out["p_counts_sum_to_used"] = sum(p.count for p in alt.patterns) == alt.used_embeddings
    out["p_possible_factorial"] = alt.possible_pattern_count == 2
    out["p_reject_dim_high"] = _refuses(mod.Input, values=[1.0] * 10, embedding_dimension=8)
    out["p_reject_dim_low"] = _refuses(mod.Input, values=[1.0] * 10, embedding_dimension=1)
    out["p_reject_bad_tie_policy"] = _refuses(mod.Input, values=[1.0] * 10, tie_policy="shuffle")
    out["p_extra_forbid"] = _refuses(mod.Input, values=[1.0] * 10, embedding_dimension=2, bogus=1)
    return out


def _probe_spectral() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = spectral_summary
    ctx = _ctx()
    n = 64
    dt = 0.5
    k = 4  # tone at bin 4
    tone = [math.sin(2.0 * math.pi * k * i / n) + 10.0 for i in range(n)]
    r = mod.execute(mod.Input(values=tone, sample_interval_seconds=dt, detrend="mean"), ctx)
    resolution = 1.0 / (n * dt)
    out["sp_peak_bin"] = math.isclose(r.peak_frequency_hz or 0.0, k * resolution, rel_tol=1e-12)
    out["sp_peak_period"] = math.isclose(r.peak_period_seconds or 0.0, 1.0 / (k * resolution))
    out["sp_resolution"] = math.isclose(r.frequency_resolution_hz, resolution)
    out["sp_positive_bins"] = (
        r.positive_frequency_bin_count == n // 2 and r.total_bin_count == n // 2 + 1
    )
    # Parseval: positive power * n equals centered energy (interior doubling).
    centered_sq = fsum((v - 10.0) ** 2 for v in tone)
    out["sp_parseval"] = math.isclose(
        r.integrated_positive_frequency_power * n, centered_sq, rel_tol=1e-9
    )
    out["sp_status_finite"] = r.spectral_status == "finite"
    out["sp_entropy_bounded"] = (
        r.spectral_entropy_nats is not None
        and r.normalized_spectral_entropy is not None
        and 0.0 <= r.normalized_spectral_entropy <= 1.0
    )
    out["sp_entropy_pure_tone_small"] = (r.normalized_spectral_entropy or 1.0) < 0.05

    const = mod.execute(mod.Input(values=[7.0] * 16, sample_interval_seconds=1.0), ctx)
    out["sp_constant_status"] = const.spectral_status == "constant_input"
    out["sp_constant_no_peak"] = (
        const.peak_frequency_hz is None and const.spectral_entropy_nats is None
    )
    out["sp_constant_zero_positive"] = (
        math.isclose(const.integrated_positive_frequency_power, 0.0)
        and const.positive_power_underflow is False
    )
    out["sp_constant_entropy_status"] = const.entropy_status == "no_positive_frequency_power"
    three = mod.execute(mod.Input(values=[1.0, 0.0, 0.0], sample_interval_seconds=1.0), ctx)
    out["sp_single_positive_bin"] = (
        three.positive_frequency_bin_count == 1
        and three.entropy_status == "single_positive_bin"
        and three.normalized_spectral_entropy is None
    )
    paged = mod.execute(
        mod.Input(
            values=tone,
            sample_interval_seconds=dt,
            detrend="mean",
            offset=2,
            limit=5,
        ),
        ctx,
    )
    out["sp_pagination"] = (
        len(paged.bins) == 5
        and paged.bins[0].index == 2
        and math.isclose(paged.bins[0].frequency_hz, 2 * resolution)
        and paged.has_more is True
    )
    tail = mod.execute(
        mod.Input(
            values=tone,
            sample_interval_seconds=dt,
            detrend="mean",
            offset=n // 2,
            limit=5,
        ),
        ctx,
    )
    out["sp_tail_no_more"] = len(tail.bins) == 1 and tail.has_more is False
    out["sp_reject_len_one"] = _refuses(mod.Input, values=[1.0], sample_interval_seconds=1.0)
    out["sp_reject_dt_zero"] = _refuses(mod.Input, values=[1.0, 2.0], sample_interval_seconds=0.0)
    out["sp_reject_bad_detrend"] = _refuses(
        mod.Input, values=[1.0, 2.0], sample_interval_seconds=1.0, detrend="linear"
    )
    out["sp_reject_nan"] = _refuses(
        mod.Input, values=[float("nan"), 1.0], sample_interval_seconds=1.0
    )
    out["sp_extra_forbid"] = _refuses(
        mod.Input, values=[1.0, 2.0], sample_interval_seconds=1.0, bogus=1
    )
    return out


def _probe_time_weighted() -> dict[str, bool]:
    out: dict[str, bool] = {}
    mod = time_weighted_mean
    ctx = _ctx()
    base = datetime(2024, 1, 1, tzinfo=UTC)

    def obs(et: str, at: str, v: float) -> Any:
        return mod.Observation(event_time=et, available_time=at, value=v)  # type: ignore[arg-type]

    # one obs live for the whole 120s window → mean = value
    r = mod.execute(
        mod.Input(
            observations=[obs(_T_INIT, _T_INIT, 2.0)],
            query_time=base,
            lookback_seconds=120,
        ),
        ctx,
    )
    out["tw_single_held"] = math.isclose(r.time_weighted_mean, 2.0)
    out["tw_full_coverage"] = math.isclose(r.coverage_fraction, 1.0) and math.isclose(
        r.covered_seconds, 120.0
    )
    # 0.0 held for first 60s of window, 4.0 for last 60s → mean 2.0
    r2 = mod.execute(
        mod.Input(
            observations=[
                obs(_T_INIT, _T_INIT, 0.0),
                obs(_T_MID, _T_MID, 4.0),
            ],
            query_time=base,
            lookback_seconds=120,
        ),
        ctx,
    )
    out["tw_mid_switch"] = math.isclose(r2.time_weighted_mean, 2.0)
    out["tw_integral"] = math.isclose(r2.integral_value_seconds, 240.0)
    out["tw_contributors"] = r2.contributing_observation_indices == [0, 1]
    # a stale event arriving late must not overwrite a newer observable event
    stale = mod.execute(
        mod.Input(
            observations=[
                obs(_T_INIT, _T_INIT, 0.0),
                obs("2023-12-31T23:58:30+00:00", _T_MID, 9.0),
                obs(_T_MID, _T_MID, 4.0),
            ],
            query_time=base,
            lookback_seconds=120,
        ),
        ctx,
    )
    # the 9.0 row activates at the same instant as the newer 4.0 event but
    # is itself older — it holds zero duration and never contributes.
    out["tw_stale_event_skipped"] = math.isclose(
        stale.time_weighted_mean, 2.0
    ) and stale.contributing_observation_indices == [0, 2]
    # availability gates activation: event inside window but published after
    # query → not observable
    gated = mod.execute(
        mod.Input(
            observations=[
                obs(_T_INIT, _T_INIT, 1.0),
                obs(_T_LATE, "2024-01-01T00:01:00+00:00", 9.0),
            ],
            query_time=base,
            lookback_seconds=120,
        ),
        ctx,
    )
    out["tw_late_available_ignored"] = (
        math.isclose(gated.time_weighted_mean, 1.0) and gated.rows_not_activated_before_query == 1
    )
    # no observable initial value → fail closed
    try:
        mod.execute(
            mod.Input(
                observations=[obs(_T_LATE, _T_LATE, 1.0)],
                query_time=base,
                lookback_seconds=120,
            ),
            ctx,
        )
        out["tw_no_initial_fails"] = False
    except ValueError:
        out["tw_no_initial_fails"] = True
    out["tw_reject_duplicate_events"] = _refuses(
        mod.Input,
        observations=[
            {
                "event_time": _T_MID,
                "available_time": _T_MID,
                "value": 1.0,
            },
            {
                "event_time": _T_MID,
                "available_time": "2023-12-31T23:59:10+00:00",
                "value": 2.0,
            },
        ],
        query_time=base,
        lookback_seconds=120,
    )
    out["tw_reject_naive_clock"] = _refuses(
        mod.Input,
        observations=[
            {
                "event_time": "2023-12-31T23:59:00",  # no tz
                "available_time": "2023-12-31T23:59:00",
                "value": 1.0,
            }
        ],
        query_time=base,
        lookback_seconds=120,
    )
    out["tw_reject_unix_clock"] = _refuses(
        mod.Input,
        observations=[
            {
                "event_time": _T_MID,
                "available_time": _T_MID,
                "value": 1.0,
            }
        ],
        query_time=1704067200,
        lookback_seconds=120,
    )
    out["tw_reject_lookback_zero"] = _refuses(
        mod.Input,
        observations=[
            {
                "event_time": _T_MID,
                "available_time": _T_MID,
                "value": 1.0,
            }
        ],
        query_time=base,
        lookback_seconds=0,
    )
    out["tw_reject_lookback_too_large"] = _refuses(
        mod.Input,
        observations=[
            {
                "event_time": _T_INIT,
                "available_time": _T_INIT,
                "value": 1.0,
            }
        ],
        query_time=base,
        lookback_seconds=31_536_001,
    )
    out["tw_hold_policy_declared"] = (
        r.hold_policy == "latest_observable_event_without_staleness_cutoff"
    )
    out["tw_extra_forbid"] = _refuses(
        mod.Input,
        observations=[
            {
                "event_time": _T_MID,
                "available_time": _T_MID,
                "value": 1.0,
            }
        ],
        query_time=base,
        lookback_seconds=120,
        bogus=1,
    )
    return out


def stats_audit() -> dict[str, bool]:
    """Every feature-math contract as literal booleans."""
    out: dict[str, bool] = {}
    out.update(_probe_descriptors())
    out.update(_probe_simple_returns())
    out.update(_probe_drawdown())
    out.update(_probe_ewma())
    out.update(_probe_rolling_zscore())
    out.update(_probe_rolling_mad())
    out.update(_probe_rolling_rank())
    out.update(_probe_rolling_autocorr())
    out.update(_probe_rolling_trend())
    out.update(_probe_bipower())
    out.update(_probe_permutation_entropy())
    out.update(_probe_spectral())
    out.update(_probe_time_weighted())
    return out


def stats_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = stats_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "stats_audit",
        "schema": "stats_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process execute() calls; no served app, no workspace I/O",
            "not_verified": [
                "registry dispatch wiring (see registry lane)",
                "audit_* data-quality ops and file readers (separate battery)",
                "PIT source selection (caller duty by contract)",
            ],
        },
        "interpretation": (
            "Feature-math contracts hold: lagged returns and drawdown paths "
            "are causal with null warmups and nonpositive drawdowns; EWMA "
            "variance forecasts are strictly pre-observation under the "
            "declared zero-mean recursion; rolling z/MAD/rank/autocorr/OLS "
            "match hand-computed fixtures with explicit warmup, zero-variance "
            "and constant-window statuses; bipower variation reproduces "
            "RV=Σr² and the π/2-normalized product sum with the n/(n−1) "
            "correction named and insufficient-pair history null; "
            "permutation entropy matches hand-counted ordinal frequencies "
            "with stable/drop/reject tie policies and the full status "
            "ladder; the spectral summary's positive-frequency power "
            "satisfies a Parseval invariant against direct time-domain "
            "energy, with constant inputs zeroing positive power rather "
            "than leaking FFT roundoff; time-weighted means integrate the "
            "left-continuous latest-observable value, gate on both event "
            "and availability clocks, skip stale revisions, and fail "
            "closed on missing initial coverage or ambiguous clocks."
            if ok
            else f"STATS AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(stats_audit_bench(), indent=2, sort_keys=True))
