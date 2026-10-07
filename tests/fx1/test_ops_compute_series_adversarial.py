"""SYNTHETIC adversarial probes for rolling/spectral fx1 compute operations.

Every fixture is synthetic and deterministic; results are correctness checks,
never market evidence. Probes target warmup/constant-window honesty, division
by zero, tie conventions, causality of forecasts, and underflow-safe scaling.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    bipower_variation as bv,
)
from fx1.operations import (
    drawdown_path as dd,
)
from fx1.operations import (
    ewma_variance as ewma,
)
from fx1.operations import (
    permutation_entropy as pe,
)
from fx1.operations import (
    rolling_autocorrelation as rac,
)
from fx1.operations import (
    rolling_linear_trend as rlt,
)
from fx1.operations import (
    rolling_mad as rmad,
)
from fx1.operations import (
    rolling_rank as rr,
)
from fx1.operations import (
    rolling_zscore as rz,
)
from fx1.operations import (
    simple_returns as sr,
)
from fx1.operations import (
    spectral_summary as spec,
)
from fx1.operations import (
    summarize_ingestion_latency as sil,
)
from fx1.operations.base import OperationContext

T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


# ------------------------------ rolling_zscore ---------------------------------


def test_zscore_warmup_and_constant_windows_are_null(context: OperationContext):
    out = rz.execute(rz.Input(values=[1.0, 1.0, 1.0, 1.0], window=3), context)
    assert out.zscores[:2] == [None, None]
    assert out.zscores[2:] == [None, None]  # constant window -> zero variance
    assert out.population_variance is True


def test_zscore_is_population_not_sample(context: OperationContext):
    # window [1, 2, 3]: mean 2, population std = sqrt(2/3)
    out = rz.execute(rz.Input(values=[1.0, 2.0, 3.0], window=3), context)
    assert out.zscores[2] == pytest.approx(1.0 / math.sqrt(2.0 / 3.0))
    assert not math.isclose(out.zscores[2], 1.0, rel_tol=1e-12)  # not the sample z


def test_zscore_tiny_variance_window_not_flushed(context: OperationContext):
    """Subnormal-scale windows must still produce finite z-scores."""
    base = 1e-150
    out = rz.execute(rz.Input(values=[base, 2 * base, 3 * base, 4 * base], window=4), context)
    z = out.zscores[3]
    assert z is not None and z > 1.0


def test_zscore_seed_crosscheck(context: OperationContext):
    rng = random.Random(5)
    values = [rng.uniform(-5, 5) for _ in range(30)]
    w = 7
    out = rz.execute(rz.Input(values=values, window=w), context)
    for i in range(len(values)):
        if i + 1 < w:
            assert out.zscores[i] is None
            continue
        win = values[i + 1 - w : i + 1]
        mean = sum(win) / w
        var = sum((v - mean) ** 2 for v in win) / w
        assert out.zscores[i] == pytest.approx((values[i] - mean) / math.sqrt(var))


# ------------------------------ rolling_rank -----------------------------------


def test_rank_all_tied_gives_midpoint(context: OperationContext):
    out = rr.execute(rr.Input(values=[7.0, 7.0, 7.0, 7.0], window=4), context)
    # smaller=0, ties=4 -> (0 + (4+1)/2)/4 = 0.625
    assert out.percentiles[3] == pytest.approx(0.625)


def test_rank_extremes_and_warmup(context: OperationContext):
    out = rr.execute(rr.Input(values=[3.0, 1.0, 4.0, 2.0], window=4), context)
    assert out.percentiles[:3] == [None, None, None]
    # window [3,1,4,2] current=2: smaller=1, ties=1 -> (1+1)/4 = 0.5
    assert out.percentiles[3] == pytest.approx(0.5)


def test_rank_unique_max_is_one(context: OperationContext):
    out = rr.execute(rr.Input(values=[1.0, 2.0, 3.0, 9.0], window=4), context)
    assert out.percentiles[3] == pytest.approx(1.0)


# ------------------------------- rolling_mad -----------------------------------


def test_mad_zero_deviation_explicit_status(context: OperationContext):
    out = rmad.execute(rmad.Input(values=[5.0, 5.0, 5.0, 5.0], window=4), context)
    assert out.medians[3] == pytest.approx(5.0)
    assert out.median_absolute_deviations[3] == pytest.approx(0.0)
    assert out.robust_zscores[3] is None
    assert out.zscore_status[3] == "zero_mad"


def test_mad_window_one_is_always_zero_mad(context: OperationContext):
    out = rmad.execute(rmad.Input(values=[3.0, 7.0, -2.0], window=1), context)
    assert out.zscore_status == ["zero_mad", "zero_mad", "zero_mad"]
    assert out.robust_zscores == [None, None, None]


def test_mad_no_consistency_factor_applied(context: OperationContext):
    """MAD is raw median absolute deviation — no 1.4826 normal scaling."""
    values = [0.0, 1.0, 2.0, 10.0]
    out = rmad.execute(rmad.Input(values=values, window=4), context)
    # median = 1.5, deviations = [1.5, 0.5, 0.5, 8.5] -> MAD = 1.0
    assert out.medians[3] == pytest.approx(1.5)
    assert out.median_absolute_deviations[3] == pytest.approx(1.0)
    assert out.robust_zscores[3] == pytest.approx((10.0 - 1.5) / 1.0)


def test_mad_even_window_median_between_floats(context: OperationContext):
    out = rmad.execute(rmad.Input(values=[1.0, 2.0, 4.0, 8.0, 16.0], window=4), context)
    # last window [2,4,8,16]: median = (4+8)/2 = 6
    assert out.medians[4] == pytest.approx(6.0)
    # deviations from 6: [4,2,2,10] -> sorted [2,2,4,10] -> median 3
    assert out.median_absolute_deviations[4] == pytest.approx(3.0)


# -------------------------- rolling_autocorrelation ----------------------------


def test_autocorr_monotone_series_lag1_is_positive_one(context: OperationContext):
    out = rac.execute(rac.Input(values=[1.0, 2.0, 3.0, 4.0, 5.0], window=5, lag=1), context)
    assert out.correlations[4] == pytest.approx(1.0)
    assert out.status[4] == "finite"


def test_autocorr_alternating_series_lag1_is_negative_one(context: OperationContext):
    out = rac.execute(rac.Input(values=[1.0, -1.0, 1.0, -1.0, 1.0], window=5, lag=1), context)
    assert out.correlations[4] == pytest.approx(-1.0)


def test_autocorr_constant_window_zero_variance(context: OperationContext):
    out = rac.execute(rac.Input(values=[2.0] * 5, window=5, lag=1), context)
    assert out.correlations[4] is None
    assert out.status[4] == "zero_variance"


def test_autocorr_uses_separate_segment_means(context: OperationContext):
    """Left/right means are per-segment, not one full-window mean."""
    values = [0.0, 10.0, 0.0, 10.0, 5.0, 5.0]
    out = rac.execute(rac.Input(values=values, window=6, lag=2), context)
    left = values[:4]
    right = values[2:]
    ml, mr = sum(left) / 4, sum(right) / 4
    cov = sum((a - ml) * (b - mr) for a, b in zip(left, right, strict=True))
    nl = math.sqrt(sum((a - ml) ** 2 for a in left))
    nr = math.sqrt(sum((b - mr) ** 2 for b in right))
    assert out.correlations[5] == pytest.approx(cov / (nl * nr))


def test_autocorr_window_minus_lag_below_minimum_pairs_rejected() -> None:
    with pytest.raises(ValidationError, match="minimum_pairs"):
        rac.Input(values=[1.0] * 5, window=5, lag=4, minimum_pairs=2)


# -------------------------- rolling_linear_trend -------------------------------


def test_trend_perfect_line_exact_slope(context: OperationContext):
    values = [2.0 + 3.0 * i for i in range(6)]
    out = rlt.execute(rlt.Input(values=values, window=6), context)
    assert out.slopes_per_observation[5] == pytest.approx(3.0)
    assert out.residual_scales[5] == pytest.approx(0.0)
    assert out.r_squared[5] == pytest.approx(1.0)
    # centered intercept = mean of the window
    assert out.centered_intercepts[5] == pytest.approx(sum(values) / 6)


def test_trend_constant_window_is_explicit(context: OperationContext):
    out = rlt.execute(rlt.Input(values=[4.0] * 5, window=5), context)
    assert out.slopes_per_observation[4] == pytest.approx(0.0)
    assert out.centered_intercepts[4] == pytest.approx(4.0)
    assert out.residual_scales[4] == pytest.approx(0.0)
    assert out.r_squared[4] is None
    assert out.status[4] == "constant"


def test_trend_warmup_rows_null(context: OperationContext):
    out = rlt.execute(rlt.Input(values=[1.0, 2.0, 4.0, 8.0], window=4), context)
    assert out.slopes_per_observation[:3] == [None, None, None]
    assert out.status[:3] == ["warmup", "warmup", "warmup"]
    assert out.residual_degrees_of_freedom == 2


def test_trend_seed_crosscheck_against_normal_equations(context: OperationContext):
    rng = random.Random(11)
    values = [rng.uniform(-3, 3) for _ in range(12)]
    w = 5
    out = rlt.execute(rlt.Input(values=values, window=w), context)
    xs = [j - (w - 1) / 2.0 for j in range(w)]
    sxx = sum(x * x for x in xs)
    for i in range(w - 1, 12):
        win = values[i + 1 - w : i + 1]
        mean = sum(win) / w
        centered = [v - mean for v in win]
        slope = sum(x * y for x, y in zip(xs, centered, strict=True)) / sxx
        assert out.slopes_per_observation[i] == pytest.approx(slope)


# ----------------------------- bipower_variation -------------------------------


def test_bipower_single_return_insufficient(context: OperationContext):
    out = bv.execute(bv.Input(returns=[0.5]), context)
    assert out.bipower_variation is None
    assert out.bipower_status == "insufficient_pairs"
    assert out.adjacent_pair_count == 0
    assert out.realized_variation == pytest.approx(0.25)


def test_bipower_known_value(context: OperationContext):
    out = bv.execute(bv.Input(returns=[1.0, -1.0, 1.0]), context)
    # RV = 3; pairs (|1||-1| + |-1||1|) = 2; BV = pi/2 * 2 = pi
    assert out.realized_variation == pytest.approx(3.0)
    assert out.bipower_variation == pytest.approx(math.pi)
    assert out.applied_finite_sample_factor == pytest.approx(1.0)


def test_bipower_finite_sample_factor(context: OperationContext):
    out = bv.execute(
        bv.Input(returns=[1.0, -1.0, 1.0], finite_sample_policy="n_over_n_minus_1"),
        context,
    )
    assert out.bipower_variation == pytest.approx(math.pi * 3.0 / 2.0)
    assert out.applied_finite_sample_factor == pytest.approx(1.5)


def test_bipower_zero_returns(context: OperationContext):
    out = bv.execute(bv.Input(returns=[0.0, 0.0, 0.0]), context)
    assert out.bipower_variation == pytest.approx(0.0)
    assert out.bipower_status == "finite"


def test_bipower_tiny_products_accumulate_without_underflow(
    context: OperationContext,
):
    """Each |r_i|*|r_j| = 1e-326 underflows to zero alone; the scaled sum does not."""
    tiny = 1e-163
    assert tiny * tiny == 0.0  # naive product underflows
    out = bv.execute(bv.Input(returns=[tiny] * 10_000), context)
    assert out.realized_variation == pytest.approx(1e-322)
    assert out.bipower_variation == pytest.approx(math.pi / 2 * 9999e-326)


# ------------------------------- ewma_variance ---------------------------------


def test_ewma_forecast_is_causal_excludes_current(context: OperationContext):
    """forecast_variances[t] must not depend on returns[t]."""
    out = ewma.execute(
        ewma.Input(returns=[2.0, 0.0, 0.0], decay=0.5, initial_variance=1.0),
        context,
    )
    # v0=1.0; forecast[0]=1.0; v1 = .5*1 + .5*4 = 2.5; forecast[1]=2.5 ...
    assert out.forecast_variances == [
        pytest.approx(1.0),
        pytest.approx(2.5),
        pytest.approx(1.25),
    ]
    assert out.next_variance == pytest.approx(0.625)
    assert out.zero_mean_assumption is True


def test_ewma_decay_zero_is_last_squared_return(context: OperationContext):
    out = ewma.execute(
        ewma.Input(returns=[3.0, -2.0], decay=0.0, initial_variance=9.0),
        context,
    )
    assert out.forecast_variances[0] == pytest.approx(9.0)
    assert out.forecast_variances[1] == pytest.approx(9.0)  # r0^2
    assert out.next_variance == pytest.approx(4.0)  # r1^2


def test_ewma_decay_one_rejected(context: OperationContext):
    with pytest.raises(ValidationError):
        ewma.Input(returns=[1.0], decay=1.0)


# ------------------------------ drawdown_path ----------------------------------


def test_drawdown_peak_reacquisition_resets_duration(context: OperationContext):
    out = dd.execute(dd.Input(prices=[100.0, 80.0, 100.0, 90.0]), context)
    assert out.drawdowns == [
        pytest.approx(0.0),
        pytest.approx(-0.2),
        pytest.approx(0.0),
        pytest.approx(-0.1),
    ]
    assert out.durations == [0, 1, 0, 1]
    assert out.high_water_marks == [100.0, 100.0, 100.0, 100.0]


def test_drawdown_equal_price_repeak_is_a_new_peak(context: OperationContext):
    out = dd.execute(dd.Input(prices=[10.0, 5.0, 10.0]), context)
    assert out.durations[2] == 0


def test_drawdown_nonpositive_price_rejected() -> None:
    with pytest.raises(ValidationError):
        dd.Input(prices=[10.0, 0.0, 12.0])
    with pytest.raises(ValidationError):
        dd.Input(prices=[10.0, -3.0])


def test_drawdown_monotone_up_all_zero(context: OperationContext):
    out = dd.execute(dd.Input(prices=[1.0, 2.0, 3.0, 4.0]), context)
    assert out.drawdowns == [0.0, 0.0, 0.0, 0.0]
    assert out.durations == [0, 0, 0, 0]


# ------------------------------ simple_returns ---------------------------------


def test_simple_returns_lag_boundary(context: OperationContext):
    out = sr.execute(sr.Input(prices=[100.0, 110.0, 121.0], lag=1), context)
    assert out.returns[0] is None
    assert out.returns[1] == pytest.approx(0.1)
    assert out.returns[2] == pytest.approx(0.1)


def test_simple_returns_lag_exceeding_length_all_null(context: OperationContext):
    out = sr.execute(sr.Input(prices=[1.0, 2.0], lag=5), context)
    assert out.returns == [None, None]


def test_simple_returns_exact_lag_indexing(context: OperationContext):
    out = sr.execute(sr.Input(prices=[10.0, 11.0, 12.0, 15.0], lag=2), context)
    assert out.returns == [
        None,
        None,
        pytest.approx(0.2),  # 12/10 - 1
        pytest.approx(15.0 / 11.0 - 1.0),
    ]


# --------------------------- permutation_entropy -------------------------------


def test_pe_all_tied_values_stable_policy_single_pattern(context: OperationContext):
    out = pe.execute(
        pe.Input(
            values=[1.0] * 30,
            embedding_dimension=3,
            delay=1,
            minimum_patterns=5,
            tie_policy="stable",
        ),
        context,
    )
    assert out.tied_embeddings == 28
    assert out.used_embeddings == 28
    assert out.observed_pattern_count == 1
    # stable tie break -> identity ordering (0,1,2)
    assert out.patterns[0].ordinal_positions == [0, 1, 2]
    assert out.entropy_nats == pytest.approx(0.0)
    assert out.normalized_entropy == pytest.approx(0.0)


def test_pe_drop_policy_discards_tied_embeddings(context: OperationContext):
    values = [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    out = pe.execute(
        pe.Input(
            values=values,
            embedding_dimension=3,
            delay=1,
            minimum_patterns=1,
            tie_policy="drop",
        ),
        context,
    )
    # embeddings = 8-2 = 6; the first (starting 0) contains the 1.0,1.0 tie
    assert out.tied_embeddings == 1
    assert out.discarded_embeddings == 1
    assert out.used_embeddings == 5


def test_pe_reject_policy_raises_on_ties(context: OperationContext):
    with pytest.raises(ValueError, match="tied values"):
        pe.execute(
            pe.Input(
                values=[1.0, 2.0, 2.0, 3.0],
                embedding_dimension=3,
                tie_policy="reject",
            ),
            context,
        )


def test_pe_too_short_series_no_embeddings(context: OperationContext):
    out = pe.execute(
        pe.Input(values=[1.0, 2.0], embedding_dimension=5, minimum_patterns=1),
        context,
    )
    assert out.total_embeddings == 0
    assert out.sample_status == "no_embeddings"
    assert out.entropy_nats is None
    assert out.patterns == []


def test_pe_monotone_series_single_ascending_pattern(context: OperationContext):
    out = pe.execute(
        pe.Input(
            values=[float(i) for i in range(30)],
            embedding_dimension=3,
            delay=1,
            minimum_patterns=5,
        ),
        context,
    )
    assert out.observed_pattern_count == 1
    assert out.patterns[0].ordinal_positions == [0, 1, 2]
    assert out.normalized_entropy == pytest.approx(0.0)
    assert out.sample_status == "meets_threshold"


def test_pe_normalized_entropy_bounded_by_log_factorial(context: OperationContext):
    rng = random.Random(3)
    values = [rng.uniform(0, 1) for _ in range(200)]
    out = pe.execute(
        pe.Input(
            values=values,
            embedding_dimension=3,
            delay=1,
            minimum_patterns=5,
        ),
        context,
    )
    assert out.possible_pattern_count == 6
    assert out.normalized_entropy is not None
    assert 0.0 <= out.normalized_entropy <= 1.0
    probs = sum(p.probability for p in out.patterns)
    assert probs == pytest.approx(1.0)


def test_pe_pattern_limit_omits_but_keeps_entropy(context: OperationContext):
    rng = random.Random(4)
    values = [rng.uniform(0, 1) for _ in range(300)]
    full = pe.execute(
        pe.Input(values=values, embedding_dimension=5, delay=1, minimum_patterns=5),
        context,
    )
    capped = pe.execute(
        pe.Input(
            values=values,
            embedding_dimension=5,
            delay=1,
            minimum_patterns=5,
            pattern_limit=2,
        ),
        context,
    )
    assert len(capped.patterns) == 2
    assert capped.observed_pattern_count == full.observed_pattern_count
    assert capped.omitted_pattern_count == full.observed_pattern_count - 2
    assert capped.entropy_nats == pytest.approx(full.entropy_nats)


# ------------------------------ spectral_summary -------------------------------


def test_spectral_constant_input_status(context: OperationContext):
    out = spec.execute(spec.Input(values=[3.0] * 16, sample_interval_seconds=1.0), context)
    assert out.spectral_status == "constant_input"
    assert out.integrated_positive_frequency_power == pytest.approx(0.0)
    assert out.peak_frequency_hz is None


def test_spectral_pure_sine_peaks_at_its_frequency(context: OperationContext):
    n = 64
    dt = 0.1
    cycles = 4  # 4 complete cycles in the window
    values = [math.sin(2 * math.pi * cycles * i / n) for i in range(n)]
    out = spec.execute(spec.Input(values=values, sample_interval_seconds=dt), context)
    expected_hz = cycles / (n * dt)  # 0.625 Hz
    assert out.spectral_status == "finite"
    assert out.peak_frequency_hz == pytest.approx(expected_hz)
    assert out.peak_period_seconds == pytest.approx(1.0 / expected_hz)


def test_spectral_even_n_nyquist_not_doubled(context: OperationContext):
    """Alternating +-a puts all positive power in the undoubled Nyquist bin."""
    n = 8
    values = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
    out = spec.execute(spec.Input(values=values, sample_interval_seconds=1.0), context)
    # rfft has 5 bins (DC,1,2,3,Nyq); positive power concentrated at Nyquist
    assert out.positive_frequency_bin_count == 4
    assert out.peak_frequency_hz == pytest.approx(0.5)


def test_spectral_two_observations_single_positive_bin(context: OperationContext):
    out = spec.execute(spec.Input(values=[1.0, 3.0], sample_interval_seconds=1.0), context)
    # rfft of [1,3] -> bins [4, -2]; exactly one positive-frequency bin
    assert out.positive_frequency_bin_count == 1
    assert out.entropy_status == "single_positive_bin"
    assert out.normalized_spectral_entropy is None


def test_spectral_bins_paginated_deterministically(context: OperationContext):
    n = 32
    rng = random.Random(8)
    values = [rng.uniform(-1, 1) for _ in range(n)]
    first = spec.execute(spec.Input(values=values, sample_interval_seconds=1.0, limit=5), context)
    rest = spec.execute(
        spec.Input(values=values, sample_interval_seconds=1.0, offset=5, limit=5),
        context,
    )
    assert len(first.bins) == 5
    assert [b.index for b in first.bins] == [0, 1, 2, 3, 4]
    assert [b.index for b in rest.bins] == [5, 6, 7, 8, 9]


# ------------------------- summarize_ingestion_latency -------------------------


def _obs_latency(avail_s: float, ingested_s: float, source: str = "s") -> sil.Observation:
    return sil.Observation(
        source=source,
        available_time=T0 + timedelta(seconds=avail_s),
        ingested_time=T0 + timedelta(seconds=ingested_s),
    )


def test_latency_negative_lags_kept_in_signed_only(context: OperationContext):
    obs = [
        _obs_latency(10.0, 5.0),  # -5s lag: ingested before available
        _obs_latency(0.0, 3.0),  # +3s
        _obs_latency(0.0, 0.0),  # 0s
    ]
    out = sil.execute(sil.Input(observations=obs), context)
    s = out.overall
    assert s.negative_lag_count == 1
    assert s.zero_lag_count == 1
    assert s.positive_lag_count == 1
    assert s.signed.count == 3
    assert s.nonnegative.count == 2
    assert s.signed.minimum_seconds == pytest.approx(-5.0)
    # negative lag lands in the leading (-inf, 0) histogram bin
    assert s.histogram[0].lower_seconds_inclusive is None
    assert s.histogram[0].count == 1


def test_latency_histogram_edges_are_left_closed(context: OperationContext):
    obs = [
        _obs_latency(0.0, 1.0),  # lands in [1,2)
        _obs_latency(0.0, 0.0),  # lands in [0,1)
        _obs_latency(0.0, 2.0),  # lands in [2,inf)
    ]
    out = sil.execute(
        sil.Input(
            observations=obs,
            nonnegative_bin_edges_seconds=[0.0, 1.0, 2.0],
        ),
        context,
    )
    bins = [b.count for b in out.overall.histogram]
    # (-inf,0) holds negatives only; zero lag is nonnegative -> [0,1)
    assert bins == [0, 1, 1, 1]


def test_latency_edges_must_start_at_zero_and_increase() -> None:
    with pytest.raises(ValidationError, match="start at zero"):
        sil.Input(
            observations=[_obs_latency(0.0, 1.0)],
            nonnegative_bin_edges_seconds=[1.0, 2.0],
        )
    with pytest.raises(ValidationError, match="strictly increasing"):
        sil.Input(
            observations=[_obs_latency(0.0, 1.0)],
            nonnegative_bin_edges_seconds=[0.0, 0.0, 2.0],
        )


def test_latency_quantiles_linear_interpolation(context: OperationContext):
    obs = [_obs_latency(0.0, float(v)) for v in [1, 2, 3, 4]]
    out = sil.execute(sil.Input(observations=obs, quantiles=[0.0, 0.5, 1.0]), context)
    qs = {q.probability: q.seconds for q in out.overall.nonnegative.quantiles}
    assert qs[0.0] == pytest.approx(1.0)
    assert qs[0.5] == pytest.approx(2.5)  # (n-1)*0.5 = 1.5 -> midpoint of 2,3
    assert qs[1.0] == pytest.approx(4.0)


def test_latency_population_std_not_sample(context: OperationContext):
    obs = [_obs_latency(0.0, v) for v in [1.0, 3.0]]
    out = sil.execute(sil.Input(observations=obs), context)
    assert out.overall.nonnegative.population_std_seconds == pytest.approx(1.0)


def test_latency_per_source_and_pagination(context: OperationContext):
    obs = [
        _obs_latency(0.0, 1.0, source="b"),
        _obs_latency(0.0, 2.0, source="a"),
        _obs_latency(0.0, 3.0, source="b"),
    ]
    out = sil.execute(sil.Input(observations=obs, source_offset=0, source_limit=1), context)
    assert out.source_count == 2
    assert len(out.sources) == 1
    assert out.sources[0].source == "a"
    assert out.next_source_offset == 1
    second = sil.execute(sil.Input(observations=obs, source_offset=1, source_limit=1), context)
    assert second.sources[0].source == "b"
    assert second.sources[0].summary.observation_count == 2
    assert second.next_source_offset is None


def test_latency_naive_clock_rejected() -> None:
    with pytest.raises(ValidationError):
        sil.Observation.model_validate(
            {
                "source": "s",
                "available_time": "2026-01-01T00:00:00",
                "ingested_time": "2026-01-01T00:00:01",
            }
        )
