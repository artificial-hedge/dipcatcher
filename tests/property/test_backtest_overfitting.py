"""Properties of the backtest-overfitting layer. Research diagnostics only."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.metrics.overfitting import (
    deflated_sharpe,
    probability_of_backtest_overfitting,
)
from quant_fund.validation.cpcv import (
    assert_fold_embargo,
    combinatorial_purged_cv,
    combinatorial_purged_indices,
    fold_embargo_report,
    rank_configs,
)
from quant_fund.validation.walk_forward import Fold, session_index


@given(
    observed=st.floats(min_value=-1.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    length=st.integers(min_value=8, max_value=400),
    skew=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    kurtosis=st.floats(min_value=1.5, max_value=12.0, allow_nan=False, allow_infinity=False),
    n_trials=st.integers(min_value=1, max_value=40),
    var_score=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_deflated_probability_does_not_exceed_undeflated(
    observed: float,
    length: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    var_score: float,
) -> None:
    """DSR is PSR at a hurdle ≥ 0, so it cannot exceed PSR against 0."""
    from quant_fund.metrics.overfitting import probabilistic_sharpe

    psr = probabilistic_sharpe(observed, 0.0, length, skew, kurtosis)
    dsr = deflated_sharpe(observed, length, skew, kurtosis, n_trials, var_score)
    if math.isfinite(psr) and math.isfinite(dsr):
        assert 0.0 <= dsr <= psr + 1e-9
        assert 0.0 <= psr <= 1.0


@given(
    data=st.lists(
        st.lists(
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=4,
        ),
        min_size=2,
        max_size=6,
    )
)
@settings(max_examples=30, deadline=None)
def test_pbo_stays_inside_the_unit_interval(data: list[list[float]]) -> None:
    width = min(len(row) for row in data)
    matrix = np.asarray([row[:width] for row in data], dtype=float)
    pbo = probability_of_backtest_overfitting(matrix, matrix[::-1])
    if math.isfinite(pbo):
        assert 0.0 <= pbo <= 1.0


@given(
    n=st.integers(min_value=16, max_value=36),
    horizon=st.integers(min_value=0, max_value=3),
    embargo=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=25, deadline=None)
def test_purge_never_leaks_overlapping_labels(n: int, horizon: int, embargo: int) -> None:
    """A kept train index never shares a label window or an embargo bar with a test block."""
    folds = combinatorial_purged_indices(n, 4, 2, label_horizon=horizon, embargo=embargo)
    bounds = [int(i * n / 4) for i in range(5)]
    groups = [set(range(bounds[i], bounds[i + 1])) for i in range(4)]
    for train, test in folds:
        train_set = set(train.tolist())
        test_set = set(test.tolist())
        assert train_set.isdisjoint(test_set)
        test_groups = [group for group in groups if group & test_set]
        for index in train_set:
            for point in test_set:
                assert not (index < point <= index + horizon)
                assert index != point
            for group in test_groups:
                lo, hi = min(group), max(group)
                assert not (hi < index <= hi + embargo)
                assert not (lo - embargo <= index < lo)


# ---------------------------------------------------------------------------
# Wave 28: bar-level embargo assertion and worst-path ranking
# ---------------------------------------------------------------------------


def _daily_times(n: int) -> list[datetime]:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    return [t0 + timedelta(days=i) for i in range(n)]


@given(
    n=st.integers(min_value=24, max_value=48),
    n_groups=st.integers(min_value=3, max_value=6),
    horizon=st.integers(min_value=0, max_value=2),
    embargo=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=20, deadline=None)
def test_cpcv_folds_always_pass_the_bar_level_assertion(
    n: int, n_groups: int, horizon: int, embargo: int
) -> None:
    """Whatever geometry the splitter emits must satisfy its own embargo guarantee.

    The datetime splitter is checked by the same bar-level predicate as the
    index splitter above. A future change that loosens purge/embargo would fail
    here rather than silently leaking a train bar adjacent to a holdout.
    """
    times = _daily_times(n)
    n_test = max(1, n_groups // 2)
    try:
        folds = combinatorial_purged_cv(
            times, n_groups, n_test, horizon_bars=horizon, embargo_bars=embargo
        )
    except ValueError:
        return  # fewer unique dates than groups; not a fold-integrity claim
    if not folds:
        return
    report = fold_embargo_report(folds, times, horizon_bars=horizon, embargo_bars=embargo)
    assert report["ok"] is True
    assert report["n_violations"] == 0
    # The assertion wrapper must not raise on a clean report.
    assert (
        assert_fold_embargo(folds, times, horizon_bars=horizon, embargo_bars=embargo)["ok"] is True
    )


@given(
    n=st.integers(min_value=24, max_value=40),
    n_groups=st.integers(min_value=3, max_value=5),
    horizon=st.integers(min_value=0, max_value=2),
    embargo=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=20, deadline=None)
def test_reintroducing_an_embargoed_bar_is_always_caught(
    n: int, n_groups: int, horizon: int, embargo: int
) -> None:
    """Re-adding a single embargoed train bar to any fold must flip the report to not-ok."""
    times = _daily_times(n)
    n_test = max(1, n_groups // 2)
    try:
        folds = combinatorial_purged_cv(
            times, n_groups, n_test, horizon_bars=horizon, embargo_bars=embargo
        )
    except ValueError:
        return
    if not folds:
        return
    idx = session_index(times)
    corrupted: list[Fold] = []
    n_leaked = 0
    for fold in folds:
        test_ids = {idx[t] for t in fold.test_times}
        train_ids = {idx[t] for t in fold.train_times}
        leaked = list(fold.train_times)
        # The bar immediately after the last test session is always inside the
        # post-block embargo window (embargo >= 1), so it must not be train.
        candidate = max(test_ids) + 1
        if candidate < n and candidate not in test_ids and candidate not in train_ids:
            leaked.append(times[candidate])
            n_leaked += 1
        corrupted.append(Fold(train_times=leaked, val_times=[], test_times=fold.test_times))
    if n_leaked == 0:
        return
    report = fold_embargo_report(corrupted, times, horizon_bars=horizon, embargo_bars=embargo)
    assert report["ok"] is False
    assert report["embargo_after_violation"] >= n_leaked
    assert report["n_violations"] >= n_leaked
    with pytest.raises(AssertionError):
        assert_fold_embargo(corrupted, times, horizon_bars=horizon, embargo_bars=embargo)


@given(
    n_paths=st.integers(min_value=2, max_value=6),
    n_configs=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=30, deadline=None)
def test_worst_path_ranking_never_picks_a_worse_worst_path_than_the_mean_winner(
    n_paths: int, n_configs: int, seed: int
) -> None:
    """The worst-path winner's minimum is >= the mean winner's minimum.

    This is the defining property of conservative selection: the config chosen
    to be robust is never worse on its worst reconstructed path than the config
    chosen for its average. Both rankings must also cover every config (a
    silently dropped config would understate the selection multiplicity).
    """
    rng = np.random.default_rng(seed)
    scores = {f"c{i}": rng.normal(size=n_paths) for i in range(n_configs)}
    by_mean = rank_configs(scores, criterion="mean")
    by_worst = rank_configs(scores, criterion="worst_path")
    assert by_mean.selected is not None and by_worst.selected is not None
    assert set(by_mean.order) == set(scores)
    assert set(by_worst.order) == set(scores)
    assert (
        by_worst.detail[by_worst.selected]["worst_path"]
        >= by_worst.detail[by_mean.selected]["worst_path"] - 1e-12
    )


@given(
    n_paths=st.integers(min_value=2, max_value=6),
    constant=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=25, deadline=None)
def test_worst_path_and_mean_agree_on_a_constant_config(n_paths: int, constant: float) -> None:
    """When every path scores identically, mean == worst_path and selection is a tie on name."""
    scores = {"only": np.full(n_paths, constant)}
    by_mean = rank_configs(scores, criterion="mean")
    by_worst = rank_configs(scores, criterion="worst_path")
    assert by_mean.selected == by_worst.selected == "only"
    assert by_mean.detail["only"]["mean"] == pytest.approx(constant)
    assert by_worst.detail["only"]["worst_path"] == pytest.approx(constant)
    assert by_worst.detail["only"]["path_dispersion"] == pytest.approx(0.0)
