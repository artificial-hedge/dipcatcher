"""SYNTHETIC numeric and causal checks; these are not market evidence."""

from math import sqrt
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fx1.operations import drawdown_path, ewma_variance, rolling_zscore, simple_returns
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def test_simple_returns_known_lags_and_short_history(context: OperationContext) -> None:
    prices = [100.0, 110.0, 99.0, 118.8]
    result = simple_returns.execute(simple_returns.Input(prices=prices), context)
    assert result.returns[0] is None
    assert result.returns[1:] == pytest.approx([0.1, -0.1, 0.2])
    two_step = simple_returns.execute(simple_returns.Input(prices=prices, lag=2), context)
    assert two_step.returns[:2] == [None, None]
    assert two_step.returns[2:] == pytest.approx([-0.01, 0.08])
    short = simple_returns.execute(simple_returns.Input(prices=[12.0], lag=3), context)
    assert short.returns == [None]


def test_rolling_zscore_population_window_and_constant_policy(context: OperationContext) -> None:
    request = rolling_zscore.Input(values=[1.0, 2.0, 3.0, 6.0], window=3)
    result = rolling_zscore.execute(request, context)
    assert result.zscores[:2] == [None, None]
    assert result.zscores[2:] == pytest.approx([sqrt(3.0 / 2.0), 7.0 / sqrt(26.0)])
    constant = rolling_zscore.execute(
        rolling_zscore.Input(values=[4.0, 4.0, 4.0], window=2), context
    )
    assert constant.zscores == [None, None, None]
    short = rolling_zscore.execute(rolling_zscore.Input(values=[1.0], window=3), context)
    assert short.zscores == [None]


def test_rolling_zscore_preserves_tiny_and_offset_differences(context: OperationContext) -> None:
    tiny = rolling_zscore.execute(
        rolling_zscore.Input(values=[1e-300, 2e-300, 3e-300], window=3), context
    )
    assert tiny.zscores[-1] == pytest.approx(sqrt(3.0 / 2.0))
    shifted = rolling_zscore.execute(
        rolling_zscore.Input(values=[1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0], window=3), context
    )
    assert shifted.zscores[-1] == pytest.approx(sqrt(3.0 / 2.0))
    rounded_mean = rolling_zscore.execute(
        rolling_zscore.Input(values=[1e16, 1e16 + 2.0], window=2), context
    )
    assert rounded_mean.zscores[-1] == pytest.approx(1.0)
    large_constant = rolling_zscore.execute(
        rolling_zscore.Input(values=[1e150] * 7, window=7), context
    )
    assert large_constant.zscores == [None] * 7


def test_ewma_forecasts_exclude_current_observation(context: OperationContext) -> None:
    request = ewma_variance.Input(returns=[2.0, -1.0, 3.0], decay=0.5, initial_variance=4.0)
    result = ewma_variance.execute(request, context)
    assert result.forecast_variances == pytest.approx([4.0, 4.0, 2.5])
    assert result.next_variance == pytest.approx(5.75)
    single = ewma_variance.execute(
        ewma_variance.Input(returns=[3.0], decay=0.0, initial_variance=2.0), context
    )
    assert single.forecast_variances == [2.0]
    assert single.next_variance == 9.0
    changed = ewma_variance.execute(
        ewma_variance.Input(returns=[2.0, -1.0, 300.0], decay=0.5, initial_variance=4.0),
        context,
    )
    assert changed.forecast_variances == result.forecast_variances
    assert changed.next_variance != result.next_variance


def test_drawdown_path_peak_ties_and_duration(context: OperationContext) -> None:
    result = drawdown_path.execute(
        drawdown_path.Input(prices=[100.0, 120.0, 90.0, 100.0, 120.0, 60.0, 150.0]), context
    )
    assert result.high_water_marks == [100.0, 120.0, 120.0, 120.0, 120.0, 120.0, 150.0]
    assert result.drawdowns == pytest.approx([0.0, 0.0, -0.25, -1.0 / 6.0, 0.0, -0.5, 0.0])
    assert result.durations == [0, 0, 1, 2, 0, 1, 0]
    single = drawdown_path.execute(drawdown_path.Input(prices=[2.0]), context)
    assert single.drawdowns == [0.0]
    assert single.high_water_marks == [2.0]
    assert single.durations == [0]


@pytest.mark.parametrize(
    ("module", "arguments", "input_key", "output_keys"),
    [
        (simple_returns, {"lag": 2}, "prices", ["returns"]),
        (rolling_zscore, {"window": 3}, "values", ["zscores"]),
        (ewma_variance, {"decay": 0.8}, "returns", ["forecast_variances"]),
        (drawdown_path, {}, "prices", ["high_water_marks", "drawdowns", "durations"]),
    ],
)
def test_every_feature_is_prefix_invariant(
    module: Any,
    arguments: dict[str, Any],
    input_key: str,
    output_keys: list[str],
    context: OperationContext,
) -> None:
    values = [3.0, 2.0, 5.0, 7.0, 4.0, 9.0]
    full = module.execute(module.Input(**arguments, **{input_key: values}), context)
    for length in range(1, len(values)):
        prefix = module.execute(module.Input(**arguments, **{input_key: values[:length]}), context)
        for output_key in output_keys:
            assert getattr(prefix, output_key) == getattr(full, output_key)[:length]


@pytest.mark.parametrize(
    ("module", "input_key"),
    [
        (simple_returns, "prices"),
        (rolling_zscore, "values"),
        (ewma_variance, "returns"),
        (drawdown_path, "prices"),
    ],
)
@pytest.mark.parametrize("invalid", [True, "1.0", float("nan"), float("inf"), -float("inf")])
def test_numeric_inputs_reject_coercion_and_nonfinite(
    module: Any, input_key: str, invalid: Any
) -> None:
    with pytest.raises(ValidationError):
        module.Input(**{input_key: [invalid]})


@pytest.mark.parametrize(
    ("module", "input_key"),
    [
        (simple_returns, "prices"),
        (rolling_zscore, "values"),
        (ewma_variance, "returns"),
        (drawdown_path, "prices"),
    ],
)
def test_feature_input_length_bounds(module: Any, input_key: str) -> None:
    with pytest.raises(ValidationError):
        module.Input(**{input_key: []})
    with pytest.raises(ValidationError):
        module.Input(**{input_key: [1.0] * 10_001})


@pytest.mark.parametrize("prices", [[0.0], [-1.0], [1e-151], [1e151]])
def test_price_operations_reject_nonpositive_or_extreme_prices(prices: list[float]) -> None:
    for module in (simple_returns, drawdown_path):
        with pytest.raises(ValidationError):
            module.Input(prices=prices)


@pytest.mark.parametrize("lag", [0, -1, 10_001, True, "2"])
def test_return_lag_bounds_and_type(lag: Any) -> None:
    with pytest.raises(ValidationError):
        simple_returns.Input(prices=[1.0], lag=lag)


@pytest.mark.parametrize("window", [1, 1_001, True, "2"])
def test_zscore_window_bounds_and_type(window: Any) -> None:
    with pytest.raises(ValidationError):
        rolling_zscore.Input(values=[1.0], window=window)


@pytest.mark.parametrize("decay", [-0.01, 1.0, True, "0.9", float("nan")])
def test_ewma_decay_bounds_and_type(decay: Any) -> None:
    with pytest.raises(ValidationError):
        ewma_variance.Input(returns=[1.0], decay=decay)


@pytest.mark.parametrize("initial", [-1.0, 1e301, True, "1.0", float("inf")])
def test_ewma_initial_variance_bounds_and_type(initial: Any) -> None:
    with pytest.raises(ValidationError):
        ewma_variance.Input(returns=[1.0], initial_variance=initial)
