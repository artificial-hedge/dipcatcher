"""kronos_base fleet head: contract shape, dep gate, disclosed input mode."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quant_fund.models.kronos_fleet import (
    KronosFleetDistribution,
    _returns_to_candle_frame,
)


class _FakePredictor:
    """Emits candle paths with close = last_close * (1 + drift)."""

    def __init__(self, drift: float = 0.01) -> None:
        self.drift = drift
        self.calls = 0

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: Any,
        y_timestamp: Any,
        pred_len: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        self.calls += 1
        last = float(df["close"].iloc[-1])
        out = pd.DataFrame(
            {
                "open": [last] * pred_len,
                "high": [last * 1.001] * pred_len,
                "low": [last * 0.999] * pred_len,
                "close": [last * (1.0 + self.drift)] * pred_len,
                "volume": [0.0] * pred_len,
                "amount": [0.0] * pred_len,
            }
        )
        return out


def _fit_stubbed(**kwargs: Any) -> tuple[KronosFleetDistribution, _FakePredictor]:
    pred = _FakePredictor()
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.01, size=300)
    model = KronosFleetDistribution(predictor=pred, **kwargs).fit(np.zeros((300, 2)), y)
    return model, pred


def test_fit_without_predictor_fails_closed() -> None:
    model = KronosFleetDistribution()
    with pytest.raises(RuntimeError, match="kronos_base requires"):
        model.fit(np.zeros((300, 2)), np.zeros(300))


def test_predict_quantile_shape_and_content() -> None:
    model, pred = _fit_stubbed(lookback=32, n_samples=8)
    q = model.predict(np.zeros((7, 2)))
    assert q.shape == (7, 3)
    # All samples agree on +1% close -> every quantile ~log(1.01).
    np.testing.assert_allclose(q[0], np.log(1.01), rtol=1e-4)
    assert pred.calls >= 8  # n_samples independent draws


def test_predict_from_history_is_causal_windows() -> None:
    model, _ = _fit_stubbed(lookback=16, n_samples=4)
    history = np.random.default_rng(1).normal(0.0, 0.01, size=50)
    q = model.predict_from_history(history)
    assert q.shape == (50 - 16 + 1, 3)
    assert np.isfinite(q).all()
    # Monotone quantile ordering.
    assert (np.diff(q, axis=1) >= 0).all()


def test_rejects_bad_history_and_unfitted() -> None:
    model, _ = _fit_stubbed(lookback=16, n_samples=4)
    with pytest.raises(ValueError, match=">= lookback"):
        model.predict_from_history(np.zeros(8))
    unfitted = KronosFleetDistribution(predictor=_FakePredictor())
    with pytest.raises(RuntimeError, match="not been fitted"):
        unfitted.predict(np.zeros((2, 2)))


def test_metadata_discloses_input_mode() -> None:
    model, _ = _fit_stubbed()
    extra = model.metadata().extra
    assert extra["degenerate_input"] is True
    assert extra["input_mode"] == "returns_synthesized_degenerate_candles"
    assert extra["timestamps"] == "synthetic_daily"
    assert extra["framework"] == "NeoQuasarAI/Kronos"


def test_returns_to_candle_frame_coherent() -> None:
    candles = _returns_to_candle_frame(np.array([0.01, -0.02, 0.005]))
    assert list(candles.columns) == ["open", "high", "low", "close", "volume", "amount"]
    # high >= max(open, close); low <= min(open, close); volume zero.
    assert (candles["high"] >= np.maximum(candles["open"], candles["close"])).all()
    assert (candles["low"] <= np.minimum(candles["open"], candles["close"])).all()
    assert (candles["volume"] == 0.0).all()


def test_fleet_registry_includes_kronos() -> None:
    from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY

    assert "kronos_base" in FLEET_HEAD_REGISTRY
