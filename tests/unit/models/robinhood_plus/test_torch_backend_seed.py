"""robinhood_plus torch_backend determinism probes (SYNTHETIC).

The official KronosPredictor samples from torch's global RNG: the
cross-section forecast must seed a forked stream per (seed, asof, name) so
repeated calls and different cross-section membership are deterministic.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import AppConfig

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="requires the nn extra (torch)")


def _frame(n_dates: int = 8, names: int = 2) -> pl.DataFrame:
    rows = []
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    for d in range(n_dates):
        for k in range(names):
            rows.append(
                {
                    "security_id": f"S{k}",
                    "event_time": t0 + timedelta(days=d),
                    "open_split_adjusted": 100.0 + d,
                    "high_split_adjusted": 101.0 + d,
                    "low_split_adjusted": 99.0 + d,
                    "close_split_adjusted": 100.5 + d,
                    "volume": 1000.0,
                    "amount": 100_000.0,
                }
            )
    return pl.DataFrame(rows)


def _cfg() -> AppConfig:
    return AppConfig.model_validate(
        {"robinhood_plus": {"lookback": 4, "pred_len": 3, "sample_count": 2}}
    )


class _RngEatingPredictor:
    """Stand-in for KronosPredictor: each predict call consumes torch RNG."""

    def predict(self, *, df: Any, pred_len: int, **kwargs: Any) -> Any:
        import pandas as pd
        import torch

        base = np.asarray(df.iloc[-1].to_numpy(), dtype=float)
        noise = torch.randn(pred_len, base.size).numpy()
        return pd.DataFrame(np.tile(base, (pred_len, 1)) + 0.01 * noise, columns=list(df.columns))


def test_cross_section_forecasts_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.models.robinhood_plus import torch_backend as tb

    monkeypatch.setattr(tb, "_cached_predictor", lambda config: _RngEatingPredictor())
    frame = _frame()
    cfg = _cfg()
    asof = datetime(2024, 1, 10, tzinfo=UTC)
    first = tb.forecast_cross_section_torch(frame, asof, cfg, ["S0", "S1"])
    second = tb.forecast_cross_section_torch(frame, asof, cfg, ["S0", "S1"])
    assert np.array_equal(first["S0"].forecast.paths, second["S0"].forecast.paths)
    assert np.array_equal(first["S1"].forecast.paths, second["S1"].forecast.paths)


def test_name_forecast_independent_of_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.models.robinhood_plus import torch_backend as tb

    monkeypatch.setattr(tb, "_cached_predictor", lambda config: _RngEatingPredictor())
    frame = _frame(names=3)
    cfg = _cfg()
    asof = datetime(2024, 1, 10, tzinfo=UTC)
    solo = tb.forecast_cross_section_torch(frame, asof, cfg, ["S0"])
    both = tb.forecast_cross_section_torch(frame, asof, cfg, ["S1", "S0"])
    assert np.array_equal(solo["S0"].forecast.paths, both["S0"].forecast.paths)


def test_global_torch_stream_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from quant_fund.models.robinhood_plus import torch_backend as tb

    monkeypatch.setattr(tb, "_cached_predictor", lambda config: _RngEatingPredictor())
    torch.manual_seed(77)
    reference = torch.randn(8)
    torch.manual_seed(77)
    tb.forecast_cross_section_torch(_frame(), datetime(2024, 1, 10, tzinfo=UTC), _cfg(), ["S0"])
    assert torch.equal(torch.randn(8), reference)


def test_forecast_seed_stable() -> None:
    from quant_fund.models.robinhood_plus.torch_backend import _forecast_seed

    asof = datetime(2024, 1, 10, tzinfo=UTC)
    a = _forecast_seed(42, asof, "S0")
    assert a == _forecast_seed(42, asof, "S0")
    assert a != _forecast_seed(42, asof, "S1")
    assert 0 <= a < 2**31 - 1
