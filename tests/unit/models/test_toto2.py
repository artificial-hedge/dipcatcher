"""toto2 fleet head: dep gate, contract shape, sample-quantile reduction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quant_fund.models.toto2 import (
    Toto2Distribution,
    _prediction_quantiles,
)


class _FakeModel:
    """Returns a (n_samples,) next-bar z-return draw for each call."""

    def __init__(self, loc: float = 0.01) -> None:
        self.loc = loc
        self.calls = 0

    def predict(self, window: Any) -> Any:
        self.calls += 1

        # Deterministic spread: 32 samples centered at loc, on a
        # .samples carrier (the tightened contract refuses bare vectors
        # that are not exactly the grid).
        class _Out:
            samples = self.loc + np.linspace(-0.02, 0.02, 32)

        return _Out()


def _fit_stubbed(**kwargs: Any) -> tuple[Toto2Distribution, _FakeModel]:
    pred = _FakeModel()
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.01, size=300)
    model = Toto2Distribution(predictor=pred, **kwargs).fit(np.zeros((300, 2)), y)
    return model, pred


def test_fit_without_dep_fails_closed() -> None:
    model = Toto2Distribution()
    with pytest.raises(RuntimeError, match="toto-2"):
        model.fit(np.zeros((300, 2)), np.zeros(300))


def test_predict_shape_and_unstandardize() -> None:
    model, _ = _fit_stubbed(lookback=32)
    q = model.predict(np.zeros((7, 2)))
    assert q.shape == (7, 3)
    # Samples ~ N(0.01, 0.02/16-ish spread); rescaled by fitted (mu, sig).
    assert np.isfinite(q).all()
    assert (np.diff(q, axis=1) > 0).all()  # strict increase across taus


def test_predict_from_history_causal_windows() -> None:
    model, _ = _fit_stubbed(lookback=16)
    history = np.random.default_rng(1).normal(0.0, 0.01, size=50)
    q = model.predict_from_history(history)
    assert q.shape == (50 - 16 + 1, 3)
    assert np.isfinite(q).all()


def test_rejects_bad_inputs() -> None:
    model, _ = _fit_stubbed(lookback=16)
    with pytest.raises(ValueError, match=">= lookback"):
        model.predict_from_history(np.zeros(4))
    with pytest.raises(RuntimeError, match="not been fitted"):
        Toto2Distribution(predictor=_FakeModel()).predict(np.zeros((2, 2)))


def test_prediction_quantiles_grid_or_samples() -> None:
    taus = np.array([0.1, 0.5, 0.9])
    # Vector on the grid.
    np.testing.assert_allclose(
        _prediction_quantiles(np.array([0.0, 1.0, 2.0]), taus), [0.0, 1.0, 2.0]
    )

    # Sample axis -> empirical quantiles (only via a .samples carrier).
    class _Sampled:
        def __init__(self, s: np.ndarray) -> None:
            self.samples = s

    q = _prediction_quantiles(_Sampled(np.linspace(0.0, 1.0, 101)), taus)
    np.testing.assert_allclose(q, [0.1, 0.5, 0.9], atol=0.01)
    # Wrong-grid vector fails closed.
    with pytest.raises(ValueError, match="cannot be reduced"):
        _prediction_quantiles(np.array([0.5, 0.6]), taus)
    with pytest.raises(ValueError, match="non-finite"):
        _prediction_quantiles(np.array([0.0, np.nan, 2.0]), taus)


def test_metadata_flags() -> None:
    model, _ = _fit_stubbed()
    extra = model.metadata().extra
    assert extra["framework"] == "toto-2"
    assert "uv.lock-incompatible" in extra["weights_note"]
    assert extra["pretrained"] is True


def test_fleet_registry_includes_toto2() -> None:
    from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY

    assert "toto2" in FLEET_HEAD_REGISTRY
