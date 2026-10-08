"""NaN-feature imputation probes for GaussianHMMRegime (raw-0 fill was biased)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.regime import GaussianHMMRegime


def _fitted(seed: int = 0) -> GaussianHMMRegime:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, 2))
    return GaussianHMMRegime(2, seed=0).fit(x)


def test_predict_proba_nan_feature_matches_training_mean_row() -> None:
    model = _fitted()
    probe_nan = np.array([[np.nan, 0.3]])
    probe_mean = np.array([[model.scaler.mean_[0], 0.3]])
    a = model.predict_proba(probe_nan)
    b = model.predict_proba(probe_mean)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_predict_smoothed_and_aic_use_same_neutral_fill() -> None:
    model = _fitted()
    probe_nan = np.array([[np.nan, 0.3], [0.1, -0.2]])
    probe_mean = probe_nan.copy()
    probe_mean[0, 0] = model.scaler.mean_[0]
    np.testing.assert_allclose(
        model.predict_smoothed_proba(probe_nan), model.predict_smoothed_proba(probe_mean)
    )
    assert model.aic_bic(probe_nan) == pytest.approx(model.aic_bic(probe_mean))


def test_fit_with_nan_rows_uses_column_mean_not_zero() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(loc=4.0, scale=0.5, size=(300, 2))
    x[::4, 0] = np.nan
    model = GaussianHMMRegime(2, seed=1).fit(x)
    # Feature 0 centers near 4; a raw-0 fill would have dragged the scaler
    # mean far below the true column mean.
    assert model.scaler.mean_[0] == pytest.approx(np.nanmean(x[:, 0]), rel=1e-6)
