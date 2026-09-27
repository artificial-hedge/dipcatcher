"""Tests for quant_fund.models.ngboost_lite — NGBoostLite natural-gradient boosting.

All randomness is seeded via np.random.default_rng(fixed_int); every test is
deterministic run-to-run.
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from quant_fund.metrics.scoring import crps_gaussian
from quant_fund.models.ngboost_lite import NGBoostLite

SEED = 20260927
N = 600


def _hetero_data(seed: int, n: int = N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y = sin(2x) + sigma(x) * eps with monotone sigma(x) (heteroskedastic)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n).reshape(-1, 1)
    mu = np.sin(2.0 * x.ravel())
    sigma = 0.3 + 0.6 * (x.ravel() + 3.0) / 6.0  # 0.3 .. 0.9, increasing in x
    y = mu + sigma * rng.standard_normal(n)
    return x, y, sigma


def _constant_baseline_crps(y: np.ndarray) -> float:
    """Global constant-(mu, sigma) baseline fit on the same data: init params."""
    mu = np.full(y.shape[0], float(np.mean(y)))
    sigma = np.full(y.shape[0], float(np.std(y)))
    return float(np.mean(crps_gaussian(y, mu, sigma)))


def test_recovers_sigma_ordering() -> None:
    """(a) predicted sigma ranks with the true heteroskedastic sigma."""
    x, y, sigma_true = _hetero_data(SEED)
    model = NGBoostLite(random_state=SEED).fit(x, y)
    _, sigma_hat = model.predict_params(x)
    rho = spearmanr(sigma_hat, sigma_true).statistic
    assert rho > 0.5, f"sigma rank correlation {rho:.3f} <= 0.5"


def test_crps_beats_constant_baseline() -> None:
    """(b) strict improvement over the constant-(mu, sigma) baseline (its init)."""
    x, y, _ = _hetero_data(SEED + 1)
    model = NGBoostLite(random_state=SEED).fit(x, y)
    assert model.mean_crps(x, y) < _constant_baseline_crps(y)


def test_mle_score_beats_constant_baseline() -> None:
    """(c) the log-score variant also strictly beats the constant baseline."""
    x, y, _ = _hetero_data(SEED + 2)
    model = NGBoostLite(score="mle", random_state=SEED).fit(x, y)
    assert model.mean_crps(x, y) < _constant_baseline_crps(y)


def test_student_t_variant() -> None:
    """(d) student_t at fixed df runs, stays finite, and beats the baseline."""
    x, y, _ = _hetero_data(SEED + 3)
    model = NGBoostLite(dist="student_t", df=5.0, random_state=SEED).fit(x, y)
    mu, sigma = model.predict_params(x)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)
    assert model.mean_crps(x, y) < _constant_baseline_crps(y)


def test_determinism_same_random_state() -> None:
    """(e) two fits with the same random_state give identical parameters."""
    x, y, _ = _hetero_data(SEED + 4)
    m1 = NGBoostLite(random_state=SEED).fit(x, y)
    m2 = NGBoostLite(random_state=SEED).fit(x, y)
    mu1, s1 = m1.predict_params(x)
    mu2, s2 = m2.predict_params(x)
    np.testing.assert_array_equal(mu1, mu2)
    np.testing.assert_array_equal(s1, s2)


def test_sample_shape_and_seeded() -> None:
    """Sampling draws (n, n_samples) and reproduces under the model seed."""
    x, y, _ = _hetero_data(SEED + 5)
    model = NGBoostLite(random_state=SEED).fit(x, y)
    draws = model.sample(x, 25)
    assert draws.shape == (x.shape[0], 25)
    assert np.all(np.isfinite(draws))
    np.testing.assert_array_equal(draws, model.sample(x, 25))


def test_fail_closed_small_n() -> None:
    """(f) n < 50 raises ValueError."""
    x, y, _ = _hetero_data(SEED, n=40)
    with pytest.raises(ValueError, match="at least 50"):
        NGBoostLite(random_state=SEED).fit(x, y)


def test_fail_closed_feature_ratio() -> None:
    """(f) n < 10*max(n_features, 2) raises ValueError."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((60, 20))  # 60 < 10*20 = 200
    y = x[:, 0] + rng.standard_normal(60)
    with pytest.raises(ValueError, match=r"10\*max"):
        NGBoostLite(random_state=SEED).fit(x, y)


def test_fail_closed_constant_y() -> None:
    """(f) constant y raises ValueError."""
    x, _, _ = _hetero_data(SEED)
    with pytest.raises(ValueError, match="constant y"):
        NGBoostLite(random_state=SEED).fit(x, np.full(N, 1.25))


def test_fail_closed_nan_inf() -> None:
    """(f) NaN/inf in X or y raises ValueError."""
    x, y, _ = _hetero_data(SEED)
    x_bad = x.copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        NGBoostLite(random_state=SEED).fit(x_bad, y)
    y_bad = y.copy()
    y_bad[1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        NGBoostLite(random_state=SEED).fit(x, y_bad)


def test_predict_before_fit_raises() -> None:
    """(f) any predict/score method before fit raises RuntimeError."""
    x, _, _ = _hetero_data(SEED)
    model = NGBoostLite(random_state=SEED)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_params(x)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.sample(x, 5)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.mean_crps(x, np.zeros(N))


def test_constructor_fail_closed() -> None:
    """Bad hyperparameters raise ValueError at construction."""
    with pytest.raises(ValueError):
        NGBoostLite(dist="gamma")
    with pytest.raises(ValueError):
        NGBoostLite(score="pinball")
    with pytest.raises(ValueError):
        NGBoostLite(n_estimators=0)
    with pytest.raises(ValueError):
        NGBoostLite(learning_rate=-0.1)
    with pytest.raises(ValueError):
        NGBoostLite(dist="student_t", df=2.0)
