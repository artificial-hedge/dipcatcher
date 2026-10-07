import numpy as np
import pytest

from quant_fund.combination.trimming import (
    median_ensemble,
    trimmed_mean_ensemble,
    winsorize_members,
)
from tests.unit.combination._helpers import crps_from_quantiles

pytestmark = pytest.mark.synthetic


def test_trimmed_mean_of_constants_is_constant() -> None:
    members = np.tile(np.linspace(-1, 1, 5), (6, 1))
    out = trimmed_mean_ensemble(members, 0.2)
    np.testing.assert_allclose(out, np.linspace(-1, 1, 5), atol=1e-12)


def test_median_ensemble_of_constants_is_constant() -> None:
    members = np.tile(np.linspace(-1, 1, 5), (5, 1))
    out = median_ensemble(members)
    np.testing.assert_allclose(out, np.linspace(-1, 1, 5), atol=1e-12)


def test_trimmed_better_than_plain_mean_with_outlier() -> None:
    rng = np.random.default_rng(120)
    from scipy.stats import norm

    taus = np.linspace(0.05, 0.95, 19)
    z = norm.ppf(taus)
    members = [z + rng.normal(0, 0.05, len(taus)) for _ in range(6)]
    members.append(z + 3.0)  # one badly biased member
    q = np.stack(members)
    y = rng.standard_normal(1500)
    pred_plain = np.broadcast_to(np.mean(q, axis=0), (len(y), len(taus)))
    pred_trim = np.broadcast_to(trimmed_mean_ensemble(q, 0.15), (len(y), len(taus)))
    crps_plain = crps_from_quantiles(y, pred_plain, taus)
    crps_trim = crps_from_quantiles(y, pred_trim, taus)
    assert crps_trim < crps_plain


def test_median_robust_to_single_wild_member() -> None:
    members = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.1, 1.1, 2.1],
            [0.2, 1.2, 2.2],
            [50.0, 60.0, 70.0],
            [0.15, 1.15, 2.15],
        ]
    )
    out = median_ensemble(members)
    np.testing.assert_allclose(out, [0.15, 1.15, 2.15], atol=1e-12)


def test_trim_with_three_members_keeps_middle() -> None:
    members = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = trimmed_mean_ensemble(members, 0.3)  # floor(0.9)=1 → keeps [3,4]
    np.testing.assert_allclose(out, [3.0, 4.0])


def test_winsorize_members_clamps_extremes() -> None:
    rng = np.random.default_rng(125)
    clean = rng.normal(0, 1, (99, 3))
    outlier = np.full((1, 3), 100.0)
    members = np.vstack([clean, outlier])
    out = winsorize_members(members, (0.05, 0.95))
    # the outlier is pulled to the cross-member 95% band (≈ 1.7σ with n=100)
    assert np.max(np.abs(out[99])) < 3.0
    # order across members is preserved by clipping
    for j in range(3):
        assert np.all(np.diff(out[:, j]) >= 0.0) == np.all(np.diff(members[:, j]) >= 0.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        trimmed_mean_ensemble(np.zeros(5), 0.1)
    with pytest.raises(ValueError):
        trimmed_mean_ensemble(np.zeros((3, 5)), 0.6)
