import numpy as np
import pytest

from quant_fund.combination.quantile_average import (
    pinball,
    probability_average_gaussian,
    quantile_average,
    quantile_average_gaussian,
)
from tests.unit.combination._helpers import crps_gaussian

pytestmark = pytest.mark.synthetic


def test_quantile_average_identical_members_is_identity() -> None:
    member = np.linspace(-2, 2, 9)
    members = np.tile(member, (4, 1))
    out = quantile_average(members, np.array([0.1, 0.2, 0.3, 0.4]))
    np.testing.assert_allclose(out, member, atol=1e-12)


def test_quantile_average_repairs_crossing() -> None:
    crossing = np.array([2.0, 1.0, 0.0])
    out = quantile_average(crossing[None, :], np.array([1.0]))
    np.testing.assert_allclose(out, [0.0, 1.0, 2.0])


def test_quantile_average_gaussian_is_narrower_than_probability_average() -> None:
    mus = np.array([0.0, 0.5])
    sigmas = np.array([0.5, 1.5])
    w = np.array([0.5, 0.5])
    taus = np.linspace(0.05, 0.95, 19)
    q_va = quantile_average_gaussian(mus, sigmas, w, taus)
    q_pa = probability_average_gaussian(mus, sigmas, w, taus)
    width_va = q_va[-1] - q_va[0]
    width_pa = q_pa[-1] - q_pa[0]
    # arithmetic mean of sigmas (1.0) vs RMS (sqrt(1.25)) — the classic
    # vincentization sharpening: quantile averaging is narrower
    assert width_va == pytest.approx(2 * 1.6448536269514722)
    assert width_va < width_pa


def test_gaussian_averages_agree_for_identical_members() -> None:
    mus = np.array([1.0, 1.0])
    sigmas = np.array([0.7, 0.7])
    w = np.array([0.3, 0.7])
    taus = np.linspace(0.1, 0.9, 9)
    np.testing.assert_allclose(
        quantile_average_gaussian(mus, sigmas, w, taus),
        probability_average_gaussian(mus, sigmas, w, taus),
        atol=1e-12,
    )


def test_quantile_average_gaussian_crps_dominance() -> None:
    # combining two miscalibrated members with complementary dispersion
    # lands near the calibrated member's CRPS on standard-normal data
    sigmas = np.array([0.4, 1.6])
    w = np.array([0.5, 0.5])
    rng = np.random.default_rng(80)
    y = rng.standard_normal(4000)
    crps_a = float(np.mean(crps_gaussian(y, np.zeros(4000), np.full(4000, 0.4))))
    crps_b = float(np.mean(crps_gaussian(y, np.zeros(4000), np.full(4000, 1.6))))
    crps_va = float(
        np.mean(crps_gaussian(y, np.zeros(4000), np.full(4000, float(np.dot(w, sigmas)))))
    )
    assert crps_va <= max(crps_a, crps_b) + 1e-9


def test_pinball_scores_quantiles() -> None:
    y = np.array([0.0, 1.0, 2.0])
    q = np.array([[0.0, 0.1], [1.0, 0.9], [2.0, 1.9]])  # (n_obs, n_quantiles)
    taus = np.array([0.25, 0.75])
    losses = pinball(y, q, taus)
    assert losses.shape == (2,)
    assert losses[0] == pytest.approx(0.0)
    assert losses[1] > 0.0


def test_pinball_validates_shapes() -> None:
    with pytest.raises(ValueError):
        pinball(np.zeros(3), np.zeros((2, 3)), np.array([0.25, 0.75, 0.9]))
