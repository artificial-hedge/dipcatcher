import numpy as np
import pytest

from quant_fund.flowbars.fracdiff import (
    adf_pvalue,
    frac_diff_apply,
    frac_diff_weights,
    frac_integrate_weights,
    min_stationary_d,
    synth_frac_integrated,
)

pytestmark = pytest.mark.synthetic


def test_weights_d1_are_first_difference() -> None:
    w = frac_diff_weights(1.0, 5)
    np.testing.assert_allclose(w, [1.0, -1.0, 0.0, 0.0, 0.0])


def test_weights_d0_is_identity() -> None:
    w = frac_diff_weights(0.0, 4)
    np.testing.assert_allclose(w, [1.0, 0.0, 0.0, 0.0])


def test_weights_fractional_negative_and_shrink() -> None:
    w = frac_diff_weights(0.4, 50)
    assert w[0] == 1.0
    # for 0 < d < 1 all higher-order weights are negative and decaying
    assert np.all(w[1:] < 0)
    assert np.all(np.abs(w[1:]) <= np.abs(w[:-1]) + 1e-12)


def test_integrate_weights_positive() -> None:
    g = frac_integrate_weights(0.4, 30)
    assert np.all(g > 0)
    assert np.all(np.diff(g) < 0)


def test_apply_d1_matches_diff() -> None:
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    y = frac_diff_apply(x, 1.0, window=2)
    np.testing.assert_allclose(y[1:], np.diff(x))
    assert y[0] == 1.0


def test_apply_d0_is_identity() -> None:
    x = np.arange(10, dtype=np.float64)
    np.testing.assert_allclose(frac_diff_apply(x, 0.0), x)


def test_apply_constant_noninteger_is_not_zero() -> None:
    # fractional differentiation of a constant is non-trivial (unlike d=1)
    x = np.full(200, 2.0)
    y = frac_diff_apply(x, 0.5, window=60)
    assert np.any(np.abs(y) > 1e-6)


def test_adf_constant_series_fail_closed() -> None:
    assert adf_pvalue(np.full(50, 1.0)) == 1.0


def test_adf_white_noise_rejects_unit_root() -> None:
    rng = np.random.default_rng(20)
    assert adf_pvalue(rng.standard_normal(1000)) < 0.05


def test_adf_random_walk_fails_to_reject() -> None:
    rng = np.random.default_rng(21)
    x = np.cumsum(rng.standard_normal(1000))
    assert adf_pvalue(x) > 0.05


def test_min_stationary_d_stationary_series_stays_at_zero() -> None:
    # an FI(0.4) series is already stationary, so the ADF scan correctly
    # reports no differentiation needed
    y = synth_frac_integrated(2500, d=0.4, seed=22)
    out = min_stationary_d(y, np.linspace(0.0, 1.0, 21), alpha=0.05)
    assert out["d"] == 0.0
    assert out["pvalue"] < 0.05


def test_min_stationary_d_on_random_walk_needs_differentiation() -> None:
    rng = np.random.default_rng(23)
    x = np.cumsum(rng.standard_normal(1500))
    out = min_stationary_d(x, np.linspace(0.0, 1.0, 21), alpha=0.05)
    # ADF has limited power against long-memory alternatives (documented
    # limitation): it starts rejecting the unit root well before d = 1, so
    # the scan lands in the fractional region rather than at exactly 1.
    assert 0.25 <= out["d"] <= 1.0
    assert out["pvalue"] < 0.05


def test_synth_frac_integrated_deterministic() -> None:
    a = synth_frac_integrated(300, d=0.3, seed=5)
    b = synth_frac_integrated(300, d=0.3, seed=5)
    np.testing.assert_array_equal(a, b)


def test_synth_frac_integrated_rejects_bad_d() -> None:
    with pytest.raises(ValueError):
        synth_frac_integrated(100, d=0.7)
