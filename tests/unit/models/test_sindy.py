"""Tests for models/sindy.py — SYNTHETIC correctness only."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.sindy import (
    _monomial_powers,
    bench_sindy,
    library_eval,
    polynomial_library,
    simulate_sindy,
    sindy_fit,
    sindy_predict,
    stlsq,
    synth_lorenz,
    synth_van_der_pol,
)


class TestPowers:
    def test_powers_cover_degrees(self) -> None:
        powers = _monomial_powers(2, 3)
        assert (0, 0) in powers  # constant
        assert (1, 0) in powers and (0, 2) in powers
        assert max(sum(p) for p in powers) == 3
        assert len(set(powers)) == len(powers)

    def test_powers_order(self) -> None:
        assert _monomial_powers(3, 1) == sorted(_monomial_powers(3, 1))


class TestLibrary:
    def test_shapes_and_names(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((50, 3))
        theta, names, powers = polynomial_library(x, order=2)
        n, d = x.shape
        assert theta.shape == (n, len(powers))
        assert len(names) == len(powers)
        assert "1" in names and any("*" in nme for nme in names)

    def test_constant_column(self) -> None:
        x = np.arange(40.0).reshape(20, 2) + 1
        theta, names, _ = polynomial_library(x, order=1)
        j = names.index("1")
        assert np.allclose(theta[:, j], 1.0)

    def test_library_eval_reproduces(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal((30, 3))
        theta, _, powers = polynomial_library(x, order=3)
        theta2 = library_eval(x, powers)
        assert np.allclose(theta, theta2)

    def test_library_eval_dim_mismatch(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.standard_normal((30, 3))
        _, _, powers = polynomial_library(x, order=2)
        with pytest.raises(ValueError):
            library_eval(rng.standard_normal((30, 2)), powers)

    def test_invalid_order(self) -> None:
        x = np.zeros((20, 2))
        with pytest.raises(ValueError):
            polynomial_library(x, order=0)
        with pytest.raises(ValueError):
            polynomial_library(x, order=7)


class TestStslq:
    def test_thresholds_small(self) -> None:
        rng = np.random.default_rng(3)
        theta = rng.standard_normal((200, 5))
        xi_true = np.array([[3.0], [0.0], [-2.0], [0.05], [0.0]])
        y = theta @ xi_true + 0.01 * rng.standard_normal((200, 1))
        xi, path, it = stlsq(theta, y, lam=0.1)
        assert abs(xi[0, 0] - 3.0) < 0.05
        assert abs(xi[2, 0] + 2.0) < 0.05
        assert xi[3, 0] == 0.0 and xi[4, 0] == 0.0
        assert 1 <= it <= 20
        assert path[-1] < 0.01  # sparse refit mse stays small after thresholding

    def test_converges_stable(self) -> None:
        rng = np.random.default_rng(4)
        theta = rng.standard_normal((150, 4))
        xi, _, it = stlsq(theta, rng.standard_normal((150, 1)), lam=0.5)
        assert it >= 1

    def test_row_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            stlsq(np.ones((10, 3)), np.ones((8, 1)))

    def test_negative_lam_raises(self) -> None:
        with pytest.raises(ValueError):
            stlsq(np.ones((10, 3)), np.ones((10, 1)), lam=-1)


class TestSindyFit:
    def test_lorenz_recovers_sparse(self) -> None:
        x, _ = synth_lorenz(n=2500, dt=0.004)
        res = sindy_fit(x, dt=0.004, order=3, lam=0.05)
        assert res.n_nonzero <= 12
        assert res.rmse < 0.5

    def test_vdp_recovers(self) -> None:
        x, _ = synth_van_der_pol(n=1500, dt=0.01)
        res = sindy_fit(x, dt=0.01, order=3, lam=0.05)
        assert res.n_nonzero <= 8
        assert res.rmse < 0.1

    def test_predict_integrates(self) -> None:
        x, _ = synth_lorenz(n=2500, dt=0.004)
        res = sindy_fit(x, dt=0.004, order=3, lam=0.05)
        t = np.arange(100) * 0.004
        pred = sindy_predict(res, x[0], t)
        assert pred.shape == (100, 3)
        assert np.all(np.isfinite(pred))

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            sindy_fit(np.ones(5), 0.01)
        with pytest.raises(ValueError):
            sindy_fit(np.ones((20, 2)), -0.01)
        with pytest.raises(ValueError):
            sindy_fit(np.full((20, 2), np.nan), 0.01)


class TestBench:
    def test_bench_keys_finite(self) -> None:
        blob = bench_sindy()
        assert len(blob) >= 8
        assert all(np.isfinite(v) for v in blob.values())
        for key in blob:
            assert key.startswith("synthetic_")
            assert {"sharpe", "sortino", "calmar", "pnl", "nav"}.isdisjoint(key.lower().split("_"))

    def test_bench_quality(self) -> None:
        blob = bench_sindy()
        assert blob["synthetic_coef_relerr"] < 0.1
        assert blob["synthetic_n_nonzero_err"] == 0.0
        assert blob["synthetic_determinism"] == 1.0
        assert blob["synthetic_traj_relerr"] < 0.05

    def test_simulate_sindy_shape(self) -> None:
        x, _ = synth_lorenz(n=2000, dt=0.004)
        res = sindy_fit(x, dt=0.004, order=3, lam=0.05)
        traj = simulate_sindy(res, x[0], n=50, dt=0.004)
        assert traj.shape == (50, 3)
        with pytest.raises(ValueError):
            simulate_sindy(res, x[0], n=1, dt=0.004)
