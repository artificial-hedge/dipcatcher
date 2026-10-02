"""Unconstrained optimizers (NM/Powell/CG/BFGS/LM)."""

from __future__ import annotations

import numpy as np

from quant_fund.models.unconstrained_optimizers import (
    bfgs,
    levenberg_marquardt,
    nelder_mead,
    nonlinear_cg,
    powell,
)


def _rosen(x: np.ndarray) -> float:
    return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)


def _rosen_g(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
            200 * (x[1] - x[0] ** 2),
        ]
    )


def _quad(x: np.ndarray) -> float:
    return float((x[0] - 3) ** 2 + 4 * (x[1] + 1) ** 2)


def _quad_g(x: np.ndarray) -> np.ndarray:
    return np.array([2 * (x[0] - 3), 8 * (x[1] + 1)])


def test_nelder_mead_quadratic():
    r = nelder_mead(_quad, np.array([0.0, 0.0]))
    assert np.linalg.norm(r["x"] - np.array([3.0, -1.0])) < 1e-3
    assert r["f"] < 1e-8


def test_nelder_mead_rosenbrock():
    r = nelder_mead(_rosen, np.array([-1.2, 1.0]))
    assert np.linalg.norm(r["x"] - 1.0) < 1e-2


def test_powell_quadratic():
    r = powell(_quad, np.array([0.0, 0.0]))
    assert np.linalg.norm(r["x"] - np.array([3.0, -1.0])) < 1e-6


def test_nonlinear_cg():
    r = nonlinear_cg(_quad, _quad_g, np.array([0.0, 0.0]))
    assert np.linalg.norm(r["x"] - np.array([3.0, -1.0])) < 1e-4


def test_bfgs_rosenbrock():
    r = bfgs(_rosen, _rosen_g, np.array([-1.2, 1.0]))
    assert np.linalg.norm(r["x"] - 1.0) < 1e-3


def test_levenberg_marquardt_linear():
    t = np.linspace(0, 2, 30)
    yd = 1.0 + 2.0 * t

    def resid(p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * t - yd

    def jac(p: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(30), t])

    r = levenberg_marquardt(resid, jac, np.array([0.0, 0.0]))
    assert np.linalg.norm(r["x"] - np.array([1.0, 2.0])) < 1e-6


def test_lm_exponential_fit():
    t = np.linspace(0, 2, 30)
    yd = 2.0 * np.exp(0.7 * t)

    def resid(p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(p[1] * t) - yd

    def jac(p: np.ndarray) -> np.ndarray:
        return np.column_stack([np.exp(p[1] * t), p[0] * t * np.exp(p[1] * t)])

    r = levenberg_marquardt(resid, jac, np.array([1.0, 0.5]))
    assert np.linalg.norm(r["x"] - np.array([2.0, 0.7])) < 1e-6


def test_bench_optimizers():
    from quant_fund.models.unconstrained_optimizers import (
        bench_optimizers,
    )

    out = bench_optimizers(seed=534)
    assert out["synthetic_rosen_nm_err"] < 1e-3
    assert out["synthetic_rosen_cg_err"] < 1e-3
    assert out["synthetic_rosen_bf_err"] < 1e-3
    assert out["synthetic_lm_param_err"] < 1e-6
