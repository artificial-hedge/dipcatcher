"""Tests for microstructure/dgm_mm.py — DGM solver for the AS inventory
kernel. All drills SYNTHETIC; torch-dependent tests skip without `nn`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.microstructure.dgm_mm import (
    DGM_MM_SCHEMA,
    InventoryPDESpec,
)

torch = pytest.importorskip("torch")

from quant_fund.microstructure.dgm_mm import (  # noqa: E402
    DGMInventorySolver,
    dgm_mm_bench,
)


def test_spec_fail_closed() -> None:
    with pytest.raises(ValueError):
        InventoryPDESpec(T=0.0)
    with pytest.raises(ValueError):
        InventoryPDESpec(gamma=-1.0)
    with pytest.raises(ValueError):
        InventoryPDESpec(sigma=float("nan"))
    with pytest.raises(ValueError):
        InventoryPDESpec(q_max=0.0)
    s = InventoryPDESpec()
    assert s.k == pytest.approx(0.5 * s.gamma * s.sigma * s.sigma)


def test_spec_exact_solution() -> None:
    s = InventoryPDESpec(T=1.0, gamma=0.5, sigma=0.2)
    t = np.asarray([0.0, 0.5, 1.0])
    q = np.asarray([0.0, 3.0, -5.0])
    s.exact(t, q)
    # at t = T the kernel is identically 1
    assert s.exact(np.asarray([s.T]), np.asarray([7.0]))[0] == 1.0
    # symmetric in q, decreasing in |q| for t < T
    assert (
        s.exact(np.asarray([0.0]), np.asarray([-2.0]))[0]
        == s.exact(np.asarray([0.0]), np.asarray([2.0]))[0]
    )
    assert s.exact(np.asarray([0.0]), np.asarray([0.0]))[0] == 1.0
    # reservation skew is odd in q around the forward half-step
    r = s.exact_reservation(np.zeros(2), np.asarray([0.0, 5.0]))
    assert r[0] == pytest.approx(s.k * s.T * 1.0 / s.gamma)
    assert r[1] > 0.0  # forward indifference price grows with inventory


def test_solver_construction_fail_closed() -> None:
    s = InventoryPDESpec()
    with pytest.raises(ValueError):
        DGMInventorySolver(s, hidden=0)
    with pytest.raises(ValueError):
        DGMInventorySolver(s, depth=0)
    with pytest.raises(ValueError):
        DGMInventorySolver(s, n_interior=0)
    with pytest.raises(ValueError):
        DGMInventorySolver(s, lr=0.0)
    with pytest.raises(ValueError):
        DGMInventorySolver(s, seed=-1)


def test_solver_learns_the_kernel() -> None:
    """The headline check: residual + closed-form recovery."""
    s = InventoryPDESpec(T=1.0, gamma=0.5, sigma=0.2, q_max=10.0)
    solver = DGMInventorySolver(s, hidden=48, depth=3, seed=0)
    solver.train(1500)
    tg = np.linspace(0.0, s.T, 11)
    qg = np.linspace(-s.q_max, s.q_max, 11)
    tt, qq = np.meshgrid(tg, qg, indexing="ij")
    u_hat = solver.predict(tt.ravel(), qq.ravel())
    u_ex = s.exact(tt.ravel(), qq.ravel())
    mse = float(np.mean(np.square(u_hat - u_ex)))
    assert mse < 5e-3
    # residual audit: the PDE must hold off the training cloud too
    assert solver.residual_rmse() < 0.05
    # reservation skew: linear in q, correct slope in the interior
    mask = np.abs(qq.ravel()) <= s.q_max - 1.0
    rho_hat = solver.reservation(tt.ravel()[mask], qq.ravel()[mask])
    rho_ex = s.exact_reservation(tt.ravel()[mask], qq.ravel()[mask])
    assert float(np.mean(np.square(rho_hat - rho_ex))) ** 0.5 < 0.05


def test_solver_determinism() -> None:
    s = InventoryPDESpec()
    a = DGMInventorySolver(s, seed=7)
    b = DGMInventorySolver(s, seed=7)
    a.train(50)
    b.train(50)
    x_t = np.linspace(0.0, s.T, 5)
    x_q = np.linspace(-s.q_max, s.q_max, 5)
    np.testing.assert_allclose(a.predict(x_t, x_q), b.predict(x_t, x_q))
    assert a.loss_history[-1] == b.loss_history[-1]


def test_solver_seed_varies() -> None:
    s = InventoryPDESpec()
    a = DGMInventorySolver(s, seed=1)
    b = DGMInventorySolver(s, seed=2)
    a.train(20)
    b.train(20)
    x = np.linspace(0.0, s.T, 4)
    assert not np.allclose(a.predict(x, np.zeros(4)), b.predict(x, np.zeros(4)))


def test_bench_receipt_schema() -> None:
    out = dgm_mm_bench(n_steps=300, n_grid=9, seed=0)
    rec = out["receipt"]
    assert rec["schema"] == DGM_MM_SCHEMA
    assert rec["kind"] == "dgm_mm"
    assert rec["data_label"] == "SYNTHETIC"
    m = rec["metrics"]
    for key in (
        "u_mse",
        "rho_rmse",
        "interior_residual_rmse",
        "final_interior_loss",
        "final_terminal_loss",
        "final_boundary_loss",
    ):
        assert math.isfinite(m[key])
    frame = out["frame"]
    assert frame.shape[0] == 81
    assert {"t", "q", "u_dgm", "u_exact", "rho_dgm", "rho_exact"} <= set(frame.columns)


def test_bench_converged_at_reference_config() -> None:
    """Full-config drill: the solver beats the pinned error budgets."""
    out = dgm_mm_bench(n_steps=2000, n_grid=15, seed=1)
    m = out["receipt"]["metrics"]
    assert m["u_mse"] < 5e-3
    assert m["rho_rmse"] < 0.05
    assert m["interior_residual_rmse"] < 0.05


def test_bench_fail_closed() -> None:
    with pytest.raises(ValueError):
        dgm_mm_bench(n_steps=0)
    with pytest.raises(ValueError):
        dgm_mm_bench(n_grid=2)


def test_reservation_uses_potential_not_memorized_map() -> None:
    """rho comes from phi-differences: a solver that only fit u pointwise
    can't shortcut the skew."""
    s = InventoryPDESpec()
    solver = DGMInventorySolver(s, seed=0)
    solver.train(400)
    q = np.asarray([0.0])
    t = np.asarray([0.0])
    rho = solver.reservation(t, q)
    expected = (solver.phi(t, q + 1.0) - solver.phi(t, q)) / s.gamma
    np.testing.assert_allclose(rho, expected)
