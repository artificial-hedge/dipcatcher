"""Wave-187 scientific-ML/PDE-solver canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.deepritz_pinn import bench_deepritz_pinn
from quant_fund.models.fbsde_solver import bench_fbsde_solver
from quant_fund.models.feynman_kac_mc import bench_feynman_kac_mc
from quant_fund.models.moc_lines import bench_moc_lines
from quant_fund.models.spectral_pde import bench_spectral_pde
from quant_fund.models.weak_form_pinn import bench_weak_form_pinn


class TestPINN:
    def test_bench(self) -> None:
        out = bench_deepritz_pinn(seed=3, iters=60)
        assert np.isfinite(out["synthetic_pinn_rel_l2"])


class TestWeak:
    def test_bench(self) -> None:
        out = bench_weak_form_pinn(seed=5, iters=60)
        assert np.isfinite(out["synthetic_weak_rel_l2"])


class TestBSDE:
    def test_bench(self) -> None:
        out = bench_fbsde_solver(seed=7, iters=40, n_steps=8)
        assert np.isfinite(out["synthetic_bsde_rel_l2"])


class TestSpectral:
    def test_bench(self) -> None:
        out = bench_spectral_pde(nx=64)
        assert out["synthetic_spectral_rel_l2"] < 0.1


class TestMOC:
    def test_bench(self) -> None:
        out = bench_moc_lines(nx=51, nt=400)
        assert out["synthetic_moc_rel_l2"] < 0.1


class TestFKMC:
    def test_bench(self) -> None:
        out = bench_feynman_kac_mc(seed=11, n_mc=500)
        assert out["synthetic_fkmc_rel_l2"] < 0.2
