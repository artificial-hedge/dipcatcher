"""Wave-185 differentiable-algorithm canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.gumbel_relax import bench_gumbel_relax
from quant_fund.models.implicit_diff import bench_implicit_diff
from quant_fund.models.ode_adjoint import bench_ode_adjoint
from quant_fund.models.perturb_map import bench_perturb_map
from quant_fund.models.smooth_argmax import bench_smooth_argmax
from quant_fund.models.st_estimator import bench_st_estimator


class TestSTE:
    def test_bench(self) -> None:
        out = bench_st_estimator(seed=3, trials=5)
        assert -1.0 <= out["synthetic_ste_grad_corr"] <= 1.0


class TestGumbel:
    def test_bench(self) -> None:
        out = bench_gumbel_relax(seed=5, trials=5)
        assert -1.0 <= out["synthetic_gumbel_grad_corr"] <= 1.0


class TestPMAP:
    def test_bench(self) -> None:
        out = bench_perturb_map(seed=7, trials=5, M=50)
        assert -1.0 <= out["synthetic_pmap_grad_corr"] <= 1.0


class TestIFT:
    def test_bench(self) -> None:
        out = bench_implicit_diff(seed=9)
        assert out["synthetic_ift_corr"] > 0.99


class TestAdjoint:
    def test_bench(self) -> None:
        out = bench_ode_adjoint(seed=11)
        assert out["synthetic_adjoint_corr"] > 0.9


class TestSmax:
    def test_bench(self) -> None:
        out = bench_smooth_argmax(seed=13, trials=5)
        assert np.isfinite(out["synthetic_smax_sharp_corr"])
