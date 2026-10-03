"""Wave-165 conditional-density canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._cd_synth import cd_data, gauss_logpdf
from quant_fund.models.crps_net import bench_crps_net
from quant_fund.models.diffusion_regressor import bench_diffusion_regressor
from quant_fund.models.flow_regression import bench_flow_regression
from quant_fund.models.het_gp import bench_het_gp
from quant_fund.models.kernel_mixture import bench_kernel_mixture
from quant_fund.models.mdn_cond import bench_mdn_cond


class TestCDSynth:
    def test_data(self) -> None:
        x, y, xt, yt = cd_data(seed=3, n=40)
        assert x.shape == (40, 3) and y.shape == (40,)

    def test_gauss_logpdf(self) -> None:
        out = gauss_logpdf(np.array([0.0]), 0.0, 1.0)
        assert out[0] < 0


class TestMDN:
    def test_bench(self) -> None:
        out = bench_mdn_cond(seed=5, iters=30)
        assert np.isfinite(out["synthetic_mdn_test_ll"])


class TestFlow:
    def test_bench(self) -> None:
        out = bench_flow_regression(seed=7, iters=30)
        assert np.isfinite(out["synthetic_cnf_test_ll"])


class TestDiffReg:
    def test_bench(self) -> None:
        out = bench_diffusion_regressor(seed=9, iters=30, T=6, n_samp=4)
        assert np.isfinite(out["synthetic_diffreg_test_ll"])


class TestHetGP:
    def test_bench(self) -> None:
        out = bench_het_gp(seed=11)
        assert np.isfinite(out["synthetic_hetgp_test_ll"])


class TestCRPS:
    def test_bench(self) -> None:
        out = bench_crps_net(seed=13, iters=30)
        assert np.isfinite(out["synthetic_crps_test_ll"])


class TestKMN:
    def test_bench(self) -> None:
        out = bench_kernel_mixture(seed=15, iters=30)
        assert np.isfinite(out["synthetic_kmn_test_ll"])
