"""Wave-194 stochastic-process canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cir_sim import bench_cir_sim
from quant_fund.models.gp_bridge import bench_gp_bridge
from quant_fund.models.hawkes_thinning import bench_hawkes_thinning
from quant_fund.models.levy_jump import bench_levy_jump
from quant_fund.models.ou_bridge import bench_ou_bridge
from quant_fund.models.poisson_thinning import bench_poisson_thinning


class TestCIR:
    def test_bench(self) -> None:
        out = bench_cir_sim(n=800)
        assert np.isfinite(out["synthetic_cir_exact_mean_err"])


class TestOUBridge:
    def test_bench(self) -> None:
        out = bench_ou_bridge(trials=60)
        assert out["synthetic_oub_endpoint_err"] < 1e-6


class TestThinning:
    def test_bench(self) -> None:
        out = bench_poisson_thinning(T=10.0)
        assert np.isfinite(out["synthetic_thin_rate_corr"])


class TestHawkes:
    def test_bench(self) -> None:
        out = bench_hawkes_thinning(T=50.0, trials=10)
        assert np.isfinite(out["synthetic_hawkes_fano"])


class TestLevy:
    def test_bench(self) -> None:
        out = bench_levy_jump(paths=80)
        assert np.isfinite(out["synthetic_merton_kurt_excess"])


class TestGPBridge:
    def test_bench(self) -> None:
        out = bench_gp_bridge()
        assert out["synthetic_gp_anchor_var"] < 1e-4
