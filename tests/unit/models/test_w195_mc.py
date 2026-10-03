"""Wave-195 ensemble/adaptive-MCMC canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.de_mcmc import bench_de_mcmc
from quant_fund.models.dram import bench_dram
from quant_fund.models.emcee_stretch import bench_emcee_stretch
from quant_fund.models.indep_mh import bench_indep_mh
from quant_fund.models.pcn_sampler import bench_pcn_sampler
from quant_fund.models.rjmcmc import bench_rjmcmc


class TestEmcee:
    def test_bench(self) -> None:
        out = bench_emcee_stretch(steps=200)
        assert np.isfinite(out["synthetic_emcee_mean_err"])


class TestDEMC:
    def test_bench(self) -> None:
        out = bench_de_mcmc(steps=250)
        assert np.isfinite(out["synthetic_demc_cov_err"])


class TestDRAM:
    def test_bench(self) -> None:
        out = bench_dram(n=700, warmup=300)
        assert 0.0 <= out["synthetic_dram_ess_frac"] <= 1.0


class TestRJ:
    def test_bench(self) -> None:
        out = bench_rjmcmc(steps=500)
        assert 0.0 <= out["synthetic_rj_p2_bimodal"] <= 1.0


class TestPCN:
    def test_bench(self) -> None:
        out = bench_pcn_sampler(steps=300)
        assert 0.0 <= out["synthetic_pcn_accept"] <= 1.0


class TestIMH:
    def test_bench(self) -> None:
        out = bench_indep_mh(n=1500)
        assert np.isfinite(out["synthetic_imh_tail_err"])
