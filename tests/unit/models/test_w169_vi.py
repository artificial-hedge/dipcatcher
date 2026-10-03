"""Wave-169 amortized-inference canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.advi_bbvi import bench_advi_bbvi
from quant_fund.models.iwae_bound import bench_iwae_bound
from quant_fund.models.nf_vi import bench_nf_vi
from quant_fund.models.sparse_gp_sv import bench_sparse_gp_sv
from quant_fund.models.structured_vi import bench_structured_vi
from quant_fund.models.vrnn_seq import bench_vrnn_seq


class TestADVI:
    def test_bench(self) -> None:
        out = bench_advi_bbvi(seed=3, iters=200)
        assert np.isfinite(out["synthetic_advi_test_logloss"])


class TestIWAE:
    def test_bench(self) -> None:
        out = bench_iwae_bound(seed=5, iters=100, K=4)
        assert np.isfinite(out["synthetic_iwae_test_logloss"])


class TestNFVI:
    def test_bench(self) -> None:
        out = bench_nf_vi(seed=7, iters=150, Kflow=2)
        assert np.isfinite(out["synthetic_nfv_test_logloss"])


class TestSVGP:
    def test_bench(self) -> None:
        out = bench_sparse_gp_sv(seed=9, M=8)
        assert np.isfinite(out["synthetic_svgp_test_nll"])


class TestStructuredVI:
    def test_bench(self) -> None:
        out = bench_structured_vi(seed=11, iters=150)
        assert np.isfinite(out["synthetic_svi_full_test_logloss"])


class TestVRNN:
    def test_bench(self) -> None:
        out = bench_vrnn_seq(seed=13, steps=100, T=8)
        assert np.isfinite(out["synthetic_vrnn_step_nll"])
