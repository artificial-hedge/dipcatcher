"""Wave-192 info-theory canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.copula_mi import bench_copula_mi
from quant_fund.models.hsic_independence import bench_hsic_independence
from quant_fund.models.lsd_deptest import bench_lsd_deptest
from quant_fund.models.mine_mi import bench_mine_mi
from quant_fund.models.mmd_two_sample import bench_mmd_two_sample
from quant_fund.models.nwj_mi import bench_nwj_mi


class TestMMD:
    def test_bench(self) -> None:
        out = bench_mmd_two_sample(seed=3, perms=60)
        assert 0.0 <= out["synthetic_mmd_pval_alt"] <= 1.0


class TestHSIC:
    def test_bench(self) -> None:
        out = bench_hsic_independence(seed=5, perms=60)
        assert 0.0 <= out["synthetic_hsic_pval_dep"] <= 1.0


class TestMINE:
    def test_bench(self) -> None:
        out = bench_mine_mi(seed=7)
        assert np.isfinite(out["synthetic_mine_mi_dep"])


class TestNWJ:
    def test_bench(self) -> None:
        out = bench_nwj_mi(seed=9)
        assert np.isfinite(out["synthetic_nwj_mi_dep"])


class TestCopulaMI:
    def test_bench(self) -> None:
        out = bench_copula_mi(seed=11)
        assert np.isfinite(out["synthetic_copula_mi_dep"])


class TestLSD:
    def test_bench(self) -> None:
        out = bench_lsd_deptest(seed=13, perms=60)
        assert 0.0 <= out["synthetic_lsd_pval_dep"] <= 1.0
