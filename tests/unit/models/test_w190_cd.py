"""Wave-190 classical causal-discovery canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.direct_lingam import bench_direct_lingam
from quant_fund.models.fci_alg import bench_fci_alg
from quant_fund.models.ges_search import bench_ges_search
from quant_fund.models.ica_lingam import bench_ica_lingam
from quant_fund.models.mmmb_select import bench_mmmb_select
from quant_fund.models.var_lingam import bench_var_lingam


class TestICAL:
    def test_bench(self) -> None:
        out = bench_ica_lingam(seed=3, trials=2)
        assert 0.0 <= out["synthetic_ical_skel_f1"] <= 1.0


class TestDLing:
    def test_bench(self) -> None:
        out = bench_direct_lingam(seed=5, trials=2)
        assert 0.0 <= out["synthetic_dling_skel_f1"] <= 1.0


class TestVARL:
    def test_bench(self) -> None:
        out = bench_var_lingam(seed=7, trials=2)
        assert np.isfinite(out["synthetic_varling_skel_f1"])


class TestGES:
    def test_bench(self) -> None:
        out = bench_ges_search(seed=9, trials=2)
        assert np.isfinite(out["synthetic_ges_shd"])


class TestFCI:
    def test_bench(self) -> None:
        out = bench_fci_alg(seed=11, trials=2)
        assert 0.0 <= out["synthetic_fci_skel_f1"] <= 1.0


class TestMMMB:
    def test_bench(self) -> None:
        out = bench_mmmb_select(seed=13, trials=2)
        assert 0.0 <= out["synthetic_mmmb_skel_f1"] <= 1.0
