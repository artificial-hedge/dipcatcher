"""Wave-179 survival-DL canon tests."""

from __future__ import annotations

from quant_fund.models.cox_time import bench_cox_time
from quant_fund.models.deephit import bench_deephit
from quant_fund.models.deepsurv import bench_deepsurv
from quant_fund.models.drsa_surv import bench_drsa_surv
from quant_fund.models.nnet_surv import bench_nnet_surv
from quant_fund.models.pchazard import bench_pchazard


class TestDeepSurv:
    def test_bench(self) -> None:
        out = bench_deepsurv(seed=3, iters=80)
        assert 0.0 <= out["synthetic_deepsurv_cindex"] <= 1.0


class TestDeepHit:
    def test_bench(self) -> None:
        out = bench_deephit(seed=5, iters=80)
        assert 0.0 <= out["synthetic_deephit_cindex"] <= 1.0


class TestCoxTime:
    def test_bench(self) -> None:
        out = bench_cox_time(seed=7, iters=40)
        assert 0.0 <= out["synthetic_coxtime_cindex"] <= 1.0


class TestNnetSurv:
    def test_bench(self) -> None:
        out = bench_nnet_surv(seed=9, iters=80)
        assert 0.0 <= out["synthetic_nnet_cindex"] <= 1.0


class TestDRSA:
    def test_bench(self) -> None:
        out = bench_drsa_surv(seed=11, iters=80)
        assert 0.0 <= out["synthetic_drsa_cindex"] <= 1.0


class TestPCHazard:
    def test_bench(self) -> None:
        out = bench_pchazard(seed=13, iters=80)
        assert 0.0 <= out["synthetic_pchazard_cindex"] <= 1.0
