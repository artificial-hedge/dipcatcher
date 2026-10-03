"""Wave-171 conformal-2 canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.aps_cp import bench_aps_cp
from quant_fund.models.cqr_pred import bench_cqr_pred
from quant_fund.models.full_cp import bench_full_cp
from quant_fund.models.ltt_cp import bench_ltt_cp
from quant_fund.models.risk_cp import bench_risk_cp
from quant_fund.models.survival_cp import bench_survival_cp


class TestCQR:
    def test_bench(self) -> None:
        out = bench_cqr_pred(seed=3)
        assert np.isfinite(out["synthetic_cqr_coverage"])


class TestSurvivalCP:
    def test_bench(self) -> None:
        out = bench_survival_cp(seed=5)
        assert np.isfinite(out["synthetic_scp_coverage"])


class TestAPS:
    def test_bench(self) -> None:
        out = bench_aps_cp(seed=7)
        assert np.isfinite(out["synthetic_aps_coverage"])


class TestLTT:
    def test_bench(self) -> None:
        out = bench_ltt_cp(seed=9)
        assert np.isfinite(out["synthetic_ltt_coverage"])


class TestFullCP:
    def test_bench(self) -> None:
        out = bench_full_cp(seed=11, n_eval=30)
        assert np.isfinite(out["synthetic_fcp_coverage"])


class TestRiskCP:
    def test_bench(self) -> None:
        out = bench_risk_cp(seed=13)
        assert np.isfinite(out["synthetic_rcp_risk"])
