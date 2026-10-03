"""Wave-188 active-learning canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.badge_embed import bench_badge_embed
from quant_fund.models.coreset_kcenter import bench_coreset_kcenter
from quant_fund.models.egl_change import bench_egl_change
from quant_fund.models.entropy_query import bench_entropy_query
from quant_fund.models.margin_sampling import bench_margin_sampling
from quant_fund.models.qbc_committee import bench_qbc_committee


class TestEQ:
    def test_bench(self) -> None:
        out = bench_entropy_query(seed=3, trials=2)
        assert 0.0 <= out["synthetic_eq_acc"] <= 1.0


class TestMS:
    def test_bench(self) -> None:
        out = bench_margin_sampling(seed=5, trials=2)
        assert 0.0 <= out["synthetic_ms_acc"] <= 1.0


class TestQBC:
    def test_bench(self) -> None:
        out = bench_qbc_committee(seed=7, trials=2)
        assert 0.0 <= out["synthetic_qbc_acc"] <= 1.0


class TestKC:
    def test_bench(self) -> None:
        out = bench_coreset_kcenter(seed=9, trials=2)
        assert 0.0 <= out["synthetic_kc_acc"] <= 1.0


class TestBADGE:
    def test_bench(self) -> None:
        out = bench_badge_embed(seed=11, trials=2)
        assert 0.0 <= out["synthetic_badge_acc"] <= 1.0


class TestEGL:
    def test_bench(self) -> None:
        out = bench_egl_change(seed=13, trials=2)
        assert np.isfinite(out["synthetic_egl_acc"])
