"""Wave-176 data-centric canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.active_bald import bench_active_bald
from quant_fund.models.data_cartography import bench_data_cartography
from quant_fund.models.el2n_scoring import bench_el2n_scoring
from quant_fund.models.forgetting_events import bench_forgetting_events
from quant_fund.models.influence_func import bench_influence_func
from quant_fund.models.proto_prune import bench_proto_prune


class TestBALD:
    def test_bench(self) -> None:
        out = bench_active_bald(seed=3, budget=30, rounds=2)
        assert 0.0 <= out["synthetic_bald_acc"] <= 1.0


class TestCartography:
    def test_bench(self) -> None:
        out = bench_data_cartography(seed=5, epochs=15)
        assert 0.0 <= out["synthetic_cart_easy_frac"] <= 1.0


class TestEL2N:
    def test_bench(self) -> None:
        out = bench_el2n_scoring(seed=7, early_epochs=4)
        assert np.isfinite(out["synthetic_el2n_gain"])


class TestForgetting:
    def test_bench(self) -> None:
        out = bench_forgetting_events(seed=9, epochs=15)
        assert 0.0 <= out["synthetic_forget_mislabel_auc"] <= 1.0


class TestInfluence:
    def test_bench(self) -> None:
        out = bench_influence_func(seed=11)
        assert out["synthetic_infl_mislabel_auc"] > 0.5


class TestProto:
    def test_bench(self) -> None:
        out = bench_proto_prune(seed=13)
        assert 0.0 <= out["synthetic_proto_clean_auc"] <= 1.0
