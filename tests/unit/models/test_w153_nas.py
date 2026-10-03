"""Wave-153 NAS canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models._nas_synth import all_archs, noisy_labels
from quant_fund.models.arch_predictor import bench_arch_predictor
from quant_fund.models.darts_nas import bench_darts_nas
from quant_fund.models.enas_controller import bench_enas_controller
from quant_fund.models.evolution_nas import bench_evolution_nas
from quant_fund.models.one_shot_nas import bench_one_shot_nas
from quant_fund.models.random_search_nas import bench_random_search_nas


class TestFixture:
    def test_space(self) -> None:
        assert len(all_archs()) == 30

    def test_noise(self) -> None:
        y = np.zeros(100, dtype=int)
        y2 = noisy_labels(y, 0, 0.2)
        assert (y2 == 1).sum() == 20


class TestRS:
    def test_bench(self) -> None:
        out = bench_random_search_nas(seed=3, n=120, budget=6, iters=15)
        assert out["synthetic_rnas_oracle"] >= out["synthetic_rnas_best"]


class TestEvo:
    def test_bench(self) -> None:
        out = bench_evolution_nas(seed=5, n=120, pop_size=4, budget=8, iters=15)
        assert 0 <= out["synthetic_enas_best"] <= 1


class TestDarts:
    def test_bench(self) -> None:
        out = bench_darts_nas(seed=7, n=120, iters=20)
        assert 0 <= out["synthetic_darts_pick_acc"] <= 1


class TestEC:
    def test_bench(self) -> None:
        out = bench_enas_controller(seed=9, n=120, steps=8, iters=15)
        assert 0 <= out["synthetic_ec_final_acc"] <= 1


class TestOS:
    def test_bench(self) -> None:
        out = bench_one_shot_nas(seed=11, n=120, iters=20, hidds=(4, 16))
        assert -1 <= out["synthetic_os_rank_rho"] <= 1


class TestAP:
    def test_bench(self) -> None:
        out = bench_arch_predictor(seed=13, n=120, n_train=8, iters=15)
        assert -1 <= out["synthetic_ap_rank_rho"] <= 1
