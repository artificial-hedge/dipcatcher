"""Wave-150 compression canon tests."""

from __future__ import annotations

from quant_fund.models._compress_synth import split
from quant_fund.models.fisher_prune import bench_fisher_prune
from quant_fund.models.kd_distill import bench_kd_distill
from quant_fund.models.lottery_ticket import bench_lottery_ticket
from quant_fund.models.lowrank_factor import bench_lowrank_factor
from quant_fund.models.magnitude_pruning import bench_magnitude_pruning
from quant_fund.models.quant_int8 import bench_quant_int8


class TestFixture:
    def test_split(self) -> None:
        x_tr, y_tr, x_te, y_te = split(0, 100)
        assert x_tr.shape == (50, 8) and x_te.shape == (50, 8)


class TestMPrune:
    def test_bench(self) -> None:
        out = bench_magnitude_pruning(seed=3, n=160, iters=40, ft_iters=20)
        assert 0 <= out["synthetic_mprune_acc"] <= 1


class TestLTH:
    def test_bench(self) -> None:
        out = bench_lottery_ticket(seed=5, n=160, iters=40)
        assert 0 <= out["synthetic_lth_ticket_acc"] <= 1


class TestQuant:
    def test_bench(self) -> None:
        out = bench_quant_int8(seed=7, n=160, iters=40)
        assert 0 <= out["synthetic_quant_acc"] <= 1


class TestKD:
    def test_bench(self) -> None:
        out = bench_kd_distill(seed=9, n=160, iters=40)
        assert 0 <= out["synthetic_kd_acc"] <= 1


class TestLR:
    def test_bench(self) -> None:
        out = bench_lowrank_factor(seed=11, n=160, iters=40, ft_iters=20)
        assert 0 <= out["synthetic_lr_acc"] <= 1


class TestFisher:
    def test_bench(self) -> None:
        out = bench_fisher_prune(seed=13, n=160, iters=40)
        assert 0 <= out["synthetic_fisher_acc"] <= 1
