"""Wave-152 privacy/federated canon tests."""

from __future__ import annotations

from quant_fund.models.canary_exposure import bench_canary_exposure
from quant_fund.models.dp_sgd import bench_dp_sgd
from quant_fund.models.fedavg_hetero import bench_fedavg_hetero
from quant_fund.models.gradient_leakage import bench_gradient_leakage
from quant_fund.models.pate_teacher import bench_pate_teacher
from quant_fund.models.secure_agg import bench_secure_agg


class TestDPSGD:
    def test_bench(self) -> None:
        out = bench_dp_sgd(seed=3, n=120, iters=10, n_sub=40)
        assert 0 <= out["synthetic_dpsgd_acc"] <= 1


class TestSecureAgg:
    def test_bench(self) -> None:
        out = bench_secure_agg(seed=5, n_clients=4, dim=8)
        assert out["synthetic_sagg_sum_err"] < 1e-8


class TestFedAvg:
    def test_bench(self) -> None:
        out = bench_fedavg_hetero(seed=7, n=120, rounds=3, epochs=2)
        assert 0 <= out["synthetic_fed_iid_acc"] <= 1


class TestPATE:
    def test_bench(self) -> None:
        out = bench_pate_teacher(seed=9, n=120, iters=10)
        assert 0 <= out["synthetic_pate_noisy_agree"] <= 1


class TestDLG:
    def test_bench(self) -> None:
        out = bench_gradient_leakage(seed=11, n=80, steps=20)
        assert out["synthetic_dlg_mse_single"] >= 0


class TestCanary:
    def test_bench(self) -> None:
        out = bench_canary_exposure(seed=13, n=120, iters=10)
        assert 0 <= out["synthetic_canary_prob"] <= 1
