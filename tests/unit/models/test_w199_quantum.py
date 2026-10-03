"""Wave-199 quantum canon tests."""

from __future__ import annotations

from quant_fund.models.grover_search import bench_grover_search
from quant_fund.models.qaoa_maxcut import bench_qaoa_maxcut
from quant_fund.models.qkernel_svm import bench_qkernel_svm
from quant_fund.models.qpe_phase import bench_qpe_phase
from quant_fund.models.quantum_walk import bench_quantum_walk
from quant_fund.models.vqe_ising import bench_vqe_ising


class TestQAOA:
    def test_bench(self) -> None:
        out = bench_qaoa_maxcut()
        assert out["synthetic_qaoa_cut"] >= out["synthetic_uniform_cut"] - 1e-9


class TestVQE:
    def test_bench(self) -> None:
        out = bench_vqe_ising()
        assert out["synthetic_vqe_energy"] >= out["synthetic_exact_gs"] - 1e-9


class TestGrover:
    def test_bench(self) -> None:
        out = bench_grover_search()
        assert out["synthetic_grover_p_marked"] > 0.8


class TestQPE:
    def test_bench(self) -> None:
        out = bench_qpe_phase()
        assert out["synthetic_qpe_err"] < 0.05


class TestQKernel:
    def test_bench(self) -> None:
        out = bench_qkernel_svm()
        assert out["synthetic_qk_acc"] >= out["synthetic_linear_acc"] - 0.05


class TestQWalk:
    def test_bench(self) -> None:
        out = bench_quantum_walk()
        assert out["synthetic_qwalk_p_peak"] > out["synthetic_cwalk_p_antipode"]
