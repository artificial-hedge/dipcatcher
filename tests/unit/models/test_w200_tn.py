"""Wave-200 tensor-network canon tests."""

from __future__ import annotations

from quant_fund.models.dmrg_tfim import bench_dmrg_tfim
from quant_fund.models.mps_fidelity import bench_mps_fidelity
from quant_fund.models.tebd_quench import bench_tebd_quench
from quant_fund.models.tensor_cross import bench_tensor_cross
from quant_fund.models.tt_round import bench_tt_round
from quant_fund.models.tt_svd import bench_tt_svd


class TestTTSVD:
    def test_bench(self) -> None:
        out = bench_tt_svd()
        assert out["synthetic_tt_err"] < 0.05
        assert out["synthetic_tt_ratio"] < 1.0


class TestDMRG:
    def test_bench(self) -> None:
        out = bench_dmrg_tfim()
        assert out["synthetic_dmrg_e"] >= out["synthetic_exact_e"] - 1e-9


class TestTCross:
    def test_bench(self) -> None:
        out = bench_tensor_cross()
        assert out["synthetic_tcross_err"] < 0.3


class TestTEBD:
    def test_bench(self) -> None:
        out = bench_tebd_quench()
        assert out["synthetic_tebd_fid"] > 0.95


class TestMPS:
    def test_bench(self) -> None:
        out = bench_mps_fidelity()
        assert out["synthetic_mps_fid_ghz"] > 0.99


class TestTTRound:
    def test_bench(self) -> None:
        out = bench_tt_round()
        assert out["synthetic_ttround_err"] < 0.1
