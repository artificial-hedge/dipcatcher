"""Wave-196 eigen canon tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.bidiag_svd import bench_bidiag_svd
from quant_fund.models.hessenberg_red import bench_hessenberg_red
from quant_fund.models.inverse_iter import bench_inverse_iter
from quant_fund.models.jacobi_eig import bench_jacobi_eig
from quant_fund.models.power_iter import bench_power_iter
from quant_fund.models.qr_eig import bench_qr_eig


class TestPower:
    def test_bench(self) -> None:
        out = bench_power_iter(k=3)
        assert out["synthetic_power_topk_err"] < 0.05


class TestInverse:
    def test_bench(self) -> None:
        out = bench_inverse_iter()
        assert out["synthetic_inviter_eig_err"] < 0.05


class TestJacobi:
    def test_bench(self) -> None:
        out = bench_jacobi_eig()
        assert out["synthetic_jacobi_spec_err"] < 0.05


class TestQR:
    def test_bench(self) -> None:
        out = bench_qr_eig()
        assert np.isfinite(out["synthetic_qr_spec_err"])


class TestHessenberg:
    def test_bench(self) -> None:
        out = bench_hessenberg_red()
        assert out["synthetic_hess_resid_sym"] < 1e-8


class TestBidiag:
    def test_bench(self) -> None:
        out = bench_bidiag_svd()
        assert out["synthetic_bidiag_sv_err"] < 0.05
