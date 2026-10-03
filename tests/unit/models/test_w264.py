"""Wave-264 numerical-linalg-3 unit tests."""

from quant_fund.models.block_lanczos import bench_block_lanczos
from quant_fund.models.divide_conquer_eig import bench_divide_conquer_eig
from quant_fund.models.dqds import bench_dqds
from quant_fund.models.fgmres import bench_fgmres
from quant_fund.models.randomized_qb import bench_randomized_qb
from quant_fund.models.sparse_cholesky import bench_sparse_cholesky


def test_dc_eig_keys() -> None:
    assert "synthetic_dc_eig_exact" in bench_divide_conquer_eig(1)


def test_dqds_keys() -> None:
    out = bench_dqds(2)
    assert "synthetic_dqds_rel_err" in out and "synthetic_dqds_close" in out


def test_block_lanczos_keys() -> None:
    assert "synthetic_block_lanczos_top" in bench_block_lanczos(3)


def test_qb_keys() -> None:
    assert "synthetic_qb_lowrank" in bench_randomized_qb(4)


def test_chol_keys() -> None:
    assert "synthetic_sparse_chol" in bench_sparse_cholesky(5)


def test_fgmres_keys() -> None:
    assert "synthetic_fgmres_conv" in bench_fgmres(6)


def test_ranges() -> None:
    assert 0.0 <= bench_divide_conquer_eig(7)["synthetic_dc_eig_exact"] <= 1.0
    assert 0.0 <= bench_fgmres(8)["synthetic_fgmres_conv"] <= 1.0


def test_determinism() -> None:
    assert bench_dqds(9) == bench_dqds(9)
