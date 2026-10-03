from quant_fund.models.character_table_s3 import bench_character_table_s3
from quant_fund.models.fourier_sn import bench_fourier_sn
from quant_fund.models.induced_rep import bench_induced_rep
from quant_fund.models.perm_rep import bench_perm_rep
from quant_fund.models.regular_rep import bench_regular_rep
from quant_fund.models.schur_ortho import bench_schur_ortho


def test_character_table_s3():
    assert bench_character_table_s3()["synthetic_character_table_s3"] == 1.0


def test_perm_rep():
    assert bench_perm_rep()["synthetic_perm_rep"] == 1.0


def test_schur_ortho():
    assert bench_schur_ortho()["synthetic_schur_ortho"] == 1.0


def test_induced_rep():
    assert bench_induced_rep()["synthetic_induced_rep"] == 1.0


def test_fourier_sn():
    assert bench_fourier_sn()["synthetic_fourier_sn"] == 1.0


def test_regular_rep():
    assert bench_regular_rep()["synthetic_regular_rep"] == 1.0
