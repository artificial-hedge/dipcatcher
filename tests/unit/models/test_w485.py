from quant_fund.models.banach_colmez2 import bench_banach_colmez2
from quant_fund.models.bc_space import bench_bc_space
from quant_fund.models.fargues_curve2 import bench_fargues_curve2
from quant_fund.models.local_shimura import bench_local_shimura
from quant_fund.models.lubin_tate2 import bench_lubin_tate2
from quant_fund.models.scholze_weinstein import bench_scholze_weinstein


def test_lubin_tate2():
    assert bench_lubin_tate2()["synthetic_lubin_tate2"] == 1.0


def test_bc_space():
    assert bench_bc_space()["synthetic_bc_space"] == 1.0


def test_local_shimura():
    assert bench_local_shimura()["synthetic_local_shimura"] == 1.0


def test_scholze_weinstein():
    assert bench_scholze_weinstein()["synthetic_scholze_weinstein"] == 1.0


def test_fargues_curve2():
    assert bench_fargues_curve2()["synthetic_fargues_curve2"] == 1.0


def test_banach_colmez2():
    assert bench_banach_colmez2()["synthetic_banach_colmez2"] == 1.0
