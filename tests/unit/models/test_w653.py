from quant_fund.models.ainf_cohom import bench_ainf_cohom
from quant_fund.models.fargues_scholze3 import bench_fargues_scholze3
from quant_fund.models.galois_padic import bench_galois_padic
from quant_fund.models.hodge_tate_padic import bench_hodge_tate_padic
from quant_fund.models.integral_padic2 import bench_integral_padic2
from quant_fund.models.period_ring import bench_period_ring


def test_fargues_scholze3():
    assert bench_fargues_scholze3()["synthetic_fargues_scholze3"] == 1.0


def test_integral_padic2():
    assert bench_integral_padic2()["synthetic_integral_padic2"] == 1.0


def test_ainf_cohom():
    assert bench_ainf_cohom()["synthetic_ainf_cohom"] == 1.0


def test_period_ring():
    assert bench_period_ring()["synthetic_period_ring"] == 1.0


def test_galois_padic():
    assert bench_galois_padic()["synthetic_galois_padic"] == 1.0


def test_hodge_tate_padic():
    assert bench_hodge_tate_padic()["synthetic_hodge_tate_padic"] == 1.0
