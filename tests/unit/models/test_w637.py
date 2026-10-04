from quant_fund.models.cocartesian_diamond import (
    bench_cocartesian_diamond,
)
from quant_fund.models.curve_padic import (
    bench_curve_padic,
)
from quant_fund.models.diamond_mod import (
    bench_diamond_mod,
)
from quant_fund.models.etale_phiphi import (
    bench_etale_phiphi,
)
from quant_fund.models.fargues_scholze2 import (
    bench_fargues_scholze2,
)
from quant_fund.models.scholze_bc import bench_scholze_bc


def test_fargues_scholze2():
    assert bench_fargues_scholze2()["synthetic_fargues_scholze2"] == 1.0


def test_curve_padic():
    assert bench_curve_padic()["synthetic_curve_padic"] == 1.0


def test_diamond_mod():
    assert bench_diamond_mod()["synthetic_diamond_mod"] == 1.0


def test_etale_phiphi():
    assert bench_etale_phiphi()["synthetic_etale_phiphi"] == 1.0


def test_cocartesian_diamond():
    assert bench_cocartesian_diamond()["synthetic_cocartesian_diamond"] == 1.0


def test_scholze_bc():
    assert bench_scholze_bc()["synthetic_scholze_bc"] == 1.0
