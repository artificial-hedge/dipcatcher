from quant_fund.models.coates_wiles import bench_coates_wiles
from quant_fund.models.gan_gross_prasad import (
    bench_gan_gross_prasad,
)
from quant_fund.models.greenberg_selmer import (
    bench_greenberg_selmer,
)
from quant_fund.models.heegner_cycle import bench_heegner_cycle
from quant_fund.models.iwasawa_lfunc import bench_iwasawa_lfunc
from quant_fund.models.kurihara_iwasawa import (
    bench_kurihara_iwasawa,
)


def test_coates_wiles():
    assert bench_coates_wiles()["synthetic_coates_wiles"] == 1.0


def test_iwasawa_lfunc():
    assert bench_iwasawa_lfunc()["synthetic_iwasawa_lfunc"] == 1.0


def test_greenberg_selmer():
    assert bench_greenberg_selmer()["synthetic_greenberg_selmer"] == 1.0


def test_kurihara_iwasawa():
    assert bench_kurihara_iwasawa()["synthetic_kurihara_iwasawa"] == 1.0


def test_heegner_cycle():
    assert bench_heegner_cycle()["synthetic_heegner_cycle"] == 1.0


def test_gan_gross_prasad():
    assert bench_gan_gross_prasad()["synthetic_gan_gross_prasad"] == 1.0
