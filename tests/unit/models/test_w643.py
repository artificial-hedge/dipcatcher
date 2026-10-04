from quant_fund.models.calc_tower import bench_calc_tower
from quant_fund.models.goodwillie_deriv import (
    bench_goodwillie_deriv,
)
from quant_fund.models.kervaire_inv import (
    bench_kervaire_inv,
)
from quant_fund.models.mahowald_inv import (
    bench_mahowald_inv,
)
from quant_fund.models.snaith_split import (
    bench_snaith_split,
)
from quant_fund.models.toda_smith import bench_toda_smith


def test_toda_smith():
    assert bench_toda_smith()["synthetic_toda_smith"] == 1.0


def test_mahowald_inv():
    assert bench_mahowald_inv()["synthetic_mahowald_inv"] == 1.0


def test_calc_tower():
    assert bench_calc_tower()["synthetic_calc_tower"] == 1.0


def test_goodwillie_deriv():
    assert bench_goodwillie_deriv()["synthetic_goodwillie_deriv"] == 1.0


def test_snaith_split():
    assert bench_snaith_split()["synthetic_snaith_split"] == 1.0


def test_kervaire_inv():
    assert bench_kervaire_inv()["synthetic_kervaire_inv"] == 1.0
