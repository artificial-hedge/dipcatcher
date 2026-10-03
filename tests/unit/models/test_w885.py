from quant_fund.models.averaging_method import (
    bench_averaging_method,
)
from quant_fund.models.entropy_stable_dg import (
    bench_entropy_stable_dg,
)
from quant_fund.models.hyperasymptotic import (
    bench_hyperasymptotic,
)
from quant_fund.models.laplace_method import (
    bench_laplace_method,
)
from quant_fund.models.ldg_flux import (
    bench_ldg_flux,
)
from quant_fund.models.wkb_turning import (
    bench_wkb_turning,
)


def test_ldg_flux():
    assert bench_ldg_flux()["synthetic_ldg_flux"] == 1.0


def test_entropy_stable_dg():
    assert bench_entropy_stable_dg()["synthetic_entropy_stable_dg"] == 1.0


def test_wkb_turning():
    assert bench_wkb_turning()["synthetic_wkb_turning"] == 1.0


def test_averaging_method():
    assert bench_averaging_method()["synthetic_averaging_method"] == 1.0


def test_laplace_method():
    assert bench_laplace_method()["synthetic_laplace_method"] == 1.0


def test_hyperasymptotic():
    assert bench_hyperasymptotic()["synthetic_hyperasymptotic"] == 1.0
