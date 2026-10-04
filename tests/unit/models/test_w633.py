from quant_fund.models.cone_theorem import (
    bench_cone_theorem,
)
from quant_fund.models.contr_rational import (
    bench_contr_rational,
)
from quant_fund.models.motivic_abelian import (
    bench_motivic_abelian,
)
from quant_fund.models.motivic_coho2 import (
    bench_motivic_coho2,
)
from quant_fund.models.motivic_compact import (
    bench_motivic_compact,
)
from quant_fund.models.motivic_landweber import (
    bench_motivic_landweber,
)


def test_motivic_coho2():
    assert bench_motivic_coho2()["synthetic_motivic_coho2"] == 1.0


def test_cone_theorem():
    assert bench_cone_theorem()["synthetic_cone_theorem"] == 1.0


def test_motivic_landweber():
    assert bench_motivic_landweber()["synthetic_motivic_landweber"] == 1.0


def test_motivic_abelian():
    assert bench_motivic_abelian()["synthetic_motivic_abelian"] == 1.0


def test_motivic_compact():
    assert bench_motivic_compact()["synthetic_motivic_compact"] == 1.0


def test_contr_rational():
    assert bench_contr_rational()["synthetic_contr_rational"] == 1.0
