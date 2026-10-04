from quant_fund.models.motivic_additive import (
    bench_motivic_additive,
)
from quant_fund.models.motivic_additive_cat import (
    bench_motivic_additive_cat,
)
from quant_fund.models.motivic_chern2 import (
    bench_motivic_chern2,
)
from quant_fund.models.motivic_cover import (
    bench_motivic_cover,
)
from quant_fund.models.motivic_filtration2 import (
    bench_motivic_filtration2,
)
from quant_fund.models.motivic_gysin2 import (
    bench_motivic_gysin2,
)


def test_motivic_additive():
    assert bench_motivic_additive()["synthetic_motivic_additive"] == 1.0


def test_motivic_additive_cat():
    assert bench_motivic_additive_cat()["synthetic_motivic_additive_cat"] == 1.0


def test_motivic_cover():
    assert bench_motivic_cover()["synthetic_motivic_cover"] == 1.0


def test_motivic_gysin2():
    assert bench_motivic_gysin2()["synthetic_motivic_gysin2"] == 1.0


def test_motivic_chern2():
    assert bench_motivic_chern2()["synthetic_motivic_chern2"] == 1.0


def test_motivic_filtration2():
    assert bench_motivic_filtration2()["synthetic_motivic_filtration2"] == 1.0
