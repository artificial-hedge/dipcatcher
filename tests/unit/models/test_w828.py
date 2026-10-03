from quant_fund.models.cross_section import (
    bench_cross_section,
)
from quant_fund.models.dellacherie_section import (
    bench_dellacherie_section,
)
from quant_fund.models.maharam_lift import (
    bench_maharam_lift,
)
from quant_fund.models.projection_theorem import (
    bench_projection_theorem,
)
from quant_fund.models.uniform_section import (
    bench_uniform_section,
)
from quant_fund.models.von_neumann_sel import (
    bench_von_neumann_sel,
)


def test_projection_theorem():
    assert bench_projection_theorem()["synthetic_projection_theorem"] == 1.0


def test_uniform_section():
    assert bench_uniform_section()["synthetic_uniform_section"] == 1.0


def test_dellacherie_section():
    assert bench_dellacherie_section()["synthetic_dellacherie_section"] == 1.0


def test_cross_section():
    assert bench_cross_section()["synthetic_cross_section"] == 1.0


def test_maharam_lift():
    assert bench_maharam_lift()["synthetic_maharam_lift"] == 1.0


def test_von_neumann_sel():
    assert bench_von_neumann_sel()["synthetic_von_neumann_sel"] == 1.0
