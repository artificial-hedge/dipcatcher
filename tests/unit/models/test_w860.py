from quant_fund.models.cut_cell import (
    bench_cut_cell,
)
from quant_fund.models.fictitious_domain import (
    bench_fictitious_domain,
)
from quant_fund.models.immersed_boundary import (
    bench_immersed_boundary,
)
from quant_fund.models.iso_geom import (
    bench_iso_geom,
)
from quant_fund.models.nurbs_elem import (
    bench_nurbs_elem,
)
from quant_fund.models.xfem import (
    bench_xfem,
)


def test_iso_geom():
    assert bench_iso_geom()["synthetic_iso_geom"] == 1.0


def test_nurbs_elem():
    assert bench_nurbs_elem()["synthetic_nurbs_elem"] == 1.0


def test_xfem():
    assert bench_xfem()["synthetic_xfem"] == 1.0


def test_immersed_boundary():
    assert bench_immersed_boundary()["synthetic_immersed_boundary"] == 1.0


def test_cut_cell():
    assert bench_cut_cell()["synthetic_cut_cell"] == 1.0


def test_fictitious_domain():
    assert bench_fictitious_domain()["synthetic_fictitious_domain"] == 1.0
