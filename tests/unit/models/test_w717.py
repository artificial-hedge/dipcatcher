from quant_fund.models.bondal_kapranov import (
    bench_bondal_kapranov,
)
from quant_fund.models.dg_enhancement import (
    bench_dg_enhancement,
)
from quant_fund.models.enhanced_triangulated import (
    bench_enhanced_triangulated,
)
from quant_fund.models.nc_k_theory import bench_nc_k_theory
from quant_fund.models.nc_motive import bench_nc_motive
from quant_fund.models.tabuada_motive import (
    bench_tabuada_motive,
)


def test_nc_motive():
    assert bench_nc_motive()["synthetic_nc_motive"] == 1.0


def test_dg_enhancement():
    assert bench_dg_enhancement()["synthetic_dg_enhancement"] == 1.0


def test_bondal_kapranov():
    assert bench_bondal_kapranov()["synthetic_bondal_kapranov"] == 1.0


def test_enhanced_triangulated():
    assert bench_enhanced_triangulated()["synthetic_enhanced_triangulated"] == 1.0


def test_tabuada_motive():
    assert bench_tabuada_motive()["synthetic_tabuada_motive"] == 1.0


def test_nc_k_theory():
    assert bench_nc_k_theory()["synthetic_nc_k_theory"] == 1.0
