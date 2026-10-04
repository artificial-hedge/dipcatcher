from quant_fund.models.harmonic_bdl import bench_harmonic_bdl
from quant_fund.models.higgs_bundle2 import bench_higgs_bundle2
from quant_fund.models.hitchin_section import (
    bench_hitchin_section,
)
from quant_fund.models.hodge_moduli import bench_hodge_moduli
from quant_fund.models.nonabelian_hodge import (
    bench_nonabelian_hodge,
)
from quant_fund.models.simpson_corr import bench_simpson_corr


def test_higgs_bundle2():
    assert bench_higgs_bundle2()["synthetic_higgs_bundle2"] == 1.0


def test_hitchin_section():
    assert bench_hitchin_section()["synthetic_hitchin_section"] == 1.0


def test_simpson_corr():
    assert bench_simpson_corr()["synthetic_simpson_corr"] == 1.0


def test_nonabelian_hodge():
    assert bench_nonabelian_hodge()["synthetic_nonabelian_hodge"] == 1.0


def test_harmonic_bdl():
    assert bench_harmonic_bdl()["synthetic_harmonic_bdl"] == 1.0


def test_hodge_moduli():
    assert bench_hodge_moduli()["synthetic_hodge_moduli"] == 1.0
