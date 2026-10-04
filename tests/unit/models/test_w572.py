from quant_fund.models.abelian_variety import bench_abelian_variety
from quant_fund.models.faltings_thm import bench_faltings_thm
from quant_fund.models.isogeny_av import bench_isogeny_av
from quant_fund.models.mordell_weil_av import bench_mordell_weil_av
from quant_fund.models.shafarevich_conj import (
    bench_shafarevich_conj,
)
from quant_fund.models.tate_module import bench_tate_module


def test_abelian_variety():
    assert bench_abelian_variety()["synthetic_abelian_variety"] == 1.0


def test_isogeny_av():
    assert bench_isogeny_av()["synthetic_isogeny_av"] == 1.0


def test_tate_module():
    assert bench_tate_module()["synthetic_tate_module"] == 1.0


def test_shafarevich_conj():
    assert bench_shafarevich_conj()["synthetic_shafarevich_conj"] == 1.0


def test_faltings_thm():
    assert bench_faltings_thm()["synthetic_faltings_thm"] == 1.0


def test_mordell_weil_av():
    assert bench_mordell_weil_av()["synthetic_mordell_weil_av"] == 1.0
