from quant_fund.models.anabelian_geo import bench_anabelian_geo
from quant_fund.models.etale_pi1 import bench_etale_pi1
from quant_fund.models.fundamental_grp import bench_fundamental_grp
from quant_fund.models.groth_tei import bench_groth_tei
from quant_fund.models.section_conj import bench_section_conj
from quant_fund.models.tamagawa_mochi import bench_tamagawa_mochi


def test_anabelian_geo():
    assert bench_anabelian_geo()["synthetic_anabelian_geo"] == 1.0


def test_section_conj():
    assert bench_section_conj()["synthetic_section_conj"] == 1.0


def test_fundamental_grp():
    assert bench_fundamental_grp()["synthetic_fundamental_grp"] == 1.0


def test_etale_pi1():
    assert bench_etale_pi1()["synthetic_etale_pi1"] == 1.0


def test_groth_tei():
    assert bench_groth_tei()["synthetic_groth_tei"] == 1.0


def test_tamagawa_mochi():
    assert bench_tamagawa_mochi()["synthetic_tamagawa_mochi"] == 1.0
