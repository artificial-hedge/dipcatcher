from quant_fund.models.bubbling_hm import bench_bubbling_hm
from quant_fund.models.eells_sampson import bench_eells_sampson
from quant_fund.models.harmonic_map import bench_harmonic_map
from quant_fund.models.heat_flow_hm import bench_heat_flow_hm
from quant_fund.models.sacks_uhlenbeck import bench_sacks_uhlenbeck
from quant_fund.models.schoen_uhlenbeck import bench_schoen_uhlenbeck


def test_harmonic_map():
    assert bench_harmonic_map()["synthetic_harmonic_map"] == 1.0


def test_eells_sampson():
    assert bench_eells_sampson()["synthetic_eells_sampson"] == 1.0


def test_schoen_uhlenbeck():
    assert bench_schoen_uhlenbeck()["synthetic_schoen_uhlenbeck"] == 1.0


def test_bubbling_hm():
    assert bench_bubbling_hm()["synthetic_bubbling_hm"] == 1.0


def test_heat_flow_hm():
    assert bench_heat_flow_hm()["synthetic_heat_flow_hm"] == 1.0


def test_sacks_uhlenbeck():
    assert bench_sacks_uhlenbeck()["synthetic_sacks_uhlenbeck"] == 1.0
