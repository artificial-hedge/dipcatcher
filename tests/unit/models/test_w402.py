from quant_fund.models.etale_space import bench_etale_space
from quant_fund.models.geometric_morph import bench_geometric_morph
from quant_fund.models.groth_topo import bench_groth_topo
from quant_fund.models.logic_topos import bench_logic_topos
from quant_fund.models.sheaf_cond import bench_sheaf_cond
from quant_fund.models.topos_subobj import bench_topos_subobj


def test_topos_subobj():
    assert bench_topos_subobj()["synthetic_topos_subobj"] == 1.0


def test_groth_topo():
    assert bench_groth_topo()["synthetic_groth_topo"] == 1.0


def test_sheaf_cond():
    assert bench_sheaf_cond()["synthetic_sheaf_cond"] == 1.0


def test_logic_topos():
    assert bench_logic_topos()["synthetic_logic_topos"] == 1.0


def test_geometric_morph():
    assert bench_geometric_morph()["synthetic_geometric_morph"] == 1.0


def test_etale_space():
    assert bench_etale_space()["synthetic_etale_space"] == 1.0
