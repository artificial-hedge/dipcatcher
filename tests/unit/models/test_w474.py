from quant_fund.models.cobordism_hyp import bench_cobordism_hyp
from quant_fund.models.heegaard_floer import bench_heegaard_floer
from quant_fund.models.khovanov import bench_khovanov
from quant_fund.models.modular_cat import bench_modular_cat
from quant_fund.models.reshet_turaev import bench_reshet_turaev
from quant_fund.models.topological_order import bench_topological_order


def test_reshet_turaev():
    assert bench_reshet_turaev()["synthetic_reshet_turaev"] == 1.0


def test_khovanov():
    assert bench_khovanov()["synthetic_khovanov"] == 1.0


def test_heegaard_floer():
    assert bench_heegaard_floer()["synthetic_heegaard_floer"] == 1.0


def test_cobordism_hyp():
    assert bench_cobordism_hyp()["synthetic_cobordism_hyp"] == 1.0


def test_modular_cat():
    assert bench_modular_cat()["synthetic_modular_cat"] == 1.0


def test_topological_order():
    assert bench_topological_order()["synthetic_topological_order"] == 1.0
