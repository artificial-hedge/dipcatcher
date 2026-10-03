from quant_fund.models.companion_conj import bench_companion_conj
from quant_fund.models.fibrant_double import bench_fibrant_double
from quant_fund.models.framed_bicat import bench_framed_bicat
from quant_fund.models.proarrow import bench_proarrow
from quant_fund.models.tabulation import bench_tabulation
from quant_fund.models.virtual_equip import bench_virtual_equip


def test_proarrow():
    assert bench_proarrow()["synthetic_proarrow"] == 1.0


def test_virtual_equip():
    assert bench_virtual_equip()["synthetic_virtual_equip"] == 1.0


def test_fibrant_double():
    assert bench_fibrant_double()["synthetic_fibrant_double"] == 1.0


def test_tabulation():
    assert bench_tabulation()["synthetic_tabulation"] == 1.0


def test_companion_conj():
    assert bench_companion_conj()["synthetic_companion_conj"] == 1.0


def test_framed_bicat():
    assert bench_framed_bicat()["synthetic_framed_bicat"] == 1.0
