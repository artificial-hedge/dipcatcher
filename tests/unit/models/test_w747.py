from quant_fund.models.balazs_seppalainen import (
    bench_balazs_seppalainen,
)
from quant_fund.models.bertini_giacomin import (
    bench_bertini_giacomin,
)
from quant_fund.models.gardina_asym import bench_gardina_asym
from quant_fund.models.quastel_valko import bench_quastel_valko
from quant_fund.models.schutz_tasep import bench_schutz_tasep
from quant_fund.models.timar_tasep import bench_timar_tasep


def test_bertini_giacomin():
    assert bench_bertini_giacomin()["synthetic_bertini_giacomin"] == 1.0


def test_gardina_asym():
    assert bench_gardina_asym()["synthetic_gardina_asym"] == 1.0


def test_schutz_tasep():
    assert bench_schutz_tasep()["synthetic_schutz_tasep"] == 1.0


def test_balazs_seppalainen():
    assert bench_balazs_seppalainen()["synthetic_balazs_seppalainen"] == 1.0


def test_quastel_valko():
    assert bench_quastel_valko()["synthetic_quastel_valko"] == 1.0


def test_timar_tasep():
    assert bench_timar_tasep()["synthetic_timar_tasep"] == 1.0
