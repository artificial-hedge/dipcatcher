from quant_fund.models.first_order import bench_first_order
from quant_fund.models.obstruction_def import (
    bench_obstruction_def,
)
from quant_fund.models.prorepresent import (
    bench_prorepresent,
)
from quant_fund.models.schlessinger2 import (
    bench_schlessinger2,
)
from quant_fund.models.semiuniversal import (
    bench_semiuniversal,
)
from quant_fund.models.versal_def import bench_versal_def


def test_schlessinger2():
    assert bench_schlessinger2()["synthetic_schlessinger2"] == 1.0


def test_prorepresent():
    assert bench_prorepresent()["synthetic_prorepresent"] == 1.0


def test_versal_def():
    assert bench_versal_def()["synthetic_versal_def"] == 1.0


def test_semiuniversal():
    assert bench_semiuniversal()["synthetic_semiuniversal"] == 1.0


def test_first_order():
    assert bench_first_order()["synthetic_first_order"] == 1.0


def test_obstruction_def():
    assert bench_obstruction_def()["synthetic_obstruction_def"] == 1.0
