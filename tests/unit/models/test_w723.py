from quant_fund.models.euler_system import bench_euler_system
from quant_fund.models.gross_zagier import bench_gross_zagier
from quant_fund.models.iwasawa_motive import bench_iwasawa_motive
from quant_fund.models.kolyvagin_sys import bench_kolyvagin_sys
from quant_fund.models.perrin_riou import bench_perrin_riou
from quant_fund.models.rubin_main_conj import (
    bench_rubin_main_conj,
)


def test_gross_zagier():
    assert bench_gross_zagier()["synthetic_gross_zagier"] == 1.0


def test_kolyvagin_sys():
    assert bench_kolyvagin_sys()["synthetic_kolyvagin_sys"] == 1.0


def test_euler_system():
    assert bench_euler_system()["synthetic_euler_system"] == 1.0


def test_iwasawa_motive():
    assert bench_iwasawa_motive()["synthetic_iwasawa_motive"] == 1.0


def test_rubin_main_conj():
    assert bench_rubin_main_conj()["synthetic_rubin_main_conj"] == 1.0


def test_perrin_riou():
    assert bench_perrin_riou()["synthetic_perrin_riou"] == 1.0
