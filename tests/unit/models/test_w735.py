from quant_fund.models.beffara_sle import bench_beffara_sle
from quant_fund.models.benoist_sle import bench_benoist_sle
from quant_fund.models.holden_sle import bench_holden_sle
from quant_fund.models.kemppainen_sle import (
    bench_kemppainen_sle,
)
from quant_fund.models.viklund_sle import bench_viklund_sle
from quant_fund.models.zykin_sle import bench_zykin_sle


def test_beffara_sle():
    assert bench_beffara_sle()["synthetic_beffara_sle"] == 1.0


def test_kemppainen_sle():
    assert bench_kemppainen_sle()["synthetic_kemppainen_sle"] == 1.0


def test_zykin_sle():
    assert bench_zykin_sle()["synthetic_zykin_sle"] == 1.0


def test_viklund_sle():
    assert bench_viklund_sle()["synthetic_viklund_sle"] == 1.0


def test_benoist_sle():
    assert bench_benoist_sle()["synthetic_benoist_sle"] == 1.0


def test_holden_sle():
    assert bench_holden_sle()["synthetic_holden_sle"] == 1.0
