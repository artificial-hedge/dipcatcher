from quant_fund.models.karlin_mcg import bench_karlin_mcg
from quant_fund.models.keilson_stieltjes import (
    bench_keilson_stieltjes,
)
from quant_fund.models.korolyuk import bench_korolyuk
from quant_fund.models.palm_khinchin import (
    bench_palm_khinchin,
)
from quant_fund.models.regen_proc import bench_regen_proc
from quant_fund.models.wold_proc import bench_wold_proc


def test_karlin_mcg():
    assert bench_karlin_mcg()["synthetic_karlin_mcg"] == 1.0


def test_keilson_stieltjes():
    assert bench_keilson_stieltjes()["synthetic_keilson_stieltjes"] == 1.0


def test_palm_khinchin():
    assert bench_palm_khinchin()["synthetic_palm_khinchin"] == 1.0


def test_regen_proc():
    assert bench_regen_proc()["synthetic_regen_proc"] == 1.0


def test_wold_proc():
    assert bench_wold_proc()["synthetic_wold_proc"] == 1.0


def test_korolyuk():
    assert bench_korolyuk()["synthetic_korolyuk"] == 1.0
