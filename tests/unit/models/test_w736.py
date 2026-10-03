from quant_fund.models.aru_powell import bench_aru_powell
from quant_fund.models.berestycki_sheffield import (
    bench_berestycki_sheffield,
)
from quant_fund.models.bisbisot_sheffield import (
    bench_bisbisot_sheffield,
)
from quant_fund.models.dhms_lqg import bench_dhms_lqg
from quant_fund.models.huang_rhodes import bench_huang_rhodes
from quant_fund.models.sheffield_gff import bench_sheffield_gff


def test_sheffield_gff():
    assert bench_sheffield_gff()["synthetic_sheffield_gff"] == 1.0


def test_berestycki_sheffield():
    assert bench_berestycki_sheffield()["synthetic_berestycki_sheffield"] == 1.0


def test_aru_powell():
    assert bench_aru_powell()["synthetic_aru_powell"] == 1.0


def test_huang_rhodes():
    assert bench_huang_rhodes()["synthetic_huang_rhodes"] == 1.0


def test_bisbisot_sheffield():
    assert bench_bisbisot_sheffield()["synthetic_bisbisot_sheffield"] == 1.0


def test_dhms_lqg():
    assert bench_dhms_lqg()["synthetic_dhms_lqg"] == 1.0
