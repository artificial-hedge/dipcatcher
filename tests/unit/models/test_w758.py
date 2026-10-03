from quant_fund.models.aru_gff import bench_aru_gff
from quant_fund.models.bolthausen_gff import bench_bolthausen_gff
from quant_fund.models.chatterjee_gff import bench_chatterjee_gff
from quant_fund.models.ding_zeitouni import bench_ding_zeitouni
from quant_fund.models.najafi_gff import bench_najafi_gff
from quant_fund.models.powell_gff import bench_powell_gff


def test_powell_gff():
    assert bench_powell_gff()["synthetic_powell_gff"] == 1.0


def test_aru_gff():
    assert bench_aru_gff()["synthetic_aru_gff"] == 1.0


def test_ding_zeitouni():
    assert bench_ding_zeitouni()["synthetic_ding_zeitouni"] == 1.0


def test_chatterjee_gff():
    assert bench_chatterjee_gff()["synthetic_chatterjee_gff"] == 1.0


def test_bolthausen_gff():
    assert bench_bolthausen_gff()["synthetic_bolthausen_gff"] == 1.0


def test_najafi_gff():
    assert bench_najafi_gff()["synthetic_najafi_gff"] == 1.0
