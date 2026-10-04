from quant_fund.models.deift_rmt import bench_deift_rmt
from quant_fund.models.erdos_yau import bench_erdos_yau
from quant_fund.models.forrester_rmt import bench_forrester_rmt
from quant_fund.models.johansson_rmt import bench_johansson_rmt
from quant_fund.models.mehta_rmt import bench_mehta_rmt
from quant_fund.models.soshnikov_rmt import bench_soshnikov_rmt


def test_soshnikov_rmt():
    assert bench_soshnikov_rmt()["synthetic_soshnikov_rmt"] == 1.0


def test_erdos_yau():
    assert bench_erdos_yau()["synthetic_erdos_yau"] == 1.0


def test_forrester_rmt():
    assert bench_forrester_rmt()["synthetic_forrester_rmt"] == 1.0


def test_mehta_rmt():
    assert bench_mehta_rmt()["synthetic_mehta_rmt"] == 1.0


def test_deift_rmt():
    assert bench_deift_rmt()["synthetic_deift_rmt"] == 1.0


def test_johansson_rmt():
    assert bench_johansson_rmt()["synthetic_johansson_rmt"] == 1.0
