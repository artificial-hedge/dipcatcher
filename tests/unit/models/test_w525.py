from quant_fund.models.bernoulli_shift import bench_bernoulli_shift
from quant_fund.models.birkhoff import bench_birkhoff
from quant_fund.models.entropy_ks import bench_entropy_ks
from quant_fund.models.mean_ergodic import bench_mean_ergodic
from quant_fund.models.mixing_weak import bench_mixing_weak
from quant_fund.models.osceledets import bench_osceledets


def test_birkhoff():
    assert bench_birkhoff()["synthetic_birkhoff"] == 1.0


def test_mean_ergodic():
    assert bench_mean_ergodic()["synthetic_mean_ergodic"] == 1.0


def test_mixing_weak():
    assert bench_mixing_weak()["synthetic_mixing_weak"] == 1.0


def test_entropy_ks():
    assert bench_entropy_ks()["synthetic_entropy_ks"] == 1.0


def test_bernoulli_shift():
    assert bench_bernoulli_shift()["synthetic_bernoulli_shift"] == 1.0


def test_osceledets():
    assert bench_osceledets()["synthetic_osceledets"] == 1.0
