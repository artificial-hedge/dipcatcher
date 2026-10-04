from quant_fund.models.bcmp_net import bench_bcmp_net
from quant_fund.models.convoy_net import bench_convoy_net
from quant_fund.models.insensitive_thm import (
    bench_insensitive_thm,
)
from quant_fund.models.kaufman_roberts import (
    bench_kaufman_roberts,
)
from quant_fund.models.mean_value import bench_mean_value
from quant_fund.models.orku_loss import bench_orku_loss


def test_bcmp_net():
    assert bench_bcmp_net()["synthetic_bcmp_net"] == 1.0


def test_mean_value():
    assert bench_mean_value()["synthetic_mean_value"] == 1.0


def test_convoy_net():
    assert bench_convoy_net()["synthetic_convoy_net"] == 1.0


def test_insensitive_thm():
    assert bench_insensitive_thm()["synthetic_insensitive_thm"] == 1.0


def test_kaufman_roberts():
    assert bench_kaufman_roberts()["synthetic_kaufman_roberts"] == 1.0


def test_orku_loss():
    assert bench_orku_loss()["synthetic_orku_loss"] == 1.0
