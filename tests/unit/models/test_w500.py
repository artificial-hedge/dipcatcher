from quant_fund.models.epsilon_factor import bench_epsilon_factor
from quant_fund.models.harris_taylor import bench_harris_taylor
from quant_fund.models.l_packet import bench_l_packet
from quant_fund.models.langlands_functoriality import bench_langlands_functoriality
from quant_fund.models.local_langlands import bench_local_langlands
from quant_fund.models.weil_group import bench_weil_group


def test_local_langlands():
    assert bench_local_langlands()["synthetic_local_langlands"] == 1.0


def test_harris_taylor():
    assert bench_harris_taylor()["synthetic_harris_taylor"] == 1.0


def test_weil_group():
    assert bench_weil_group()["synthetic_weil_group"] == 1.0


def test_langlands_functoriality():
    assert bench_langlands_functoriality()["synthetic_langlands_functoriality"] == 1.0


def test_epsilon_factor():
    assert bench_epsilon_factor()["synthetic_epsilon_factor"] == 1.0


def test_l_packet():
    assert bench_l_packet()["synthetic_l_packet"] == 1.0
