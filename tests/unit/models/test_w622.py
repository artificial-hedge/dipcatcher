from quant_fund.models.delta_ring import bench_delta_ring
from quant_fund.models.hodge_tate import bench_hodge_tate
from quant_fund.models.nygaard2 import bench_nygaard2
from quant_fund.models.prism2 import bench_prism2
from quant_fund.models.prismatic_crystal import (
    bench_prismatic_crystal,
)
from quant_fund.models.prismatic_site import (
    bench_prismatic_site,
)


def test_prism2():
    assert bench_prism2()["synthetic_prism2"] == 1.0


def test_prismatic_site():
    assert bench_prismatic_site()["synthetic_prismatic_site"] == 1.0


def test_delta_ring():
    assert bench_delta_ring()["synthetic_delta_ring"] == 1.0


def test_prismatic_crystal():
    assert bench_prismatic_crystal()["synthetic_prismatic_crystal"] == 1.0


def test_hodge_tate():
    assert bench_hodge_tate()["synthetic_hodge_tate"] == 1.0


def test_nygaard2():
    assert bench_nygaard2()["synthetic_nygaard2"] == 1.0
