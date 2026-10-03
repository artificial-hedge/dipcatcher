from quant_fund.models.cohen_mac import bench_cohen_mac
from quant_fund.models.depth_ring import bench_depth_ring
from quant_fund.models.free_resolution import bench_free_resolution
from quant_fund.models.groebner_syz import bench_groebner_syz
from quant_fund.models.hilbert_syzygy import bench_hilbert_syzygy
from quant_fund.models.regular_seq import bench_regular_seq


def test_groebner_syz():
    assert bench_groebner_syz()["synthetic_groebner_syz"] == 1.0


def test_free_resolution():
    assert bench_free_resolution()["synthetic_free_resolution"] == 1.0


def test_hilbert_syzygy():
    assert bench_hilbert_syzygy()["synthetic_hilbert_syzygy"] == 1.0


def test_regular_seq():
    assert bench_regular_seq()["synthetic_regular_seq"] == 1.0


def test_depth_ring():
    assert bench_depth_ring()["synthetic_depth_ring"] == 1.0


def test_cohen_mac():
    assert bench_cohen_mac()["synthetic_cohen_mac"] == 1.0
