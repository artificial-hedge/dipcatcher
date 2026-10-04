from quant_fund.models.homotopy_class2 import bench_homotopy_class2
from quant_fund.models.homotopy_limit import bench_homotopy_limit
from quant_fund.models.homotopy_tower import bench_homotopy_tower
from quant_fund.models.spectral_sequence5 import (
    bench_spectral_sequence5,
)
from quant_fund.models.stable_bousfield import (
    bench_stable_bousfield,
)
from quant_fund.models.stable_mapping import bench_stable_mapping


def test_homotopy_limit():
    assert bench_homotopy_limit()["synthetic_homotopy_limit"] == 1.0


def test_homotopy_tower():
    assert bench_homotopy_tower()["synthetic_homotopy_tower"] == 1.0


def test_spectral_sequence5():
    assert bench_spectral_sequence5()["synthetic_spectral_sequence5"] == 1.0


def test_homotopy_class2():
    assert bench_homotopy_class2()["synthetic_homotopy_class2"] == 1.0


def test_stable_mapping():
    assert bench_stable_mapping()["synthetic_stable_mapping"] == 1.0


def test_stable_bousfield():
    assert bench_stable_bousfield()["synthetic_stable_bousfield"] == 1.0
