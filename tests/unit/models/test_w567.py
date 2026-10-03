from quant_fund.models.berenstein_zelevinsky import (
    bench_berenstein_zelevinsky,
)
from quant_fund.models.honeycomb_tiling import bench_honeycomb_tiling
from quant_fund.models.knuth_rsk import bench_knuth_rsk
from quant_fund.models.littlewood_richardson import (
    bench_littlewood_richardson,
)
from quant_fund.models.macdonald_poly import bench_macdonald_poly
from quant_fund.models.schubert_calc import bench_schubert_calc


def test_littlewood_richardson():
    assert bench_littlewood_richardson()["synthetic_littlewood_richardson"] == 1.0


def test_knuth_rsk():
    assert bench_knuth_rsk()["synthetic_knuth_rsk"] == 1.0


def test_macdonald_poly():
    assert bench_macdonald_poly()["synthetic_macdonald_poly"] == 1.0


def test_schubert_calc():
    assert bench_schubert_calc()["synthetic_schubert_calc"] == 1.0


def test_honeycomb_tiling():
    assert bench_honeycomb_tiling()["synthetic_honeycomb_tiling"] == 1.0


def test_berenstein_zelevinsky():
    assert bench_berenstein_zelevinsky()["synthetic_berenstein_zelevinsky"] == 1.0
