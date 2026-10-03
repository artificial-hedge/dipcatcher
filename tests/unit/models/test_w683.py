from quant_fund.models.blue_shift2 import bench_blue_shift2
from quant_fund.models.chromatic_fracture2 import (
    bench_chromatic_fracture2,
)
from quant_fund.models.fgsl_group2 import bench_fgsl_group2
from quant_fund.models.k_n_local2 import bench_k_n_local2
from quant_fund.models.morava_stabilizer2 import (
    bench_morava_stabilizer2,
)
from quant_fund.models.tate_spec2 import bench_tate_spec2


def test_blue_shift2():
    assert bench_blue_shift2()["synthetic_blue_shift2"] == 1.0


def test_chromatic_fracture2():
    assert bench_chromatic_fracture2()["synthetic_chromatic_fracture2"] == 1.0


def test_morava_stabilizer2():
    assert bench_morava_stabilizer2()["synthetic_morava_stabilizer2"] == 1.0


def test_fgsl_group2():
    assert bench_fgsl_group2()["synthetic_fgsl_group2"] == 1.0


def test_tate_spec2():
    assert bench_tate_spec2()["synthetic_tate_spec2"] == 1.0


def test_k_n_local2():
    assert bench_k_n_local2()["synthetic_k_n_local2"] == 1.0
