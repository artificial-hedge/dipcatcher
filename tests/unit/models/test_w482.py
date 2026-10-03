from quant_fund.models.blue_shift import bench_blue_shift
from quant_fund.models.chromatic_fracture import bench_chromatic_fracture
from quant_fund.models.fgsl_group import bench_fgsl_group
from quant_fund.models.morava_stabilizer import bench_morava_stabilizer
from quant_fund.models.red_shift import bench_red_shift
from quant_fund.models.tate_spec import bench_tate_spec


def test_chromatic_fracture():
    assert bench_chromatic_fracture()["synthetic_chromatic_fracture"] == 1.0


def test_morava_stabilizer():
    assert bench_morava_stabilizer()["synthetic_morava_stabilizer"] == 1.0


def test_fgsl_group():
    assert bench_fgsl_group()["synthetic_fgsl_group"] == 1.0


def test_tate_spec():
    assert bench_tate_spec()["synthetic_tate_spec"] == 1.0


def test_blue_shift():
    assert bench_blue_shift()["synthetic_blue_shift"] == 1.0


def test_red_shift():
    assert bench_red_shift()["synthetic_red_shift"] == 1.0
