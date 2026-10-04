from quant_fund.models.chromatic_l3 import bench_chromatic_l3
from quant_fund.models.morava_e2 import bench_morava_e2
from quant_fund.models.morava_k3 import bench_morava_k3
from quant_fund.models.picard_spec2 import bench_picard_spec2
from quant_fund.models.red_shift2 import bench_red_shift2
from quant_fund.models.telescope_tower3 import (
    bench_telescope_tower3,
)


def test_morava_k3():
    assert bench_morava_k3()["synthetic_morava_k3"] == 1.0


def test_morava_e2():
    assert bench_morava_e2()["synthetic_morava_e2"] == 1.0


def test_chromatic_l3():
    assert bench_chromatic_l3()["synthetic_chromatic_l3"] == 1.0


def test_telescope_tower3():
    assert bench_telescope_tower3()["synthetic_telescope_tower3"] == 1.0


def test_picard_spec2():
    assert bench_picard_spec2()["synthetic_picard_spec2"] == 1.0


def test_red_shift2():
    assert bench_red_shift2()["synthetic_red_shift2"] == 1.0
