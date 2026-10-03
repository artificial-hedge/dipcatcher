from quant_fund.models.chromatic_base import bench_chromatic_base
from quant_fund.models.chromatic_layer import (
    bench_chromatic_layer,
)
from quant_fund.models.chromatic_square2 import (
    bench_chromatic_square2,
)
from quant_fund.models.elliptic_morava import (
    bench_elliptic_morava,
)
from quant_fund.models.lubin_tate3 import bench_lubin_tate3
from quant_fund.models.morava_maven import bench_morava_maven


def test_chromatic_layer():
    assert bench_chromatic_layer()["synthetic_chromatic_layer"] == 1.0


def test_morava_maven():
    assert bench_morava_maven()["synthetic_morava_maven"] == 1.0


def test_chromatic_square2():
    assert bench_chromatic_square2()["synthetic_chromatic_square2"] == 1.0


def test_lubin_tate3():
    assert bench_lubin_tate3()["synthetic_lubin_tate3"] == 1.0


def test_elliptic_morava():
    assert bench_elliptic_morava()["synthetic_elliptic_morava"] == 1.0


def test_chromatic_base():
    assert bench_chromatic_base()["synthetic_chromatic_base"] == 1.0
