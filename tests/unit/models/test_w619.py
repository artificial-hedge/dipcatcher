from quant_fund.models.andersen_lannes import (
    bench_andersen_lannes,
)
from quant_fund.models.chromatic_hopkins import (
    bench_chromatic_hopkins,
)
from quant_fund.models.devissage_ss import bench_devissage_ss
from quant_fund.models.tame_htpy import bench_tame_htpy
from quant_fund.models.thick_spectrum import (
    bench_thick_spectrum,
)
from quant_fund.models.unstable_htpy import bench_unstable_htpy


def test_unstable_htpy():
    assert bench_unstable_htpy()["synthetic_unstable_htpy"] == 1.0


def test_tame_htpy():
    assert bench_tame_htpy()["synthetic_tame_htpy"] == 1.0


def test_devissage_ss():
    assert bench_devissage_ss()["synthetic_devissage_ss"] == 1.0


def test_andersen_lannes():
    assert bench_andersen_lannes()["synthetic_andersen_lannes"] == 1.0


def test_chromatic_hopkins():
    assert bench_chromatic_hopkins()["synthetic_chromatic_hopkins"] == 1.0


def test_thick_spectrum():
    assert bench_thick_spectrum()["synthetic_thick_spectrum"] == 1.0
