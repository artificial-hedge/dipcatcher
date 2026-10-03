from quant_fund.models.chromatic_completion import (
    bench_chromatic_completion,
)
from quant_fund.models.chromatic_l2 import bench_chromatic_l2
from quant_fund.models.morava_k2 import bench_morava_k2
from quant_fund.models.periodicity_height import (
    bench_periodicity_height,
)
from quant_fund.models.picard_spec import bench_picard_spec
from quant_fund.models.telescope_tower2 import (
    bench_telescope_tower2,
)


def test_morava_k2():
    assert bench_morava_k2()["synthetic_morava_k2"] == 1.0


def test_telescope_tower2():
    assert bench_telescope_tower2()["synthetic_telescope_tower2"] == 1.0


def test_chromatic_l2():
    assert bench_chromatic_l2()["synthetic_chromatic_l2"] == 1.0


def test_picard_spec():
    assert bench_picard_spec()["synthetic_picard_spec"] == 1.0


def test_periodicity_height():
    assert bench_periodicity_height()["synthetic_periodicity_height"] == 1.0


def test_chromatic_completion():
    assert bench_chromatic_completion()["synthetic_chromatic_completion"] == 1.0
