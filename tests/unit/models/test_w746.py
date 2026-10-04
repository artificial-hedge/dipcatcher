from quant_fund.models.derrida_tasep import bench_derrida_tasep
from quant_fund.models.ferrari_tasep import bench_ferrari_tasep
from quant_fund.models.liggett_exclusion import (
    bench_liggett_exclusion,
)
from quant_fund.models.sasamoto_tasep import bench_sasamoto_tasep
from quant_fund.models.spitzer_exclusion import (
    bench_spitzer_exclusion,
)
from quant_fund.models.tracy_widom_tasep import (
    bench_tracy_widom_tasep,
)


def test_liggett_exclusion():
    assert bench_liggett_exclusion()["synthetic_liggett_exclusion"] == 1.0


def test_spitzer_exclusion():
    assert bench_spitzer_exclusion()["synthetic_spitzer_exclusion"] == 1.0


def test_sasamoto_tasep():
    assert bench_sasamoto_tasep()["synthetic_sasamoto_tasep"] == 1.0


def test_tracy_widom_tasep():
    assert bench_tracy_widom_tasep()["synthetic_tracy_widom_tasep"] == 1.0


def test_derrida_tasep():
    assert bench_derrida_tasep()["synthetic_derrida_tasep"] == 1.0


def test_ferrari_tasep():
    assert bench_ferrari_tasep()["synthetic_ferrari_tasep"] == 1.0
