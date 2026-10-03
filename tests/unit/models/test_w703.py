from quant_fund.models.homotopy_fiber3 import (
    bench_homotopy_fiber3,
)
from quant_fund.models.homotopy_spectrum2 import (
    bench_homotopy_spectrum2,
)
from quant_fund.models.homotopy_suspension2 import (
    bench_homotopy_suspension2,
)
from quant_fund.models.homotopy_vn import bench_homotopy_vn
from quant_fund.models.stable_derivator import (
    bench_stable_derivator,
)
from quant_fund.models.stable_excisive import (
    bench_stable_excisive,
)


def test_homotopy_suspension2():
    assert bench_homotopy_suspension2()["synthetic_homotopy_suspension2"] == 1.0


def test_homotopy_fiber3():
    assert bench_homotopy_fiber3()["synthetic_homotopy_fiber3"] == 1.0


def test_stable_derivator():
    assert bench_stable_derivator()["synthetic_stable_derivator"] == 1.0


def test_homotopy_spectrum2():
    assert bench_homotopy_spectrum2()["synthetic_homotopy_spectrum2"] == 1.0


def test_stable_excisive():
    assert bench_stable_excisive()["synthetic_stable_excisive"] == 1.0


def test_homotopy_vn():
    assert bench_homotopy_vn()["synthetic_homotopy_vn"] == 1.0
