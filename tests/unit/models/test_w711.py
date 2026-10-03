from quant_fund.models.homotopy_local import (
    bench_homotopy_local,
)
from quant_fund.models.homotopy_sheaf2 import (
    bench_homotopy_sheaf2,
)
from quant_fund.models.homotopy_stable4 import (
    bench_homotopy_stable4,
)
from quant_fund.models.stable_coalgebra import (
    bench_stable_coalgebra,
)
from quant_fund.models.stable_inf_cat import (
    bench_stable_inf_cat,
)
from quant_fund.models.stable_sheaf2 import (
    bench_stable_sheaf2,
)


def test_homotopy_sheaf2():
    assert bench_homotopy_sheaf2()["synthetic_homotopy_sheaf2"] == 1.0


def test_stable_inf_cat():
    assert bench_stable_inf_cat()["synthetic_stable_inf_cat"] == 1.0


def test_homotopy_stable4():
    assert bench_homotopy_stable4()["synthetic_homotopy_stable4"] == 1.0


def test_homotopy_local():
    assert bench_homotopy_local()["synthetic_homotopy_local"] == 1.0


def test_stable_sheaf2():
    assert bench_stable_sheaf2()["synthetic_stable_sheaf2"] == 1.0


def test_stable_coalgebra():
    assert bench_stable_coalgebra()["synthetic_stable_coalgebra"] == 1.0
