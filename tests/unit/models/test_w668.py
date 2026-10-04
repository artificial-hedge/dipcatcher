from quant_fund.models.beilinson_regulator2 import (
    bench_beilinson_regulator2,
)
from quant_fund.models.hodge_motive2 import (
    bench_hodge_motive2,
)
from quant_fund.models.motivic_galois2 import (
    bench_motivic_galois2,
)
from quant_fund.models.norimotive3 import bench_norimotive3
from quant_fund.models.period_realization2 import (
    bench_period_realization2,
)
from quant_fund.models.tannakian_motive2 import (
    bench_tannakian_motive2,
)


def test_norimotive3():
    assert bench_norimotive3()["synthetic_norimotive3"] == 1.0


def test_motivic_galois2():
    assert bench_motivic_galois2()["synthetic_motivic_galois2"] == 1.0


def test_tannakian_motive2():
    assert bench_tannakian_motive2()["synthetic_tannakian_motive2"] == 1.0


def test_period_realization2():
    assert bench_period_realization2()["synthetic_period_realization2"] == 1.0


def test_beilinson_regulator2():
    assert bench_beilinson_regulator2()["synthetic_beilinson_regulator2"] == 1.0


def test_hodge_motive2():
    assert bench_hodge_motive2()["synthetic_hodge_motive2"] == 1.0
