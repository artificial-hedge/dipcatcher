from quant_fund.models.fulton_mclarty import (
    bench_fulton_mclarty,
)
from quant_fund.models.motivic_base_change import (
    bench_motivic_base_change,
)
from quant_fund.models.motivic_homotopy2 import (
    bench_motivic_homotopy2,
)
from quant_fund.models.motivic_proper import (
    bench_motivic_proper,
)
from quant_fund.models.motivic_smooth import (
    bench_motivic_smooth,
)
from quant_fund.models.six_op_motivic import (
    bench_six_op_motivic,
)


def test_motivic_base_change():
    assert bench_motivic_base_change()["synthetic_motivic_base_change"] == 1.0


def test_six_op_motivic():
    assert bench_six_op_motivic()["synthetic_six_op_motivic"] == 1.0


def test_motivic_smooth():
    assert bench_motivic_smooth()["synthetic_motivic_smooth"] == 1.0


def test_motivic_proper():
    assert bench_motivic_proper()["synthetic_motivic_proper"] == 1.0


def test_fulton_mclarty():
    assert bench_fulton_mclarty()["synthetic_fulton_mclarty"] == 1.0


def test_motivic_homotopy2():
    assert bench_motivic_homotopy2()["synthetic_motivic_homotopy2"] == 1.0
