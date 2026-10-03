from quant_fund.models.f_motive2 import bench_f_motive2
from quant_fund.models.milnor_operations2 import (
    bench_milnor_operations2,
)
from quant_fund.models.motivic_bordism import (
    bench_motivic_bordism,
)
from quant_fund.models.motivic_eilenberg2 import (
    bench_motivic_eilenberg2,
)
from quant_fund.models.motivic_ss2 import bench_motivic_ss2
from quant_fund.models.slice_filtration2 import (
    bench_slice_filtration2,
)


def test_slice_filtration2():
    assert bench_slice_filtration2()["synthetic_slice_filtration2"] == 1.0


def test_milnor_operations2():
    assert bench_milnor_operations2()["synthetic_milnor_operations2"] == 1.0


def test_motivic_bordism():
    assert bench_motivic_bordism()["synthetic_motivic_bordism"] == 1.0


def test_motivic_eilenberg2():
    assert bench_motivic_eilenberg2()["synthetic_motivic_eilenberg2"] == 1.0


def test_f_motive2():
    assert bench_f_motive2()["synthetic_f_motive2"] == 1.0


def test_motivic_ss2():
    assert bench_motivic_ss2()["synthetic_motivic_ss2"] == 1.0
