from quant_fund.models.motivic_atiyah import (
    bench_motivic_atiyah,
)
from quant_fund.models.motivic_coniveau import (
    bench_motivic_coniveau,
)
from quant_fund.models.motivic_deligne import (
    bench_motivic_deligne,
)
from quant_fund.models.motivic_residue import (
    bench_motivic_residue,
)
from quant_fund.models.motivic_trace import (
    bench_motivic_trace,
)
from quant_fund.models.motivic_transfer2 import (
    bench_motivic_transfer2,
)


def test_motivic_trace():
    assert bench_motivic_trace()["synthetic_motivic_trace"] == 1.0


def test_motivic_transfer2():
    assert bench_motivic_transfer2()["synthetic_motivic_transfer2"] == 1.0


def test_motivic_coniveau():
    assert bench_motivic_coniveau()["synthetic_motivic_coniveau"] == 1.0


def test_motivic_atiyah():
    assert bench_motivic_atiyah()["synthetic_motivic_atiyah"] == 1.0


def test_motivic_deligne():
    assert bench_motivic_deligne()["synthetic_motivic_deligne"] == 1.0


def test_motivic_residue():
    assert bench_motivic_residue()["synthetic_motivic_residue"] == 1.0
