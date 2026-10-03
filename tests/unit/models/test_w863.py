from quant_fund.models.bddc_lite import (
    bench_bddc_lite,
)
from quant_fund.models.coarse_correction import (
    bench_coarse_correction,
)
from quant_fund.models.feti_lite import (
    bench_feti_lite,
)
from quant_fund.models.mortar_dd import (
    bench_mortar_dd,
)
from quant_fund.models.schwarz_add import (
    bench_schwarz_add,
)
from quant_fund.models.schwarz_mult import (
    bench_schwarz_mult,
)


def test_schwarz_add():
    assert bench_schwarz_add()["synthetic_schwarz_add"] == 1.0


def test_schwarz_mult():
    assert bench_schwarz_mult()["synthetic_schwarz_mult"] == 1.0


def test_coarse_correction():
    assert bench_coarse_correction()["synthetic_coarse_correction"] == 1.0


def test_mortar_dd():
    assert bench_mortar_dd()["synthetic_mortar_dd"] == 1.0


def test_feti_lite():
    assert bench_feti_lite()["synthetic_feti_lite"] == 1.0


def test_bddc_lite():
    assert bench_bddc_lite()["synthetic_bddc_lite"] == 1.0
