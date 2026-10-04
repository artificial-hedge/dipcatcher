from quant_fund.models.burkholder_davis import (
    bench_burkholder_davis,
)
from quant_fund.models.doleans_meas import bench_doleans_meas
from quant_fund.models.gundy_mart import bench_gundy_mart
from quant_fund.models.local_mart import bench_local_mart
from quant_fund.models.predictable_proc import (
    bench_predictable_proc,
)
from quant_fund.models.square_bracket import (
    bench_square_bracket,
)


def test_doleans_meas():
    assert bench_doleans_meas()["synthetic_doleans_meas"] == 1.0


def test_predictable_proc():
    assert bench_predictable_proc()["synthetic_predictable_proc"] == 1.0


def test_local_mart():
    assert bench_local_mart()["synthetic_local_mart"] == 1.0


def test_square_bracket():
    assert bench_square_bracket()["synthetic_square_bracket"] == 1.0


def test_burkholder_davis():
    assert bench_burkholder_davis()["synthetic_burkholder_davis"] == 1.0


def test_gundy_mart():
    assert bench_gundy_mart()["synthetic_gundy_mart"] == 1.0
