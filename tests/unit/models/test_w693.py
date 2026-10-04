from quant_fund.models.derived_cartesian import (
    bench_derived_cartesian,
)
from quant_fund.models.derived_etale import (
    bench_derived_etale,
)
from quant_fund.models.derived_flat import bench_derived_flat
from quant_fund.models.derived_quasi_coherent import (
    bench_derived_quasi_coherent,
)
from quant_fund.models.derived_represent import (
    bench_derived_represent,
)
from quant_fund.models.derived_smooth2 import (
    bench_derived_smooth2,
)


def test_derived_etale():
    assert bench_derived_etale()["synthetic_derived_etale"] == 1.0


def test_derived_flat():
    assert bench_derived_flat()["synthetic_derived_flat"] == 1.0


def test_derived_smooth2():
    assert bench_derived_smooth2()["synthetic_derived_smooth2"] == 1.0


def test_derived_quasi_coherent():
    assert bench_derived_quasi_coherent()["synthetic_derived_quasi_coherent"] == 1.0


def test_derived_represent():
    assert bench_derived_represent()["synthetic_derived_represent"] == 1.0


def test_derived_cartesian():
    assert bench_derived_cartesian()["synthetic_derived_cartesian"] == 1.0
