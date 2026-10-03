from quant_fund.models.discrete_ordinates import (
    bench_discrete_ordinates,
)
from quant_fund.models.moc_transport import (
    bench_moc_transport,
)
from quant_fund.models.pn_closure import (
    bench_pn_closure,
)
from quant_fund.models.spherical_harmonics import (
    bench_spherical_harmonics,
)
from quant_fund.models.spn_equations import (
    bench_spn_equations,
)
from quant_fund.models.transport_sn import (
    bench_transport_sn,
)


def test_transport_sn():
    assert bench_transport_sn()["synthetic_transport_sn"] == 1.0


def test_discrete_ordinates():
    assert bench_discrete_ordinates()["synthetic_discrete_ordinates"] == 1.0


def test_spherical_harmonics():
    assert bench_spherical_harmonics()["synthetic_spherical_harmonics"] == 1.0


def test_spn_equations():
    assert bench_spn_equations()["synthetic_spn_equations"] == 1.0


def test_moc_transport():
    assert bench_moc_transport()["synthetic_moc_transport"] == 1.0


def test_pn_closure():
    assert bench_pn_closure()["synthetic_pn_closure"] == 1.0
