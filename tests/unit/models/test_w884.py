from quant_fund.models.boundary_element import (
    bench_boundary_element,
)
from quant_fund.models.marquina_flux import (
    bench_marquina_flux,
)
from quant_fund.models.multidomain_sem import (
    bench_multidomain_sem,
)
from quant_fund.models.nodal_dg import (
    bench_nodal_dg,
)
from quant_fund.models.second_gen_wavelet import (
    bench_second_gen_wavelet,
)
from quant_fund.models.wavelet_matrix import (
    bench_wavelet_matrix,
)


def test_wavelet_matrix():
    assert bench_wavelet_matrix()["synthetic_wavelet_matrix"] == 1.0


def test_second_gen_wavelet():
    assert bench_second_gen_wavelet()["synthetic_second_gen_wavelet"] == 1.0


def test_nodal_dg():
    assert bench_nodal_dg()["synthetic_nodal_dg"] == 1.0


def test_multidomain_sem():
    assert bench_multidomain_sem()["synthetic_multidomain_sem"] == 1.0


def test_boundary_element():
    assert bench_boundary_element()["synthetic_boundary_element"] == 1.0


def test_marquina_flux():
    assert bench_marquina_flux()["synthetic_marquina_flux"] == 1.0
