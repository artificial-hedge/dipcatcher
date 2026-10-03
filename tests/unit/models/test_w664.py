from quant_fund.models.e_ring_moduli import bench_e_ring_moduli
from quant_fund.models.elliptic_spec2 import bench_elliptic_spec2
from quant_fund.models.spectral_artstack import (
    bench_spectral_artstack,
)
from quant_fund.models.spectral_moduli import (
    bench_spectral_moduli,
)
from quant_fund.models.structured_spec import (
    bench_structured_spec,
)
from quant_fund.models.tmf_stack import bench_tmf_stack


def test_spectral_moduli():
    assert bench_spectral_moduli()["synthetic_spectral_moduli"] == 1.0


def test_e_ring_moduli():
    assert bench_e_ring_moduli()["synthetic_e_ring_moduli"] == 1.0


def test_tmf_stack():
    assert bench_tmf_stack()["synthetic_tmf_stack"] == 1.0


def test_spectral_artstack():
    assert bench_spectral_artstack()["synthetic_spectral_artstack"] == 1.0


def test_structured_spec():
    assert bench_structured_spec()["synthetic_structured_spec"] == 1.0


def test_elliptic_spec2():
    assert bench_elliptic_spec2()["synthetic_elliptic_spec2"] == 1.0
