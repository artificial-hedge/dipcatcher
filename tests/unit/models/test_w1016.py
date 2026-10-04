"""Wave-1016 optics-2 canon tests."""

from __future__ import annotations

from quant_fund.models.coherence_theory import bench_coherence_theory
from quant_fund.models.diffraction_grating import bench_diffraction_grating
from quant_fund.models.fourier_optics import bench_fourier_optics
from quant_fund.models.holography import bench_holography
from quant_fund.models.interference_fringes import bench_interference_fringes
from quant_fund.models.polarization_states import bench_polarization_states


def test_diffraction_grating():
    assert bench_diffraction_grating()["synthetic_diffraction_grating"] == 1.0


def test_fourier_optics():
    assert bench_fourier_optics()["synthetic_fourier_optics"] == 1.0


def test_interference_fringes():
    assert bench_interference_fringes()["synthetic_interference_fringes"] == 1.0


def test_polarization_states():
    assert bench_polarization_states()["synthetic_polarization_states"] == 1.0


def test_coherence_theory():
    assert bench_coherence_theory()["synthetic_coherence_theory"] == 1.0


def test_holography():
    assert bench_holography()["synthetic_holography"] == 1.0
