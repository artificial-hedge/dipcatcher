"""Wave-1011 condensed-matter canon tests."""

from __future__ import annotations

from quant_fund.models.band_structure import bench_band_structure
from quant_fund.models.bloch_theorem import bench_bloch_theorem
from quant_fund.models.hubbard_model import bench_hubbard_model
from quant_fund.models.kondo_effect import bench_kondo_effect
from quant_fund.models.phonon_spectrum import bench_phonon_spectrum
from quant_fund.models.tight_binding import bench_tight_binding


def test_bloch_theorem():
    assert bench_bloch_theorem()["synthetic_bloch_theorem"] == 1.0


def test_tight_binding():
    assert bench_tight_binding()["synthetic_tight_binding"] == 1.0


def test_phonon_spectrum():
    assert bench_phonon_spectrum()["synthetic_phonon_spectrum"] == 1.0


def test_band_structure():
    assert bench_band_structure()["synthetic_band_structure"] == 1.0


def test_hubbard_model():
    assert bench_hubbard_model()["synthetic_hubbard_model"] == 1.0


def test_kondo_effect():
    assert bench_kondo_effect()["synthetic_kondo_effect"] == 1.0
