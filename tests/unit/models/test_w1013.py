"""Wave-1013 atomic/molecular-physics canon tests."""

from __future__ import annotations

from quant_fund.models.born_oppenheimer import bench_born_oppenheimer
from quant_fund.models.hartree_fock import bench_hartree_fock
from quant_fund.models.molecular_orbitals import bench_molecular_orbitals
from quant_fund.models.rotational_spectra import bench_rotational_spectra
from quant_fund.models.vibrational_spectra import bench_vibrational_spectra
from quant_fund.models.zeeman_effect import bench_zeeman_effect


def test_hartree_fock():
    assert bench_hartree_fock()["synthetic_hartree_fock"] == 1.0


def test_born_oppenheimer():
    assert bench_born_oppenheimer()["synthetic_born_oppenheimer"] == 1.0


def test_molecular_orbitals():
    assert bench_molecular_orbitals()["synthetic_molecular_orbitals"] == 1.0


def test_rotational_spectra():
    assert bench_rotational_spectra()["synthetic_rotational_spectra"] == 1.0


def test_vibrational_spectra():
    assert bench_vibrational_spectra()["synthetic_vibrational_spectra"] == 1.0


def test_zeeman_effect():
    assert bench_zeeman_effect()["synthetic_zeeman_effect"] == 1.0
