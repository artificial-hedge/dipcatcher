"""Wave-1006 quantum-mechanics canon tests."""

from __future__ import annotations

from quant_fund.models.fock_space import bench_fock_space
from quant_fund.models.harmonic_oscillator import bench_harmonic_oscillator
from quant_fund.models.hydrogen_atom import bench_hydrogen_atom
from quant_fund.models.schrodinger_eq import bench_schrodinger_eq
from quant_fund.models.spin_half import bench_spin_half
from quant_fund.models.wigner_wick import bench_wigner_wick


def test_schrodinger_eq():
    assert bench_schrodinger_eq()["synthetic_schrodinger_eq"] == 1.0


def test_hydrogen_atom():
    assert bench_hydrogen_atom()["synthetic_hydrogen_atom"] == 1.0


def test_harmonic_oscillator():
    assert bench_harmonic_oscillator()["synthetic_harmonic_oscillator"] == 1.0


def test_spin_half():
    assert bench_spin_half()["synthetic_spin_half"] == 1.0


def test_wigner_wick():
    assert bench_wigner_wick()["synthetic_wigner_wick"] == 1.0


def test_fock_space():
    assert bench_fock_space()["synthetic_fock_space"] == 1.0
