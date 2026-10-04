"""Wave-1006 quantum-mechanics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fock_space import bench_fock_space
from quant_fund.models.harmonic_oscillator import bench_harmonic_oscillator
from quant_fund.models.hydrogen_atom import bench_hydrogen_atom
from quant_fund.models.schrodinger_eq import bench_schrodinger_eq
from quant_fund.models.spin_half import bench_spin_half
from quant_fund.models.wigner_wick import bench_wigner_wick

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_schrodinger_eq_family(seed: int = _SEED + 39400) -> dict[str, float]:
    return _finite_blob(bench_schrodinger_eq(seed))


def bench_hydrogen_atom_family(seed: int = _SEED + 39401) -> dict[str, float]:
    return _finite_blob(bench_hydrogen_atom(seed))


def bench_harmonic_oscillator_family(seed: int = _SEED + 39402) -> dict[str, float]:
    return _finite_blob(bench_harmonic_oscillator(seed))


def bench_spin_half_family(seed: int = _SEED + 39403) -> dict[str, float]:
    return _finite_blob(bench_spin_half(seed))


def bench_wigner_wick_family(seed: int = _SEED + 39404) -> dict[str, float]:
    return _finite_blob(bench_wigner_wick(seed))


def bench_fock_space_family(seed: int = _SEED + 39405) -> dict[str, float]:
    return _finite_blob(bench_fock_space(seed))
