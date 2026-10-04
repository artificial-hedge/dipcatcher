"""Wave-1013 atomic/molecular-physics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.born_oppenheimer import bench_born_oppenheimer
from quant_fund.models.hartree_fock import bench_hartree_fock
from quant_fund.models.molecular_orbitals import bench_molecular_orbitals
from quant_fund.models.rotational_spectra import bench_rotational_spectra
from quant_fund.models.vibrational_spectra import bench_vibrational_spectra
from quant_fund.models.zeeman_effect import bench_zeeman_effect

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


def bench_hartree_fock_family(seed: int = _SEED + 40100) -> dict[str, float]:
    return _finite_blob(bench_hartree_fock(seed))


def bench_born_oppenheimer_family(seed: int = _SEED + 40101) -> dict[str, float]:
    return _finite_blob(bench_born_oppenheimer(seed))


def bench_molecular_orbitals_family(seed: int = _SEED + 40102) -> dict[str, float]:
    return _finite_blob(bench_molecular_orbitals(seed))


def bench_rotational_spectra_family(seed: int = _SEED + 40103) -> dict[str, float]:
    return _finite_blob(bench_rotational_spectra(seed))


def bench_vibrational_spectra_family(seed: int = _SEED + 40104) -> dict[str, float]:
    return _finite_blob(bench_vibrational_spectra(seed))


def bench_zeeman_effect_family(seed: int = _SEED + 40105) -> dict[str, float]:
    return _finite_blob(bench_zeeman_effect(seed))
