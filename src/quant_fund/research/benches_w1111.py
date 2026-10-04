"""Wave-1111 chemistry-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.medicinal_chemistry import bench_medicinal_chemistry
from quant_fund.models.photochemistry import bench_photochemistry
from quant_fund.models.quantum_chemistry import bench_quantum_chemistry
from quant_fund.models.spectroscopy import bench_spectroscopy
from quant_fund.models.stereochemistry import bench_stereochemistry
from quant_fund.models.supramolecular_chemistry import bench_supramolecular_chemistry

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


def bench_quantum_chemistry_family(seed: int = _SEED + 49900) -> dict[str, float]:
    return _finite_blob(bench_quantum_chemistry(seed))


def bench_spectroscopy_family(seed: int = _SEED + 49901) -> dict[str, float]:
    return _finite_blob(bench_spectroscopy(seed))


def bench_photochemistry_family(seed: int = _SEED + 49902) -> dict[str, float]:
    return _finite_blob(bench_photochemistry(seed))


def bench_stereochemistry_family(seed: int = _SEED + 49903) -> dict[str, float]:
    return _finite_blob(bench_stereochemistry(seed))


def bench_supramolecular_chemistry_family(seed: int = _SEED + 49904) -> dict[str, float]:
    return _finite_blob(bench_supramolecular_chemistry(seed))


def bench_medicinal_chemistry_family(seed: int = _SEED + 49905) -> dict[str, float]:
    return _finite_blob(bench_medicinal_chemistry(seed))
