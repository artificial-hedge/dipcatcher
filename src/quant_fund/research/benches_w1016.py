"""Wave-1016 optics-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.coherence_theory import bench_coherence_theory
from quant_fund.models.diffraction_grating import bench_diffraction_grating
from quant_fund.models.fourier_optics import bench_fourier_optics
from quant_fund.models.holography import bench_holography
from quant_fund.models.interference_fringes import bench_interference_fringes
from quant_fund.models.polarization_states import bench_polarization_states

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


def bench_diffraction_grating_family(seed: int = _SEED + 40400) -> dict[str, float]:
    return _finite_blob(bench_diffraction_grating(seed))


def bench_fourier_optics_family(seed: int = _SEED + 40401) -> dict[str, float]:
    return _finite_blob(bench_fourier_optics(seed))


def bench_interference_fringes_family(seed: int = _SEED + 40402) -> dict[str, float]:
    return _finite_blob(bench_interference_fringes(seed))


def bench_polarization_states_family(seed: int = _SEED + 40403) -> dict[str, float]:
    return _finite_blob(bench_polarization_states(seed))


def bench_coherence_theory_family(seed: int = _SEED + 40404) -> dict[str, float]:
    return _finite_blob(bench_coherence_theory(seed))


def bench_holography_family(seed: int = _SEED + 40405) -> dict[str, float]:
    return _finite_blob(bench_holography(seed))
