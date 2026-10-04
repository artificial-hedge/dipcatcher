"""Wave-1011 condensed-matter canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.band_structure import bench_band_structure
from quant_fund.models.bloch_theorem import bench_bloch_theorem
from quant_fund.models.hubbard_model import bench_hubbard_model
from quant_fund.models.kondo_effect import bench_kondo_effect
from quant_fund.models.phonon_spectrum import bench_phonon_spectrum
from quant_fund.models.tight_binding import bench_tight_binding

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


def bench_bloch_theorem_family(seed: int = _SEED + 39900) -> dict[str, float]:
    return _finite_blob(bench_bloch_theorem(seed))


def bench_tight_binding_family(seed: int = _SEED + 39901) -> dict[str, float]:
    return _finite_blob(bench_tight_binding(seed))


def bench_phonon_spectrum_family(seed: int = _SEED + 39902) -> dict[str, float]:
    return _finite_blob(bench_phonon_spectrum(seed))


def bench_band_structure_family(seed: int = _SEED + 39903) -> dict[str, float]:
    return _finite_blob(bench_band_structure(seed))


def bench_hubbard_model_family(seed: int = _SEED + 39904) -> dict[str, float]:
    return _finite_blob(bench_hubbard_model(seed))


def bench_kondo_effect_family(seed: int = _SEED + 39905) -> dict[str, float]:
    return _finite_blob(bench_kondo_effect(seed))
