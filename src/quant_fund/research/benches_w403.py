"""Wave-403 homotopy-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.hopf_invariant import bench_hopf_invariant
from quant_fund.models.j_hom_toy import bench_j_hom_toy
from quant_fund.models.pi_stems import bench_pi_stems
from quant_fund.models.spectral_atiyah import bench_spectral_atiyah
from quant_fund.models.thom_spectrum import bench_thom_spectrum
from quant_fund.models.toda_bracket import bench_toda_bracket

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_j_hom_toy_family(seed: int = _SEED + 2324) -> dict[str, float]:
    return _floats(_finite_blob("j_hom_toy", bench_j_hom_toy(seed)))


def bench_toda_bracket_family(seed: int = _SEED + 2325) -> dict[str, float]:
    return _floats(_finite_blob("toda_bracket", bench_toda_bracket(seed)))


def bench_spectral_atiyah_family(
    seed: int = _SEED + 2326,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_atiyah", bench_spectral_atiyah(seed)))


def bench_pi_stems_family(seed: int = _SEED + 2327) -> dict[str, float]:
    return _floats(_finite_blob("pi_stems", bench_pi_stems(seed)))


def bench_hopf_invariant_family(
    seed: int = _SEED + 2328,
) -> dict[str, float]:
    return _floats(_finite_blob("hopf_invariant", bench_hopf_invariant(seed)))


def bench_thom_spectrum_family(
    seed: int = _SEED + 2329,
) -> dict[str, float]:
    return _floats(_finite_blob("thom_spectrum", bench_thom_spectrum(seed)))
