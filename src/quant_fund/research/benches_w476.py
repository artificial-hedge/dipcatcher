"""Wave-476 chromatic-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adams_novikov import bench_adams_novikov
from quant_fund.models.bp_spectrum import bench_bp_spectrum
from quant_fund.models.greek_letter import bench_greek_letter
from quant_fund.models.landweber_exact import bench_landweber_exact
from quant_fund.models.picard_grp import bench_picard_grp
from quant_fund.models.smith_toda import bench_smith_toda

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


def bench_bp_spectrum_family(seed: int = _SEED + 2762) -> dict[str, float]:
    return _floats(_finite_blob("bp_spectrum", bench_bp_spectrum(seed)))


def bench_adams_novikov_family(seed: int = _SEED + 2763) -> dict[str, float]:
    return _floats(_finite_blob("adams_novikov", bench_adams_novikov(seed)))


def bench_landweber_exact_family(seed: int = _SEED + 2764) -> dict[str, float]:
    return _floats(_finite_blob("landweber_exact", bench_landweber_exact(seed)))


def bench_greek_letter_family(seed: int = _SEED + 2765) -> dict[str, float]:
    return _floats(_finite_blob("greek_letter", bench_greek_letter(seed)))


def bench_smith_toda_family(seed: int = _SEED + 2766) -> dict[str, float]:
    return _floats(_finite_blob("smith_toda", bench_smith_toda(seed)))


def bench_picard_grp_family(seed: int = _SEED + 2767) -> dict[str, float]:
    return _floats(_finite_blob("picard_grp", bench_picard_grp(seed)))
