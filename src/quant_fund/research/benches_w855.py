"""Wave-855 wavelet-Galerkin bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adapt_wavelet import (
    bench_adapt_wavelet,
)
from quant_fund.models.coiflet_basis import (
    bench_coiflet_basis,
)
from quant_fund.models.daubechies_basis import (
    bench_daubechies_basis,
)
from quant_fund.models.spline_wavelet import (
    bench_spline_wavelet,
)
from quant_fund.models.wavelet_collocation import (
    bench_wavelet_collocation,
)
from quant_fund.models.wavelet_galerkin import (
    bench_wavelet_galerkin,
)

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


def bench_wavelet_galerkin_family(
    seed: int = _SEED + 24300,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wavelet_galerkin",
            bench_wavelet_galerkin(seed),
        )
    )


def bench_daubechies_basis_family(
    seed: int = _SEED + 24301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "daubechies_basis",
            bench_daubechies_basis(seed),
        )
    )


def bench_coiflet_basis_family(
    seed: int = _SEED + 24302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "coiflet_basis",
            bench_coiflet_basis(seed),
        )
    )


def bench_spline_wavelet_family(
    seed: int = _SEED + 24303,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spline_wavelet",
            bench_spline_wavelet(seed),
        )
    )


def bench_wavelet_collocation_family(
    seed: int = _SEED + 24304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wavelet_collocation",
            bench_wavelet_collocation(seed),
        )
    )


def bench_adapt_wavelet_family(
    seed: int = _SEED + 24305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "adapt_wavelet",
            bench_adapt_wavelet(seed),
        )
    )
