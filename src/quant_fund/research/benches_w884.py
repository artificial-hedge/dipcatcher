"""Wave-884 wavelet-2/spectral-elem bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.boundary_element import (
    bench_boundary_element,
)
from quant_fund.models.marquina_flux import (
    bench_marquina_flux,
)
from quant_fund.models.multidomain_sem import (
    bench_multidomain_sem,
)
from quant_fund.models.nodal_dg import (
    bench_nodal_dg,
)
from quant_fund.models.second_gen_wavelet import (
    bench_second_gen_wavelet,
)
from quant_fund.models.wavelet_matrix import (
    bench_wavelet_matrix,
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


def bench_wavelet_matrix_family(
    seed: int = _SEED + 27200,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "wavelet_matrix",
            bench_wavelet_matrix(seed),
        )
    )


def bench_second_gen_wavelet_family(
    seed: int = _SEED + 27201,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "second_gen_wavelet",
            bench_second_gen_wavelet(seed),
        )
    )


def bench_nodal_dg_family(
    seed: int = _SEED + 27202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nodal_dg",
            bench_nodal_dg(seed),
        )
    )


def bench_multidomain_sem_family(
    seed: int = _SEED + 27203,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "multidomain_sem",
            bench_multidomain_sem(seed),
        )
    )


def bench_boundary_element_family(
    seed: int = _SEED + 27204,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "boundary_element",
            bench_boundary_element(seed),
        )
    )


def bench_marquina_flux_family(
    seed: int = _SEED + 27205,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "marquina_flux",
            bench_marquina_flux(seed),
        )
    )
