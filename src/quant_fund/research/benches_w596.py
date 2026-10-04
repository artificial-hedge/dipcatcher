"""Wave-596 p-adic-Hodge bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.breuil_mod import bench_breuil_mod
from quant_fund.models.etale_phi import bench_etale_phi
from quant_fund.models.finite_height import (
    bench_finite_height,
)
from quant_fund.models.galois_lattice import (
    bench_galois_lattice,
)
from quant_fund.models.kisin_mod import bench_kisin_mod
from quant_fund.models.padic_hodge import bench_padic_hodge

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


def bench_breuil_mod_family(
    seed: int = _SEED + 3482,
) -> dict[str, float]:
    return _floats(_finite_blob("breuil_mod", bench_breuil_mod(seed)))


def bench_kisin_mod_family(seed: int = _SEED + 3483) -> dict[str, float]:
    return _floats(_finite_blob("kisin_mod", bench_kisin_mod(seed)))


def bench_galois_lattice_family(
    seed: int = _SEED + 3484,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "galois_lattice",
            bench_galois_lattice(seed),
        )
    )


def bench_padic_hodge_family(
    seed: int = _SEED + 3485,
) -> dict[str, float]:
    return _floats(_finite_blob("padic_hodge", bench_padic_hodge(seed)))


def bench_finite_height_family(
    seed: int = _SEED + 3486,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "finite_height",
            bench_finite_height(seed),
        )
    )


def bench_etale_phi_family(
    seed: int = _SEED + 3487,
) -> dict[str, float]:
    return _floats(_finite_blob("etale_phi", bench_etale_phi(seed)))
