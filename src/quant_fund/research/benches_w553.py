"""Wave-553 mirror-symmetry bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.frobenius_mfd import bench_frobenius_mfd
from quant_fund.models.givental_j import bench_givental_j
from quant_fund.models.mirror_symmetry import bench_mirror_symmetry
from quant_fund.models.quantum_cohomology import bench_quantum_cohomology
from quant_fund.models.quintic_invariants import bench_quintic_invariants
from quant_fund.models.toric_mirror import bench_toric_mirror

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


def bench_mirror_symmetry_family(
    seed: int = _SEED + 3224,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mirror_symmetry",
            bench_mirror_symmetry(seed),
        )
    )


def bench_givental_j_family(seed: int = _SEED + 3225) -> dict[str, float]:
    return _floats(_finite_blob("givental_j", bench_givental_j(seed)))


def bench_quantum_cohomology_family(
    seed: int = _SEED + 3226,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quantum_cohomology",
            bench_quantum_cohomology(seed),
        )
    )


def bench_quintic_invariants_family(
    seed: int = _SEED + 3227,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "quintic_invariants",
            bench_quintic_invariants(seed),
        )
    )


def bench_toric_mirror_family(seed: int = _SEED + 3228) -> dict[str, float]:
    return _floats(_finite_blob("toric_mirror", bench_toric_mirror(seed)))


def bench_frobenius_mfd_family(seed: int = _SEED + 3229) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "frobenius_mfd",
            bench_frobenius_mfd(seed),
        )
    )
