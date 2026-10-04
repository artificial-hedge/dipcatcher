"""Wave-546 knot-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.alexander_poly import bench_alexander_poly
from quant_fund.models.jones_poly import bench_jones_poly
from quant_fund.models.knot_group import bench_knot_group
from quant_fund.models.knot_invariant import bench_knot_invariant
from quant_fund.models.knot_signature import bench_knot_signature
from quant_fund.models.vassiliev_inv import bench_vassiliev_inv

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


def bench_knot_invariant_family(
    seed: int = _SEED + 3182,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "knot_invariant",
            bench_knot_invariant(seed),
        )
    )


def bench_jones_poly_family(seed: int = _SEED + 3183) -> dict[str, float]:
    return _floats(_finite_blob("jones_poly", bench_jones_poly(seed)))


def bench_alexander_poly_family(
    seed: int = _SEED + 3184,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "alexander_poly",
            bench_alexander_poly(seed),
        )
    )


def bench_knot_group_family(seed: int = _SEED + 3185) -> dict[str, float]:
    return _floats(_finite_blob("knot_group", bench_knot_group(seed)))


def bench_knot_signature_family(
    seed: int = _SEED + 3186,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "knot_signature",
            bench_knot_signature(seed),
        )
    )


def bench_vassiliev_inv_family(seed: int = _SEED + 3187) -> dict[str, float]:
    return _floats(_finite_blob("vassiliev_inv", bench_vassiliev_inv(seed)))
