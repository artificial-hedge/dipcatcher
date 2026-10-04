"""Wave-552 contact-topology bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.contact_form import bench_contact_form
from quant_fund.models.convex_surface import bench_convex_surface
from quant_fund.models.giroux_corr import bench_giroux_corr
from quant_fund.models.legendrian_knot import bench_legendrian_knot
from quant_fund.models.overtwisted import bench_overtwisted
from quant_fund.models.tight_contact import bench_tight_contact

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


def bench_contact_form_family(seed: int = _SEED + 3218) -> dict[str, float]:
    return _floats(_finite_blob("contact_form", bench_contact_form(seed)))


def bench_legendrian_knot_family(
    seed: int = _SEED + 3219,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "legendrian_knot",
            bench_legendrian_knot(seed),
        )
    )


def bench_overtwisted_family(seed: int = _SEED + 3220) -> dict[str, float]:
    return _floats(_finite_blob("overtwisted", bench_overtwisted(seed)))


def bench_tight_contact_family(seed: int = _SEED + 3221) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tight_contact",
            bench_tight_contact(seed),
        )
    )


def bench_giroux_corr_family(seed: int = _SEED + 3222) -> dict[str, float]:
    return _floats(_finite_blob("giroux_corr", bench_giroux_corr(seed)))


def bench_convex_surface_family(seed: int = _SEED + 3223) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "convex_surface",
            bench_convex_surface(seed),
        )
    )
