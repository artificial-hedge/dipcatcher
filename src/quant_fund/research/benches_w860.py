"""Wave-860 isogeometric/immersed-methods bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cut_cell import (
    bench_cut_cell,
)
from quant_fund.models.fictitious_domain import (
    bench_fictitious_domain,
)
from quant_fund.models.immersed_boundary import (
    bench_immersed_boundary,
)
from quant_fund.models.iso_geom import (
    bench_iso_geom,
)
from quant_fund.models.nurbs_elem import (
    bench_nurbs_elem,
)
from quant_fund.models.xfem import (
    bench_xfem,
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


def bench_iso_geom_family(
    seed: int = _SEED + 24800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "iso_geom",
            bench_iso_geom(seed),
        )
    )


def bench_nurbs_elem_family(
    seed: int = _SEED + 24801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nurbs_elem",
            bench_nurbs_elem(seed),
        )
    )


def bench_xfem_family(
    seed: int = _SEED + 24802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "xfem",
            bench_xfem(seed),
        )
    )


def bench_immersed_boundary_family(
    seed: int = _SEED + 24803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "immersed_boundary",
            bench_immersed_boundary(seed),
        )
    )


def bench_cut_cell_family(
    seed: int = _SEED + 24804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cut_cell",
            bench_cut_cell(seed),
        )
    )


def bench_fictitious_domain_family(
    seed: int = _SEED + 24805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "fictitious_domain",
            bench_fictitious_domain(seed),
        )
    )
