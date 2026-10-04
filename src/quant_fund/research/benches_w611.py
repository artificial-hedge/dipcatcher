"""Wave-611 category-8 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.mate_dual import bench_mate_dual
from quant_fund.models.modification import bench_modification
from quant_fund.models.pasting_diag import bench_pasting_diag
from quant_fund.models.pseudo_naturality import (
    bench_pseudo_naturality,
)
from quant_fund.models.two_adjoint import bench_two_adjoint
from quant_fund.models.whisker_comp import bench_whisker_comp

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


def bench_pasting_diag_family(
    seed: int = _SEED + 3572,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pasting_diag",
            bench_pasting_diag(seed),
        )
    )


def bench_mate_dual_family(seed: int = _SEED + 3573) -> dict[str, float]:
    return _floats(_finite_blob("mate_dual", bench_mate_dual(seed)))


def bench_whisker_comp_family(
    seed: int = _SEED + 3574,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "whisker_comp",
            bench_whisker_comp(seed),
        )
    )


def bench_pseudo_naturality_family(
    seed: int = _SEED + 3575,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pseudo_naturality",
            bench_pseudo_naturality(seed),
        )
    )


def bench_two_adjoint_family(
    seed: int = _SEED + 3576,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "two_adjoint",
            bench_two_adjoint(seed),
        )
    )


def bench_modification_family(
    seed: int = _SEED + 3577,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "modification",
            bench_modification(seed),
        )
    )
