"""Wave-853 spectral-element bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gll_nodes import (
    bench_gll_nodes,
)
from quant_fund.models.hp_refinement import (
    bench_hp_refinement,
)
from quant_fund.models.mortar_method import (
    bench_mortar_method,
)
from quant_fund.models.sem_grid import (
    bench_sem_grid,
)
from quant_fund.models.spectral_element import (
    bench_spectral_element,
)
from quant_fund.models.tensor_product_sem import (
    bench_tensor_product_sem,
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


def bench_sem_grid_family(
    seed: int = _SEED + 24100,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sem_grid",
            bench_sem_grid(seed),
        )
    )


def bench_gll_nodes_family(
    seed: int = _SEED + 24101,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gll_nodes",
            bench_gll_nodes(seed),
        )
    )


def bench_spectral_element_family(
    seed: int = _SEED + 24102,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "spectral_element",
            bench_spectral_element(seed),
        )
    )


def bench_mortar_method_family(
    seed: int = _SEED + 24103,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "mortar_method",
            bench_mortar_method(seed),
        )
    )


def bench_tensor_product_sem_family(
    seed: int = _SEED + 24104,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tensor_product_sem",
            bench_tensor_product_sem(seed),
        )
    )


def bench_hp_refinement_family(
    seed: int = _SEED + 24105,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hp_refinement",
            bench_hp_refinement(seed),
        )
    )
