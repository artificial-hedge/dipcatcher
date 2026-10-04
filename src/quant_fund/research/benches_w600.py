"""Wave-600 operad-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.a_infty_alg import bench_a_infty_alg
from quant_fund.models.koszul_duality import (
    bench_koszul_duality,
)
from quant_fund.models.l_infty_alg import bench_l_infty_alg
from quant_fund.models.minimal_model_op import (
    bench_minimal_model_op,
)
from quant_fund.models.operad_cobar import bench_operad_cobar
from quant_fund.models.operadic_bar import bench_operadic_bar

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


def bench_a_infty_alg_family(
    seed: int = _SEED + 3506,
) -> dict[str, float]:
    return _floats(_finite_blob("a_infty_alg", bench_a_infty_alg(seed)))


def bench_l_infty_alg_family(
    seed: int = _SEED + 3507,
) -> dict[str, float]:
    return _floats(_finite_blob("l_infty_alg", bench_l_infty_alg(seed)))


def bench_koszul_duality_family(
    seed: int = _SEED + 3508,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "koszul_duality",
            bench_koszul_duality(seed),
        )
    )


def bench_minimal_model_op_family(
    seed: int = _SEED + 3509,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "minimal_model_op",
            bench_minimal_model_op(seed),
        )
    )


def bench_operadic_bar_family(
    seed: int = _SEED + 3510,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operadic_bar",
            bench_operadic_bar(seed),
        )
    )


def bench_operad_cobar_family(
    seed: int = _SEED + 3511,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operad_cobar",
            bench_operad_cobar(seed),
        )
    )
