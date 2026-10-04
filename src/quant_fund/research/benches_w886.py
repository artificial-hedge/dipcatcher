"""Wave-886 QMC/tensor bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.faure_seq import (
    bench_faure_seq,
)
from quant_fund.models.gauss_hermite import (
    bench_gauss_hermite,
)
from quant_fund.models.gauss_laguerre import (
    bench_gauss_laguerre,
)
from quant_fund.models.hiot_decomp import (
    bench_hiot_decomp,
)
from quant_fund.models.importance_mc import (
    bench_importance_mc,
)
from quant_fund.models.tensor_train import (
    bench_tensor_train,
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


def bench_faure_seq_family(
    seed: int = _SEED + 27400,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "faure_seq",
            bench_faure_seq(seed),
        )
    )


def bench_importance_mc_family(
    seed: int = _SEED + 27401,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "importance_mc",
            bench_importance_mc(seed),
        )
    )


def bench_gauss_hermite_family(
    seed: int = _SEED + 27402,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gauss_hermite",
            bench_gauss_hermite(seed),
        )
    )


def bench_gauss_laguerre_family(
    seed: int = _SEED + 27403,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gauss_laguerre",
            bench_gauss_laguerre(seed),
        )
    )


def bench_tensor_train_family(
    seed: int = _SEED + 27404,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tensor_train",
            bench_tensor_train(seed),
        )
    )


def bench_hiot_decomp_family(
    seed: int = _SEED + 27405,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hiot_decomp",
            bench_hiot_decomp(seed),
        )
    )
