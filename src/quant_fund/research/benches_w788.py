"""Wave-788 Malliavin-calculus bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.clark_ocone import bench_clark_ocone
from quant_fund.models.divergence_op import (
    bench_divergence_op,
)
from quant_fund.models.nourdin_peccati import (
    bench_nourdin_peccati,
)
from quant_fund.models.nualart_pardoux import (
    bench_nualart_pardoux,
)
from quant_fund.models.skorohod_int import (
    bench_skorohod_int,
)
from quant_fund.models.wiener_chaos import (
    bench_wiener_chaos,
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


def bench_clark_ocone_family(
    seed: int = _SEED + 17700,
) -> dict[str, float]:
    return _floats(_finite_blob("clark_ocone", bench_clark_ocone(seed)))


def bench_nualart_pardoux_family(
    seed: int = _SEED + 17701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nualart_pardoux",
            bench_nualart_pardoux(seed),
        )
    )


def bench_divergence_op_family(
    seed: int = _SEED + 17702,
) -> dict[str, float]:
    return _floats(_finite_blob("divergence_op", bench_divergence_op(seed)))


def bench_wiener_chaos_family(
    seed: int = _SEED + 17703,
) -> dict[str, float]:
    return _floats(_finite_blob("wiener_chaos", bench_wiener_chaos(seed)))


def bench_skorohod_int_family(
    seed: int = _SEED + 17704,
) -> dict[str, float]:
    return _floats(_finite_blob("skorohod_int", bench_skorohod_int(seed)))


def bench_nourdin_peccati_family(
    seed: int = _SEED + 17705,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nourdin_peccati",
            bench_nourdin_peccati(seed),
        )
    )
