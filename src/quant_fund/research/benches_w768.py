"""Wave-768 large-deviation bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dw_ldp import bench_dw_ldp
from quant_fund.models.freidlin_wentzell import (
    bench_freidlin_wentzell,
)
from quant_fund.models.mogulskii_thm import (
    bench_mogulskii_thm,
)
from quant_fund.models.sanov_thm import bench_sanov_thm
from quant_fund.models.schider_thm import bench_schider_thm
from quant_fund.models.varadhan_ldp import bench_varadhan_ldp

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


def bench_varadhan_ldp_family(
    seed: int = _SEED + 15700,
) -> dict[str, float]:
    return _floats(_finite_blob("varadhan_ldp", bench_varadhan_ldp(seed)))


def bench_freidlin_wentzell_family(
    seed: int = _SEED + 15701,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "freidlin_wentzell",
            bench_freidlin_wentzell(seed),
        )
    )


def bench_dw_ldp_family(
    seed: int = _SEED + 15702,
) -> dict[str, float]:
    return _floats(_finite_blob("dw_ldp", bench_dw_ldp(seed)))


def bench_sanov_thm_family(
    seed: int = _SEED + 15703,
) -> dict[str, float]:
    return _floats(_finite_blob("sanov_thm", bench_sanov_thm(seed)))


def bench_mogulskii_thm_family(
    seed: int = _SEED + 15704,
) -> dict[str, float]:
    return _floats(_finite_blob("mogulskii_thm", bench_mogulskii_thm(seed)))


def bench_schider_thm_family(
    seed: int = _SEED + 15705,
) -> dict[str, float]:
    return _floats(_finite_blob("schider_thm", bench_schider_thm(seed)))
