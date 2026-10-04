"""Wave-754 random-cluster bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.brydges_spencer import (
    bench_brydges_spencer,
)
from quant_fund.models.caracciolo_pelissetto import (
    bench_caracciolo_pelissetto,
)
from quant_fund.models.glasner_aizenman import (
    bench_glasner_aizenman,
)
from quant_fund.models.grimmett_rc import bench_grimmett_rc
from quant_fund.models.hara_hara import bench_hara_hara
from quant_fund.models.sokal_bcc import bench_sokal_bcc

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


def bench_sokal_bcc_family(
    seed: int = _SEED + 14300,
) -> dict[str, float]:
    return _floats(_finite_blob("sokal_bcc", bench_sokal_bcc(seed)))


def bench_caracciolo_pelissetto_family(
    seed: int = _SEED + 14301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "caracciolo_pelissetto",
            bench_caracciolo_pelissetto(seed),
        )
    )


def bench_grimmett_rc_family(
    seed: int = _SEED + 14302,
) -> dict[str, float]:
    return _floats(_finite_blob("grimmett_rc", bench_grimmett_rc(seed)))


def bench_hara_hara_family(
    seed: int = _SEED + 14303,
) -> dict[str, float]:
    return _floats(_finite_blob("hara_hara", bench_hara_hara(seed)))


def bench_brydges_spencer_family(
    seed: int = _SEED + 14304,
) -> dict[str, float]:
    return _floats(_finite_blob("brydges_spencer", bench_brydges_spencer(seed)))


def bench_glasner_aizenman_family(
    seed: int = _SEED + 14305,
) -> dict[str, float]:
    return _floats(_finite_blob("glasner_aizenman", bench_glasner_aizenman(seed)))
