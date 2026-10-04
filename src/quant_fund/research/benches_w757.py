"""Wave-757 CLE bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.camia_newman import bench_camia_newman
from quant_fund.models.dubedat_cle import bench_dubedat_cle
from quant_fund.models.kemppainen_werner import (
    bench_kemppainen_werner,
)
from quant_fund.models.miller_watson_cle import (
    bench_miller_watson_cle,
)
from quant_fund.models.rivera_cle import bench_rivera_cle
from quant_fund.models.sheffield_werner_cle import (
    bench_sheffield_werner_cle,
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


def bench_sheffield_werner_cle_family(
    seed: int = _SEED + 14600,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sheffield_werner_cle",
            bench_sheffield_werner_cle(seed),
        )
    )


def bench_miller_watson_cle_family(
    seed: int = _SEED + 14601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "miller_watson_cle",
            bench_miller_watson_cle(seed),
        )
    )


def bench_camia_newman_family(
    seed: int = _SEED + 14602,
) -> dict[str, float]:
    return _floats(_finite_blob("camia_newman", bench_camia_newman(seed)))


def bench_dubedat_cle_family(
    seed: int = _SEED + 14603,
) -> dict[str, float]:
    return _floats(_finite_blob("dubedat_cle", bench_dubedat_cle(seed)))


def bench_kemppainen_werner_family(
    seed: int = _SEED + 14604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "kemppainen_werner",
            bench_kemppainen_werner(seed),
        )
    )


def bench_rivera_cle_family(
    seed: int = _SEED + 14605,
) -> dict[str, float]:
    return _floats(_finite_blob("rivera_cle", bench_rivera_cle(seed)))
