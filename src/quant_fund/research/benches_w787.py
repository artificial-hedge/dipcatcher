"""Wave-787 regularity-structure bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gubinelli_sewing import (
    bench_gubinelli_sewing,
)
from quant_fund.models.ito_signature import (
    bench_ito_signature,
)
from quant_fund.models.lyons_extension import (
    bench_lyons_extension,
)
from quant_fund.models.step_signature import (
    bench_step_signature,
)
from quant_fund.models.tame_map import bench_tame_map
from quant_fund.models.young_integral import (
    bench_young_integral,
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


def bench_ito_signature_family(
    seed: int = _SEED + 17600,
) -> dict[str, float]:
    return _floats(_finite_blob("ito_signature", bench_ito_signature(seed)))


def bench_lyons_extension_family(
    seed: int = _SEED + 17601,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "lyons_extension",
            bench_lyons_extension(seed),
        )
    )


def bench_tame_map_family(
    seed: int = _SEED + 17602,
) -> dict[str, float]:
    return _floats(_finite_blob("tame_map", bench_tame_map(seed)))


def bench_step_signature_family(
    seed: int = _SEED + 17603,
) -> dict[str, float]:
    return _floats(_finite_blob("step_signature", bench_step_signature(seed)))


def bench_gubinelli_sewing_family(
    seed: int = _SEED + 17604,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gubinelli_sewing",
            bench_gubinelli_sewing(seed),
        )
    )


def bench_young_integral_family(
    seed: int = _SEED + 17605,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "young_integral",
            bench_young_integral(seed),
        )
    )
