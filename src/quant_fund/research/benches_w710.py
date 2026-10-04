"""Wave-710 higher-algebra-11 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.floyd_farey import bench_floyd_farey
from quant_fund.models.higher_algebra8 import (
    bench_higher_algebra8,
)
from quant_fund.models.little_discs3 import (
    bench_little_discs3,
)
from quant_fund.models.operad_infty4 import (
    bench_operad_infty4,
)
from quant_fund.models.operad_swiss3 import (
    bench_operad_swiss3,
)
from quant_fund.models.operad_twisted import (
    bench_operad_twisted,
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


def bench_higher_algebra8_family(
    seed: int = _SEED + 9900,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "higher_algebra8",
            bench_higher_algebra8(seed),
        )
    )


def bench_operad_infty4_family(
    seed: int = _SEED + 9901,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operad_infty4",
            bench_operad_infty4(seed),
        )
    )


def bench_floyd_farey_family(
    seed: int = _SEED + 9902,
) -> dict[str, float]:
    return _floats(_finite_blob("floyd_farey", bench_floyd_farey(seed)))


def bench_operad_swiss3_family(
    seed: int = _SEED + 9903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operad_swiss3",
            bench_operad_swiss3(seed),
        )
    )


def bench_little_discs3_family(
    seed: int = _SEED + 9904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "little_discs3",
            bench_little_discs3(seed),
        )
    )


def bench_operad_twisted_family(
    seed: int = _SEED + 9905,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "operad_twisted",
            bench_operad_twisted(seed),
        )
    )
