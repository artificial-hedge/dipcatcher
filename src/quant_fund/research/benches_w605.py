"""Wave-605 motivic-8 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dk_motive import bench_dk_motive
from quant_fund.models.motivic_adem import (
    bench_motivic_adem,
)
from quant_fund.models.motivic_steenrod import (
    bench_motivic_steenrod,
)
from quant_fund.models.motivic_transfer import (
    bench_motivic_transfer,
)
from quant_fund.models.power_operations import (
    bench_power_operations,
)
from quant_fund.models.simplicial_motive import (
    bench_simplicial_motive,
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


def bench_motivic_steenrod_family(
    seed: int = _SEED + 3536,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_steenrod",
            bench_motivic_steenrod(seed),
        )
    )


def bench_motivic_adem_family(
    seed: int = _SEED + 3537,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_adem",
            bench_motivic_adem(seed),
        )
    )


def bench_power_operations_family(
    seed: int = _SEED + 3538,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "power_operations",
            bench_power_operations(seed),
        )
    )


def bench_simplicial_motive_family(
    seed: int = _SEED + 3539,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "simplicial_motive",
            bench_simplicial_motive(seed),
        )
    )


def bench_dk_motive_family(
    seed: int = _SEED + 3540,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "dk_motive",
            bench_dk_motive(seed),
        )
    )


def bench_motivic_transfer_family(
    seed: int = _SEED + 3541,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_transfer",
            bench_motivic_transfer(seed),
        )
    )
