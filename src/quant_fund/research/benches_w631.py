"""Wave-631 etale-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.constructible_sh import (
    bench_constructible_sh,
)
from quant_fund.models.etale_cover3 import (
    bench_etale_cover3,
)
from quant_fund.models.etale_site3 import (
    bench_etale_site3,
)
from quant_fund.models.ql_sheaf import bench_ql_sheaf
from quant_fund.models.torsion_sheaf import (
    bench_torsion_sheaf,
)
from quant_fund.models.weil_sheaf import bench_weil_sheaf

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


def bench_etale_cover3_family(
    seed: int = _SEED + 3692,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "etale_cover3",
            bench_etale_cover3(seed),
        )
    )


def bench_etale_site3_family(
    seed: int = _SEED + 3693,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "etale_site3",
            bench_etale_site3(seed),
        )
    )


def bench_constructible_sh_family(
    seed: int = _SEED + 3694,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "constructible_sh",
            bench_constructible_sh(seed),
        )
    )


def bench_weil_sheaf_family(
    seed: int = _SEED + 3695,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "weil_sheaf",
            bench_weil_sheaf(seed),
        )
    )


def bench_torsion_sheaf_family(
    seed: int = _SEED + 3696,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "torsion_sheaf",
            bench_torsion_sheaf(seed),
        )
    )


def bench_ql_sheaf_family(
    seed: int = _SEED + 3697,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "ql_sheaf",
            bench_ql_sheaf(seed),
        )
    )
