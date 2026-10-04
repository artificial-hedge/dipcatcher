"""Wave-602 cyclic-homology bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cyclotomic_spec import (
    bench_cyclotomic_spec,
)
from quant_fund.models.negative_cyclic import (
    bench_negative_cyclic,
)
from quant_fund.models.periodic_cyclic import (
    bench_periodic_cyclic,
)
from quant_fund.models.tate_construction import (
    bench_tate_construction,
)
from quant_fund.models.tc_spec import bench_tc_spec
from quant_fund.models.tr_structure import bench_tr_structure

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


def bench_cyclotomic_spec_family(
    seed: int = _SEED + 3518,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cyclotomic_spec",
            bench_cyclotomic_spec(seed),
        )
    )


def bench_tr_structure_family(
    seed: int = _SEED + 3519,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tr_structure",
            bench_tr_structure(seed),
        )
    )


def bench_tc_spec_family(seed: int = _SEED + 3520) -> dict[str, float]:
    return _floats(_finite_blob("tc_spec", bench_tc_spec(seed)))


def bench_negative_cyclic_family(
    seed: int = _SEED + 3521,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "negative_cyclic",
            bench_negative_cyclic(seed),
        )
    )


def bench_periodic_cyclic_family(
    seed: int = _SEED + 3522,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "periodic_cyclic",
            bench_periodic_cyclic(seed),
        )
    )


def bench_tate_construction_family(
    seed: int = _SEED + 3523,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tate_construction",
            bench_tate_construction(seed),
        )
    )
