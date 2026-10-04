"""Wave-694 motivic-21 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.motivic_atiyah import (
    bench_motivic_atiyah,
)
from quant_fund.models.motivic_coniveau import (
    bench_motivic_coniveau,
)
from quant_fund.models.motivic_deligne import (
    bench_motivic_deligne,
)
from quant_fund.models.motivic_residue import (
    bench_motivic_residue,
)
from quant_fund.models.motivic_trace import (
    bench_motivic_trace,
)
from quant_fund.models.motivic_transfer2 import (
    bench_motivic_transfer2,
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


def bench_motivic_trace_family(
    seed: int = _SEED + 8300,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_trace", bench_motivic_trace(seed)))


def bench_motivic_transfer2_family(
    seed: int = _SEED + 8301,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_transfer2",
            bench_motivic_transfer2(seed),
        )
    )


def bench_motivic_coniveau_family(
    seed: int = _SEED + 8302,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_coniveau", bench_motivic_coniveau(seed)))


def bench_motivic_atiyah_family(
    seed: int = _SEED + 8303,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_atiyah", bench_motivic_atiyah(seed)))


def bench_motivic_deligne_family(
    seed: int = _SEED + 8304,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_deligne", bench_motivic_deligne(seed)))


def bench_motivic_residue_family(
    seed: int = _SEED + 8305,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_residue", bench_motivic_residue(seed)))
