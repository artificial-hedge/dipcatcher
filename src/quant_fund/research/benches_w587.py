"""Wave-587 birational-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.abundance_conj import (
    bench_abundance_conj,
)
from quant_fund.models.bdd_fano import bench_bdd_fano
from quant_fund.models.canonical_sing2 import (
    bench_canonical_sing2,
)
from quant_fund.models.klt_mmp import bench_klt_mmp
from quant_fund.models.mmp_flip import bench_mmp_flip
from quant_fund.models.terminal_sing import (
    bench_terminal_sing,
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


def bench_terminal_sing_family(
    seed: int = _SEED + 3428,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "terminal_sing",
            bench_terminal_sing(seed),
        )
    )


def bench_canonical_sing2_family(
    seed: int = _SEED + 3429,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "canonical_sing2",
            bench_canonical_sing2(seed),
        )
    )


def bench_klt_mmp_family(seed: int = _SEED + 3430) -> dict[str, float]:
    return _floats(_finite_blob("klt_mmp", bench_klt_mmp(seed)))


def bench_mmp_flip_family(seed: int = _SEED + 3431) -> dict[str, float]:
    return _floats(_finite_blob("mmp_flip", bench_mmp_flip(seed)))


def bench_abundance_conj_family(
    seed: int = _SEED + 3432,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "abundance_conj",
            bench_abundance_conj(seed),
        )
    )


def bench_bdd_fano_family(seed: int = _SEED + 3433) -> dict[str, float]:
    return _floats(_finite_blob("bdd_fano", bench_bdd_fano(seed)))
