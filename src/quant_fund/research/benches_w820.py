"""Wave-820 semimartingale-decomp bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.canonical_decomp import (
    bench_canonical_decomp,
)
from quant_fund.models.doom_decomp import (
    bench_doom_decomp,
)
from quant_fund.models.pcdt import (
    bench_pcdt,
)
from quant_fund.models.sem_loc_char import (
    bench_sem_loc_char,
)
from quant_fund.models.special_sem import (
    bench_special_sem,
)
from quant_fund.models.triplet_char import (
    bench_triplet_char,
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


def bench_doom_decomp_family(
    seed: int = _SEED + 20800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "doom_decomp",
            bench_doom_decomp(seed),
        )
    )


def bench_pcdt_family(
    seed: int = _SEED + 20801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "pcdt",
            bench_pcdt(seed),
        )
    )


def bench_special_sem_family(
    seed: int = _SEED + 20802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "special_sem",
            bench_special_sem(seed),
        )
    )


def bench_canonical_decomp_family(
    seed: int = _SEED + 20803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "canonical_decomp",
            bench_canonical_decomp(seed),
        )
    )


def bench_sem_loc_char_family(
    seed: int = _SEED + 20804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "sem_loc_char",
            bench_sem_loc_char(seed),
        )
    )


def bench_triplet_char_family(
    seed: int = _SEED + 20805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "triplet_char",
            bench_triplet_char(seed),
        )
    )
