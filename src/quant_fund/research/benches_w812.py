"""Wave-812 regenerative-structure bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.epsilon_coupling import (
    bench_epsilon_coupling,
)
from quant_fund.models.nummelin import (
    bench_nummelin,
)
from quant_fund.models.petite_set import (
    bench_petite_set,
)
from quant_fund.models.regenerative import (
    bench_regenerative,
)
from quant_fund.models.small_set import (
    bench_small_set,
)
from quant_fund.models.split_chain import (
    bench_split_chain,
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


def bench_regenerative_family(
    seed: int = _SEED + 20000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "regenerative",
            bench_regenerative(seed),
        )
    )


def bench_epsilon_coupling_family(
    seed: int = _SEED + 20001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "epsilon_coupling",
            bench_epsilon_coupling(seed),
        )
    )


def bench_small_set_family(
    seed: int = _SEED + 20002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "small_set",
            bench_small_set(seed),
        )
    )


def bench_petite_set_family(
    seed: int = _SEED + 20003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "petite_set",
            bench_petite_set(seed),
        )
    )


def bench_split_chain_family(
    seed: int = _SEED + 20004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "split_chain",
            bench_split_chain(seed),
        )
    )


def bench_nummelin_family(
    seed: int = _SEED + 20005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "nummelin",
            bench_nummelin(seed),
        )
    )
