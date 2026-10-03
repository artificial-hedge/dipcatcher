"""Wave-741 percolation-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aiten_chayes import bench_aiten_chayes
from quant_fund.models.beffara_nolin import bench_beffara_nolin
from quant_fund.models.gandre_liggett import bench_gandre_liggett
from quant_fund.models.hara_slade import bench_hara_slade
from quant_fund.models.heyman_redner import bench_heyman_redner
from quant_fund.models.newman_percolation import (
    bench_newman_percolation,
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


def bench_beffara_nolin_family(
    seed: int = _SEED + 13000,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "beffara_nolin",
            bench_beffara_nolin(seed),
        )
    )


def bench_hara_slade_family(
    seed: int = _SEED + 13001,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hara_slade",
            bench_hara_slade(seed),
        )
    )


def bench_gandre_liggett_family(
    seed: int = _SEED + 13002,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gandre_liggett",
            bench_gandre_liggett(seed),
        )
    )


def bench_heyman_redner_family(
    seed: int = _SEED + 13003,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "heyman_redner",
            bench_heyman_redner(seed),
        )
    )


def bench_aiten_chayes_family(
    seed: int = _SEED + 13004,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "aiten_chayes",
            bench_aiten_chayes(seed),
        )
    )


def bench_newman_percolation_family(
    seed: int = _SEED + 13005,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "newman_percolation",
            bench_newman_percolation(seed),
        )
    )
