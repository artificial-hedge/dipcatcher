"""Wave-729 motivic-A1-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.hauwas_nori import bench_hauwas_nori
from quant_fund.models.jogiad_motive import (
    bench_jogiad_motive,
)
from quant_fund.models.motivic_pipe import bench_motivic_pipe
from quant_fund.models.roald_suslin import (
    bench_roald_suslin,
)
from quant_fund.models.thom_mgl2 import bench_thom_mgl2
from quant_fund.models.voev_suslin import bench_voev_suslin

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


def bench_roald_suslin_family(
    seed: int = _SEED + 11800,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "roald_suslin",
            bench_roald_suslin(seed),
        )
    )


def bench_jogiad_motive_family(
    seed: int = _SEED + 11801,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "jogiad_motive",
            bench_jogiad_motive(seed),
        )
    )


def bench_hauwas_nori_family(
    seed: int = _SEED + 11802,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hauwas_nori",
            bench_hauwas_nori(seed),
        )
    )


def bench_motivic_pipe_family(
    seed: int = _SEED + 11803,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "motivic_pipe",
            bench_motivic_pipe(seed),
        )
    )


def bench_thom_mgl2_family(
    seed: int = _SEED + 11804,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thom_mgl2",
            bench_thom_mgl2(seed),
        )
    )


def bench_voev_suslin_family(
    seed: int = _SEED + 11805,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "voev_suslin",
            bench_voev_suslin(seed),
        )
    )
