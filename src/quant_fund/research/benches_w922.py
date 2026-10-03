"""Wave-922 distributed-systems-6 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.abcast_lite import bench_abcast_lite
from quant_fund.models.cap_theorem import bench_cap_theorem
from quant_fund.models.cbc_bcast import bench_cbc_bcast
from quant_fund.models.lake_wisc import bench_lake_wisc
from quant_fund.models.slush_consensus import bench_slush_consensus
from quant_fund.models.snowflake_consensus import bench_snowflake_consensus

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_abcast_lite_family(seed: int = _SEED + 31000) -> dict[str, float]:
    return _finite_blob(bench_abcast_lite(seed))


def bench_cbc_bcast_family(seed: int = _SEED + 31001) -> dict[str, float]:
    return _finite_blob(bench_cbc_bcast(seed))


def bench_slush_consensus_family(seed: int = _SEED + 31002) -> dict[str, float]:
    return _finite_blob(bench_slush_consensus(seed))


def bench_snowflake_consensus_family(seed: int = _SEED + 31003) -> dict[str, float]:
    return _finite_blob(bench_snowflake_consensus(seed))


def bench_cap_theorem_family(seed: int = _SEED + 31004) -> dict[str, float]:
    return _finite_blob(bench_cap_theorem(seed))


def bench_lake_wisc_family(seed: int = _SEED + 31005) -> dict[str, float]:
    return _finite_blob(bench_lake_wisc(seed))
