"""Wave-238 adapters: networking canon — TCP AIMD, sliding-window ARQ,
token bucket, Jacobson RTT, NAT table, HTTP/2 flow control — SYNTHETIC
correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.http2_flow import bench_http2_flow
from quant_fund.models.nat_table import bench_nat_table
from quant_fund.models.rtt_estimator import bench_rtt_estimator
from quant_fund.models.sliding_window import bench_sliding_window
from quant_fund.models.tcp_aimd import bench_tcp_aimd
from quant_fund.models.token_bucket import bench_token_bucket

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_http2_flow_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("http2_flow", bench_http2_flow(seed=_SEED + 1150)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"http2_flow bench failed: {exc}") from exc


def bench_nat_table_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nat_table", bench_nat_table(seed=_SEED + 1151)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nat_table bench failed: {exc}") from exc


def bench_rtt_estimator_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rtt_estimator", bench_rtt_estimator(seed=_SEED + 1152)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rtt_estimator bench failed: {exc}") from exc


def bench_sliding_window_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sliding_window", bench_sliding_window(seed=_SEED + 1153)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sliding_window bench failed: {exc}") from exc


def bench_tcp_aimd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tcp_aimd", bench_tcp_aimd(seed=_SEED + 1154)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tcp_aimd bench failed: {exc}") from exc


def bench_token_bucket_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("token_bucket", bench_token_bucket(seed=_SEED + 1155)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"token_bucket bench failed: {exc}") from exc
