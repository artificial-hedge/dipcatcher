"""Wave-286 security-defensive canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.beacon_detect import bench_beacon_detect
from quant_fund.models.cred_stuffing import bench_cred_stuffing
from quant_fund.models.entropy_dns import bench_entropy_dns
from quant_fund.models.exfil_zscore import bench_exfil_zscore
from quant_fund.models.impossible_travel import bench_impossible_travel
from quant_fund.models.sig_score import bench_sig_score

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


def bench_beacon_detect_family(seed: int = _SEED + 1622) -> dict[str, float]:
    return _floats(_finite_blob("beacon_detect", bench_beacon_detect(seed)))


def bench_entropy_dns_family(seed: int = _SEED + 1623) -> dict[str, float]:
    return _floats(_finite_blob("entropy_dns", bench_entropy_dns(seed)))


def bench_cred_stuffing_family(seed: int = _SEED + 1624) -> dict[str, float]:
    return _floats(_finite_blob("cred_stuffing", bench_cred_stuffing(seed)))


def bench_impossible_travel_family(seed: int = _SEED + 1625) -> dict[str, float]:
    return _floats(_finite_blob("impossible_travel", bench_impossible_travel(seed)))


def bench_exfil_zscore_family(seed: int = _SEED + 1626) -> dict[str, float]:
    return _floats(_finite_blob("exfil_zscore", bench_exfil_zscore(seed)))


def bench_sig_score_family(seed: int = _SEED + 1627) -> dict[str, float]:
    return _floats(_finite_blob("sig_score", bench_sig_score(seed)))
