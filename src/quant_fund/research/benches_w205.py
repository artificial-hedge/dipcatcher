"""Wave-203 adapters: information-geometry canon — vickrey_auction,
first_price_auction, all_pay_auction, ascending_clock, double_auction, gsp_auction —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.all_pay_auction import bench_all_pay_auction
from quant_fund.models.ascending_clock import bench_ascending_clock
from quant_fund.models.double_auction import bench_double_auction
from quant_fund.models.first_price_auction import bench_first_price_auction
from quant_fund.models.gsp_auction import bench_gsp_auction
from quant_fund.models.vickrey_auction import bench_vickrey_auction

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


def bench_double_auction_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("double_auction", bench_double_auction(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"double_auction bench failed: {exc}") from exc


def bench_vickrey_auction_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vickrey_auction", bench_vickrey_auction(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vickrey_auction bench failed: {exc}") from exc


def bench_ascending_clock_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ascending_clock", bench_ascending_clock(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ascending_clock bench failed: {exc}") from exc


def bench_first_price_auction_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("first_price_auction", bench_first_price_auction(seed=_SEED + 963))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"first_price_auction bench failed: {exc}") from exc


def bench_all_pay_auction_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("all_pay_auction", bench_all_pay_auction(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"all_pay_auction bench failed: {exc}") from exc


def bench_gsp_auction_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gsp_auction", bench_gsp_auction(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gsp_auction bench failed: {exc}") from exc
