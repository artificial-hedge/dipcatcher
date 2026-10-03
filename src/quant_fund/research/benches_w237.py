"""Wave-237 adapters: parser canon — recursive descent, Pratt, Earley,
SLR(1), PEG packrat, LL(1) table — SYNTHETIC correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.earley_parser import bench_earley_parser
from quant_fund.models.ll1_table import bench_ll1_table
from quant_fund.models.peg_packrat import bench_peg_packrat
from quant_fund.models.pratt_parser import bench_pratt_parser
from quant_fund.models.recursive_descent import bench_recursive_descent
from quant_fund.models.slr_parser import bench_slr_parser

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


def bench_earley_parser_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("earley_parser", bench_earley_parser(seed=_SEED + 1140)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"earley_parser bench failed: {exc}") from exc


def bench_ll1_table_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ll1_table", bench_ll1_table(seed=_SEED + 1141)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ll1_table bench failed: {exc}") from exc


def bench_peg_packrat_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("peg_packrat", bench_peg_packrat(seed=_SEED + 1142)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"peg_packrat bench failed: {exc}") from exc


def bench_pratt_parser_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pratt_parser", bench_pratt_parser(seed=_SEED + 1143)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pratt_parser bench failed: {exc}") from exc


def bench_recursive_descent_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("recursive_descent", bench_recursive_descent(seed=_SEED + 1144))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"recursive_descent bench failed: {exc}") from exc


def bench_slr_parser_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("slr_parser", bench_slr_parser(seed=_SEED + 1145)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"slr_parser bench failed: {exc}") from exc
