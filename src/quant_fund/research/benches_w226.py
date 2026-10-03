"""Wave-226 adapters: compiler/formal-language canon — regex_engine,
dfa_minimize, cyk_parser, dominance_tree, liveness_dce, linscan_regalloc —
benched on SYNTHETIC grammars/CFGs/instruction lists. Adapters flatten to a
finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cyk_parser import bench_cyk_parser
from quant_fund.models.dfa_minimize import bench_dfa_minimize
from quant_fund.models.dominance_tree import bench_dominance_tree
from quant_fund.models.linscan_regalloc import bench_linscan_regalloc
from quant_fund.models.liveness_dce import bench_liveness_dce
from quant_fund.models.regex_engine import bench_regex_engine

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


def bench_cyk_parser_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cyk_parser", bench_cyk_parser(seed=_SEED + 1030)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cyk_parser bench failed: {exc}") from exc


def bench_dfa_minimize_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dfa_minimize", bench_dfa_minimize(seed=_SEED + 1031)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dfa_minimize bench failed: {exc}") from exc


def bench_dominance_tree_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dominance_tree", bench_dominance_tree(seed=_SEED + 1032)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dominance_tree bench failed: {exc}") from exc


def bench_linscan_regalloc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("linscan_regalloc", bench_linscan_regalloc(seed=_SEED + 1033)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"linscan_regalloc bench failed: {exc}") from exc


def bench_liveness_dce_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("liveness_dce", bench_liveness_dce(seed=_SEED + 1034)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"liveness_dce bench failed: {exc}") from exc


def bench_regex_engine_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("regex_engine", bench_regex_engine(seed=_SEED + 1035)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"regex_engine bench failed: {exc}") from exc
