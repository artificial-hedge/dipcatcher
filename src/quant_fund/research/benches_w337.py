"""Wave-337 lambda-calculus/rewriting canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.church_encoding import bench_church_encoding
from quant_fund.models.de_bruijn import bench_de_bruijn
from quant_fund.models.knuth_bendix import bench_knuth_bendix
from quant_fund.models.lambda_typing import bench_lambda_typing
from quant_fund.models.ski_combinator import bench_ski_combinator
from quant_fund.models.unification import bench_unification

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


def bench_ski_combinator_family(seed: int = _SEED + 1929) -> dict[str, float]:
    return _floats(_finite_blob("ski_combinator", bench_ski_combinator(seed)))


def bench_de_bruijn_family(seed: int = _SEED + 1930) -> dict[str, float]:
    return _floats(_finite_blob("de_bruijn", bench_de_bruijn(seed)))


def bench_church_encoding_family(seed: int = _SEED + 1931) -> dict[str, float]:
    return _floats(_finite_blob("church_encoding", bench_church_encoding(seed)))


def bench_lambda_typing_family(seed: int = _SEED + 1932) -> dict[str, float]:
    return _floats(_finite_blob("lambda_typing", bench_lambda_typing(seed)))


def bench_unification_family(seed: int = _SEED + 1933) -> dict[str, float]:
    return _floats(_finite_blob("unification", bench_unification(seed)))


def bench_knuth_bendix_family(seed: int = _SEED + 1934) -> dict[str, float]:
    return _floats(_finite_blob("knuth_bendix", bench_knuth_bendix(seed)))
